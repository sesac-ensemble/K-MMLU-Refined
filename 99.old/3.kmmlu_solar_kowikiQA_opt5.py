#!/usr/bin/env python
# -*- coding: utf-8 -*-
# run_kmmlu_solar_kowikiQA_opt4.py
"""
KoWikiQA SFT + KMMLU 평가 (오류 수정 + 안정 버전)
- 정확도 계산: all_correct / all_total
- KMMLU 45개 subset 및 supercategory(4종) 완전 매핑
- Few-shot 방식 선택: --use_manual_fewshots (기본=False → dev에서 랜덤 5개)
- KoWikiQA 학습 데이터: 모든 train split 명시적 병합 + --max_train_samples로 제한
- Robust: tokenizer/FlashAttention/bfloat16/데이터 필드/토큰 인덱싱 안전 처리
"""

import os
import re
import time
import random
import argparse
from datetime import datetime

import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig


# =========================================================
# 수동 5-shot 예시 (옵션으로 사용)
# =========================================================
MANUAL_FEWSHOTS = [
    {
        "question": "반도체에서 전자가 이동하는 주요 경로는 무엇인가?",
        "A": "원자핵",
        "B": "전도대",
        "C": "가전자대",
        "D": "절연층",
        "answer": "2",
    },
    {
        "question": "형법상 '고의'의 의미는 무엇인가?",
        "A": "결과 발생을 예견하지 못함",
        "B": "결과 발생을 원하거나 용인함",
        "C": "단순한 과실",
        "D": "의무 위반에 대한 무지",
        "answer": "2",
    },
    {
        "question": "패션 트렌드가 순환적으로 반복되는 현상을 무엇이라 하는가?",
        "A": "패션 사이클",
        "B": "패션 모듈",
        "C": "패션 블렌드",
        "D": "패션 세그먼트",
        "answer": "1",
    },
    {
        "question": "재난 대응 단계 중 현장지휘체계를 구성하는 주체는?",
        "A": "중앙정부",
        "B": "지방자치단체장",
        "C": "소방본부장",
        "D": "경찰청장",
        "answer": "2",
    },
    {
        "question": "세포 내 에너지 생성의 주요 기관은?",
        "A": "핵",
        "B": "미토콘드리아",
        "C": "리보솜",
        "D": "소포체",
        "answer": "2",
    },
]


# =========================================================
# 모델 및 LoRA 유틸
# =========================================================
def _is_bf16_supported() -> bool:
    return (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )


def load_unsloth_model(
    model_id: str, max_seq_length: int = 4096, use_flash_attn2: bool = True
):
    print(f"\n{'='*60}\n모델 로딩 중: {model_id}\n{'='*60}\n")
    dtype = torch.bfloat16 if _is_bf16_supported() else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
    )
    # FlashAttention2 (미지원 환경 graceful fallback)
    try:
        FastLanguageModel.for_inference(model, use_flash_attention_2=use_flash_attn2)
        print("FlashAttention2 활성화됨")
    except Exception:
        print("FlashAttention2 미지원 환경 — 기본 모드로 로드")

    # 토크나이저 패딩/토큰 세이프가드
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass
    if (
        getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is None
    ):
        # 어떤 토크나이저는 eos/pad 모두 없음 → 임시 세팅
        tokenizer.add_special_tokens({"eos_token": "</s>", "pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    if (
        getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    elif getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = "<pad>"

    model.eval()
    return model, tokenizer


def inject_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM"
    )
    model = FastLanguageModel.get_peft_model(model, cfg)
    print(f"LoRA 주입 완료 (r={r}, α={alpha}, dropout={dropout})")
    return model


# =========================================================
# 메인 파이프라인
# =========================================================
class KoWikiQA2KMMLU:
    def __init__(
        self,
        model_id: str,
        dataset_id: str,
        batch_size: int = 2,  # 이전 버전까지 4-> 메모리 이슈로 변경. 필요시 4, 8 배열로 수정 가능
        seed: int = 42,
        use_flash_attn2: bool = True,
        use_manual_fewshots: bool = False,  # 기본값: 벤치마크 일반 관행(랜덤 dev 5-shot)
        max_train_samples: int = 10000,
        num_shots: int = 5,
        output_dir: str = "results",
    ):
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.batch_size = batch_size
        self.seed = seed
        self.use_flash_attn2 = use_flash_attn2
        self.use_manual_fewshots = use_manual_fewshots
        self.max_train_samples = max_train_samples
        self.num_shots = max(1, int(num_shots))
        self.output_dir = output_dir
        # prompting_strategy 자동 지정 (manual vs dev_sample)
        self.prompting_strategy = "manual" if use_manual_fewshots else "dev_sample"

        # Random seed 고정
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # 모델 및 토크나이저 로드
        self.model, self.tokenizer = load_unsloth_model(
            model_id, use_flash_attn2=use_flash_attn2
        )
        self.subsets = self._get_official_subsets()
        self.supercats = self._get_supercategories()

        # 숫자/문자 정답 모두 대응 (안전 매핑)
        self.letter_map: Dict[str, int] = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "1": 0,
            "2": 1,
            "3": 2,
            "4": 3,
        }

    # -----------------------------
    # SFT 학습
    # -----------------------------
    def train_on_kowikiQA(self):
        print("\nKoWikiQA 데이터셋 로딩 중...")
        ds = load_dataset(self.dataset_id)

        # 모든 train split 명시적으로 병합
        trains = []
        for k in ds.keys():
            if "train" in k.lower():
                trains.append(ds[k])
        if not trains:
            # 예외적으로 split 이름이 train이 아닌 경우 전체 병합
            trains = [ds[k] for k in ds.keys()]
        merged = concatenate_datasets(trains) if len(trains) > 1 else trains[0]

        # 개수 제한
        use_n = min(self.max_train_samples, len(merged))
        train_ds = merged.select(range(use_n))
        print(f"학습 데이터 개수: {len(train_ds)} 사용")

        # 포맷 변환 (필드 안전 접근)
        def fmt(ex):
            q = ex.get("instruction", "")
            a = ex.get("output", "")
            return {"text": f"질문: {q}\n답변: {a}"}

        train_ds = train_ds.map(
            fmt, remove_columns=[c for c in train_ds.column_names if c != "text"]
        )

        # LoRA 주입
        self.model = inject_lora(self.model)

        args = TrainingArguments(
            output_dir="results/lora_kowikiQA",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            warmup_ratio=0.03,
            logging_steps=10,
            save_steps=1000,
            report_to="none",
        )
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_ds,
            args=args,
        )
        print("\nSFT 학습 시작...")
        trainer.train()
        os.makedirs("results/lora_kowikiQA", exist_ok=True)
        trainer.save_model("results/lora_kowikiQA")
        print("SFT 학습 완료 및 LoRA 가중치 저장 완료.")

    # -----------------------------
    # Few-shot 생성
    # -----------------------------
    def _get_fewshots(self, subset: str):
        """--use_manual_fewshots가 False면 dev/validation에서 5개 랜덤 사용"""
        if self.use_manual_fewshots:
            return MANUAL_FEWSHOTS

        try:
            ds = load_dataset("HAERAE-HUB/KMMLU", subset)
            candidates = None
            for sp in ["dev", "validation", "test", "train"]:
                if sp in ds:
                    candidates = list(ds[sp])
                    break
            if not candidates:
                return MANUAL_FEWSHOTS

            # 정답이 1~4로 존재하는 것만 후보
            valid = [
                ex
                for ex in candidates
                if ex.get("answer") in [1, 2, 3, 4, "1", "2", "3", "4"]
            ]
            if not valid:
                return MANUAL_FEWSHOTS

            k = min(5, len(valid))
            return random.sample(valid, k)
        except Exception as e:
            print(f"Few-shot 생성 실패({subset}): {e}")
            return MANUAL_FEWSHOTS

    # -----------------------------
    # 프롬프트 구성
    # -----------------------------
    @staticmethod
    def _norm(t):
        return re.sub(r"\s+", " ", str(t)).strip()

    def _make_prompt(self, ex, fewshots):
        prompt = ""
        for fs in fewshots:
            prompt += f"문제: {self._norm(fs.get('question',''))}\n"
            for c in ["A", "B", "C", "D"]:
                prompt += f"{c}. {self._norm(fs.get(c,''))}\n"
            prompt += f"정답: {fs.get('answer','')}\n\n"
        prompt += f"문제: {self._norm(ex.get('question',''))}\n"
        for c in ["A", "B", "C", "D"]:
            prompt += f"{c}. {self._norm(ex.get(c,''))}\n"
        prompt += "정답: "
        return prompt

    # -----------------------------
    # 정답 처리 (KMMLU: 1/2/3/4 → 0~3)
    # -----------------------------
    @staticmethod
    def _extract_answer_index(ex):
        ans = ex.get("answer")
        if isinstance(ans, int):
            return ans - 1 if 1 <= ans <= 4 else None
        if isinstance(ans, str) and ans.isdigit():
            n = int(ans)
            return n - 1 if 1 <= n <= 4 else None
        return None

    # -----------------------------
    # 마지막 토큰 로짓에서 ABCD 첫 토큰만 비교
    # -----------------------------
    def _predict_logits(self, inputs):
        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :]
        choice_ids = []
        for ch in ["A", "B", "C", "D"]:
            toks = self.tokenizer.encode(ch, add_special_tokens=False)
            choice_ids.append(toks[0] if toks else 0)
        preds = torch.argmax(logits[:, choice_ids], dim=-1)
        return preds.cpu().tolist()

    # -----------------------------
    # KMMLU 평가
    # -----------------------------
    def evaluate_kmmlu(self):
        results = []
        all_correct = 0
        all_total = 0
        start = time.time()

        for subset in tqdm(self.subsets, desc="KMMLU 전체 평가"):
            try:
                dataset = load_dataset("HAERAE-HUB/KMMLU", subset)

                test = None
                for split in ["test", "validation", "dev", "train"]:
                    if split in dataset:
                        test = list(dataset[split])
                        break
                if not test:
                    print(f"{subset}: 평가 가능한 split 없음 → skip")
                    continue

                # 유효 정답만 필터
                test = [
                    ex
                    for ex in test
                    if ex.get("answer") in [1, 2, 3, 4, "1", "2", "3", "4"]
                ]
                if not test:
                    print(f"{subset}: 유효 정답 없음 → skip")
                    continue

                fewshots = self._get_fewshots(subset)

                prompts = [self._make_prompt(t, fewshots) for t in test]
                truths = [self._extract_answer_index(t) for t in test]
                correct = total = 0

                for i in range(0, len(prompts), self.batch_size):
                    batch = prompts[i : i + self.batch_size]
                    truth = truths[i : i + self.batch_size]
                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=3072,
                    ).to(self.model.device)
                    preds = self._predict_logits(inputs)
                    for p, t in zip(preds, truth):
                        if t is not None:
                            total += 1
                            if p == t:
                                correct += 1

                acc = correct / total if total else 0.0
                print(f"{subset}: {acc:.4f} ({correct}/{total})")
                results.append(
                    {
                        "Subset": subset,
                        "SuperCategory": self.supercats.get(subset, "Unknown"),
                        "Accuracy": acc,
                    }
                )
                all_correct += correct
                all_total += total

            except Exception as e:
                print(f"{subset} 오류: {e}")
                import traceback

                traceback.print_exc()
                results.append(
                    {
                        "Subset": subset,
                        "SuperCategory": self.supercats.get(subset, "Unknown"),
                        "Accuracy": 0.0,
                    }
                )

        # 저장 및 요약
        os.makedirs("results", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = f"results/kowikiQA2KMMLU_opt4_{ts}.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")

        overall_acc = (all_correct / all_total) if all_total else 0.0
        print(f"\n전체 평균: {overall_acc:.4f} ({all_correct}/{all_total})")
        print(f"CSV 저장 완료 → {out_csv}")
        print(f"총 경과시간: {(time.time()-start)/60:.1f}분")

        # SuperCategory별 평균도 함께 출력
        try:
            df = pd.DataFrame(results)
            sc_df = (
                df.groupby("SuperCategory", as_index=False)["Accuracy"]
                .mean()
                .sort_values("Accuracy", ascending=False)
            )
            out_csv_sc = f"results/kowikiQA2KMMLU_opt4_supercats_{ts}.csv"
            sc_df.to_csv(out_csv_sc, index=False, encoding="utf-8-sig")
            print("\n[SuperCategory 평균 정확도]")
            print(sc_df.to_string(index=False))
            print(f"SuperCategory CSV 저장 → {out_csv_sc}")
        except Exception as e:
            print(f"SuperCategory 평균 계산 실패: {e}")

    # =====================================================
    # KMMLU Subsets & SuperCategories
    # =====================================================
    @staticmethod
    def _get_official_subsets():
        # 공식 45개 구성
        return [
            "Accounting",
            "Agricultural-Sciences",
            "Aviation-Engineering-and-Maintenance",
            "Biology",
            "Chemical-Engineering",
            "Chemistry",
            "Civil-Engineering",
            "Computer-Science",
            "Construction",
            "Criminal-Law",
            "Ecology",
            "Economics",
            "Education",
            "Electrical-Engineering",
            "Electronics-Engineering",
            "Energy-Management",
            "Environmental-Science",
            "Fashion",
            "Food-Processing",
            "Gas-Technology-and-Engineering",
            "Geomatics",
            "Health",
            "Industrial-Engineer",
            "Information-Technology",
            "Interior-Architecture-and-Design",
            "Law",
            "Machine-Design-and-Manufacturing",
            "Management",
            "Maritime-Engineering",
            "Marketing",
            "Materials-Engineering",
            "Mechanical-Engineering",
            "Nondestructive-Testing",
            "Patent",
            "Political-Science-and-Sociology",
            "Psychology",
            "Public-Safety",
            "Railway-and-Automotive-Engineering",
            "Real-Estate",
            "Refrigerating-Machinery",
            "Social-Welfare",
            "Taxation",
            "Telecommunications-and-Wireless-Technology",
            "Korean-History",
            "Math",
        ]

    @staticmethod
    def _get_supercategories():
        cats = {
            "STEM": [
                "Biology",
                "Chemical-Engineering",
                "Chemistry",
                "Civil-Engineering",
                "Computer-Science",
                "Ecology",
                "Electrical-Engineering",
                "Information-Technology",
                "Materials-Engineering",
                "Mechanical-Engineering",
                "Math",
            ],
            "HUMSS": [
                "Accounting",
                "Criminal-Law",
                "Economics",
                "Education",
                "Law",
                "Management",
                "Political-Science-and-Sociology",
                "Psychology",
                "Social-Welfare",
                "Taxation",
                "Korean-History",
            ],
            "Applied Science": [
                "Aviation-Engineering-and-Maintenance",
                "Electronics-Engineering",
                "Energy-Management",
                "Environmental-Science",
                "Gas-Technology-and-Engineering",
                "Geomatics",
                "Industrial-Engineer",
                "Machine-Design-and-Manufacturing",
                "Maritime-Engineering",
                "Nondestructive-Testing",
                "Railway-and-Automotive-Engineering",
                "Telecommunications-and-Wireless-Technology",
            ],
            "Other": [
                "Agricultural-Sciences",
                "Construction",
                "Fashion",
                "Food-Processing",
                "Health",
                "Interior-Architecture-and-Design",
                "Marketing",
                "Patent",
                "Public-Safety",
                "Real-Estate",
                "Refrigerating-Machinery",
            ],
        }
        mapping = {}
        for cat, subs in cats.items():
            for s in subs:
                mapping[s] = cat
        return mapping


# =========================================================
# Main
# =========================================================
def main():
    """KMMLU 평가 실행 진입점"""
    parser = argparse.ArgumentParser(description="KMMLU 평가")
    parser.add_argument(
        "--model_id",
        type=str,
        default="upstage/SOLAR-10.7B-Instruct-v1.0",
        help="평가할 HuggingFace 모델 ID",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="배치 크기 (GPU 메모리에 따라 조정)"
    )  # 메모리 부족시 2, 메모리 충분 시 8
    parser.add_argument("--seed", type=int, default=42, help="Random seed (재현성)")
    parser.add_argument(
        "--num_shots",
        type=int,
        default=5,
        help="Few-shot 예시 개수 (0=zero-shot, 5=5-shot)",
    )
    parser.add_argument(
        "--prompting_strategy",
        type=str,
        default="random",
        choices=[
            "random",
            "cot",
            "similarity",
            "meta_prompt",
            "gradient",
            "zero_shot",
            "self_consistency",
        ],
        help="프롬프트 전략",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="출력 파일명 prefix (기본: 모델명_타임스탬프)",
    )
    parser.add_argument(
        "--test_subsets",
        type=str,
        default=None,
        help="콤마로 구분된 서브셋 이름 리스트 (예: humanities,STEM)",
    )
    parser.add_argument(
        "--use_manual_fewshots",
        action="store_true",
        help="manual few-shot prompt 사용 여부 (기본값: False)",
    )

    args = parser.parse_args()

    test_subsets = args.test_subsets.split(",") if args.test_subsets else None

    evaluator = KoWikiQA2KMMLU(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        batch_size=args.batch_size,
        seed=args.seed,
        test_subsets=test_subsets,
        use_flash_attn2=True,
        use_manual_fewshots=args.use_manual_fewshots,
        max_train_samples=args.max_train_samples,
        num_shots=args.num_shots,
        output_dir=args.output_dir,
    )

    evaluator = KoWikiQA2KMMLU(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        batch_size=args.batch_size,
        seed=args.seed,
        num_shots=args.num_shots,
        prompting_strategy=args.prompting_strategy,
        output_prefix=args.output_prefix,
        test_subsets=test_subsets,
        use_flash_attn2=True,
        use_manual_fewshots=args.use_manual_fewshots,
        max_train_samples=args.max_train_samples,
        num_shots=args.num_shots,
        output_dir=args.output_dir,
    )

    evaluator.evaluate()


if __name__ == "__main__":
    main()


# 학습 + 평가 (랜덤 dev 5-shot, 10K 샘플 학습)
# python run_kmmlu_solar_kowikiQA_opt4.py \
#   --model_id upstage/SOLAR-10.7B-Instruct-v1.0 \
#   --dataset_id maywell/ko_wikidata_QA \
#   --max_train_samples 10000

# 평가만 (수동 few-shot 사용)
# python run_kmmlu_solar_kowikiQA_opt4.py --eval_only --use_manual_fewshots
