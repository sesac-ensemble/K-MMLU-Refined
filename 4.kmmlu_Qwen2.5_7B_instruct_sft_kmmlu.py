#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 4.kmmlu_Qwen2.5_7B_instruct_5shot_sft.py

# -------------------------------------------------------------
# -  dataset+ KMMLU Few-shot 평가 통합 스크립트
# - Unsloth FastLanguageModel + LoRA + 4bit 양자화 지원
# - 평가 정확도: all_correct / all_total (안전 분모 체크)
# - KMMLU 45개 subset / 4개 supercategory 매핑
# - Few-shot: --use_manual_fewshots (기본 False → subset의 dev에서 5개 랜덤)
# - 학습: 모든 train split 병합 + --max_train_samples 제한
# -
# -------------------------------------------------------------

import os
import re
import json
import random
import argparse
from datetime import datetime
from typing import List, Dict, Any

import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
from unsloth import FastLanguageModel


class dataset_KMMLU:
    """데이터셋 + KMMLU 통합 Evaluator
    기능:
      - (옵션) KoWikiQA , KMMLU 데이터셋으로 LoRA 기반 SFT 수행
      - KMMLU 45개 subset Few-shot 평가 (5-shot)
      - (default)supercategory별 평균/전체 평균 계산, CSV/JSON 저장
      - --eval_subsets humss 사용 시 11개의 HUMSS subset 자동 설정
    """

    HUMSS_SUBSETS = [
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
    ]

    def __init__(
        self,
        model_id: str,
        dataset_id: str,
        batch_size: int = 2,
        seed: int = 42,
        use_flash_attn2: bool = True,
        use_manual_fewshots: bool = False,
        max_train_samples: int = 10000,
        num_shots: int = 5,
        output_dir: str = "results",
        output_prefix: str = None,
        eval_subsets: list = None,
    ):
        """모델/데이터 환경 초기화 및 로더 준비"""
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.batch_size = batch_size
        self.seed = seed
        self.use_flash_attn2 = use_flash_attn2
        self.use_manual_fewshots = use_manual_fewshots
        self.max_train_samples = max_train_samples
        self.num_shots = max(1, int(num_shots))
        self.output_dir = output_dir
        self.output_prefix = output_prefix or self._generate_output_prefix()
        self.prompting_strategy = "manual" if self.use_manual_fewshots else "random"

        # KMMLU 평가용 subset / 상위카테고리
        # self.subsets = self._get_official_subsets()

        # 평가 대상 subset 지정
        if eval_subsets == ["humss"]:
            self.subsets = self.HUMSS_SUBSETS
        else:
            self.subsets = eval_subsets or self._get_official_subsets()

        # KMMLU의 상위 카테고리 매핑
        self.supercategories = self._get_supercategories()

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 모델/토크나이저 로드 + 속도 최적화
        self.model, self.tokenizer = self._load_model()

        # LoRA 가중치 자동 로딩: 평가 전용 경로인 경우 적용 
        # 평가 실행 시, 이미 LoRA 주입(FastLanguageModel.get_peft_model)이 된 모델에 다시 PeftModel.from_pretrained()를 호출하기 때문에 중복 적용 발생으로 주석처리
        # adapter_config_path = os.path.join(self.model_id, "adapter_config.json")
        #
        # if os.path.exists(adapter_config_path):
        #     try:
        #         self.model = PeftModel.from_pretrained(self.model, self.model_id)
        #         print(f"LoRA 가중치 적용 완료: {self.model_id}")
        #     except Exception as e:
        #         print(f"LoRA 가중치 로딩 실패: {e}")

        # LoRA 가중치 자동 로딩 (중복 방지 포함)
        adapter_config_path = os.path.join(self.model_id, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            if hasattr(self.model, "peft_config"):
                print("이미 LoRA 어댑터가 적용된 모델입니다. 중복 로딩을 건너뜁니다.")
            else:
                try:
                    self.model = PeftModel.from_pretrained(self.model, self.model_id)
                    print(f"LoRA 가중치 적용 완료: {self.model_id}")
                except Exception as e:
                    print(f"LoRA 가중치 로딩 실패: {e}")

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
    def _generate_output_prefix(self):
        """출력 파일명용 prefix 생성"""
        model_name = self.model_id.split("/")[-1]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_{ts}"

    # -----------------------------
    def _load_model(self):
        """모델 및 토크나이저 로드 (Unsloth FastLanguageModel + 4bit + FlashAttn2)"""
        print(f"\n{'='*60}\n모델 로딩 중: {self.model_id}\n{'='*60}\n")

        # bf16 지원 시 우선 적용, 미지원 시 fp16
        try:
            bf16_ok = bool(
                torch.cuda.is_available()
                and hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            )
        except Exception:
            bf16_ok = False
        # dtype = torch.bfloat16 if bf16_ok else torch.float16
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=4096,
            dtype=dtype,
            load_in_4bit=True,
        )

        # FlashAttention2 활성화 시도 (미지원 환경은 조용히 fallback)
        try:
            FastLanguageModel.for_inference(
                model, use_flash_attention_2=self.use_flash_attn2
            )
            print("FlashAttention2 활성화됨")
        except Exception:
            print("FlashAttention2 미지원 환경 — 기본 모드로 로드")

        # padding 토큰 안전 설정
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or "<pad>"
        tokenizer.padding_side = "left"

        model.eval()
        return model, tokenizer

    # -----------------------------
    def train_on_kmmlu(self):
        """데이터셋 기반 SFT (LoRA) 학습"""
        print("\n데이터셋 로딩 중...")
        # 전체 데이터셋 로딩
        # ds = load_dataset(self.dataset_id)

        # 모든 train split 병합 (향후 train-1, train-2 등의 분할에도 대응)
        # trains = [ds[k] for k in ds.keys() if "train" in k.lower()] or [
        #     ds[k] for k in ds.keys()
        # merged = concatenate_datasets(trains) if len(trains) > 1 else trains[0]
        humss_subsets = self.HUMSS_SUBSETS

        datasets = []
        for subset in humss_subsets:
            print(f"  - {subset} loading...")
            try:
                ds = load_dataset(self.dataset_id, subset)
                if "train" in ds:
                    datasets.append(ds["train"])
                else:
                    print(f"{subset}에 train split 없음 → skip")
            except Exception as e:
                print(f"{subset} 로딩 실패: {e}")

        if not datasets:
            print("학습 가능한 subset이 없습니다. 데이터를 확인해주세요.")
            return

        # HUMSS subset만 데이터셋으로 합치기
        merged = concatenate_datasets(datasets)

        # use_n = min(self.max_train_samples, len(merged))
        # self.max_train_samples가 None이면 전체 데이터 사용
        use_n = (
            len(merged)
            if self.max_train_samples is None
            else min(self.max_train_samples, len(merged))
        )
        train_ds = merged.select(range(use_n))
        print(f"학습 데이터 개수: {len(train_ds)} 사용")

        # kowikiQA 데이터셋 기준: instruction / output → text 필드로 포맷팅
        # def to_text(ex):
        #     q = str(ex.get("instruction", "")).strip()
        #     a = str(ex.get("output", "")).strip()
        #     return {"text": f"질문: {q}\n답변: {a}"}

        # KMMLU데이터셋 기반 훈련 시 --------
        def to_text(ex):
            q = str(ex.get("question", "")).strip()
            choices = "\n".join(
                f"{opt}. {str(ex.get(opt, '')).strip()}" for opt in ["A", "B", "C", "D"]
            )
            # int가 아닌 문자열로 감싸줌.
            a_idx = str(ex.get("answer", "")).strip()
            try:
                a_idx = int(a_idx) - 1
                answer = ["A", "B", "C", "D"][a_idx]
            except:
                answer = ""
            return {"text": f"문제: {q}\n{choices}\n정답: {answer}"}

        # ------------------------------------
        keep_cols = ["text"]
        train_ds = train_ds.map(
            to_text,
            remove_columns=[c for c in train_ds.column_names if c not in keep_cols],
            batched=False,  # 또는 생략 (기본값 False지만 명시 추천)
        )

        # LoRA 설정 (경량 파인튜닝)
        self.model = FastLanguageModel.get_peft_model(
            model=self.model,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            # task_type="CAUSAL_LM"
        )
        print("LoRA 추가 완료")
        # lora_cfg = LoraConfig(
        #     r=int(8),
        #     lora_alpha=16,
        #     lora_dropout=0.05,
        #     bias="none",
        #     task_type="CAUSAL_LM",
        # )
        # self.model = FastLanguageModel.get_peft_model(self.model, lora_cfg)
        # print("LoRA 추가 완료")

        # 학습 설정
        args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, "lora_kmmlu"),
            num_train_epochs=5,  # 기본 1에서 변경 가능
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            warmup_ratio=0.03,
            logging_steps=10,
            save_steps=1000,
            report_to="none",
            fp16=False,  # 추가
            bf16=True,  # 추가
            optim="adamw_torch_fused",  # 또는 adamw_hf
        )

        # SFTTrainer는 text 단일 필드를 사용하여 SFT 수행
        trainer = SFTTrainer(
            model=self.model,
            # tokenizer=self.tokenizer,
            train_dataset=train_ds,
            args=args,
        )

        print("\nSFT 학습 시작...")
        trainer.train()

        # 모델 저장 (모델명 기반 하위 폴더 생성)
        save_path = os.path.join(
            self.output_dir, f"lora_ft_{self.model_id.split('/')[-1]}"
        )
        os.makedirs(save_path, exist_ok=True)
        trainer.save_model(save_path)
        print(f"SFT 학습 완료 및 모델 저장: {save_path}")

        # trainer.save_model(os.path.join(self.output_dir, "lora_kmmlu"))
        # print("SFT 학습 완료 및 LoRA 가중치 저장 완료")

    # -----------------------------
    def _normalize(self, text: Any) -> str:
        """문자열 공백 정규화"""
        return re.sub(r"\s+", " ", str(text)).strip()

    # -----------------------------
    def _make_prompt(
        self, test_ex: Dict[str, Any], fewshots: List[Dict[str, Any]]
    ) -> str:
        """Few-shot 예시와 테스트 문항 결합

        - fewshots: 정답 포함
        - test_ex: 정답 비포함 (모델 예측 대상)
        """
        prompt = ""
        for fs in fewshots:
            prompt += f"문제: {self._normalize(fs.get('question', ''))}\n"
            for c in ["A", "B", "C", "D"]:
                prompt += f"{c}. {self._normalize(fs.get(c, ''))}\n"
            prompt += f"정답: {self._normalize(fs.get('answer', ''))}\n\n"

        prompt += f"문제: {self._normalize(test_ex.get('question', ''))}\n"
        for c in ["A", "B", "C", "D"]:
            prompt += f"{c}. {self._normalize(test_ex.get(c, ''))}\n"
        prompt += "정답: "
        return prompt

    # -----------------------------
    def _extract_answer_index(self, ex: Dict[str, Any]) -> int:
        """문항의 정답을 0~3 인덱스로 변환 (문자/숫자 모두 허용)"""
        ans = str(ex.get("answer", "")).strip().upper()
        if ans in self.letter_map:
            return self.letter_map[ans]
        if ans.isdigit():
            num = int(ans)
            if 1 <= num <= 4:
                return num - 1
        return None

    # -----------------------------
    def _predict_logits(self, inputs) -> List[int]:
        """선택지별 로짓(logit)을 계산하고 argmax로 예측"""
        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :]

        # 각 선택지를 표현하는 첫 토큰 ID만 사용 (안전 가드 포함)
        choice_ids = []
        for ch in ["A", "B", "C", "D"]:
            tok = self.tokenizer.encode(ch, add_special_tokens=False)
            choice_ids.append(tok[0] if len(tok) > 0 else 0)

        preds = torch.argmax(logits[:, choice_ids], dim=-1)
        return preds.cpu().tolist()

    # -----------------------------
    def _get_fewshots(self, subset: str) -> List[Dict[str, Any]]:
        """--use_manual_fewshots 옵션에 따라 few-shot 예시 생성

        기본(False): subset의 train에서 랜덤 5개, 없으면 train → dev
        True: 수동 예시 사용
        """
        manual = [
            {
                "question": "세포 내 에너지 생성의 주요 기관은?",
                "A": "핵",
                "B": "미토콘드리아",
                "C": "리보솜",
                "D": "소포체",
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
                "question": "반도체에서 전자가 이동하는 주요 경로는?",
                "A": "원자핵",
                "B": "전도대",
                "C": "가전자대",
                "D": "절연층",
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
        ]
        if self.use_manual_fewshots:
            return manual

        ds = load_dataset("HAERAE-HUB/KMMLU", subset)

        # "validation", "test" 포함되어있어 주석 처리
        # for split in ["dev", "validation", "test", "train"]:
        #     if split in ds:
        #         data = [
        #             ex
        #             for ex in ds[split]
        #             if str(ex.get("answer", "")).strip() in ["1", "2", "3", "4"]
        #         ]
        #         if data:
        #             k = min(self.num_shots, len(data))
        #             return random.sample(data, k)
        # return manual

        for split in ["train", "dev"]:
            if split in ds:
                data = [
                    ex
                    for ex in ds[split]
                    if str(ex.get("answer", "")).strip() in ["1", "2", "3", "4"]
                ]
                if data:
                    k = min(self.num_shots, len(data))
                    return random.sample(data, k)

        # train, dev 둘 다 없는 경우 → skip 처리 (manual fallback 제거)
        print(f"{subset}: train/dev split에 유효한 few-shot 데이터 없음 → skip")
        return []

    # -----------------------------
    def evaluate(self):
        """KMMLU 전체 subset 평가 수행 (정확도: all_correct / all_total)"""
        results, all_correct, all_total = [], 0, 0
        start_time = datetime.now()

        for subset in tqdm(self.subsets, desc="KMMLU 평가 진행 중"):
            try:
                ds = load_dataset("HAERAE-HUB/KMMLU", subset)

                # 평가 split: test 우선, 없으면 validation → dev → train 순
                eval_split = next(
                    (s for s in ["test", "validation", "dev", "train"] if s in ds), None
                )
                if not eval_split:
                    print(f"{subset}: 평가 가능한 split 없음 → skip")
                    continue

                # 정답 필터링 (answer가 1~4인 데이터만 사용)
                data = [
                    ex
                    for ex in ds[eval_split]
                    if str(ex.get("answer", "")).strip() in ["1", "2", "3", "4"]
                ]
                if not data:
                    print(f"{subset}: 유효한 정답 데이터 없음 → skip")
                    continue

                fewshots = self._get_fewshots(subset)
                if not fewshots:  # fewshot 예시가 없으면 skip
                    print(f"{subset}: few-shot 예시 없음 → skip")
                    continue

                # Prompt 및 Truth 생성
                prompts = [self._make_prompt(ex, fewshots) for ex in data]
                truths = [self._extract_answer_index(ex) for ex in data]

                # Batch 평가
                correct = total = 0
                for i in range(0, len(prompts), self.batch_size):
                    batch_prompts = prompts[i : i + self.batch_size]
                    batch_truths = truths[i : i + self.batch_size]

                    inputs = self.tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=3072,
                    ).to(self.model.device)

                    preds = self._predict_logits(inputs)

                    for p, t in zip(preds, batch_truths):
                        if t is not None:
                            total += 1
                            if p == t:
                                correct += 1

                acc = (correct / total) if total > 0 else 0.0

                # 출력 형식
                print(f"{subset}: {acc:.4f} ({correct}/{total})")

                results.append(
                    {
                        "Subset": subset,
                        "Category": self.supercategories.get(subset, "N/A"),
                        "Accuracy": acc,
                        "Correct": correct,
                        "Total": total,
                    }
                )

                all_correct += correct
                all_total += total

            except Exception as e:
                print(f"{subset} 오류 발생: {e}")
                results.append(
                    {
                        "Subset": subset,
                        "Category": self.supercategories.get(subset, "N/A"),
                        "Accuracy": 0.0,
                        "Correct": 0,
                        "Total": 0,
                    }
                )

        self._summarize(results, all_correct, all_total, datetime.now() - start_time)

    # -----------------------------
    def _summarize(
        self, results: List[Dict[str, Any]], correct: int, total: int, time_elapsed
    ):
        """평가 결과를 요약하여 출력 및 저장 (CSV, JSON)"""
        os.makedirs(self.output_dir, exist_ok=True)
        df = pd.DataFrame(results)

        # 전체 평균: all_correct / all_total
        overall_acc = (correct / total) if total > 0 else 0.0
        overall_percent = overall_acc * 100

        # 카테고리별 평균
        if not df.empty:
            cat_mean = df.groupby("Category")["Accuracy"].mean().sort_index()
        else:
            cat_mean = pd.Series(dtype=float)

        print("\n" + "=" * 60)
        print("분야별 평균 정확도")
        print("-" * 60)
        if len(cat_mean) > 0:
            print(cat_mean.to_string(float_format="%.4f"))
        else:
            print("집계할 결과가 없습니다.")
        print("-" * 60)
        print(f"전체 평균 정확도: {overall_acc:.4f} ({correct}/{total})")
        print(f"총 소요 시간: {time_elapsed}")
        print("=" * 60)

        # CSV 파일 저장
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, f"dataset_KMMLU_eval_{ts}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 상세 JSON 저장 (요약 + 세부정보 통합)
        detailed_results = {
            "model_id": self.model_id,
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "time_elapsed": str(time_elapsed),
            "time_elapsed_seconds": round(time_elapsed.total_seconds(), 2),
            "experiment_config": {
                "seed": self.seed,
                "batch_size": self.batch_size,
                "num_shots": self.num_shots,
                "prompting_strategy": f"{self.prompting_strategy}_{self.num_shots}shot",
            },
            "summary": {
                "overall_accuracy": round(overall_acc, 4),
                "correct_answers": correct,
                "total_questions": total,
                "category_accuracy": {
                    k: round(v, 4) for k, v in cat_mean.to_dict().items()
                },
            },
            "subset_scores": [
                {
                    "subset": row["Subset"],
                    "category": row["Category"],
                    "accuracy": round(row["Accuracy"], 4),
                }
                for _, row in df.iterrows()
            ],
        }

        json_filename = f"kmmlu_{self.output_prefix}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        print(f"=== JSON 저장 완료: {json_filename}")

    # -----------------------------
    def _get_official_subsets(self) -> List[str]:
        """KMMLU 45개 공식 subset 목록 반환"""
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

    # -----------------------------
    def _get_supercategories(self) -> Dict[str, str]:
        """KMMLU 상위 분야(Supercategory) 매핑"""
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
        return {s: cat for cat, subs in cats.items() for s in subs}


def main():
    """명령행 인자 파서 및 실행 진입점"""
    parser = argparse.ArgumentParser(description="Dataset SFT + KMMLU 평가 (Few-shot)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--dataset_id", type=str, default="HAERAE-HUB/KMMLU"
    )  # 기존: maywell/ko_wikidata_QA
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_manual_fewshots", action="store_true", help="수동 few-shot 예시 사용"
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,  # 전체 학습 개별 적용시 1000 과 같이 적용 가능
        help="SFT 학습에 사용할 최대 샘플 수",
    )
    parser.add_argument("--num_shots", type=int, default=5, help="few-shot 예시 개수")
    parser.add_argument(
        "--eval_only", action="store_true", help="평가만 수행 (SFT 생략)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--eval_subsets",
        nargs="+",
        type=str,
        default=None,
        help="평가할 subset 이름 (예: --eval_subsets humss 또는 Accounting Criminal-Law)",
    )

    args = parser.parse_args()

    evaluator = dataset_KMMLU(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        batch_size=args.batch_size,
        seed=args.seed,
        use_flash_attn2=True,
        use_manual_fewshots=args.use_manual_fewshots,
        max_train_samples=args.max_train_samples,
        num_shots=args.num_shots,
        output_dir=args.output_dir,
    )

    # eval_only 옵션이 없으면 SFT → 평가, 있으면 평가만
    if not args.eval_only:
        evaluator.train_on_kmmlu()
    evaluator.evaluate()


if __name__ == "__main__":

    main()

    # 예시 실행:
    # 1) 훈련 + 평가 실시
    #    python 4.kmmlu_Qwen2.5_7B_instruct_sft_kmmlu.py
    #
    # 2) 학습된 모델을 불러와서 humss만 평가(학습과 평가를 따로 진행할 때)
    # python 4.kmmlu_Qwen2.5_7B_instruct_sft_kmmlu.py --model_id results/lora_ft_Qwen2.5-7B-Instruct --eval_only --eval_subsets humss
    #
    # 3) 학습(SFT)은 건너뛰고, 평가만 수행(전체 45개 subset)
    # python 4.kmmlu_Qwen2.5_7B_instruct_sft_kmmlu.py --eval_only
    #
    # 4) 학습(SFT)을 건너뛰고, 평가만 수행(HUMSS 관련 11개 subset)
    # python 4.kmmlu_Qwen2.5_7B_instruct_sft_kmmlu.py --eval_only --eval_subsets humss
    #
    #
