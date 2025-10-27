#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2.kmmlu_solar_fewshot_only.py

import os
import torch
import pandas as pd
import re
import random
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datetime import datetime
import json


class KMMLUEvaluator:
    """KMMLU Few-shot Evaluator
    --------------------------------
    - KMMLU 45개 subset 각각에 대해 5-shot few-shot 프롬프트 평가를 수행합니다.
    - dev split에서 few-shot 예시를 추출하며, 존재하지 않으면 test 일부를 사용합니다.
    - 분야별(supercategory) 평균, 전체 평균을 함께 계산하며 CSV/JSON으로 저장합니다.
    """

    def __init__(
        self,
        model_id: str,
        batch_size: int = 4,
        seed: int = 42,
        output_prefix: str = None,
        num_shots: int = 0,
        prompting_strategy: str = "zero_shot_cot",  # zero_shot_cot 으로 변경해야 CoT가 적용된다.
    ):
        """모델 초기화 및 few-shot 설정"""
        self.model_id = model_id
        self.batch_size = batch_size
        self.seed = seed
        self.output_prefix = output_prefix or self._generate_output_prefix()
        self.num_shots = num_shots
        self.prompting_strategy = prompting_strategy

        # Random seed 고정
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.tokenizer, self.model = self._load_model()
        self.subsets = self._get_official_subsets()
        self.supercategories = self._get_supercategories()

        # 숫자/문자 정답 모두 대응
        self.letter_map = {
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
        """모델과 토크나이저 로드 (4bit 양자화 지원)"""
        print(f"\n{'='*60}")
        print(f"모델 로딩 중: {self.model_id}")
        print(f"Random Seed: {self.seed}")
        print(f"{'='*60}\n")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.eval()
        print("모델 로딩 완료.\n")
        return tokenizer, model

    # -----------------------------
    def _normalize_text(self, text):
        """문자열 공백 정규화"""
        if not isinstance(text, str):
            return str(text)
        return re.sub(r"\s+", " ", text).strip()

    # -----------------------------
    def _format_example(self, ex, include_answer=True):
        """문항을 few-shot 프롬프트 형식으로 변환"""
        prompt = f"문제: {self._normalize_text(ex['question'])}\n"
        for c in ["A", "B", "C", "D"]:
            prompt += f"{c}. {self._normalize_text(ex[c])}\n"
        prompt += "정답:"
        if include_answer and ex.get("answer"):
            prompt += f" {ex['answer']}\n\n"
        else:
            prompt += " "
        return prompt

    # -----------------------------
    # def _make_prompt(self, fewshots, test_ex):
    #     """few-shot 예시와 테스트 문항을 결합하여 최종 프롬프트 구성"""
    #     prompt = ""
    #     for fs in fewshots:
    #         prompt += self._format_example(fs, include_answer=True)
    #     prompt += self._format_example(test_ex, include_answer=False)
    #     return prompt

    # ----- CoT 조건 추가(1개 텍스트만 실행할 때) --
    # def _make_prompt(self, few_shot, test_ex):
    #     base_prompt = "".join(
    #         [self._format_example(e) for e in few_shot]
    #     ) + self._format_example(test_ex, include_answer=False)

    #     # Zero-shot CoT 추가
    #     if self.prompting_strategy == "zero_shot_cot" and self.num_shots == 0:
    #         base_prompt += " Let's think step by step."

    #     return base_prompt

    # ----- Instruction 조건 추가(단일 또는 다수 실행할 때) --
    def _make_prompt(self, few_shot, test_ex):
        base_prompt = "".join(
            [self._format_example(e) for e in few_shot]
        ) + self._format_example(test_ex, include_answer=False)

        # 여러 Instruction 문장을 무작위 추가
        if self.prompting_strategy == "zero_shot_cot" and self.num_shots == 0:
            cot_phrases = [
                "Let’s think step by step.",
                "First, let’s read the question carefully.",
                "Let’s break this problem down into smaller steps.",
                "Let’s identify the key requirement of the question.",
                "Let’s think through this logically.",
                "Let’s reason through this realistically and step by step.",
                # "First, we need to understand what the question is asking.",
                # "Let’s start by identifying the essential elements of the question.",
                # "We need to distinguish between what the question requires and the background explanation.",
                # "Can we divide this problem into simpler components?",
                # "Let’s break it down into smaller, manageable parts and organize them.",
                # "Let’s clarify our assumptions step by step to solve the problem.",
                # "Let's think step by step.",
                # "First, let's understand the question clearly.",
                # "We can solve this by analyzing each option.",
                # "Let’s reason this out carefully.",
                # "Let’s eliminate incorrect choices one by one.",
                # "What is the question really asking?",
                # "Let’s recall relevant knowledge.",
                # "Let’s consider all possibilities before choosing.",
                # "Let’s break this down logically.",
                # "Let’s approach this methodically."
            ]

            # 방법 1: 랜덤하게 하나만 추가
            # base_prompt += " " + random.choice(cot_phrases)

            # 방법 2: 여러 문장을 함께 추가 (더 자연스러운 프롬프트 생성 가능)
            base_prompt += "\n" + "\n".join(cot_phrases[:5])  # 원하는 개수만큼 추가

        return base_prompt

    # -----------------------------
    def _extract_answer_index(self, ex):
        """문항의 정답을 0~3 인덱스로 변환"""
        ans = str(ex.get("answer", "")).strip().upper()
        if ans in self.letter_map:
            return self.letter_map[ans]
        elif ans.isdigit() and 1 <= int(ans) <= 4:
            return int(ans) - 1
        return None

    # -----------------------------
    def _predict_logits(self, inputs):
        """선택지별 로짓(logit)을 계산하고 argmax로 예측"""
        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :]
        choice_ids = [
            self.tokenizer.encode(ch, add_special_tokens=False)[0]
            for ch in ["A", "B", "C", "D"]
        ]
        preds = torch.argmax(logits[:, choice_ids], dim=-1)
        return preds.cpu().tolist()

    # -----------------------------
    def evaluate(self):
        """KMMLU 전체 subset에 대해 Few-shot 평가 수행"""
        results, all_correct, all_total = [], 0, 0
        start_time = datetime.now()

        for subset in tqdm(self.subsets, desc="KMMLU 전체 평가"):
            try:
                dataset = load_dataset("HAERAE-HUB/KMMLU", subset)

                if "train" in dataset:
                    data = [
                        ex
                        for ex in dataset["train"]
                        if str(ex.get("answer", "")).strip() in ["1", "2", "3", "4"]
                    ]
                    if data:
                        fewshot_data = random.sample(
                            data, min(self.num_shots, len(data))
                        )

                elif "dev" in dataset:
                    data = [
                        ex
                        for ex in dataset["dev"]
                        if str(ex.get("answer", "")).strip() in ["1", "2", "3", "4"]
                    ]
                    if data:
                        fewshot_data = random.sample(
                            data, min(self.num_shots, len(data))
                        )

                # train, dev 둘 다 없으면 skip
                if not fewshot_data:
                    print(
                        f"{subset}: train/dev split에 유효한 few-shot 데이터 없음 → skip"
                    )
                    continue
                # ----------------- 여기까지 수정 ------------------------
                fewshot_data = []
                if "test" not in dataset:
                    print(f"{subset}: test split 없음 → skip")
                    continue
                test_data = list(dataset["test"])

                # fewshots = random.sample(dev_data, min(self.num_shots, len(dev_data)))
                fewshots = fewshot_data
                prompts = [self._make_prompt(fewshots, t) for t in test_data]
                truths = [self._extract_answer_index(t) for t in test_data]

                correct = total = 0
                for i in tqdm(
                    range(0, len(prompts), self.batch_size),
                    desc=f"{subset} 평가 중",
                    leave=False,
                ):
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

                acc = correct / total if total > 0 else 0.0
                print(f"  - {subset}: {acc:.4f} ({correct}/{total})")

                results.append(
                    {
                        "Subset": subset,
                        "Category": self.supercategories.get(subset, "N/A"),
                        "Accuracy": acc,
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

        time_elapsed = datetime.now() - start_time
        self._summarize(results, all_correct, all_total, time_elapsed)

    # -----------------------------
    def _summarize(self, results, correct, total, time_elapsed):
        """평가 결과를 요약하여 출력 및 저장 (CSV, JSON)"""
        os.makedirs(self.output_dir, exist_ok=True)
        df = pd.DataFrame(results)

        # 카테고리별 평균
        if not df.empty:
            cat_mean = df.groupby("Category")["Accuracy"].mean().sort_index()
        else:
            cat_mean = pd.Series(dtype=float)

        cat_mean_percent = cat_mean * 100

        # 전체 평균
        overall_acc = correct / total if total > 0 else 0.0
        overall_percent = overall_acc * 100

        print("\n" + "=" * 60)
        print("분야별 평균 정확도")
        print("-" * 60)
        for cat, acc in cat_mean.items():
            print(f"{cat:20s}: {acc:.4f} ({acc*100:.2f}%)")
        print("-" * 60)
        print(f"\n** 전체 평균 정확도: {overall_acc:.4f} ({overall_percent:.2f}%) **")
        print(f"정답: {correct} / 전체: {total}")
        print(f"총 소요 시간: {time_elapsed}")
        print("=" * 60)

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

        # CSV 파일 저장
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # csv_path = os.path.join(self.output_dir, f"dataset_KMMLU_eval_{ts}.csv")
        csv_path = f"kmmlu_{self.output_prefix}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"=== CSV 저장 완료: {csv_path}")

        json_filename = f"kmmlu_{self.output_prefix}_summary.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        print(f"=== JSON 저장 완료: {json_filename}")

    # -----------------------------
    def _get_official_subsets(self):
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

    def _get_supercategories(self):
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
        mapping = {s: cat for cat, subs in cats.items() for s in subs}
        return mapping


def main():
    """KMMLU 평가 실행 진입점"""
    parser = argparse.ArgumentParser(description="KMMLU 평가")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
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
            "zero_shot_cot",
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

    args = parser.parse_args()

    test_subsets = args.test_subsets.split(",") if args.test_subsets else None

    evaluator = KMMLUEvaluator(
        model_id=args.model_id,
        batch_size=args.batch_size,
        seed=args.seed,
        num_shots=args.num_shots,
        prompting_strategy=args.prompting_strategy,
        output_prefix=args.output_prefix,
        # test_subsets=args.test_subsets,  # 개별 subsets 추론 없이, 전체를 사용하기 위해 주석처리
    )

    evaluator.evaluate()


if __name__ == "__main__":
    main()


# python 2.kmmlu_solar_fewshot_only.py
# python 2.kmmlu_solar_fewshot_only.py
# python 2.kmmlu_solar_fewshot_only.py --model_id "your-username/your-finetuned-model"
# python 2.kmmlu_solar_fewshot_only.py --batch_size 2  # 메모리 부족 시
# python 2.kmmlu_solar_fewshot_only.py --batch_size 8  # 메모리 충분 시
# python 2.kmmlu_solar_fewshot_only.py --output_prefix "baseline_v1"
# # 결과: kmmlu_baseline_v1.csv, kmmlu_baseline_v1_summary.json
# python 2.kmmlu_solar_fewshot_only.py --seed 123
# # compare_models.sh
# python 2.kmmlu_solar_fewshot_only.py --model_id "Qwen/Qwen2.5-7B-Instruct" --output_prefix "baseline"
# python 2.kmmlu_solar_fewshot_only.py --model_id "your-username/finetuned-model" --output_prefix "finetuned"
