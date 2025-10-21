#!/usr/bin/env python
# -*- coding: utf-8 -*-
# run_kmmlu_solar_fewshot_only.py

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

    def __init__(self, model_id: str, batch_size: int = 4, seed: int = 42, output_prefix: str = None, num_shots: int = 5):
        """모델 초기화 및 few-shot 설정"""
        self.model_id = model_id
        self.batch_size = batch_size
        self.seed = seed
        self.num_shots = num_shots
        self.output_prefix = output_prefix or self._generate_output_prefix()

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.tokenizer, self.model = self._load_model()
        self.subsets = self._get_official_subsets()
        self.supercategories = self._get_supercategories()

        # 숫자/문자 정답 모두 대응
        self.letter_map = {
            "A": 0, "B": 1, "C": 2, "D": 3,
            "1": 0, "2": 1, "3": 2, "4": 3
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
    def _make_prompt(self, fewshots, test_ex):
        """few-shot 예시와 테스트 문항을 결합하여 최종 프롬프트 구성"""
        prompt = ""
        for fs in fewshots:
            prompt += self._format_example(fs, include_answer=True)
        prompt += self._format_example(test_ex, include_answer=False)
        return prompt

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
        choice_ids = [self.tokenizer.encode(ch, add_special_tokens=False)[0] for ch in ["A", "B", "C", "D"]]
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

                # dev 존재 시 few-shot 추출, 없으면 test 일부 사용
                if "dev" in dataset:
                    dev_data = list(dataset["dev"])
                elif "train" in dataset:
                    dev_data = random.sample(list(dataset["train"]), min(self.num_shots, len(dataset["train"])))
                else:
                    dev_data = list(dataset["test"][:self.num_shots])

                if "test" not in dataset:
                    print(f"{subset}: test split 없음 → skip")
                    continue
                test_data = list(dataset["test"])

                fewshots = random.sample(dev_data, min(self.num_shots, len(dev_data)))
                prompts = [self._make_prompt(fewshots, t) for t in test_data]
                truths = [self._extract_answer_index(t) for t in test_data]

                correct = total = 0
                for i in tqdm(range(0, len(prompts), self.batch_size),
                              desc=f"{subset} 평가 중", leave=False):
                    batch_prompts = prompts[i:i+self.batch_size]
                    batch_truths = truths[i:i+self.batch_size]

                    inputs = self.tokenizer(batch_prompts,
                                            return_tensors="pt",
                                            padding=True,
                                            truncation=True,
                                            max_length=3072).to(self.model.device)
                    preds = self._predict_logits(inputs)

                    for p, t in zip(preds, batch_truths):
                        if t is not None:
                            total += 1
                            if p == t:
                                correct += 1

                acc = correct / total if total > 0 else 0.0
                print(f"  - {subset}: {acc:.4f} ({correct}/{total})")

                results.append({
                    "Subset": subset,
                    "Category": self.supercategories.get(subset, "N/A"),
                    "Accuracy": acc
                })
                all_correct += correct
                all_total += total

            except Exception as e:
                print(f"{subset} 오류 발생: {e}")
                results.append({
                    "Subset": subset,
                    "Category": self.supercategories.get(subset, "N/A"),
                    "Accuracy": 0.0
                })

        time_elapsed = datetime.now() - start_time
        self._summarize(results, all_correct, all_total, time_elapsed)

    # -----------------------------
    def _summarize(self, results, correct, total, time_elapsed):
        """평가 결과를 요약하여 출력 및 저장 (CSV, JSON)"""
        df = pd.DataFrame(results)
        cat_mean = df.groupby("Category")["Accuracy"].mean().sort_index()
        overall_acc = correct / total if total > 0 else 0.0

        print("\n" + "="*60)
        print("분야별 평균 정확도")
        print("-"*60)
        print(cat_mean.to_string(float_format="%.4f"))
        print("-"*60)
        print(f"** 전체 평균 정확도: {overall_acc:.4f} ({correct}/{total}) **")
        print(f"총 소요 시간: {time_elapsed}")
        print("="*60)

        csv_file = f"kmmlu_fewshot_{self.output_prefix}.csv"
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        print(f"\n결과 CSV 저장 완료: {csv_file}")

        summary = {
            "model_id": self.model_id,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": self.seed,
            "num_shots": self.num_shots,
            "overall_accuracy": round(overall_acc, 4),
            "correct": correct,
            "total": total,
            "elapsed_time": str(time_elapsed),
            "category_accuracy": {k: round(v, 4) for k, v in cat_mean.to_dict().items()},
        }
        json_file = f"kmmlu_fewshot_{self.output_prefix}_summary.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"요약 JSON 저장 완료: {json_file}\n")

    # -----------------------------
    def _get_official_subsets(self):
        """KMMLU 45개 공식 subset 목록 반환"""
        return [
            "Accounting", "Agricultural-Sciences", "Aviation-Engineering-and-Maintenance", "Biology",
            "Chemical-Engineering", "Chemistry", "Civil-Engineering", "Computer-Science", "Construction",
            "Criminal-Law", "Ecology", "Economics", "Education", "Electrical-Engineering",
            "Electronics-Engineering", "Energy-Management", "Environmental-Science", "Fashion",
            "Food-Processing", "Gas-Technology-and-Engineering", "Geomatics", "Health", "Industrial-Engineer",
            "Information-Technology", "Interior-Architecture-and-Design", "Law", "Machine-Design-and-Manufacturing",
            "Management", "Maritime-Engineering", "Marketing", "Materials-Engineering", "Mechanical-Engineering",
            "Nondestructive-Testing", "Patent", "Political-Science-and-Sociology", "Psychology",
            "Public-Safety", "Railway-and-Automotive-Engineering", "Real-Estate", "Refrigerating-Machinery",
            "Social-Welfare", "Taxation", "Telecommunications-and-Wireless-Technology",
            "Korean-History", "Math"
        ]

    def _get_supercategories(self):
        """KMMLU 상위 분야(Supercategory) 매핑"""
        cats = {
            "STEM": ["Biology", "Chemical-Engineering", "Chemistry", "Civil-Engineering", "Computer-Science",
                     "Ecology", "Electrical-Engineering", "Information-Technology", "Materials-Engineering",
                     "Mechanical-Engineering", "Math"],
            "HUMSS": ["Accounting", "Criminal-Law", "Economics", "Education", "Law", "Management",
                      "Political-Science-and-Sociology", "Psychology", "Social-Welfare", "Taxation",
                      "Korean-History"],
            "Applied Science": ["Aviation-Engineering-and-Maintenance", "Electronics-Engineering",
                                "Energy-Management", "Environmental-Science", "Gas-Technology-and-Engineering",
                                "Geomatics", "Industrial-Engineer", "Machine-Design-and-Manufacturing",
                                "Maritime-Engineering", "Nondestructive-Testing",
                                "Railway-and-Automotive-Engineering", "Telecommunications-and-Wireless-Technology"],
            "Other": ["Agricultural-Sciences", "Construction", "Fashion", "Food-Processing", "Health",
                      "Interior-Architecture-and-Design", "Marketing", "Patent", "Public-Safety", "Real-Estate",
                      "Refrigerating-Machinery"]
        }
        mapping = {s: cat for cat, subs in cats.items() for s in subs}
        return mapping


def main():
    parser = argparse.ArgumentParser(description="KMMLU Few-shot 모델 평가 도구")
    parser.add_argument("--model_id", type=str,
                        default="upstage/SOLAR-10.7B-Instruct-v1.0",
                        help="평가할 HuggingFace 모델 ID")
    parser.add_argument("--batch_size", type=int, default=4, help="배치 크기")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_shots", type=int, default=5, help="Few-shot 예시 개수")
    parser.add_argument("--output_prefix", type=str, default=None, help="출력 파일명 prefix")

    args = parser.parse_args()

    evaluator = KMMLUEvaluator(
        args.model_id, args.batch_size, args.seed, args.output_prefix, args.num_shots
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()



# python kmmlu_evaluator.py
# python kmmlu_evaluator.py --model_id "your-username/your-finetuned-model"
# python kmmlu_evaluator.py --batch_size 2  # 메모리 부족 시
# python kmmlu_evaluator.py --batch_size 8  # 메모리 충분 시
# python kmmlu_evaluator.py --output_prefix "baseline_v1"
# # 결과: kmmlu_baseline_v1.csv, kmmlu_baseline_v1_summary.json
# python kmmlu_evaluator.py --seed 123
# # compare_models.sh
# python kmmlu_evaluator.py --model_id "Bllossom/llama-3.2-Korean-Bllossom-3B" --output_prefix "baseline"
# python kmmlu_evaluator.py --model_id "your-username/finetuned-model" --output_prefix "finetuned"
