# kmmlu_evaluator2.py
# pip install torch transformers datasets pandas tqdm bitsandbytes accelerate

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
    """
     KMMLU Evaluator (모듈화 버전)
    - 한국어 LLM 모델을 KMMLU 벤치마크로 평가
    - 여러 모델을 쉽게 평가할 수 있도록 모듈화

    (추가 수정 사항)
    - 정답 변환 로직 개선
    - Accuracy 계산 명확화
    """

    def __init__(
        self,
        model_id: str,
        batch_size: int = 4,
        seed: int = 42,
        num_shots: int = 5, 
        prompting_strategy: str = "random",
        output_prefix: str = None,
        test_subsets: list = None,
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        self.seed = seed
        self.num_shots = num_shots
        self.prompting_strategy = prompting_strategy
        self.output_prefix = output_prefix or self._generate_output_prefix()
        self.test_subsets = test_subsets  # None이면 전체, 리스트면 특정 subset만

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.tokenizer, self.model = self._load_model()
        self.subsets = self._get_official_subsets()
        self.supercategories = self._get_supercategories()
        self.letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    def _generate_output_prefix(self):
        model_name = self.model_id.split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_{timestamp}"

    def _load_model(self):
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
            self.model_id, quantization_config=bnb_config, device_map="auto"
        )
        model.eval()
        print("모델 로딩 완료.\n")
        return tokenizer, model

    def _normalize_text(self, text):
        if not isinstance(text, str):
            return str(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _format_example(self, ex, include_answer=True):
        prompt = f"문제: {self._normalize_text(ex['question'])}\n"
        for c in ["A", "B", "C", "D"]:
            prompt += f"{c}. {self._normalize_text(ex[c])}\n"
        prompt += "정답:"

        if include_answer:
            ans = self._answer_to_letter(ex["answer"])
            prompt += f" {ans}\n\n"
        return prompt

    def _answer_to_letter(self, answer):
        """정답 값을 A, B, C, D 문자로 변환 (Few-shot 예시용)"""
        if isinstance(answer, str):
            answer = answer.strip().upper()
            if answer in ["A", "B", "C", "D"]:
                return answer
            elif answer.isdigit():
                idx = int(answer) - 1
                if 0 <= idx <= 3:
                    return ["A", "B", "C", "D"][idx]
        elif isinstance(answer, int):
            idx = answer - 1
            if 0 <= idx <= 3:
                return ["A", "B", "C", "D"][idx]

        print(f"경고: 알 수 없는 정답 형식 - {answer}")
        return "A"  # 기본값

    def _extract_answer_index(self, ex):
        """
        데이터셋의 정답을 0, 1, 2, 3 인덱스로 변환
        KMMLU answer 형식: 1, 2, 3, 4 (숫자)
        """
        ans = ex["answer"]

        # 정수형 처리 (1, 2, 3, 4 → 0, 1, 2, 3)
        if isinstance(ans, int):
            if 1 <= ans <= 4:
                return ans - 1
            else:
                print(f"경고: 범위 밖 정답 - {ans}")
                return None

        # 문자열 숫자 처리 ("1", "2", "3", "4" → 0, 1, 2, 3)
        if isinstance(ans, str):
            ans = ans.strip()
            if ans.isdigit():
                num = int(ans)
                if 1 <= num <= 4:
                    return num - 1
                else:
                    print(f"경고: 범위 밖 문자열 정답 - {ans}")
                    return None

        print(f"경고: 변환 불가 정답 형식 - {ans} (타입: {type(ans)})")
        return None

    def _make_prompt(self, few_shot, test_ex):
        return "".join(
            [self._format_example(e) for e in few_shot]
        ) + self._format_example(test_ex, include_answer=False)

    def _predict_logits(self, inputs):
        """
        마지막 토큰 위치에서 A, B, C, D 토큰의 logit 확률로 예측
        주의: 이 방법은 모델이 생성 기반이 아닌 분류 태스크로 학습되지 않은 경우 부정확할 수 있음
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]

        # A, B, C, D 각각에 해당하는 토큰 ID들
        choice_ids = [
            self.tokenizer.encode(ch, add_special_tokens=False)
            for ch in ["A", "B", "C", "D"]
        ]

        avg_scores = []
        for cid in choice_ids:
            # 해당 선택지의 토큰들에 대한 logit 평균
            token_logits = logits[:, cid].mean(dim=-1)
            avg_scores.append(token_logits)

        probs = torch.stack(avg_scores, dim=1)
        preds = torch.argmax(probs, dim=-1)
        return preds.cpu().tolist()

    def evaluate(self):
        """전체 KMMLU subset 평가"""
        # 시간 측정 시작
        start_time = datetime.now()
        results, all_correct, all_total = [], 0, 0
        
        # test_subsets가 지정되면 해당 subset만, 아니면 전체
        subsets_to_evaluate = self.test_subsets if self.test_subsets else self.subsets
        
        for subset in tqdm(subsets_to_evaluate, desc="KMMLU 평가"):
            try:
                dataset = load_dataset("HAERAE-HUB/KMMLU", subset)

                if "dev" in dataset:
                    dev = list(dataset["dev"])
                elif "train" in dataset:
                    dev = list(dataset["train"])
                else:
                    print(f"{subset}: dev/train 없음 → 건너뜀")
                    continue

                if "test" in dataset:
                    test = list(dataset["test"])
                elif "validation" in dataset:
                    test = list(dataset["validation"])
                else:
                    print(f"{subset}: test/validation 없음 → 건너뜀")
                    continue

                random.seed(self.seed)
                few_shot = random.sample(dev, min(self.num_shots, len(dev)))

                prompts = [self._make_prompt(few_shot, t) for t in test]
                truths = [self._extract_answer_index(t) for t in test]

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

                # Accuracy 계산: 정답 개수 / 전체 문제 수
                acc = correct / total if total > 0 else 0.0
                acc_percent = acc * 100  # 백분율 변환

                print(
                    f"  - {subset}: {acc:.4f} ({acc_percent:.2f}%) - {correct}/{total} 정답"
                )

                results.append(
                    {
                        "Subset": subset,
                        "Category": self.supercategories.get(subset, "N/A"),
                        "Accuracy": acc,  # 0~1 비율
                        "Accuracy_Percent": acc_percent,  # 0~100 백분율
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
                        "Accuracy_Percent": 0.0,
                        "Correct": 0,
                        "Total": 0,
                    }
                )
                
        # 시간 측정 종료 및 계산
        time_elapsed = datetime.now() - start_time 
        print("\n","소요 시간:", time_elapsed)
        
        self._summarize(results, all_correct, all_total, time_elapsed)


    def _summarize(self, results, correct, total, time_elapsed):
        """평가 결과 요약 및 저장"""
        df = pd.DataFrame(results)

        # 카테고리별 평균 (비율)
        cat_mean = df.groupby("Category")["Accuracy"].mean().sort_index()
        cat_mean_percent = cat_mean * 100

        # 전체 평균
        overall_acc = correct / total if total > 0 else 0.0
        overall_percent = overall_acc * 100

        print("\n" + "=" * 60)
        print("          분야별 평균 정확도")
        print("-" * 60)
        for cat, acc in cat_mean.items():
            print(f"{cat:20s}: {acc:.4f} ({acc*100:.2f}%)")
        print("-" * 60)
        print(f"\n** 전체 평균 정확도: {overall_acc:.4f} ({overall_percent:.2f}%) **")
        print(f"   정답: {correct} / 전체: {total}")
        print("=" * 60)

        # CSV 파일 저장
        csv_filename = f"kmmlu_{self.output_prefix}.csv"
        df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
        print(f"\n결과 저장 완료: {csv_filename}")

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
                "prompting_strategy": f"{self.prompting_strategy}_{self.num_shots}shot"
        },
            "summary": {
                "overall_accuracy": round(overall_acc, 4),
                "correct_answers": correct,
                "total_questions": total,
                "category_accuracy": {k: round(v, 4) for k, v in cat_mean.to_dict().items()}
            },
            "subset_scores": [
                {
                    "subset": row["Subset"],
                    "category": row["Category"],
                    "accuracy": round(row["Accuracy"], 4)
                }
                for _, row in df.iterrows()
            ]
        }

        json_filename = f"kmmlu_{self.output_prefix}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        print(f"=== JSON 저장 완료: {json_filename}")

    def _get_official_subsets(self):
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


def main():
    parser = argparse.ArgumentParser(description='KMMLU 평가')
    parser.add_argument('--model_id', type=str, 
                        default="Bllossom/llama-3.2-Korean-Bllossom-3B",
                        help='평가할 HuggingFace 모델 ID')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='배치 크기 (GPU 메모리에 따라 조정)') # 메모리 부족시 2, 메모리 충분 시 8
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (재현성)')
    parser.add_argument("--num_shots", type=int, default=5, 
                        help="Few-shot 예시 개수 (0=zero-shot, 5=5-shot)")
    parser.add_argument("--prompting_strategy", type=str, default="random",
                        choices=["random", "cot", "similarity", "meta_prompt", 
                             "gradient", "zero_shot", "self_consistency"],
                        help="프롬프트 전략")
    parser.add_argument('--output_prefix', type=str, default=None,
                        help='출력 파일명 prefix (기본: 모델명_타임스탬프)')
    parser.add_argument("--test_subsets", type=str, default=None,
                        help="테스트할 subset (콤마로 구분, 예: 'Math,Accounting')")
    
    args = parser.parse_args()
    
    # test_subsets 파싱
    test_subsets = None
    if args.test_subsets:
        test_subsets = [s.strip() for s in args.test_subsets.split(',')]

    evaluator = KMMLUEvaluator(
        model_id=args.model_id,
        batch_size=args.batch_size,
        seed=args.seed,
        num_shots=args.num_shots,
        prompting_strategy=args.prompting_strategy,
        output_prefix=args.output_prefix,
        test_subsets=test_subsets
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()