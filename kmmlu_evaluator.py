import unsloth
import torch
import pandas as pd
import re
import random
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datetime import datetime
from unsloth import FastLanguageModel
import os
import json
from kmmlu_parser import KMMLUArgumentManager


class KMMLUEvaluator:
    """
    KMMLU Evaluator (메모리 최적화 버전)
    - 이터레이터 방식으로 데이터 처리
    """
    def __init__(self, model_id: str, batch_size: int = 4, seed: int = 42,
                num_shots: int = 5, prompting_strategy: str = "random",
                output_prefix: str = None, subsets_to_test: list = None):
        self.model_id = model_id
        self.batch_size = batch_size
        self.seed = seed
        
        self.prompting_strategy = prompting_strategy
        self.zero_shot_cot = (prompting_strategy == "zero_shot_cot")

        if self.zero_shot_cot:
            self.num_shots = 0
            print("\n[알림] Zero-shot CoT 평가 모드로 실행합니다. (num_shots=0 강제)\n")
        else:
            self.num_shots = num_shots
        
        self.output_prefix = output_prefix or self._generate_output_prefix()
        
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        self.tokenizer, self.model = self._load_model()
        
        all_subsets = self._get_official_subsets()
        if subsets_to_test:
            self.subsets = [s for s in subsets_to_test if s in all_subsets]
            print(f"\n[알림] {len(self.subsets)}개의 지정된 subset만 평가합니다.")
        else:
            self.subsets = all_subsets
            
        self.supercategories = self._get_supercategories()
        self.letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    def _generate_output_prefix(self):
        model_name = self.model_id.split('/')[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.zero_shot_cot:
            strategy = "zero_shot_cot"
        else:
            strategy = f"{self.num_shots}shot_{self.prompting_strategy}"
        return f"{model_name}_{strategy}_{timestamp}"

    def _load_model(self):
        print(f"\n{'='*60}")
        print(f"모델 로딩 중: {self.model_id} (Strategy: {self.prompting_strategy})")
        print(f"Random Seed: {self.seed}")
        print(f"{'='*60}\n")
        
        # === 모델 로드 메모리 측정 시작 ===
        torch.cuda.reset_peak_memory_stats()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_id,
            max_seq_length = 4096,
            dtype = None, 
            load_in_4bit = True,
        )
        
        # === 모델 로드 메모리 측정 완료 및 출력 (GB 단위) ===
        self.peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\n[메모리] 모델 로딩 완료. 최대 VRAM 사용량: {self.peak_mem_gb:.2f} GB\n")
        
        tokenizer.padding_side = "left"
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        print("모델 로딩 완료.")
        return tokenizer, model
    
    def _normalize_text(self, text):
        if not isinstance(text, str):
            return str(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _format_example(self, ex, include_answer=True):
        prompt = f"문제: {self._normalize_text(ex['question'])}\n"
        for c in ["A", "B", "C", "D"]:
            prompt += f"{c}. {self._normalize_text(ex[c])}\n"
        prompt += "정답:"
        if include_answer:
            ans = str(ex['answer']).strip().upper()
            if ans.isdigit():
                ans = ["A", "B", "C", "D"][int(ans)-1] if 1 <= int(ans) <= 4 else ans
            prompt += f" {ans}\n\n"
        return prompt

    def _make_prompt(self, few_shot, test_ex):
        return "".join([self._format_example(e) for e in few_shot]) + \
            self._format_example(test_ex, include_answer=False)

    def _format_example_cot(self, ex):
        prompt = f"문제: {self._normalize_text(ex['question'])}\n"
        for c in ["A", "B", "C", "D"]:
            prompt += f"{c}. {self._normalize_text(ex[c])}\n"
        prompt += "정답: Let's think step by step. 이후 '최종 정답: $LETTER' 형식으로 답해주세요."
        return prompt

    def _make_prompt_cot(self, test_ex):
        return self._format_example_cot(test_ex)

    def _extract_answer_index(self, ex):
        ans = ex["answer"]
        if isinstance(ans, str):
            ans = ans.strip().upper()
            if ans in self.letter_map:
                return self.letter_map[ans]
            elif ans.isdigit():
                return int(ans) - 1
        elif isinstance(ans, int):
            return ans - 1
        return None

    def _predict_logits(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]

        choice_ids = [self.tokenizer.encode(ch, add_special_tokens=False)
                      for ch in ["A", "B", "C", "D"]]

        avg_scores = []
        for cid in choice_ids:
            token_logits = logits[:, cid].mean(dim=-1)
            avg_scores.append(token_logits)

        probs = torch.stack(avg_scores, dim=1)
        preds = torch.argmax(probs, dim=-1)
        return preds.cpu().tolist()

    def _predict_generate_cot(self, inputs):
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False, 
                num_beams=1,
            )

        input_len = inputs['input_ids'].shape[1]
        generated_ids = outputs[:, input_len:]
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        predictions = []
        for idx, text in enumerate(generated_texts):
            match = re.search(r"최종 정답:\s*([ABCD])", text)
            if match:
                ans_letter = match.group(1)
                predictions.append(self.letter_map[ans_letter])
            else:
                fallback_match = re.findall(r"([ABCD])", text)
                if fallback_match:
                    ans_letter = fallback_match[-1]
                    predictions.append(self.letter_map[ans_letter])
                else:
                    print(f"[경고] 배치 {idx}번 응답 파싱 실패: {text[:100]}...")
                    predictions.append(-1)
        
        return predictions

    def _sample_few_shot_examples(self, dev_dataset, num_samples):
        """
        Few-shot 예제만 샘플링 (메모리 효율적)
        """
        # dev 데이터셋 크기 확인
        dev_size = len(dev_dataset)
        num_samples = min(num_samples, dev_size)
        
        # 랜덤 인덱스 생성
        indices = random.sample(range(dev_size), num_samples)
        
        # few_shot할 때, 필요한 인덱스만 선택해서 메모리에 로드
        few_shot = [dev_dataset[i] for i in indices]
        return few_shot

    def _process_test_batches(self, test_dataset, few_shot=None):
        """
        메모리 최적화를 위해 test 데이터를 배치 단위로 이터레이터 방식 처리
        
        Args:
            test_dataset: HuggingFace Dataset 객체 (리스트 변환 X)
            few_shot: Few-shot 예제 리스트 (None이면 CoT 모드)
        
        Yields:
            (batch_prompts, batch_truths) 튜플
        """
        batch_prompts = []
        batch_truths = []
        
        # 이터레이터로 순회 (전체를 메모리에 올리지 않음)
        for example in test_dataset:
            # 프롬프트 생성
            if self.zero_shot_cot:
                prompt = self._make_prompt_cot(example)
            else:
                prompt = self._make_prompt(few_shot, example)
            
            truth = self._extract_answer_index(example)
            
            batch_prompts.append(prompt)
            batch_truths.append(truth)
            
            # 배치 크기 도달 시 yield
            if len(batch_prompts) == self.batch_size:
                yield batch_prompts, batch_truths
                batch_prompts = []
                batch_truths = []
        
        # 남은 데이터 처리
        if batch_prompts:
            yield batch_prompts, batch_truths

    def evaluate(self):
        start_time = datetime.now()
        results, all_correct, all_total = [], 0, 0
        skipped_subsets = []
        
        # === [추론 메모리 측정] 루프 시작 전 리셋 ===
        torch.cuda.reset_peak_memory_stats()

        for subset in tqdm(self.subsets, desc="KMMLU 전체 평가"):
            try:
                dataset = load_dataset("HAERAE-HUB/KMMLU", subset)
                
                # Dev 데이터 확인 (Dataset 객체 그대로 유지)
                if "dev" in dataset: 
                    dev_dataset = dataset["dev"]
                elif "train" in dataset: 
                    dev_dataset = dataset["train"]
                else: 
                    print(f"[경고] {subset}: dev/train 데이터 없음, 스킵")
                    skipped_subsets.append(subset)
                    continue
                
                # Test 데이터 확인 (Dataset 객체 그대로 유지)
                if "test" in dataset: 
                    test_dataset = dataset["test"]
                elif "validation" in dataset: 
                    test_dataset = dataset["validation"]
                else: 
                    print(f"[경고] {subset}: test/validation 데이터 없음, 스킵")
                    skipped_subsets.append(subset)
                    continue

                # Few-shot 예제만 샘플링 (CoT가 아닌 경우)
                few_shot = None
                if not self.zero_shot_cot:
                    few_shot = self._sample_few_shot_examples(dev_dataset, self.num_shots)
                
                # 평가 진행 (이터레이터 방식)
                correct = total = 0
                test_size = len(test_dataset)
                
                pbar = tqdm(total=test_size, desc=f"{subset} 평가 중", leave=False)
                
                for batch_prompts, batch_truths in self._process_test_batches(test_dataset, few_shot):
                    # 토크나이징
                    inputs = self.tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=4096
                    ).to(self.model.device)
                    
                    # 예측
                    if self.zero_shot_cot:
                        preds = self._predict_generate_cot(inputs)
                    else:
                        preds = self._predict_logits(inputs)
                    
                    # 정확도 계산
                    for p, t in zip(preds, batch_truths):
                        if t is not None:
                            total += 1
                            if p == t:
                                correct += 1
                    
                    pbar.update(len(batch_prompts))
                
                pbar.close()
                
                acc = correct / total if total else 0.0
                print(f"  - {subset}: {acc:.4f} ({correct}/{total})")

                results.append({
                    "Subset": subset,
                    "Category": self.supercategories.get(subset, "N/A"),
                    "Accuracy": acc
                })
                all_correct += correct
                all_total += total

            except Exception as e:
                import traceback
                print(f"[에러] {subset} 평가 실패:")
                print(traceback.format_exc())
                results.append({
                    "Subset": subset,
                    "Category": self.supercategories.get(subset, "N/A"),
                    "Accuracy": 0.0
                })
        
        if skipped_subsets:
            print(f"\n[요약] 스킵된 subset: {', '.join(skipped_subsets)}")
        
        end_time = datetime.now()
        time_elapsed = end_time - start_time
        print("\n소요 시간:", time_elapsed)
        
        # === [추론 메모리 측정] 루프 종료 후 최대값 확인 ===
        peak_inference_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\n[메모리] 추론 완료. 최대 VRAM 사용량: {peak_inference_mem_gb:.2f} GB")
        
        self._summarize(results, all_correct, all_total, time_elapsed, peak_inference_mem_gb)

    def _summarize(self, results, correct, total, time_elapsed, peak_inference_mem_gb):
        df = pd.DataFrame(results)
        cat_mean = df.groupby("Category")["Accuracy"].mean().sort_index()
        overall_acc = correct / total if total else 0.0

        print("\n" + "="*60)
        print(f"          분야별 평균 정확도 ({self.prompting_strategy})")
        print("-"*60)
        print(cat_mean.to_string(float_format="%.4f"))
        print("-"*60)
        print(f"\n** 전체 평균 정확도: {overall_acc:.2f}% ({correct}/{total}) **")
        print("="*60)
        
        result_dir = "result"
        os.makedirs(result_dir, exist_ok=True)

        csv_filename = os.path.join(result_dir, f"kmmlu_{self.output_prefix}.csv")
        df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
        print(f"\n결과 저장 완료: {csv_filename}")

        detailed_results = {
            "model_id": self.model_id,
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "time_elapsed": str(time_elapsed),
            "time_elapsed_seconds": round(time_elapsed.total_seconds(), 2),
            "experiment_config": {
                "seed": self.seed,
                "batch_size": self.batch_size,
                "num_shots": self.num_shots,
                "prompting_strategy": self.prompting_strategy
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
            ],
            "peak_memory_usage": {
                "model_load": f"{self.peak_mem_gb:.2f} GB",
                "inference": f"{peak_inference_mem_gb:.2f} GB"
            }
        }

        json_filename = os.path.join(result_dir, f"kmmlu_{self.output_prefix}_summary.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        print(f"요약 저장 완료: {json_filename}\n")

    def _get_official_subsets(self):
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
        mapping = {}
        for cat, subs in cats.items():
            for s in subs:
                mapping[s] = cat
        return mapping


def main():
    parser = KMMLUArgumentManager.get_eval_parser()
    args = parser.parse_args()
    evaluator = KMMLUEvaluator(
        model_id=args.model_id,
        batch_size=args.batch_size,
        seed=args.seed,
        num_shots=args.num_shots,
        prompting_strategy=args.prompting_strategy,
        output_prefix=args.output_prefix,
        subsets_to_test=args.subsets,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()