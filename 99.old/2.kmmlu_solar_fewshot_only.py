#!/usr/bin/env python
# -*- coding: utf-8 -*-
# run_kmmlu_solar_fewshot_only.py
"""
KMMLU 평가 전용 (Few-shot 기반)
- SFT(학습) 단계 제거
- Few-shot: MANUAL 또는 dev 랜덤 5개 선택 (--use_manual_fewshots)
- KMMLU 45개 subset 평가 + SuperCategory 평균 출력
"""

import os, re, time, random, argparse
from datetime import datetime
import torch, pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from unsloth import FastLanguageModel

# -----------------------------------------------------
# Manual 5-shot 예시
# -----------------------------------------------------
MANUAL_FEWSHOTS = [
    {"question": "반도체에서 전자가 이동하는 주요 경로는 무엇인가?", "A": "원자핵", "B": "전도대", "C": "가전자대", "D": "절연층", "answer": "2"},
    {"question": "형법상 '고의'의 의미는 무엇인가?", "A": "결과 발생을 예견하지 못함", "B": "결과 발생을 원하거나 용인함", "C": "단순한 과실", "D": "의무 위반에 대한 무지", "answer": "2"},
    {"question": "패션 트렌드가 순환적으로 반복되는 현상을 무엇이라 하는가?", "A": "패션 사이클", "B": "패션 모듈", "C": "패션 블렌드", "D": "패션 세그먼트", "answer": "1"},
    {"question": "재난 대응 단계 중 현장지휘체계를 구성하는 주체는?", "A": "중앙정부", "B": "지방자치단체장", "C": "소방본부장", "D": "경찰청장", "answer": "2"},
    {"question": "세포 내 에너지 생성의 주요 기관은?", "A": "핵", "B": "미토콘드리아", "C": "리보솜", "D": "소포체", "answer": "2"},
]

# -----------------------------------------------------
# 모델 로드
# -----------------------------------------------------
def _is_bf16_supported():
    return torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()

def load_unsloth_model(model_id: str, use_flash_attn2: bool = True):
    print(f"\n{'='*60}\n모델 로딩 중: {model_id}\n{'='*60}\n")
    dtype = torch.bfloat16 if _is_bf16_supported() else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=4096,
        dtype=dtype,
        load_in_4bit=True,
    )
    try:
        FastLanguageModel.for_inference(model, use_flash_attention_2=use_flash_attn2)
        print("FlashAttention2 활성화됨")
    except Exception:
        print("FlashAttention2 미지원 환경 — 기본 모드로 로드")

    tokenizer.padding_side = "left"
    if not getattr(tokenizer, "pad_token", None):
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    model.eval()
    return model, tokenizer


# -----------------------------------------------------
# 평가 클래스
# -----------------------------------------------------
class KMMLUEvaluator:
    def __init__(self, model_id, batch_size=2, use_flash_attn2=True, use_manual_fewshots=False):
        self.model, self.tokenizer = load_unsloth_model(model_id, use_flash_attn2)
        self.batch_size = batch_size
        self.use_manual_fewshots = use_manual_fewshots
        self.subsets = self._get_official_subsets()
        self.supercats = self._get_supercategories()

    # Few-shot 선택
    def _get_fewshots(self, subset):
        if self.use_manual_fewshots:
            return MANUAL_FEWSHOTS
        try:
            ds = load_dataset("HAERAE-HUB/KMMLU", subset)
            for split in ["dev", "validation", "test", "train"]:
                if split in ds:
                    valid = [ex for ex in ds[split] if str(ex.get("answer")) in ["1","2","3","4"]]
                    if valid:
                        return random.sample(valid, min(5, len(valid)))
            return MANUAL_FEWSHOTS
        except:
            return MANUAL_FEWSHOTS

    # Prompt 구성
    def _make_prompt(self, ex, fewshots):
        norm = lambda t: re.sub(r"\s+", " ", str(t)).strip()
        prompt = ""
        for fs in fewshots:
            prompt += f"문제: {norm(fs['question'])}\n"
            for c in ["A","B","C","D"]:
                prompt += f"{c}. {norm(fs.get(c,''))}\n"
            prompt += f"정답: {fs['answer']}\n\n"
        prompt += f"문제: {norm(ex['question'])}\n"
        for c in ["A","B","C","D"]:
            prompt += f"{c}. {norm(ex.get(c,''))}\n"
        prompt += "정답: "
        return prompt

    # 정답 변환 (1~4 → 0~3)
    def _extract_answer_index(self, ex):
        a = ex.get("answer")
        if isinstance(a, int): return a-1 if 1 <= a <= 4 else None
        if isinstance(a, str) and a.isdigit(): return int(a)-1
        return None

    def _predict_logits(self, inputs):
        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :]
        choice_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in ["A","B","C","D"]]
        preds = torch.argmax(logits[:, choice_ids], dim=-1)
        return preds.cpu().tolist()

    # 평가 실행
    def evaluate(self):
        results, all_correct, all_total = [], 0, 0
        start = time.time()
        for subset in tqdm(self.subsets, desc="KMMLU 전체 평가"):
            try:
                ds = load_dataset("HAERAE-HUB/KMMLU", subset)
                for split in ["test","validation","dev","train"]:
                    if split in ds:
                        test = list(ds[split])
                        break
                else:
                    continue
                test = [t for t in test if str(t.get("answer")) in ["1","2","3","4"]]
                if not test: continue

                fewshots = self._get_fewshots(subset)
                prompts = [self._make_prompt(t,fewshots) for t in test]
                truths = [self._extract_answer_index(t) for t in test]
                correct = total = 0

                for i in range(0,len(prompts),self.batch_size):
                    batch = prompts[i:i+self.batch_size]
                    truth = truths[i:i+self.batch_size]
                    inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=3072).to(self.model.device)
                    preds = self._predict_logits(inputs)
                    for p,t in zip(preds,truth):
                        if t is not None:
                            total+=1
                            if p==t: correct+=1
                acc = correct/total if total else 0.0
                results.append({"Subset":subset,"SuperCategory":self.supercats.get(subset,"Unknown"),"Accuracy":acc})
                all_correct+=correct; all_total+=total
                print(f"{subset}: {acc:.4f} ({correct}/{total})")

            except Exception as e:
                print(f"{subset} 오류: {e}")
                import traceback; traceback.print_exc()
                results.append({"Subset":subset,"SuperCategory":self.supercats.get(subset,"Unknown"),"Accuracy":0.0})

        # 요약 저장
        os.makedirs("results",exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(results)
        df.to_csv(f"results/kmmlu_eval_{ts}.csv",index=False,encoding="utf-8-sig")

        overall = all_correct/all_total if all_total else 0
        print(f"\n전체 평균: {overall:.4f} ({all_correct}/{all_total})")

        sc = df.groupby("SuperCategory")["Accuracy"].mean().sort_values(ascending=False)
        sc.to_csv(f"results/kmmlu_eval_supercat_{ts}.csv",encoding="utf-8-sig")
        print("\n[SuperCategory 평균 정확도]")
        print(sc)

    # 45개 Subset / SuperCategory
    def _get_official_subsets(self):
        return [
            "Accounting","Agricultural-Sciences","Aviation-Engineering-and-Maintenance","Biology",
            "Chemical-Engineering","Chemistry","Civil-Engineering","Computer-Science","Construction",
            "Criminal-Law","Ecology","Economics","Education","Electrical-Engineering",
            "Electronics-Engineering","Energy-Management","Environmental-Science","Fashion",
            "Food-Processing","Gas-Technology-and-Engineering","Geomatics","Health","Industrial-Engineer",
            "Information-Technology","Interior-Architecture-and-Design","Law","Machine-Design-and-Manufacturing",
            "Management","Maritime-Engineering","Marketing","Materials-Engineering","Mechanical-Engineering",
            "Nondestructive-Testing","Patent","Political-Science-and-Sociology","Psychology",
            "Public-Safety","Railway-and-Automotive-Engineering","Real-Estate","Refrigerating-Machinery",
            "Social-Welfare","Taxation","Telecommunications-and-Wireless-Technology",
            "Korean-History","Math"
        ]

    def _get_supercategories(self):
        cats = {
            "STEM":["Biology","Chemical-Engineering","Chemistry","Civil-Engineering","Computer-Science",
                     "Ecology","Electrical-Engineering","Information-Technology","Materials-Engineering",
                     "Mechanical-Engineering","Math"],
            "HUMSS":["Accounting","Criminal-Law","Economics","Education","Law","Management",
                     "Political-Science-and-Sociology","Psychology","Social-Welfare","Taxation","Korean-History"],
            "Applied Science":["Aviation-Engineering-and-Maintenance","Electronics-Engineering",
                               "Energy-Management","Environmental-Science","Gas-Technology-and-Engineering",
                               "Geomatics","Industrial-Engineer","Machine-Design-and-Manufacturing",
                               "Maritime-Engineering","Nondestructive-Testing",
                               "Railway-and-Automotive-Engineering","Telecommunications-and-Wireless-Technology"],
            "Other":["Agricultural-Sciences","Construction","Fashion","Food-Processing","Health",
                     "Interior-Architecture-and-Design","Marketing","Patent","Public-Safety",
                     "Real-Estate","Refrigerating-Machinery"]
        }
        mapping={s:cat for cat,subs in cats.items() for s in subs}
        return mapping

# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="upstage/SOLAR-10.7B-Instruct-v1.0")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--no_flash_attn2", action="store_true")
    parser.add_argument("--use_manual_fewshots", action="store_true")
    args = parser.parse_args()

    evaluator = KMMLUEvaluator(
        model_id=args.model_id,
        batch_size=args.batch_size,
        use_flash_attn2=not args.no_flash_attn2,
        use_manual_fewshots=args.use_manual_fewshots
    )
    evaluator.evaluate()

if __name__ == "__main__":
    main()
