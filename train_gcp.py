import json
import glob
import random
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import torch
import os
from collections import defaultdict
from huggingface_hub import HfApi
from dotenv import load_dotenv
from unsloth import FastLanguageModel

load_dotenv()

def load_filtered_samples(file_path):
    """로컬에서 생성된 필터링 데이터 로드"""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)
    return samples

def format_instruction(sample):
    return f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""

def train_model_with_lora(instruction_data, output_dir="./output/lora_model"):
    print("\n=== LoRA 학습 시작 ===")
    print(f"학습 데이터 개수: {len(instruction_data)}개")
    
    dataset = Dataset.from_list(instruction_data)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="skt/A.X-4.0-Light",
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_steps=50,
        report_to="wandb",
        run_name="kmmlu-lora-ax4-light"
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        formatting_func=format_instruction,
        max_seq_length=1024,
    )
    
    print("\n학습 시작...")
    trainer.train()
    
    print(f"\n모델 저장: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("학습 완료")
    return model, tokenizer

def upload_to_huggingface(model_path, repo_name):
    print(f"\n=== 허깅페이스 업로드 시작 ===")
    print(f"모델 경로: {model_path}")
    print(f"리포지토리: {repo_name}")
    
    api = HfApi()
    
    try:      
        try:
            api.create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)
            print(f"리포지토리 생성/확인 완료: {repo_name}")
        except Exception as e:
            print(f"리포지토리 생성 중 경고: {e}")
        
        print("\n파일 업로드 중...")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            repo_type="model",
            commit_message="Upload LoRA model trained on KMMLU-HUMSS instruction tunning with A.X-4.0-Light"
        )
        
        print(f"\n업로드 완료: https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        print(f"\n업로드 중 오류 발생: {e}")
        return False

def main():
    print("=" * 50)
    print("GCP: KMMLU LoRA 학습")
    print("=" * 50)
    
    # 로컬에서 생성된 instruction 데이터 로드
    instruction_data = load_filtered_samples("./data/instruction_data.jsonl")
    print(f"Instruction 데이터 로드 완료: {len(instruction_data)}개")
    
    # LoRA 학습
    model, tokenizer = train_model_with_lora(instruction_data, output_dir="./output/lora_model")
    
    # 메모리 정리
    print("\n메모리 정리 중...")
    del model
    torch.cuda.empty_cache()
    
    # 허깅페이스 업로드
    upload_to_huggingface(
        model_path="./output/lora_model",
        repo_name="sagittarius5/kmmlu-lora-skt-ax4-light2nd"  
    )
    
    print("\n" + "=" * 50)
    print("GCP 작업 완료")
    print("=" * 50)

if __name__ == "__main__":
    main()
