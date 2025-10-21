import torch
import argparse
import os
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from parser import KMMLUArgumentManager

# 1. Argparser 설정 및 인자 파싱
def parse_args() -> argparse.Namespace:
    parser = KMMLUArgumentManager.get_train_parser()
    args = parser.parse_args()
    return args

# 2. LoRA 설정 정의 함수 (Argparse 인자를 사용)
def create_lora_config(args: argparse.Namespace) -> LoraConfig:
    # 모든 Attention 및 FFN 레이어에 LoRA 적용 (성능 향상 목적)
    target_modules = [
        "q_proj", "v_proj", "o_proj", "k_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    lora_config = LoraConfig(
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        target_modules = target_modules,
        lora_dropout = args.lora_dropout,
        bias = "none",
        task_type = "CAUSAL_LM",
    )
    return lora_config

# 3. 모델 및 토크나이저 로드 (Unsloth QLoRA 모드)
def load_unsloth_model(args: argparse.Namespace, lora_config: LoraConfig):
    print(f"모델 로딩 중: {args.model_id} (QLoRA 4bit)")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_id,
        max_seq_length = args.max_seq_length,
        dtype = None,           # 자동으로 최적의 dtype (bf16) 선택
        load_in_4bit = True,    # QLoRA (4bit 양자화) 적용
    )
    
    # LoRA Config를 모델에 적용 (PEFT)
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_config.r,                           # r 값 전달
        target_modules = lora_config.target_modules, # target_modules 전달
        lora_alpha = lora_config.lora_alpha,         # lora_alpha 전달
        lora_dropout = lora_config.lora_dropout,     # lora_dropout 전달
        bias = lora_config.bias,                     # bias 전달
        use_gradient_checkpointing = "unsloth",      # 메모리 절약 기능 활성화
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

# 4. 데이터셋 로드 및 포맷팅 (사용자가 구축한 Instruction Data셋에 맞춰 변경 필요)
def load_and_format_data(tokenizer):
    # 🚨🚨 주의: 이 데이터셋은 Alpaca 예시이며, 실제로는 KMMLU 취약 분야 데이터로 대체해야 합니다. 
    print("데이터셋 로드 중...")
    dataset = load_dataset("nayohan/math-gpt-4o-200k-ko", split="train[:1000]") 

    # Instruction Tuning 포맷 함수 (Qwen Instruct 포맷)
    def formatting_prompts_func(examples):
        texts = []
        # kmmlu dataset용
        # for instruction, input_text, output_text in zip(examples["instruction"], examples["input"], examples["output"]):
        #     prompt = f"### Instruction:\n{instruction}\n\n"
        #     if input_text:
        #         prompt += f"### Input:\n{input_text}\n\n"
        #     prompt += f"### Response:\n{output_text}"
        #     texts.append(prompt)
        
        # nayohan/math-gpt-4o-200k-ko dataset용
        for prompt_text, response_text in zip(examples["prompt"], examples["response"]):
            prompt = f"### Instruction:\n{prompt_text}\n\n"
            prompt += f"### Response:\n{response_text}"
            texts.append(prompt)
        
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset

# 5. 메인 실행 함수
def main():
    args = parse_args() 
    lora_config = create_lora_config(args)
    
    model, tokenizer = load_unsloth_model(args, lora_config)
    train_dataset = load_and_format_data(tokenizer)

    # 학습 인자 설정
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        warmup_steps=50,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_available(),
        bf16=torch.cuda.is_available(), # A100 환경에서 bf16 사용
        logging_steps=1,
        output_dir=args.output_dir,
        optim="adamw_8bit",
        seed=args.seed,
        save_strategy="epoch",
    )

    # SFT Trainer 설정 및 학습 시작
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
    )
    
    print("\n" + "="*60)
    print("SFT 학습 시작")
    print("="*60 + "\n")

    trainer.train()

    # LoRA 가중치 병합 및 저장
    print("\n학습 완료! LoRA 가중치 병합 및 저장 중...")
    output_path = os.path.join(args.output_dir, "merged_model")
    model.save_pretrained_merged(
        output_path, 
        tokenizer, 
        save_method = "merged_4bit_forced",
    )
    print(f"최종 모델 저장 완료: {output_path}")

if __name__ == "__main__":
    main()