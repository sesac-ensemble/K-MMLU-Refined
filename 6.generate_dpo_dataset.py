
# ============================================================
# 스크립트 설명: DPO(Directional Preference Optimization) 학습용
# ------------------------------------------------------------
# 목적
#   - Qwen/Qwen2.5-7B-Instruct 모델을 DPO 방식으로 미세조정
#   - 사용자 선호 데이터(`dpo_dataset.jsonl`) 기반 응답 품질 개선
#
# 사용된 기법
#   - QLoRA (4-bit 양자화, nf4, double quant)
#   - LoRA (q_proj, k_proj, v_proj, o_proj 모듈 대상)
#   - DPO (Direct Preference Optimization)
#
# 모델 및 환경
#   - 모델: Qwen/Qwen2.5-7B-Instruct
#   - 데이터: dpo_dataset.jsonl (jsonl 형식, prompt / chosen / rejected 포함)
#   - 학습 구성: 에폭 3, 배치 크기 1, gradient_accum 8
#   - 출력 디렉토리: ./dpo_ft_Qwen2.5-7B-Instruct
#
# 수정이 필요한 주요 지점:
# ------------------------------------------------------------
# 1. 모델 변경 시
#    → model_id = "..." (줄 9 근처)
#
# 2. DPO 학습 데이터 변경 시
#    → dataset = load_dataset("json", data_files="...") (줄 42 근처)
#
# 3. LoRA 적용 모듈 변경 시
#    → target_modules=["q_proj", ...] (줄 31 근처)
#
# 4. 학습 파라미터(에폭, 배치 등) 변경 시
#    → DPOConfig(...) (줄 46 ~ 61 근처)
#
# 5. 출력 저장 위치 변경 시
#    → save_path = "./dpo_ft_..." (줄 71 근처)
#
# 주의사항:
#   - CUDA 환경에서는 `expandable_segments` 설정 유지 권장
#   - tokenizer pad_token 설정 누락 시 오류 발생 가능
#
# ============================================================



from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import torch
import os

# (메모리 조각화 방지)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 1. 모델 설정
model_id = "Qwen/Qwen2.5-7B-Instruct-"  # unsloth/Qwen2.5-7B-Instruct

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
)

# 2. PEFT (LoRA) 설정
base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 모델에 따라 다를 수 있음
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)

# 3. DPO 학습용 데이터셋
dataset = load_dataset("json", data_files="dpo_dataset.jsonl")["train"]

# 데이터 순서를 무작위로 섞기 (재현 가능한 시드 고정)
dataset = dataset.shuffle(seed=42)

# 4. DPO 설정
dpo_config = DPOConfig(
    beta=0.1,
    max_prompt_length=256,
    max_completion_length=64,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    learning_rate=5e-6,
    bf16=True,
    remove_unused_columns=False,
    report_to="none",
    output_dir="./dpo_output",
    generate_during_eval=False,
)

# 5. DPO 트레이너
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    train_dataset=dataset,
)

# 6. 학습
trainer.train()

# 7. 모델 저장
save_path = f"./dpo_ft_{model_id.split('/')[-1]}"
trainer.save_model(save_path)
