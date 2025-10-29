#!/bin/bash
export TZ='Asia/Seoul'
set -e
# ===============================================================
# run_all_checkmem.sh
# - Unsloth + LoRA vs HF 기본모델 비교 실행 자동화 스크립트
# - GPU 메모리 로그를 CSV로 저장
# ===============================================================

MODEL_SCRIPT="4.kmmlu_Qwen2.5_7B_instruct_sft_kmmlu_checkmem.py"
LOG_DIR="logs_checkmem"

# 현재 시각을 파일 이름 태그로 사용
RUN_TAG=$(date '+%Y%m%d_%H%M')

CSV_LOG="$LOG_DIR/gpu_memory_log_${RUN_TAG}.csv"
# UNSLOTH_LOG="$LOG_DIR/unsloth_run_${RUN_TAG}.log"
# HF_LOG="$LOG_DIR/hf_run_${RUN_TAG}.log"
NO_SFT_LOG="$LOG_DIR/no_sft_run_${RUN_TAG}.log"
SFT_LOG="$LOG_DIR/sft_run_${RUN_TAG}.log"

mkdir -p "$LOG_DIR"

echo "timestamp,mode,gpu_used_MB,gpu_free_MB,gpu_total_MB" > "$CSV_LOG"

# GPU 메모리 체크 함수
log_gpu() {
  TAG=$1
  nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits | \
  awk -v mode="$TAG" -v ts="$(date '+%Y-%m-%d %H:%M:%S')" \
  -F',' '{print ts "," mode "," $1 "," $2 "," $3}' >> "$CSV_LOG"
}

# --------------------------
# Unsloth + LoRA ON
# --------------------------
echo "▶ [START] no sft 모드 실행"
log_gpu "no_sft_start"

# python "$MODEL_SCRIPT" \
#   --use_unsloth --use_lora --num_shots 0 \
#   --enable_gpu_memlog --gpu_memlog_every_sec 300 \
#   --eval_subsets humss | tee "$UNSLOTH_LOG"

python 4.kmmlu_Qwen2.5_7B_instruct_sft_kmmlu_checkmem.py \
  --eval_only \
  --num_shots 0 \
  --eval_subsets humss \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --output_prefix no_sft_0shot \
  --enable_gpu_memlog  \
  --gpu_memlog_every_sec 60  | tee "$NO_SFT_LOG"

log_gpu "unsloth_end"
echo "[DONE] no sft 모드 완료"
sleep 5

# # --------------------------
# # HF 기본 모델 (Unsloth OFF)
# # --------------------------
# echo "▶ [START] HF 기본 모델 실행"
# log_gpu "hf_start"

# python "$MODEL_SCRIPT" \
#   --num_shots 0 \
#   --enable_gpu_memlog --gpu_memlog_every_sec 300 \
#   --eval_subsets humss | ts '[%Y-%m-%d %H:%M:%S]' | tee "$HF_LOG"

# log_gpu "hf_end"
# echo "[DONE] HF 기본 모델 완료"

echo "▶ [START] SFT 기본 모델 실행"
log_gpu "sft_end"

python 4.kmmlu_Qwen2.5_7B_instruct_sft_kmmlu_checkmem.py \
  --eval_only \
  --num_shots 0 \
  --eval_subsets humss \
  --use_lora \
  --use_unsloth \
  --model_id results/lora_ft_Qwen2.5-7B-Instruct \
  --output_prefix sft_0shot \
  --enable_gpu_memlog \
  --gpu_memlog_every_sec 60  | tee "$SFT_LOG"

log_gpu "sft_end"
echo "[DONE] SFT 기본 모델 완료"

# --------------------------
# 결과 요약 출력
# --------------------------
echo ""
echo "실행 결과 요약:"
echo "----------------------------------------"
echo "GPU 메모리 로그 파일: $CSV_LOG"
# echo "Unsloth 실행 로그:    $UNSLOTH_LOG"
echo "No SFT 실행 로그:         $NO_SFT_LOG"
# echo "HF 실행 로그:         $HF_LOG"
echo "SFT 실행 로그:         $SFT_LOG"
echo "----------------------------------------"
