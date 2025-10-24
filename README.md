# KMMLU 평가 스크립트 요약

## 1. kmmlu_Base_evaluator.py

- 평가 방식: Zero-shot
- 모델: HuggingFace 기반 LLM
- 기법: 4-bit QLoRA (BitsAndBytesConfig)
- 주요 기능:
  - KMMLU 전체 subset 평가
  - 분야별(supercategory) 및 전체 정확도 계산
  - 출력: CSV, JSON 저장

## 2. kmmlu_Qwen2.5_7B_instruct_fewshot.py

- 평가 방식: Few-shot (5-shot)
- 모델: Qwen2.5-7B-Instruct
- 기법: 4-bit QLoRA
- 주요 기능:
  - train에서 few-shot 예시 추출
  - KMMLU 45개 subset 평가
  - 정확도 출력 및 저장

## 3. kmmlu_Qwen2.5_7B_instruct_fewshot_cot.py

- 평가 방식: Zero-shot CoT (Chain of Thought)
- 모델: Qwen2.5-7B-Instruct
- 기법: 4-bit QLoRA + CoT 프롬프트 추가
- 주요 기능:
  - CoT 프롬프트 추가 (랜덤 선택된 reasoning 문장 포함)
  - few-shot 또는 zero-shot 조건 설정 가능
  - 평가 결과 저장

## 4. kmmlu_Qwen2.5_7B_instruct_sft_kmmlu

- 평가 방식: 
  - (1) KMMLU HUMSS subset 기반 SFT
  - (2) KMMLU Few-shot 평가
- 모델: Qwen2.5-7B (Unsloth 최적화)
- 기법:
  - SFT (Supervised Fine-Tuning) 지원
  - Unsloth로 모델 최적화
  - LoRA (Low-Rank Adaptation)
  - 4-bit 양자화
- 주요 기능:
  - KMMLU HUMSS 데이터로 SFT 가능
  - KMMLU HUMSS 만 학습, 평가 가능
  - KMMLU HUMSS 만 학습, KMMLU 전체 데이터셋 평가 가능
  - 수동 few-shot 예시 지정 옵션 제공

## 5. kmmlu_Qwen2.5_7B_instruct_zeorcot_DPO.py

- 평가 방식: Zero-shot CoT
- 모델: Qwen2.5-7B-Instruct
- 기법:
  - DPO (Direct Preference Optimization) 후 모델
  - 4-bit QLoRA
  - CoT 프롬프트
- 주요 기능:
  - zero-shot CoT 평가
  - CoT 프롬프트에 여러 reasoning 문장 삽입
  - DPO 모델을 백업하여 저장(DPO 모델학습은 별도 코드로 진행)

## 기법 요약표

| 기법      | 사용 여부 | 파일 번호     |
|-----------|-----------|---------------|
| QLoRA     | 사용함    | 1, 2, 3, 5     |
| SFT       | 사용함    | 4              |
| LoRA      | 사용함    | 4              |
| Unsloth   | 사용함    | 4              |
| CoT       | 사용함    | 3, 5           |
| DPO       | 사용함    | 5              |
| Few-shot  | 사용함    | 2, 3, 4, 5     |
| Zero-shot | 사용함    | 1, 3, 5        |


