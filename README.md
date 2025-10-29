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

## 4. kmmlu_Qwen2.5_7B_instruct_sft_kmmlu_checkmem.py

- 평가 방식: 
  - 하나의 스크립트에서 SFT(선택) + KMMLU 평가까지 실행
  - 기본값은 zero-shot 평가
  - HUMSS 11개 subset으로 SFT를 수행하고, 평가 대상은 HUMSS 또는 지정한 서브셋/전체로 선택 가능
- 모델/옵션
  - 기본 모델: Qwen/Qwen2.5-7B-Instruct
  - SFT (Supervised Fine-Tuning) 지원
  - Unsloth 사용(선택)
  - LoRA (Low-Rank Adaptation) 사용(선택)
  - 4-bit 양자화
- 주요 기능:
  - KMMLU HUMSS 만 학습, 평가 가능
  - KMMLU HUMSS 만 학습, KMMLU 전체 데이터셋 평가 가능
  - 수동 few-shot 예시 지정 옵션 제공
  - Zero-shot 옵션 제공



## 기법 요약표

| 기법      | 사용 여부 | 파일 번호     |
|-----------|-----------|---------------|
| QLoRA     | 사용함    | 1, 2, 3,       |
| SFT       | 사용함    | 4              |
| LoRA      | 사용함    | 4              |
| Unsloth   | 사용함    | 4              |
| CoT       | 사용함    | 3,             |
| DPO       | 사용안함  |                |
| Few-shot  | 사용함    | 2, 3, 4        |
| Zero-shot | 사용함    | 1, 3           |


