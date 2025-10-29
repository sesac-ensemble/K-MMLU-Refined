# K-MMLU-Refined
K-MMLU 벤치마크 최적화 프로젝트

<br />

## 변경사항
### kmmlu_evaluator.py
1. 환경 설정을 위한 bash 폴더 추가
2. unsloth 적용
3. subsets parser 추가
4. dataset load 최적화 적용
5. csv, json 파일 result 폴더에 추가
6. 최대 메모리 사용량 추적 추가

<br />

## LangSmith 평가 로깅 스크립트 (K-MMLU-Refined)

팀 프로젝트의 평가 결과를 LangSmith에 기록하기 위한 스크립트입니다.

### 1. `logging_langsmith_unique.py` (중복 방지)
* **목적:** 최종 리포팅을 위한 공식 평가 결과를 로깅합니다.
* **특징:** Run Name이 같으면 **로깅을 건너뛰어** (Skip) 데이터 중복을 방지합니다.

### 2. `logginng_langsmith_timestamp.py` (자유 테스트)
* **목적:** 하이퍼파라미터 튜닝이나 디버깅 등 테스트 기록을 위해 사용합니다.
* **특징:** Run Name에 타임스탬프를 추가하여 **매번 새로운 Run으로 기록**합니다.