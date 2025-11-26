## 변경사항
### kmmlu_evaluator.py
1. 환경 설정을 위한 bash 폴더 추가
2. unsloth 적용
3. subsets parser 추가
4. dataset load 최적화 적용
5. csv, json 파일 result 폴더에 추가
6. 최대 메모리 사용량 추적 추가

<br />

# K-MMLU 벤치마크 성능 향상 프로젝트

## 📌 프로젝트 개요
제한된 리소스(L4/A100 GPU)와 시간(1주일) 내에서 **K-MMLU(한국어 다중 과제 이해) 벤치마크** 성능을 극대화하는 것을 목표로 한 프로젝트입니다. 다양한 LLM 비교 후 `skt/A.X-4.0-Light`를 베이스 모델로 선정하여 최적화 및 튜닝을 진행했습니다.

<br />

## 🏆 최종 성과
**오답 기반 CoT Instruction Tuning** 전략을 통해 5-shot 베이스라인 대비 **7.8%p** 성능 향상을 달성했습니다.

<br />

| 구분 | 모델 (7B) | 방식 | 정확도 (Accuracy) | 비고 |
|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | skt/A.X-4.0-Light | Zero-shot | 56.25% | 베이스 선정 |
| **Reference**| skt/A.X-4.0-Light | 5-shot (No Tuning) | 51.03% | 비교군 |
| **Final** | **skt/A.X-4.0-Light** | **오답 CoT Tuning** | **58.83%** | **Best (+7.8%p)** |

<br />

## 🔑 핵심 접근 방식
1.  **오답 기반 CoT Instruction Tuning (Main Strategy)**
    * 5-shot 평가 수행 후 오답 데이터 추출 및 K-means 클러스터링(100개).
    * 대표 오답 유형에 대한 Chain-of-Thought(CoT) 데이터 생성 및 학습.
    * 전체 오답 데이터의 약 0.58%(17,155개 중 일부)만 활용하여 효율적 성능 개선 달성.
2.  **Unsloth 적용**
    * QLoRA 4bit 양자화를 통해 학습 속도 31.1% 향상 및 추론 메모리 20% 절감.
3.  **취약 과목 SFT**
    * 평균 점수 하위 과목(Math, History 등) 집중 학습을 통한 보완.

<br />

## 🛠️ 설치 및 환경 (Requirements)
본 프로젝트는 `unsloth` 라이브러리를 기반으로 합니다.

```bash
pip install -r requirements.txt
```

<br />

## 🚀 실행 방법 (Usage)

### 1. 모델 학습 (Train)
`train.py`를 실행하여 QLoRA SFT를 수행합니다.

```bash
python train.py \
    --model_id "skt/A.X-4.0-Light" \
    --output_dir "./result/checkpoint" \
    --num_epochs 3 \
    --batch_size 2
```

### 2. 모델 평가 (Evaluation)
`kmmlu_evaluator.py`를 사용하여 벤치마크를 평가합니다.

```bash
python kmmlu_evaluator.py \
    --model_id "./result/checkpoint/merged_model" \
    --num_shots 5 \
    --prompting_strategy "random"
```

<br />

## 📂 파일 구조
* `kmmlu_evaluator.py`: K-MMLU 평가 스크립트 (Unsloth 최적화 및 이터레이터 방식 적용).
* `train.py`: Unsloth 기반 QLoRA SFT 학습 스크립트.
* `kmmlu_parser.py`: 학습 및 평가에 사용되는 Argument Parser 모듈.
* `requirements.txt`: 필수 패키지 목록 (Torch 2.4.1, Unsloth, LangSmith 등).
  
### 1. `logging_langsmith_unique.py` (중복 방지)
* **목적:** 최종 리포팅을 위한 공식 평가 결과를 로깅합니다.
* **특징:** Run Name이 같으면 **로깅을 건너뛰어** (Skip) 데이터 중복을 방지합니다.

### 2. `logginng_langsmith_timestamp.py` (자유 테스트)
* **목적:** 하이퍼파라미터 튜닝이나 디버깅 등 테스트 기록을 위해 사용합니다.
* **특징:** Run Name에 타임스탬프를 추가하여 **매번 새로운 Run으로 기록**합니다.

<br />

## 👥 팀원 (Team 2)
이아민, 이지현, 장용석, 허재정
