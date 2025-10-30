import os
import json
import glob
import random
import numpy as np
from sklearn.cluster import KMeans
from langchain_upstage import UpstageEmbeddings
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.wrappers import wrap_openai
import re

load_dotenv()


def load_wrong_answers_from_file(file_path):
    wrong_samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            wrong_samples.append(sample)
    return wrong_samples

def load_correct_answers_from_file(file_path):
    correct_samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            correct_samples.append(sample)
    return correct_samples

def balance_by_subject(samples, target_ratio):
    print(f"\n=== 과목별 분포 ===")
    subject_groups = defaultdict(list)
    for sample in samples:
        subject = sample.get('subject', 'unknown')
        subject_groups[subject].append(sample)
    
    for subject, items in subject_groups.items():
        print(f"  {subject}: {len(items)}개")
    
    balanced = []
    for subject, items in subject_groups.items():
        target_count = max(1, int(len(items) * target_ratio))
        selected = random.sample(items, min(target_count, len(items)))
        balanced.extend(selected)
        print(f"  {subject} 선택: {len(selected)}개 (타겟: {target_count}개)")
    
    return balanced

def filter_wrong_answers_optimized(wrong_answers, target_ratio=0.05, api_key=None):
    print("\n=== 오답 필터링 시작 ===")
    print(f"전체 오답 샘플: {len(wrong_answers)}개")
    
    if len(wrong_answers) == 0:
        return []
    
    balanced_samples = balance_by_subject(wrong_answers, target_ratio)
    print(f"\n과목별 균형 조정 후: {len(balanced_samples)}개")
    
    if len(balanced_samples) == 0:
        return []
    
    print("\nUpstage 임베딩 생성 중...")
    
    # LangChain UpstageEmbeddings 사용
    embeddings_model = UpstageEmbeddings(
        api_key=api_key,
        model="embedding-query"
    )
    
    texts = [item['question'] for item in balanced_samples]
    embeddings = embeddings_model.embed_documents(texts)
    embeddings = np.array(embeddings)
    
    # 클러스터 개수 조정
    n_clusters = max(10, int(len(balanced_samples) * 0.3))
    print(f"\n클러스터 개수: {n_clusters}개 (샘플: {len(balanced_samples)}개)")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    
    filtered_samples = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
        if len(cluster_indices) > 0:
            selected_idx = random.choice(cluster_indices)
            filtered_samples.append(balanced_samples[selected_idx])
    
    print(f"\n최종 선택된 샘플: {len(filtered_samples)}개")
    print(f"데이터 감소율: {(1 - len(filtered_samples)/len(wrong_answers))*100:.2f}%")
    print("=" * 50)
    return filtered_samples

def select_correct_samples(correct_answers, n_samples=1000):
    print(f"\n=== 정답 샘플 선택 ===")
    print(f"전체 정답 샘플: {len(correct_answers)}개")
    
    random.seed(42)
    n_samples = min(n_samples, len(correct_answers))
    selected = random.sample(correct_answers, n_samples)
    
    print(f"선택된 정답 샘플: {len(selected)}개")
    return selected

@traceable(name="generate_cot_reasoning", run_type="llm", project_name="kmmlu-instruction-tuning")
def generate_cot_explanation(question, choices, answer, api_key):
    client = wrap_openai(OpenAI(
        api_key=api_key,
        base_url="https://api.upstage.ai/v1/solar"
    ))
    
    choices_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
    
    prompt = f"""다음 객관식 문제에 대해 단계별 추론 과정(Chain of Thought)을 생성해주세요.

문제: {question}

선택지:
{choices_text}

정답: {answer}

요구사항:
1. 문제를 이해하는 과정
2. 각 선택지를 분석하는 과정
3. 정답을 도출하는 논리적 추론 과정
4. 최종 답안

한국어로 작성해주세요."""
    
    try:
        response = client.chat.completions.create(
            model="solar-pro",
            messages=[
                {"role": "system", "content": "당신은 교육 전문가입니다. 학생들이 이해하기 쉽도록 단계별 추론 과정을 명확하게 설명해주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"CoT 생성 실패: {e}")
        return f"정답: {answer}"

def create_instruction_dataset_with_cot(samples, api_key):
    print("\n=== 오답 샘플 CoT 생성 시작 ===")
    print(f"총 {len(samples)}개 샘플에 대해 CoT 생성")
    
    instruction_data = []
    cot_records = []
    
    for idx, sample in enumerate(samples):
        if idx % 10 == 0:
            print(f"\n진행: {idx + 1}/{len(samples)}")
        
        question = sample.get('question', '')
        choices = sample.get('choices', [])
        answer = sample.get('answer', '')
        subject = sample.get('subject', 'unknown')
        
        cot_explanation = generate_cot_explanation(question, choices, answer, api_key)
        
        choices_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
        
        instruction = f"""다음 문제를 단계별로 풀어주세요.

문제: {question}

선택지:
{choices_text}

단계별 추론:"""
        
        instruction_data.append({
            "instruction": instruction,
            "input": "",
            "output": cot_explanation,
            "subject": subject,
            "type": "wrong_with_cot"
        })
        
        cot_records.append({
            "question": question,
            "choices": choices,
            "answer": answer,
            "subject": subject,
            "cot_explanation": cot_explanation,
            "type": "wrong_with_cot"
        })
    
    print(f"\n총 {len(instruction_data)}개의 오답 CoT 데이터 생성 완료")
    return instruction_data, cot_records

def create_simple_instruction_dataset(samples):
    print("\n=== 정답 샘플 단순 포맷 생성 ===")
    print(f"총 {len(samples)}개 샘플 처리")
    
    instruction_data = []
    
    for sample in samples:
        question = sample.get('question', '')
        choices = sample.get('choices', [])
        answer = sample.get('answer', '')
        
        choices_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
        
        instruction = f"""다음 문제를 풀어주세요.

문제: {question}

선택지:
{choices_text}"""
        
        output = f"정답은 {answer}입니다."
        
        instruction_data.append({
            "instruction": instruction,
            "input": "",
            "output": output,
            "subject": sample.get('subject', 'unknown'),
            "type": "correct_simple"
        })
    
    print(f"총 {len(instruction_data)}개의 정답 단순 데이터 생성 완료")
    return instruction_data

def save_instruction_data(instruction_data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in instruction_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"\nInstruction 데이터 저장 완료: {output_path}")

def save_cot_records(cot_records, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in cot_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"CoT 추론 과정 저장 완료: {output_path}")

def main():
    print("=" * 50)
    print("로컬: 데이터 준비 및 CoT 생성")
    print("=" * 50)
    
    # results_dir = "./results/skt__A.X-4.0-Light"
    output_base = "./output/skt__A.X-4.0-Light"
    os.makedirs(output_base, exist_ok=True)
    
    upstage_api_key = os.getenv('UPSTAGE_API_KEY')
    langsmith_api_key = os.getenv('LANGCHAIN_API_KEY')
    
    if not upstage_api_key or not langsmith_api_key:
        raise ValueError(".env 파일에 UPSTAGE_API_KEY와 LANGSMITH_API_KEY를 설정하세요")
    
    # print("\n1. 오답, 정답 데이터 추출")
    # wrong_answers, correct_answers = extract_wrong_answers(results_dir)
    # print(f"총 오답: {len(wrong_answers)}, 총 정답: {len(correct_answers)}")
    
    # with open(f"{output_base}/wrong_answers2.jsonl", 'w', encoding='utf-8') as f:
    #     for item in wrong_answers:
    #         f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # with open(f"{output_base}/correct_answers2.jsonl", 'w', encoding='utf-8') as f:
    #     for item in correct_answers:
    #         f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("1. 기존 오답/정답 파일 로드")
    wrong_answers = load_wrong_answers_from_file(f"{output_base}/wrong_answers_with_choices.jsonl")   # 2068개, humss 11개 서브셋
    correct_answers = load_correct_answers_from_file(f"{output_base}/correct_answers.jsonl")  # 3062
    print(f"총 오답: {len(wrong_answers)}, 총 정답: {len(correct_answers)}")
    
    print("\n2. 오답 필터링 ")
    filtered_wrong = filter_wrong_answers_optimized(
        wrong_answers, 
        target_ratio=0.15,   # 0.05 (30개) -> 0.15(90개 변경) 
        api_key=upstage_api_key
    )
    
    print("\n3. 정답 랜덤 선택 (1000개)")
    selected_correct = select_correct_samples(correct_answers, n_samples=1000)
    
    print("\n4. 오답 CoT 생성")
    wrong_instruction_data, cot_records = create_instruction_dataset_with_cot(
        filtered_wrong, 
        upstage_api_key
    )
    
    print("\n5. 정답 단순 포맷 생성")
    correct_instruction_data = create_simple_instruction_dataset(selected_correct)
    
    print("\n6. 데이터 결합")
    print(f"오답 샘플 (CoT): {len(wrong_instruction_data)}개")
    print(f"정답 샘플 (단순): {len(correct_instruction_data)}개")
    
    combined_data = wrong_instruction_data + correct_instruction_data
    print(f"총 학습 데이터: {len(combined_data)}개")
    
    random.seed(42)
    random.shuffle(combined_data)
    
    print("\n7. 데이터 저장")
    save_instruction_data(combined_data, f"{output_base}/instruction_data2.jsonl")
    save_cot_records(cot_records, f"{output_base}/cot_explanations2.jsonl")
    
    print("\n" + "=" * 50)
    print("로컬 작업 완료")
    print("생성된 파일:")
    print(f"  - {output_base}/instruction_data.jsonl: 학습용 데이터")
    print(f"  - {output_base}/cot_explanations.jsonl: CoT 추론 과정 기록")
    print(f"  - {output_base}/wrong_answers.jsonl: 추출된 오답")
    print(f"  - {output_base}/correct_answers.jsonl: 추출된 정답")
    print("instruction_data.jsonl을 GCP로 전송하여 학습을 진행하세요.")
    print("=" * 50)

if __name__ == "__main__":
    main()
