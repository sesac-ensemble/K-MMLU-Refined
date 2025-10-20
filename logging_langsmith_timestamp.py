""" json 파일 langsmith에 logging  : 시분초 run_name에 있어서 중복가능함

.env파일에 langchain api key 저장 필요함
LANGCHAIN_API_KEY="key"
"""

import os
from dotenv import load_dotenv
import json
from langsmith import Client
from datetime import datetime, timezone
import uuid
import traceback
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--update', action='store_true', help='기존 run을 업데이트합니다')
args = parser.parse_args()

load_dotenv()
client = Client()

####################################
# 
# 변경 할 것 4개
#   json_file_name 
#   RUNNER_INITIAL
#   CURRENT_DATE 
#   tags_for_run
#
####################################

# --- 1. JSON 파일 로드 ---
json_file_name = 'kmmlu_baseline_v1.json'           ################# 수정
try:
    with open(json_file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"오류: 파일 '{json_file_name}'을 찾을 수 없습니다.")
    exit()

# Run 설정
RUNNER_INITIAL = "JH"  # 이름 이니셜              ################# 수정
CURRENT_DATE = "251019"                         ################# 수정
MODEL_VERSION = data['model_id'].split('/')[-1]

# 타임스탬프 추가하여 매번 다른 run_id 생성
timestamp = datetime.now().strftime("%H%M%S")
run_name = f"{CURRENT_DATE}-{timestamp}-{RUNNER_INITIAL}-{MODEL_VERSION}"
tags_for_run = ['K-MMLU-Evaluation', 'Baseline', 'Random-5Shot']   ################# 수정

# run_name을 기반으로 UUID 생성
run_id = uuid.uuid5(uuid.NAMESPACE_DNS, run_name)
trace_id = run_id

start_time = datetime.now(timezone.utc)
end_time = datetime.now(timezone.utc)

try:
    client.create_run(
        name=run_name,
        id=run_id,
        trace_id=trace_id,
        run_type="chain",
        project_name="K-MMLU-Refined_dev",
        inputs={
            "model_id": data['model_id'],
            "evaluation_date": data['evaluation_date'],
            "seed": data['experiment_config']['seed'],
            "batch_size": data['experiment_config']['batch_size'],
            "num_shots": data['experiment_config']['num_shots'],
            "prompting_strategy": data['experiment_config']['prompting_strategy'],
            "total_questions": data['summary']['total_questions']
        },
        outputs={
            "overall_accuracy": data['summary']['overall_accuracy'],
            "correct_answers": data['summary']['correct_answers'],
            "category_accuracy": data['summary']['category_accuracy']
        },
        start_time=start_time,
        end_time=end_time,
        tags=tags_for_run,
        dotted_order=f"{start_time.strftime('%Y%m%dT%H%M%S%fZ')}{str(trace_id)}"
    )
    
    client.flush()
    
    print(f"====== LangSmith에 평가 결과가 성공적으로 로깅되었습니다.")
    print(f"   Run Name: {run_name}")
    print(f"   Run ID: {run_id}")
    print(f"   Timestamp: {timestamp}")
    
except Exception as e:
    print(f"❌ LangSmith 로깅 중 오류 발생: {e.__class__.__name__}: {e}")
    traceback.print_exc() # 상세한 오류 정보 출력