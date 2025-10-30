from huggingface_hub import HfApi

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
            commit_message="Upload LoRA model trained on KMMLU-HUMSS instruction tuning with A.X-4.0-Light"
        )
        
        # 체크포인트 제외 필요한 파일만 업로드 할 때
        # api.upload_folder(
        #     folder_path="output/lora_model",
        #     repo_id=repo_name,
        #     repo_type="model",
        #     ignore_patterns=["checkpoint-*", "*.jsonl"],
        #     commit_message="Upload LoRA model trained on KMMLU-HUMSS instruction tuning with A.X-4.0-Light"
        # )

        
        print(f"\n업로드 완료: https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        print(f"\n업로드 중 오류 발생: {e}")
        return False

upload_to_huggingface("output/lora_model", "sagittarius5/kmmlu-lora-qwen3-8b-inst-test")
