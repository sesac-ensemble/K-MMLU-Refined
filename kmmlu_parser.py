import argparse
from typing import List, Optional

class KMMLUArgumentManager:
    """KMMLU 평가 및 학습에 공통으로 사용되는 ArgumentParser를 관리하는 클래스"""
    
    @staticmethod
    def get_eval_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
        """평가 스크립트 (kmmlu_evaluator.py)용 ArgumentParser를 생성합니다."""
        
        parser = argparse.ArgumentParser(
            prog=prog,
            description='KMMLU 모델 평가 도구',
            formatter_class=argparse.RawTextHelpFormatter # 긴 헬프 메시지 포맷팅
        )
        
        # 모델 및 기본 설정
        parser.add_argument('--model_id', type=str, 
                            default="Bllossom/llama-3.2-Korean-Bllossom-3B",
                            help='평가할 HuggingFace 모델 ID')
        parser.add_argument('--seed', type=int, default=42,
                            help='Random seed (재현성)')
        
        # 평가 관련 설정
        parser.add_argument('--batch_size', type=int, default=4,
                            help='배치 크기 (GPU 메모리에 따라 조정)')
        parser.add_argument("--num_shots", type=int, default=5, 
                            help="Few-shot 예시 개수 (0=zero-shot, 5=5-shot)")
        parser.add_argument("--prompting_strategy", type=str, default="random",
                            choices=["random", "zero_shot_cot", "similarity", "meta_prompt", 
                                     "gradient", "zero_shot", "self_consistency"],
                            help="프롬프트 전략")
        parser.add_argument('--subsets', type=str, nargs='+', default=None,
                            help='테스트할 subset 이름 목록 (예: Accounting Biology). 지정하지 않으면 전체 48개 평가.')

        # 출력 관련 설정
        parser.add_argument('--output_prefix', type=str, default=None,
                            help='출력 파일명 prefix (기본: 모델명_타임스탬프)')
        
        return parser
    
    @staticmethod
    def get_train_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
        """학습 스크립트 (train.py)용 ArgumentParser를 생성합니다. (eval parser 기반 확장)"""
        
        # 🌟 get_eval_parser를 사용하여 기본 인자를 가져옵니다.
        parser = KMMLUArgumentManager.get_eval_parser(prog)
        parser.description = 'Unsloth QLoRA SFT Training Script (KMMLU)'
        
        # 학습 관련 인자 추가/재정의
        parser.add_argument('--output_dir', type=str, 
                            default="Qwen/Qwen2.5-7B-Instruct",
                            help='학습 결과(체크포인트, 병합 모델) 저장 폴더')
        
        parser.add_argument('--max_seq_length', type=int, default=4096, 
                            help='최대 시퀀스 길이 (KMMLU 프롬프트 길이 고려)')
        parser.add_argument('--grad_acc_steps', type=int, default=4, 
                            help='그래디언트 누적 단계')
        parser.add_argument('--learning_rate', type=float, default=5e-5, 
                            help='학습률 (LoRA/QLoRA에 적합한 값)')
        parser.add_argument('--num_epochs', type=int, default=3, 
                            help='학습 에폭 수')
        
        # PEFT (LoRA) 설정 추가
        parser.add_argument('--lora_r', type=int, default=16, 
                            help='LoRA 랭크 (r)')
        parser.add_argument('--lora_alpha', type=int, default=32, 
                            help='LoRA 알파 (alpha)')
        parser.add_argument('--lora_dropout', type=float, default=0.05, 
                            help='LoRA 드롭아웃')
        
        return parser