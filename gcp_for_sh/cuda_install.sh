#!/bin/bash

# GPU 타입별 CUDA 및 NVIDIA 드라이버 설치 스크립트
# 사용법: ./cuda_install.sh <l4|t4|v100>
# 드라이버 정책: 550-server 계열 통일, 유틸 패키지 포함 설치

GPU_TYPE=${1:-""}

echo "=== GPU 타입별 CUDA 설치 스크립트 (550-server 계열) ==="
echo "사용법: $0 <l4|t4|v100>"

if [ -z "$GPU_TYPE" ]; then
    echo "오류: GPU 타입 인자가 필요합니다."
    echo "사용법: $0 <l4|t4|v100>"
    exit 1
fi

# GPU 타입에 따른 스크립트 실행
case $GPU_TYPE in
    "l4")
        echo "L4 GPU용 설치를 시작합니다..."
        chmod +x cuda_install_l4.sh
        ./cuda_install_l4.sh
        ;;
    "t4")
        echo "T4 GPU용 설치를 시작합니다..."
        chmod +x cuda_install_t4.sh
        ./cuda_install_t4.sh
        ;;
    "v100")
        echo "V100 GPU용 설치를 시작합니다..."
        chmod +x cuda_install_v100.sh
        ./cuda_install_v100.sh
        ;;
    *)
        echo "지원되지 않는 GPU 타입입니다: $GPU_TYPE"
        echo "사용법: $0 <l4|t4|v100>"
        exit 1
        ;;
esac

echo "=== 설치 완료 ==="
echo "시스템 재부팅이 필요할 수 있습니다."
echo "재부팅 후 'nvidia-smi' 명령어로 GPU 인식 확인하세요."