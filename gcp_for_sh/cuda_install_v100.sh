#!/bin/bash

# V100 GPU용 CUDA 및 NVIDIA 드라이버 설치 스크립트
# V100 GPU는 Volta 아키텍처를 사용하므로 호환성 있는 드라이버 버전 사용

echo "=== V100 GPU용 CUDA 설치 시작 ==="

# 시스템 업데이트
sudo apt-get update && echo "System update complete."

# 기존 NVIDIA 드라이버 제거 (있다면)
sudo apt-get purge -y nvidia-* libnvidia-*
sudo apt-get autoremove -y

# 필요한 패키지 설치
sudo apt-get install -y wget gnupg2 software-properties-common build-essential dkms linux-headers-$(uname -r)

# NVIDIA 공식 GPG 키 추가
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# 패키지 리스트 업데이트
sudo apt-get update

# V100 GPU용 NVIDIA 드라이버 설치 (550-server 계열 + 유틸 포함)
sudo apt-get install -y \
  nvidia-driver-550-server \
  nvidia-utils-550-server \
  nvidia-compute-utils-550-server \
  libnvidia-compute-550-server \
  libnvidia-decode-550-server \
  libnvidia-encode-550-server \
  libnvidia-fbc1-550-server

# CUDA Toolkit 12.2 설치 (통일 버전)
sudo apt-get install -y cuda-toolkit-12-2

# 환경 변수 설정
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# CUDA 버전 확인
echo "=== 설치된 CUDA 버전 확인 ==="
nvcc --version

echo "=== NVIDIA 드라이버 버전 확인 ==="
nvidia-smi

echo "=== V100 GPU 설치 완료 ==="
echo "주의: 시스템 재부팅이 필요할 수 있습니다."
echo "재부팅 후 'nvidia-smi' 명령어로 GPU 인식 확인하세요."
sudo reboot
