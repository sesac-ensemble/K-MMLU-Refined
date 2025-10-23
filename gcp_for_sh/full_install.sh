#!/bin/bash

echo "=== AI 개발환경 자동화 설치 스크립트 (CUDA/pyenv/venv 등) ==="

# 1. CUDA 설치 단계 (이미 설치됐으면 스킵)
if ! nvidia-smi &> /dev/null; then
    echo "[1단계] CUDA 및 NVIDIA Driver 자동 설치 시작..."
    UBU_VER=$(lsb_release -rs 2>/dev/null || echo "")
    if [ -z "$UBU_VER" ]; then
        echo "[안내] Ubuntu 22.04 (Jammy) 환경에서 동작을 검증했습니다. 다른 버전에서는 실패할 수 있습니다."
    elif [ "$UBU_VER" != "22.04" ]; then
        echo "[경고] 현재 Ubuntu $UBU_VER 입니다. Ubuntu 22.04 (Jammy) 사용을 권장합니다."
    else
        echo "[확인] Ubuntu 22.04 (Jammy) 환경입니다."
    fi

    echo "어떤 GPU를 사용합니까?"
    echo "1) L4"
    echo "2) T4"
    echo "3) V100"
    read -p "선택 (1/2/3): " GPUTYPE
    case $GPUTYPE in
        1) CUDA_TYPE="l4" ;;
        2) CUDA_TYPE="t4" ;;
        3) CUDA_TYPE="v100" ;;
        *) echo "[에러] 잘못 입력하셨습니다. 스크립트 종료."; exit 1 ;;
    esac

    bash cuda_install.sh "$CUDA_TYPE"
    echo ""
    echo "==============================="
    echo "[안내] CUDA 설치가 완료되었습니다."
    echo "지금 곧 시스템이 자동으로 재부팅됩니다."
    echo "재부팅이 끝난 후, 반드시 이 스크립트(full_install.sh)를 다시 실행하세요!"
    echo "==============================="
    sleep 5
    sudo reboot
    exit 0
else
    echo "[확인] CUDA 및 NVIDIA 드라이버가 이미 설치되어 있습니다. (nvidia-smi OK)"
fi

# 2. pyenv 설치 단계 (이미 설치됐으면 스킵)
if ! command -v pyenv &> /dev/null; then
    echo "[2단계] pyenv 종속성/설치 자동화 시작..."
    bash dependencies_install.sh
    bash pyenv_setup.sh
    echo ""
    echo "==============================="
    echo "[안내] pyenv 설치가 완료되었습니다."
    echo ""
    echo "아래 명령어를 복사해서 터미널에 붙여넣으세요:"
    echo ""
    echo "    source ~/.bashrc"
    echo ""
    echo "명령 입력 후, 반드시 다시 아래 명령을 실행하세요:"
    echo ""
    echo "    bash full_install.sh"
    echo ""
    echo "==============================="
    exit 0
else
    echo "[확인] pyenv가 이미 설치되어 있습니다."
fi
#pyenv 초기화로 반드시 쉘 재시작 필요 (source ~/.bashrc)


# pyenv 환경변수 강제 적용 (중복 적용 무해)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# 3. 파이썬 버전 및 가상환경 체크/생성
PYTHON_VERSION="3.10.8"
VENV_NAME="my_env"

if pyenv versions --bare | grep -q "^$PYTHON_VERSION$"; then
    echo "[확인] Python $PYTHON_VERSION 이미 설치됨."
else
    echo "[3단계] Python $PYTHON_VERSION 설치 시작..."
    pyenv install $PYTHON_VERSION
fi

if pyenv virtualenvs --bare | grep -q "^$VENV_NAME$"; then
    echo "[확인] 가상환경 '$VENV_NAME' 이미 생성됨."
else
    echo "[3단계] 가상환경 '$VENV_NAME' 생성 시작..."
    pyenv virtualenv $PYTHON_VERSION "$VENV_NAME"
fi

pyenv activate "$VENV_NAME" && echo "[확인] 가상환경 '$VENV_NAME' 활성화됨."

# **PyTorch CUDA 메모리 할당 최적화 환경변수 추가**
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[설정] PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 적용됨."

# 4. requirements.txt 설치 (경로 반드시 확인)
REQ_PATH="$(dirname "$0")/requirements.txt"
if [ -f "$REQ_PATH" ]; then
    echo "[4단계] requirements.txt 발견! pip install 시작..."
    pip install --upgrade pip
    pip install -r "$REQ_PATH"
else
    echo "[경고] requirements.txt 파일이 없습니다! ($REQ_PATH)"
fi

echo "=== 전체 개발환경 설치 완료! ==="