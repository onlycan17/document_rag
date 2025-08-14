#!/bin/bash

# 가상환경이 활성화되어 있는지 확인
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "🔄 가상환경 활성화 중..."
    source venv/bin/activate
fi

# ChromaDB 텔레메트리 비활성화
export ANONYMIZED_TELEMETRY=False

# PyTorch MPS 메모리 안정화 옵션
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}
export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}

# A.X 멀티모달 메모리 튜닝(필요시 환경변수로 조정)
export AX_MAX_NEW_TOKENS=${AX_MAX_NEW_TOKENS:-128}
export AX_IMAGE_MAX_SIZE=${AX_IMAGE_MAX_SIZE:-896}

# Streamlit 앱 실행
echo "🚀 RAG 챗봇을 시작합니다..."
streamlit run app.py