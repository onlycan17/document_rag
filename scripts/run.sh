#!/bin/bash

# 가상환경이 활성화되어 있는지 확인
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "🔄 가상환경 활성화 중..."
    source venv/bin/activate
fi

# ChromaDB 텔레메트리 비활성화
export ANONYMIZED_TELEMETRY=False

# Streamlit 앱 실행
echo "🚀 RAG 챗봇을 시작합니다..."
streamlit run app.py