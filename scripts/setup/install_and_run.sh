#!/bin/bash

echo "🚀 RAG 챗봇 설치 및 실행 스크립트"

# 가상환경 확인 및 생성
if [ ! -d "venv" ]; then
    echo "📦 가상환경 생성 중..."
    python3 -m venv venv
fi

# 가상환경 활성화
echo "✅ 가상환경 활성화..."
source venv/bin/activate

# 패키지 업데이트
echo "📚 패키지 설치/업데이트 중..."
pip install --upgrade pip
pip install -r requirements.txt

# .env 파일 확인
if [ ! -f .env ]; then
    echo "📝 .env 파일 생성..."
    cp .env.example .env
    echo "⚠️  .env 파일을 열어 필요한 API 키를 설정해주세요."
fi

# ChromaDB 텔레메트리 비활성화
export ANONYMIZED_TELEMETRY=False
export CHROMA_TELEMETRY=False

# 앱 실행
echo "🚀 Streamlit 앱을 시작합니다..."
streamlit run app.py