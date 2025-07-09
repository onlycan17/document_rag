#!/bin/bash

echo "🚀 RAG 챗봇 환경 설정 시작..."

# 가상환경 생성
echo "📦 가상환경 생성 중..."
python3 -m venv venv

# 가상환경 활성화
echo "✅ 가상환경 활성화..."
source venv/bin/activate

# 패키지 설치
echo "📚 필요한 패키지 설치 중..."
pip install -r requirements.txt

# .env 파일 생성
if [ ! -f .env ]; then
    echo "📝 .env 파일 생성..."
    cp .env.example .env
    echo "⚠️  .env 파일을 열어 필요한 API 키를 설정해주세요."
fi

echo "✨ 설정 완료!"
echo ""
echo "다음 명령어로 앱을 실행하세요:"
echo "source venv/bin/activate"
echo "streamlit run app.py"