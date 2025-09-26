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

# 선택적: Upstage 임베딩 클라이언트는 tokenizers 종속성 범위가 좁아 충돌 가능성 있음
# 최신 transformers와 함께 사용할 때는 --no-deps로 설치하고, 필요한 상위 deps는 requirements로 해결
if ! python -c "import importlib; importlib.import_module('langchain_upstage')" >/dev/null 2>&1; then
    echo "📦 langchain-upstage를 별도로 설치합니다 (--no-deps)"
    pip install --no-deps langchain-upstage==0.7.1 || true
fi

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
streamlit run app.py --server.port=8503