#!/usr/bin/env python3
"""
간단한 테스트 스크립트 - FAISS 기반
"""

import os
import sys
from pathlib import Path

# ChromaDB 텔레메트리 비활성화
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

DEPENDENCIES_AVAILABLE = True
MISSING_DEPENDENCY = ""

try:
    from src.loaders import DocumentLoader
    from src.vectorstore import VectorDatabase
    from src.rag import RAGChain
except ModuleNotFoundError as exc:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPENDENCY = getattr(exc, "name", "") or str(exc)
    DocumentLoader = VectorDatabase = RAGChain = None  # type: ignore


def run_answer_formatter_smoke() -> None:
    """AnswerFormatter 핵심 동작을 빠르게 점검한다."""
    from src.utils.answer_formatter import AnswerFormatter

    formatter = AnswerFormatter()
    sample_text = (
        "백제의 수도인 위례성 일대는 몽촌토성과 풍납토성이 함께 지킵니다.\n\n"
        "1. 몽촌토성은 왕궁 방어선이자 생활 터전이었습니다.\n"
        "2. 풍납토성은 행정과 의례를 담당했습니다."
    )
    formatted = formatter.format(sample_text, [])
    assert formatted.startswith("### 핵심 요약")
    assert "### 상세 설명" in formatted
    assert "- 몽촌토성은" in formatted
    print("✅ AnswerFormatter 스모크 테스트 통과")


def main():
    print("🔧 RAG 시스템 테스트 시작...")
    run_answer_formatter_smoke()

    if not DEPENDENCIES_AVAILABLE:
        print(f"⚠️ 환경에 필요한 모듈이 없어 RAG 통합 검증을 건너뜁니다: {MISSING_DEPENDENCY}")
        return
    
    # 1. 문서 로더 테스트
    print("\n1️⃣ 문서 로더 초기화...")
    loader = DocumentLoader()
    
    # 2. domain.md 파일 로드
    if os.path.exists("domain.md"):
        print("\n2️⃣ domain.md 파일 로드 중...")
        documents = loader.load_document("domain.md")
        print(f"✅ {len(documents)}개의 청크로 분할됨")
        
        # 3. 벡터 DB 초기화 및 문서 추가
        print("\n3️⃣ 벡터 데이터베이스 초기화...")
        vector_db = VectorDatabase()
        vector_db.add_documents(documents)
        print(f"✅ 벡터 DB에 저장된 문서 수: {vector_db.get_document_count()}")
        
        # 4. RAG 체인 테스트
        print("\n4️⃣ RAG 체인 초기화...")
        rag_chain = RAGChain()
        
        # 5. 질문 테스트
        test_question = "정보시스템 사업이란 무엇인가요?"
        print(f"\n5️⃣ 테스트 질문: {test_question}")
        
        response = rag_chain.query(test_question)
        print(f"\n답변: {response['answer'][:200]}...")
        print(f"상태: {response['status']}")
        print(f"참고 문서 수: {len(response['sources'])}")
        
    else:
        print("❌ domain.md 파일을 찾을 수 없습니다.")
    
    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    main()