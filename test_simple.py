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

sys.path.append(str(Path(__file__).parent))

from src.loaders import DocumentLoader
from src.vectorstore import VectorDatabase
from src.rag import RAGChain

def main():
    print("🔧 RAG 시스템 테스트 시작...")
    
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