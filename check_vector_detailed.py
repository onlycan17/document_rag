#!/usr/bin/env python3
"""
벡터 데이터베이스에 저장된 문서들을 상세히 확인하는 스크립트
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.vectorstore import VectorDatabase
import pickle
import os

def main():
    """벡터 DB 내용 상세 확인"""
    
    try:
        vector_db = VectorDatabase()
        print("벡터 데이터베이스 연결 성공")
        
        # 문서 수 확인
        doc_count = vector_db.get_document_count()
        print(f"총 저장된 문서 청크 수: {doc_count}개")
        
        if doc_count == 0:
            print("벡터 데이터베이스가 비어있습니다.")
            return
            
    except Exception as e:
        print(f"벡터 데이터베이스 연결 실패: {e}")
        return
    
    # FAISS 인덱스 파일 직접 확인
    faiss_path = "vector_db/faiss_index.pkl"
    if os.path.exists(faiss_path):
        try:
            with open(faiss_path, 'rb') as f:
                data = pickle.load(f)
                print(f"\nFAISS 인덱스 파일 정보:")
                print(f"- 파일 크기: {os.path.getsize(faiss_path) / (1024*1024):.1f} MB")
                
                if isinstance(data, dict):
                    print(f"- 데이터 키들: {list(data.keys())}")
                    
                    # documents_cache 확인
                    if 'documents_cache' in data:
                        docs = data['documents_cache']
                        print(f"- 캐시된 문서 수: {len(docs)}")
                        
                        # 파일별 문서 수 집계
                        file_count = {}
                        for doc in docs:
                            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                                source = doc.metadata['source']
                                file_count[source] = file_count.get(source, 0) + 1
                        
                        print(f"\n=== 파일별 청크 수 ===")
                        for file, count in sorted(file_count.items()):
                            filename = os.path.basename(file)
                            print(f"- {filename}: {count}개 청크")
                        
                        # 첫 몇 개 문서의 상세 정보 확인
                        print(f"\n=== 첫 5개 문서 상세 정보 ===")
                        for i, doc in enumerate(docs[:5]):
                            print(f"\n--- 문서 {i+1} ---")
                            if hasattr(doc, 'metadata'):
                                meta = doc.metadata
                                print(f"파일명: {meta.get('source', 'Unknown')}")
                                print(f"페이지: {meta.get('page', 'Unknown')}")
                                print(f"청크 ID: {meta.get('chunk_id', 'Unknown')}")
                            
                            # 내용 확인 (전체 내용)
                            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                            print(f"내용 길이: {len(content)}자")
                            print(f"내용 (처음 500자):\n{content[:500]}")
                            if len(content) > 500:
                                print("...")
                            
                        # 몽촌토성 관련 문서 확인
                        print(f"\n=== 몽촌토성 관련 문서 확인 ===")
                        mongchon_docs = [doc for doc in docs if '몽촌토성' in str(doc)]
                        print(f"몽촌토성 관련 청크 수: {len(mongchon_docs)}")
                        
                        # 내용이 짧은 문서들 확인
                        print(f"\n=== 내용이 짧은 문서들 (100자 미만) ===")
                        short_docs = [doc for doc in docs if len(doc.page_content) < 100]
                        print(f"짧은 문서 수: {len(short_docs)}")
                        
                        if short_docs:
                            for i, doc in enumerate(short_docs[:10]):
                                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                                source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                                print(f"{i+1}. {os.path.basename(source)} (길이: {len(content)}자)")
                                print(f"   내용: {content.strip()}")
                        
                        # 내용이 긴 문서들 확인
                        print(f"\n=== 내용이 긴 문서들 (1000자 이상) ===")
                        long_docs = [doc for doc in docs if len(doc.page_content) > 1000]
                        print(f"긴 문서 수: {len(long_docs)}")
                        
                        if long_docs:
                            for i, doc in enumerate(long_docs[:5]):
                                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                                source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                                print(f"{i+1}. {os.path.basename(source)} (길이: {len(content)}자)")
                                print(f"   내용 미리보기: {content[:200].replace('\\n', ' ')}...")
                                
                    else:
                        print("- 문서 캐시를 찾을 수 없습니다.")
                        
        except Exception as e:
            print(f"FAISS 인덱스 파일 읽기 실패: {e}")
    
    # 실제 검색 테스트 (상세)
    print(f"\n=== 상세 검색 테스트 ===")
    try:
        test_queries = ["몽촌토성 발굴", "백제 왕성", "토성 구조"]
        for query in test_queries:
            print(f"\n'{query}' 검색 결과:")
            results = vector_db.search(query, k=5)
            print(f"- 검색된 문서 수: {len(results)}")
            for i, (doc, score) in enumerate(results):
                source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                print(f"\n  {i+1}. {os.path.basename(source)}")
                print(f"     점수: {score:.6f}")
                print(f"     내용 길이: {len(content)}자")
                print(f"     내용: {content[:300].replace('\\n', ' ')}...")
                    
    except Exception as e:
        print(f"상세 검색 테스트 실패: {e}")

if __name__ == "__main__":
    main() 