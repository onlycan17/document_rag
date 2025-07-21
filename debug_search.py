#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
벡터 검색 디버그 스크립트
"""

from src.vectorstore import VectorDatabase
from src.embeddings import EmbeddingModel
from config import settings

def debug_search():
    """검색 디버그"""
    vector_db = VectorDatabase()
    
    # 1. 단순 키워드로 직접 검색
    print("=== 1. 단순 키워드 '우각형파수편' 검색 ===")
    simple_results = vector_db.vector_store.similarity_search_with_score('우각형파수편', k=20)
    
    for i, (doc, score) in enumerate(simple_results[:10]):
        contains_target = '우각형파수편' in doc.page_content
        print(f"\n[{i+1}] 점수: {score:.4f} {'✅' if contains_target else '❌'}")
        print(f"청크 ID: {doc.metadata.get('chunk_id', 'Unknown')}")
        if contains_target:
            # 우각형파수편 주변 텍스트 추출
            import re
            match = re.search(r'.{0,100}우각형파수편.{0,100}', doc.page_content)
            if match:
                print(f"내용: ...{match.group()}...")
    
    # 2. RAG search 메서드 테스트
    print("\n\n=== 2. VectorDB search 메서드 테스트 ===")
    rag_results = vector_db.search('우각형파수편', k=20)
    
    print(f"검색 결과 수: {len(rag_results)}")
    print(f"FAISS 임계값: {settings.search_threshold_faiss}")
    
    for i, (doc, score) in enumerate(rag_results[:10]):
        contains_target = '우각형파수편' in doc.page_content
        print(f"\n[{i+1}] 점수: {score:.4f} {'✅' if contains_target else '❌'}")
        print(f"청크 ID: {doc.metadata.get('chunk_id', 'Unknown')}")
        if contains_target:
            import re
            match = re.search(r'.{0,100}우각형파수편.{0,100}', doc.page_content)
            if match:
                print(f"내용: ...{match.group()}...")
    
    # 3. 필터링 테스트
    print("\n\n=== 3. 필터링 분석 ===")
    all_results = vector_db.vector_store.similarity_search_with_score('우각형파수편', k=50)
    
    filtered_count = 0
    target_found_position = None
    
    for i, (doc, score) in enumerate(all_results):
        if score <= settings.search_threshold_faiss:
            filtered_count += 1
            if '우각형파수편' in doc.page_content and target_found_position is None:
                target_found_position = i + 1
    
    print(f"전체 검색 결과: {len(all_results)}개")
    print(f"임계값({settings.search_threshold_faiss}) 통과: {filtered_count}개")
    
    if target_found_position:
        print(f"✅ 우각형파수편은 {target_found_position}번째 위치에 있음")
    else:
        # 우각형파수편이 있는 문서 찾기
        for i, (doc, score) in enumerate(all_results):
            if '우각형파수편' in doc.page_content:
                print(f"❌ 우각형파수편은 {i+1}번째에 있지만 점수가 {score:.4f}로 임계값을 초과")
                break

if __name__ == '__main__':
    debug_search()