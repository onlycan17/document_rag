#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 컨텍스트 검증 스크립트
우각형파수편 검색 시 LLM에 전달되는 컨텍스트 확인
"""

from src.rag.rag_chain import RAGChain
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_context():
    """RAG 컨텍스트 테스트"""
    # RAG 초기화
    rag = RAGChain()
    
    # 테스트 쿼리
    query = '우각형파수편이 뭐야?'
    print(f'=== RAG 컨텍스트 테스트: {query} ===\n')
    
    # 1. 쿼리 전처리 확인
    processed_query = rag._preprocess_query(query)
    print(f'원본 쿼리: {query}')
    print(f'처리된 쿼리: {processed_query}\n')
    
    # 2. 문서 검색
    relevant_docs = rag.vector_db.search(processed_query, k=8)
    print(f'검색된 문서 수: {len(relevant_docs)}')
    
    # 3. 컨텍스트 구성
    context = rag._format_documents(relevant_docs, query)
    print(f'\n=== LLM에 전달되는 컨텍스트 ===')
    print(f'컨텍스트 길이: {len(context)}자\n')
    print('--- 컨텍스트 내용 ---')
    print(context)
    print('\n--- 컨텍스트 끝 ---\n')
    
    # 4. 우각형파수편 관련 내용 확인
    if '우각형파수편' in context:
        print('✅ 컨텍스트에 "우각형파수편" 포함됨!')
        
        # 우각형파수편 주변 문맥 추출
        import re
        matches = list(re.finditer(r'.{0,200}우각형파수편.{0,200}', context))
        print(f'\n우각형파수편 언급 횟수: {len(matches)}회')
        
        for i, match in enumerate(matches):
            print(f'\n[언급 {i+1}]')
            print(match.group())
    else:
        print('❌ 컨텍스트에 "우각형파수편" 없음!')
    
    # 5. 프롬프트 템플릿 확인
    print(f'\n=== 프롬프트 템플릿 ===')
    prompt = rag.prompt_template.format(context=context[:500] + '...', question=query)
    print(prompt)

if __name__ == '__main__':
    test_rag_context()