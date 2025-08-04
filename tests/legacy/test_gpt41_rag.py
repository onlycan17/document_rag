#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT-4.1-mini 모델로 RAG 테스트
"""

import os
from src.rag.rag_chain import RAGChain
from config import settings

def test_gpt41_mini_rag():
    """GPT-4.1-mini로 RAG 테스트"""
    # 설정 변경
    original_provider = settings.llm_provider
    original_model = settings.openai_model
    
    try:
        # GPT-4.1-mini로 설정
        settings.llm_provider = "openai"
        settings.openai_model = "gpt-4.1-mini"
        
        # RAG 체인 생성
        rag = RAGChain()
        
        # 모델 정보 출력
        print(f"현재 모델: {settings.llm_provider} - {settings.openai_model}")
        print(f"OpenAI API 키 설정: {'Yes' if settings.openai_api_key else 'No'}")
        print("=" * 60)
        
        # 테스트 질문들
        test_questions = [
            "우각형파수편이 뭐야?",
            "몽촌토성에서 발견된 백제 토기의 특징은?",
            "우각형파수편의 제작 기법을 설명해줘"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[질문 {i}] {question}")
            print("-" * 60)
            
            # RAG 쿼리 실행
            result = rag.query(question)
            
            if result['status'] == 'success':
                answer = result['answer']
                print("답변:")
                print(answer)
                
                # 우각형파수편 관련 답변 검증
                if '우각형파수편' in question:
                    if '백제' in answer and ('토기' in answer or '파수' in answer):
                        print("\n✅ 정답: 백제 토기/파수로 올바르게 설명")
                    elif '광학' in answer or 'optical' in answer.lower() or 'wave plate' in answer.lower():
                        print("\n❌ 오답: 광학 기기로 잘못 설명")
                    else:
                        print("\n⚠️ 답변 내용 확인 필요")
                
                # 소스 정보
                print(f"\n참조 문서: {len(result.get('sources', []))}개")
                for source in result.get('sources', [])[:3]:
                    print(f"  - {source.get('file_name')} (관련도: {source.get('relevance_percent')}%)")
            else:
                print(f"오류: {result.get('error', 'Unknown error')}")
            
            print("=" * 60)
            
    except Exception as e:
        print(f"테스트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 원래 설정으로 복원
        settings.llm_provider = original_provider
        settings.openai_model = original_model

if __name__ == '__main__':
    print("GPT-4.1-mini 모델로 RAG 테스트 시작")
    print("=" * 60)
    test_gpt41_mini_rag()