#!/usr/bin/env python3
"""
키워드 확장 기능 테스트 스크립트

새로 구현된 동적 키워드 확장 기능을 테스트합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from src.utils.keyword_expander import KeywordExpander
from src.utils.text_processing import TextProcessor
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_keyword_expander():
    """KeywordExpander 클래스 테스트"""
    print("="*60)
    print("🔍 KeywordExpander 테스트")
    print("="*60)
    
    # KeywordExpander 인스턴스 생성
    expander = KeywordExpander()
    
    # 테스트 쿼리들
    test_queries = [
        "정보시스템 구축 방법은 무엇인가요?",
        "데이터베이스 보안 관리 절차",
        "공공기관 시스템 운영 지침",
        "소프트웨어 개발 표준",
        "행정기관 정보보호 정책",
        "시스템 유지보수는 어떻게 하나요?",
        "네트워크 보안 점검 절차",
        "문서 관리 시스템 요구사항"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 테스트 {i}: {query}")
        print("-" * 50)
        
        # 핵심 키워드 추출
        core_keywords = expander._extract_core_keywords(query)
        print(f"핵심 키워드: {core_keywords}")
        
        # 키워드 확장
        expanded_keywords = expander.expand_keywords(query, max_terms=20)
        print(f"확장된 키워드 ({len(expanded_keywords)}개): {expanded_keywords}")
        
        # 확장 통계
        stats = expander.get_expansion_stats(query)
        print(f"확장 비율: {stats['expansion_ratio']:.2f}")
        print(f"질문 의도: {stats['query_intent']}")

def test_text_processor_integration():
    """TextProcessor와 통합 테스트"""
    print("\n" + "="*60)
    print("🔗 TextProcessor 통합 테스트")
    print("="*60)
    
    test_queries = [
        "정보시스템이란 무엇인가요?",
        "데이터베이스 백업 방법",
        "보안 관리 절차는 어떻게 되나요?",
        "시스템 성능 최적화 방안"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 테스트 {i}: {query}")
        print("-" * 50)
        
        # 기존 정적 확장
        static_result = TextProcessor.expand_query(
            query, 
            use_static_expansion=True, 
            use_dynamic_expansion=False
        )
        print(f"정적 확장: {static_result[:100]}...")
        
        # 새로운 동적 확장
        dynamic_result = TextProcessor.expand_query(
            query, 
            use_static_expansion=False, 
            use_dynamic_expansion=True
        )
        print(f"동적 확장: {dynamic_result[:100]}...")
        
        # 두 방법 모두 사용
        combined_result = TextProcessor.expand_query(
            query, 
            use_static_expansion=True, 
            use_dynamic_expansion=True
        )
        print(f"통합 확장: {combined_result[:100]}...")

def test_intent_analysis():
    """질문 의도 분석 테스트"""
    print("\n" + "="*60)
    print("🎯 질문 의도 분석 테스트")
    print("="*60)
    
    expander = KeywordExpander()
    
    intent_tests = [
        ("정보시스템이란 무엇인가요?", "what"),
        ("데이터베이스를 어떻게 구축하나요?", "how"),
        ("시스템 점검은 언제 하나요?", "when"),
        ("서버실은 어디에 있나요?", "where"),
        ("보안이 왜 중요한가요?", "why"),
        ("누가 시스템을 관리하나요?", "who"),
        ("성능 향상 방법", "other")
    ]
    
    for query, expected_intent in intent_tests:
        detected_intent = expander._analyze_query_intent(query)
        status = "✅" if detected_intent == expected_intent else "❌"
        print(f"{status} '{query}' -> 감지: {detected_intent}, 예상: {expected_intent}")

def test_synonym_expansion():
    """동의어 확장 테스트"""
    print("\n" + "="*60)
    print("📚 동의어 확장 테스트")
    print("="*60)
    
    expander = KeywordExpander()
    
    test_keywords = ["시스템", "관리", "보안", "데이터", "구축", "운영"]
    
    for keyword in test_keywords:
        synonyms = expander._expand_with_synonyms([keyword])
        print(f"'{keyword}' -> 동의어: {synonyms[:10]}")

def main():
    """메인 테스트 함수"""
    print("🚀 키워드 확장 기능 종합 테스트 시작")
    
    try:
        # 1. KeywordExpander 기본 테스트
        test_keyword_expander()
        
        # 2. TextProcessor 통합 테스트
        test_text_processor_integration()
        
        # 3. 질문 의도 분석 테스트
        test_intent_analysis()
        
        # 4. 동의어 확장 테스트
        test_synonym_expansion()
        
        print("\n" + "="*60)
        print("✅ 모든 테스트 완료!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 