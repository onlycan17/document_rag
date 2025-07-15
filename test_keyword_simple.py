#!/usr/bin/env python3
"""
개선된 키워드 확장 테스트 (복합어 생성 비활성화 후)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.keyword_expander import KeywordExpander

def test_improved_expansion():
    """개선된 키워드 확장 테스트"""
    
    expander = KeywordExpander()
    
    test_cases = [
        "몽촌토성에 대해 알려줘",
        "정보시스템 구축 방법은?", 
        "데이터베이스 보안 관리"
    ]
    
    print("🔍 개선된 키워드 확장 테스트")
    print("=" * 50)
    
    for query in test_cases:
        print(f"\n📝 쿼리: '{query}'")
        
        # 확장 전 키워드
        original_keywords = expander._extract_core_keywords(query)
        print(f"   원본 키워드: {original_keywords}")
        
        # 확장 후
        expanded = expander.expand_keywords(query)
        print(f"   확장 키워드: {expanded}")
        print(f"   확장 비율: {len(original_keywords)} → {len(expanded)}개 ({len(expanded)/len(original_keywords):.1f}배)")

if __name__ == "__main__":
    test_improved_expansion() 