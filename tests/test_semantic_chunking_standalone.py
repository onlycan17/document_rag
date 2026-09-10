#!/usr/bin/env python3
"""
의미 기반 청킹 알고리즘 독립 테스트 스크립트
외부 의존성 없이 SemanticChunker만 테스트합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def test_semantic_chunker_basic():
    """SemanticChunker 기본 기능 테스트"""
    print("🔍 SemanticChunker 독립 테스트")

    try:
        # SemanticChunker 직접 import
        from src.utils.semantic_chunker import SemanticChunker

        print("✓ SemanticChunker import 성공")

        # 인스턴스 생성
        chunker = SemanticChunker(min_chunk_size=200, max_chunk_size=800)
        print("✓ SemanticChunker 인스턴스 생성 성공")

        # 테스트 텍스트
        test_text = """
        ## 제1장 서론
        
        몽촌토성은 백제의 중요한 유적이다. 이곳에서 많은 발굴조사가 이루어졌다.
        토기와 철기 등 다양한 유물이 출토되었다.
        
        ## 제2장 연구 방법
        
        그러나 본 연구에서는 새로운 접근법을 시도하였다. 
        특히 방사성탄소 연대측정을 활용하였다.
        
        따라서 이전 연구와 차별화된 결과를 얻을 수 있었다.
        풍납토성과의 비교 연구도 수행하였다.
        """

        # 의미 기반 청킹 실행
        chunks = chunker.create_semantic_chunks(test_text)
        print(f"✓ 청킹 실행 성공: {len(chunks)}개 청크 생성")

        # 각 청크 정보 출력
        for i, chunk in enumerate(chunks):
            print(f"\n청크 {i+1}:")
            print(f"  크기: {len(chunk['text'])} 문자")
            print(f"  키워드 수: {len(chunk['semantic_info']['keywords'])}")
            if chunk["semantic_info"]["keywords"]:
                top_keywords = [word for word, _ in chunk["semantic_info"]["keywords"][:3]]
                print(f"  상위 키워드: {top_keywords}")
            print(f"  내용: {chunk['text'][:100]}...")

        # 통계 확인
        stats = chunker.get_chunk_statistics(chunks)
        print("\n통계:")
        print(f"  총 청크: {stats['total_chunks']}")
        print(f"  평균 크기: {stats['avg_chunk_size']:.0f} 문자")
        print(f"  평균 일관성: {stats['avg_coherence']:.2f}")
        print(f"  평균 도메인 관련성: {stats['avg_domain_relevance']:.2f}")

        return True

    except Exception as e:
        print(f"✗ 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_keyword_extraction_standalone():
    """키워드 추출 독립 테스트"""
    print("\n🔑 키워드 추출 독립 테스트")

    try:
        from src.utils.semantic_chunker import SemanticChunker

        chunker = SemanticChunker()

        test_text = (
            "몽촌토성은 백제 한성시기의 왕성으로 추정되는 중요한 유적이다. 발굴조사를 통해 토기와 철기가 출토되었다."
        )

        keywords = chunker._extract_keywords(test_text)
        print(f"✓ 키워드 추출 성공: {len(keywords)}개")

        # 상위 5개 키워드 출력
        for i, (word, score) in enumerate(keywords[:5]):
            print(f"  {i+1}. {word}: {score:.3f}")

        # 중요 키워드 포함 확인
        keyword_words = [word for word, _ in keywords]
        important_found = any(word in keyword_words for word in ["몽촌토성", "백제", "토기", "발굴"])

        if important_found:
            print("✓ 중요 키워드 추출 확인")
            return True
        else:
            print("✗ 중요 키워드 누락")
            print(f"추출된 키워드: {keyword_words}")
            return False

    except Exception as e:
        print(f"✗ 키워드 추출 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    print("🚀 의미 기반 청킹 독립 테스트 시작\n")

    tests = [
        ("SemanticChunker 기본 기능", test_semantic_chunker_basic),
        ("키워드 추출 기능", test_keyword_extraction_standalone),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 테스트 중 오류: {e}")
            results.append((test_name, False))

    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 독립 테스트 결과")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n총 {passed}/{len(results)}개 테스트 통과")

    if passed == len(results):
        print("\n🎉 의미 기반 청킹 독립 테스트 성공!")
    else:
        print(f"\n⚠️ {len(results)-passed}개 테스트 실패")
