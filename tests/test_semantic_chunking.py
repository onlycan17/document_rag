#!/usr/bin/env python3
"""
의미 기반 청킹 알고리즘 테스트 스크립트
Phase 4에서 구현된 의미 기반 청킹 기능이 올바르게 작동하는지 검증합니다.
"""

import sys
from pathlib import Path
import tempfile
import shutil

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def test_semantic_chunker_functionality():
    """SemanticChunker 기본 기능 테스트"""
    print("🔍 SemanticChunker 기본 기능 테스트")

    try:
        from src.utils.semantic_chunker import SemanticChunker

        # SemanticChunker 인스턴스 생성
        chunker = SemanticChunker(min_chunk_size=300, max_chunk_size=1500, similarity_threshold=0.3)

        # 필수 메서드 확인
        required_methods = [
            "create_semantic_chunks",
            "_split_into_paragraphs",
            "_extract_keywords",
            "_detect_topic_transitions",
            "_adjust_boundaries_by_similarity",
            "_create_adaptive_chunks",
            "get_chunk_statistics",
        ]

        all_exist = True
        for method in required_methods:
            if hasattr(chunker, method):
                print(f"✓ {method} 메서드 존재")
            else:
                print(f"✗ {method} 메서드 누락")
                all_exist = False

        return all_exist

    except Exception as e:
        print(f"✗ SemanticChunker 테스트 실패: {e}")
        return False


def test_keyword_extraction():
    """키워드 추출 기능 테스트"""
    print("\\n🔑 키워드 추출 기능 테스트")

    try:
        from src.utils.semantic_chunker import SemanticChunker

        chunker = SemanticChunker()

        # 테스트 텍스트 (고고학 관련)
        test_text = """
        몽촌토성은 백제 한성시기의 왕성으로 추정되는 중요한 유적이다. 
        발굴조사 결과 다양한 토기와 철기 유물이 출토되었으며, 
        방사성탄소 연대측정을 통해 3세기-5세기에 해당하는 시기로 편년되었다.
        특히 풍납토성과의 비교 연구를 통해 백제 왕성의 변천사를 파악할 수 있다.
        """

        keywords = chunker._extract_keywords(test_text)

        print(f"추출된 키워드 개수: {len(keywords)}")
        print("상위 10개 키워드:")
        for i, (word, score) in enumerate(keywords[:10]):
            print(f"  {i+1}. {word}: {score:.3f}")

        # 도메인 키워드가 포함되어 있는지 확인
        keyword_words = [word for word, _ in keywords]
        domain_found = any(word in keyword_words for word in ["몽촌토성", "백제", "토기", "발굴", "연대측정"])

        if domain_found:
            print("✓ 도메인 관련 키워드 추출 성공")
            return True
        else:
            print("✗ 도메인 관련 키워드 추출 실패")
            return False

    except Exception as e:
        print(f"✗ 키워드 추출 테스트 실패: {e}")
        return False


def test_topic_transition_detection():
    """주제 전환점 감지 테스트"""
    print("\\n🔄 주제 전환점 감지 테스트")

    try:
        from src.utils.semantic_chunker import SemanticChunker

        chunker = SemanticChunker()

        # 테스트 문단들 (명확한 주제 전환 포함)
        paragraphs = [
            "몽촌토성의 위치와 규모에 대해 살펴보겠다. 이 토성은 서울 송파구에 위치한다.",
            "그러나 발굴조사 결과는 다른 양상을 보여준다. 다양한 유물이 출토되었다.",
            "제1절 토기 분석 결과",  # 구조적 전환
            "출토된 토기는 크게 세 가지 유형으로 분류된다.",
            "따라서 이러한 결과를 종합하면 다음과 같은 결론을 내릴 수 있다.",  # 논리적 전환
        ]

        # 각 문단의 키워드 추출
        paragraph_keywords = []
        for paragraph in paragraphs:
            keywords = chunker._extract_keywords(paragraph)
            paragraph_keywords.append(keywords)

        # 주제 전환점 감지
        transitions = chunker._detect_topic_transitions(paragraphs, paragraph_keywords)

        print(f"감지된 주제 전환점: {transitions}")

        # 예상되는 전환점들 (인덱스 기준)
        expected_transitions = [1, 2, 4]  # "그러나", "제1절", "따라서"

        # 실제로 전환점이 감지되었는지 확인
        success_count = 0
        for expected in expected_transitions:
            if expected in transitions:
                print(f"✓ 전환점 {expected} 정상 감지")
                success_count += 1
            else:
                print(f"✗ 전환점 {expected} 감지 실패")

        success_rate = success_count / len(expected_transitions)
        print(f"전환점 감지 성공률: {success_rate:.1%}")

        return success_rate >= 0.6  # 60% 이상 성공하면 통과

    except Exception as e:
        print(f"✗ 주제 전환점 감지 테스트 실패: {e}")
        return False


def test_semantic_chunking_process():
    """전체 의미 기반 청킹 프로세스 테스트"""
    print("\\n📝 전체 의미 기반 청킹 프로세스 테스트")

    try:
        from src.utils.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(
            min_chunk_size=200,  # 테스트를 위해 작게 설정
            max_chunk_size=800,
            similarity_threshold=0.3,
        )

        # 복합적인 테스트 텍스트
        test_text = """
        ## 제1장 몽촌토성 개요
        
        몽촌토성은 서울 송파구에 위치한 백제시대의 토성이다. 이 토성은 한강 유역에서 발견된 
        대표적인 백제 유적 중 하나로, 백제 한성시기의 왕성으로 추정되고 있다.
        
        토성의 둘레는 약 2.7km이며, 평면 형태는 타원형을 이루고 있다. 성벽의 높이는 
        현재 4-7m 정도이지만, 원래는 더 높았을 것으로 추정된다.
        
        ## 제2장 발굴조사 성과
        
        그러나 본격적인 발굴조사는 1980년대부터 시작되었다. 수차례의 발굴조사를 통해 
        다양한 유구와 유물이 확인되었다.
        
        특히 주목할 만한 것은 대량의 토기 유물이다. 출토된 토기는 크게 생활용기와 
        의례용기로 구분되며, 각각 다른 특징을 보인다.
        
        ### 2.1 토기 분석
        
        생활용기로는 항아리, 시루, 그릇 등이 있으며, 의례용기로는 고배, 장경호 등이 확인되었다.
        이들 토기의 형태와 제작기법을 통해 백제 토기의 변천과정을 파악할 수 있다.
        
        따라서 몽촌토성에서 출토된 토기는 백제사 연구에 매우 중요한 자료가 된다.
        특히 풍납토성과의 비교 연구를 통해 백제 왕성의 변천사를 밝힐 수 있을 것이다.
        """

        # 의미 기반 청킹 실행
        chunks = chunker.create_semantic_chunks(test_text)

        print(f"생성된 청크 수: {len(chunks)}")

        # 각 청크 정보 출력
        for i, chunk in enumerate(chunks):
            chunk_info = chunk["semantic_info"]
            print(f"\\n청크 {i+1}:")
            print(f"  크기: {len(chunk['text'])} 문자")
            print(f"  주제 일관성: {chunk_info['topic_coherence']:.2f}")
            print(f"  도메인 관련성: {chunk_info['domain_relevance']:.2f}")
            print(f"  상위 키워드: {[word for word, _ in chunk_info['keywords'][:5]]}")
            print(f"  내용 미리보기: {chunk['text'][:100]}...")

        # 청킹 통계 출력
        stats = chunker.get_chunk_statistics(chunks)
        print("\\n청킹 통계:")
        print(f"  총 청크 수: {stats['total_chunks']}")
        print(f"  평균 크기: {stats['avg_chunk_size']:.0f} 문자")
        print(f"  크기 범위: {stats['min_chunk_size']} ~ {stats['max_chunk_size']} 문자")
        print(f"  평균 일관성: {stats['avg_coherence']:.2f}")
        print(f"  평균 도메인 관련성: {stats['avg_domain_relevance']:.2f}")

        # 성공 기준 확인
        success_criteria = [
            len(chunks) >= 2,  # 최소 2개 이상의 청크 생성
            stats["avg_chunk_size"] >= 200,  # 평균 크기가 최소 기준 이상
            stats["avg_coherence"] > 0.3,  # 적절한 일관성
            all(len(chunk["text"]) >= 100 for chunk in chunks),  # 모든 청크가 최소 크기 이상
        ]

        success_count = sum(success_criteria)
        print(f"\\n성공 기준: {success_count}/{len(success_criteria)} 충족")

        return success_count >= 3  # 4개 중 3개 이상 충족하면 성공

    except Exception as e:
        print(f"✗ 의미 기반 청킹 프로세스 테스트 실패: {e}")
        return False


def test_pdf_converter_integration():
    """PDF 변환기와 의미 기반 청킹 통합 테스트"""
    print("\\n🔗 PDF 변환기 통합 테스트")

    try:
        from src.utils.pdf_converter import ImprovedPDFConverter

        # 의미 기반 청킹 활성화한 PDF 변환기 생성
        temp_dir = tempfile.mkdtemp()
        converter = ImprovedPDFConverter(output_dir=temp_dir, enable_semantic_chunking=True)

        # 통합 관련 메서드 확인
        integration_methods = ["create_semantic_chunks", "_create_basic_chunks", "convert_pdf_to_semantic_chunks"]

        all_exist = True
        for method in integration_methods:
            if hasattr(converter, method):
                print(f"✓ {method} 메서드 존재")
            else:
                print(f"✗ {method} 메서드 누락")
                all_exist = False

        # SemanticChunker 인스턴스 확인
        if hasattr(converter, "semantic_chunker") and converter.semantic_chunker:
            print("✓ SemanticChunker 인스턴스 생성됨")
        else:
            print("✗ SemanticChunker 인스턴스 생성 실패")
            all_exist = False

        # 의미 기반 청킹 설정 확인
        if converter.enable_semantic_chunking:
            print("✓ 의미 기반 청킹 활성화됨")
        else:
            print("✗ 의미 기반 청킹 비활성화됨")
            all_exist = False

        # 테스트 텍스트로 청킹 시도
        test_text = """
        # 테스트 문서
        
        이것은 PDF 변환기와 의미 기반 청킹의 통합을 테스트하기 위한 샘플 텍스트입니다.
        
        ## 제1절 배경
        몽촌토성은 백제의 중요한 유적입니다. 많은 발굴조사가 이루어져 왔습니다.
        
        ## 제2절 결과  
        그러나 새로운 발견들이 계속되고 있습니다. 특히 토기 유물이 주목받고 있습니다.
        따라서 지속적인 연구가 필요합니다.
        """

        metadata = {"test": True}
        chunks = converter.create_semantic_chunks(test_text, metadata)

        if chunks and len(chunks) > 0:
            print(f"✓ 테스트 청킹 성공: {len(chunks)}개 청크 생성")

            # 첫 번째 청크 구조 확인
            first_chunk = chunks[0]
            required_keys = ["text", "metadata", "semantic_info"]

            for key in required_keys:
                if key in first_chunk:
                    print(f"✓ 청크 구조 확인: {key} 키 존재")
                else:
                    print(f"✗ 청크 구조 오류: {key} 키 누락")
                    all_exist = False
        else:
            print("✗ 테스트 청킹 실패")
            all_exist = False

        # 임시 디렉토리 정리
        shutil.rmtree(temp_dir)

        return all_exist

    except Exception as e:
        print(f"✗ PDF 변환기 통합 테스트 실패: {e}")
        return False


def test_performance_comparison():
    """기본 청킹 vs 의미 기반 청킹 성능 비교"""
    print("\\n⚡ 성능 비교 테스트")

    try:
        from src.utils.pdf_converter import ImprovedPDFConverter
        import time

        # 테스트용 긴 텍스트 생성
        test_text = (
            """
        ## 몽촌토성 연구사
        
        몽촌토성에 대한 연구는 일제강점기부터 시작되었다. 초기 연구자들은 이곳을 단순한 토성으로만 인식했다.
        그러나 해방 이후 본격적인 학술 연구가 시작되면서 이곳의 중요성이 재평가되기 시작했다.
        
        1970년대부터는 체계적인 발굴조사가 이루어졌다. 특히 1980년대 대규모 발굴조사는 획기적인 성과를 거두었다.
        이때 발견된 유물들은 백제사 연구에 새로운 지평을 열어주었다.
        
        ## 입지와 구조
        
        몽촌토성은 한강과 탄천이 만나는 지점의 높은 구릉에 위치한다. 이러한 입지는 방어와 교통의 요지로서 최적이었다.
        토성의 평면형태는 북동-남서 방향의 타원형이며, 둘레는 약 2.7km에 달한다.
        
        성벽은 판축기법으로 축조되었으며, 현재 높이는 4-7m 정도이다. 원래는 더 높았을 것으로 추정된다.
        성문은 동문, 서문, 남문 등 3개소가 확인되었으며, 각각 독특한 구조를 보인다.
        
        ## 출토 유물
        
        그러나 가장 주목할 만한 것은 토기 유물이다. 출토된 토기는 수천 점에 달하며, 
        이들은 백제 토기 연구의 기준자료가 되고 있다.
        
        토기는 크게 생활용기와 의례용기로 구분된다. 생활용기로는 항아리, 시루, 그릇류가 있고,
        의례용기로는 고배, 장경호, 단경호 등이 있다.
        
        ### 토기의 특징
        
        특히 주목되는 것은 토기의 제작기법과 문양이다. 백제 토기 특유의 정교한 제작기법을 보여준다.
        문양으로는 격자문, 승문, 파상문 등이 확인되며, 이들은 시기별로 변화 양상을 보인다.
        
        따라서 몽촌토성 출토 토기는 백제 토기의 편년과 지역성 연구에 핵심적인 자료이다.
        앞으로도 지속적인 연구를 통해 백제사의 새로운 면모를 밝혀나가야 할 것이다.
        """
            * 3
        )  # 텍스트를 3배로 늘려서 성능 테스트

        temp_dir = tempfile.mkdtemp()

        # 1. 기본 청킹 테스트
        converter_basic = ImprovedPDFConverter(output_dir=temp_dir + "_basic", enable_semantic_chunking=False)

        start_time = time.time()
        basic_chunks = converter_basic.create_semantic_chunks(test_text)  # 기본 청킹 사용
        basic_time = time.time() - start_time

        # 2. 의미 기반 청킹 테스트
        converter_semantic = ImprovedPDFConverter(output_dir=temp_dir + "_semantic", enable_semantic_chunking=True)

        start_time = time.time()
        semantic_chunks = converter_semantic.create_semantic_chunks(test_text)
        semantic_time = time.time() - start_time

        # 결과 비교
        print("기본 청킹:")
        print(f"  처리 시간: {basic_time:.3f}초")
        print(f"  생성 청크 수: {len(basic_chunks)}")
        print(f"  평균 청크 크기: {sum(len(c['text']) for c in basic_chunks) / len(basic_chunks):.0f} 문자")

        print("\\n의미 기반 청킹:")
        print(f"  처리 시간: {semantic_time:.3f}초")
        print(f"  생성 청크 수: {len(semantic_chunks)}")
        print(f"  평균 청크 크기: {sum(len(c['text']) for c in semantic_chunks) / len(semantic_chunks):.0f} 문자")

        if semantic_chunks and hasattr(converter_semantic.semantic_chunker, "get_chunk_statistics"):
            stats = converter_semantic.semantic_chunker.get_chunk_statistics(semantic_chunks)
            print(f"  평균 일관성: {stats['avg_coherence']:.2f}")
            print(f"  평균 도메인 관련성: {stats['avg_domain_relevance']:.2f}")

        # 성능 비교
        time_ratio = semantic_time / basic_time if basic_time > 0 else float("inf")
        print("\\n성능 비교:")
        print(f"  처리 시간 비율: {time_ratio:.1f}x (의미 기반 청킹 / 기본 청킹)")

        # 정리
        shutil.rmtree(temp_dir + "_basic", ignore_errors=True)
        shutil.rmtree(temp_dir + "_semantic", ignore_errors=True)

        # 성공 기준: 의미 기반 청킹이 기본 청킹보다 10배 이상 느리지 않으면 성공
        return time_ratio < 10.0 and len(semantic_chunks) > 0

    except Exception as e:
        print(f"✗ 성능 비교 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 실행"""
    print("🚀 의미 기반 청킹 알고리즘 테스트 시작\\n")

    tests = [
        ("SemanticChunker 기본 기능", test_semantic_chunker_functionality),
        ("키워드 추출 기능", test_keyword_extraction),
        ("주제 전환점 감지", test_topic_transition_detection),
        ("의미 기반 청킹 프로세스", test_semantic_chunking_process),
        ("PDF 변환기 통합", test_pdf_converter_integration),
        ("성능 비교", test_performance_comparison),
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
    print("\\n" + "=" * 60)
    print("📊 의미 기반 청킹 테스트 결과 요약")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\\n총 {passed}/{len(results)}개 테스트 통과")

    if passed == len(results):
        print("\\n🎉 모든 테스트 통과! Phase 4 의미 기반 청킹 알고리즘이 성공적으로 구현되었습니다.")
        print("\\n구현된 기능:")
        print("• 키워드 기반 주제 전환점 감지")
        print("• 문단 간 유사도 측정을 통한 의미적 경계 식별")
        print("• 적응적 청크 크기 조정")
        print("• 한글 고고학/역사 문서 특화 처리")
        print("• PDF 변환기와의 완전 통합")
        print("• 주제 일관성 및 도메인 관련성 측정")

        print("\\n사용 방법:")
        print("1. ImprovedPDFConverter(enable_semantic_chunking=True) 로 생성")
        print("2. convert_pdf_to_semantic_chunks() 메서드 사용")
        print("3. 생성된 청크에서 semantic_info로 상세 정보 확인")
        print("4. RAG 시스템에서 더 의미적으로 일관된 검색 결과 확인")
    else:
        print(f"\\n⚠️ {len(results)-passed}개 테스트 실패. Phase 5 진행 전 문제를 해결해야 합니다.")


if __name__ == "__main__":
    main()
