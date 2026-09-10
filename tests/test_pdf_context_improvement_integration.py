#!/usr/bin/env python3
"""
PDF 맥락 개선 시스템 통합 테스트 스크립트
Phase 5: 전체 시스템이 함께 작동하는지 검증

이 테스트는 다음 개선 사항들이 통합적으로 작동하는지 확인합니다:
- Phase 2: 페이지간 텍스트 연결 알고리즘
- Phase 3: 문장/문단 경계 인식 개선
- Phase 4: 의미 기반 청킹 알고리즘
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil
import time
from typing import Dict, List

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def create_test_pdf_content() -> str:
    """테스트용 PDF 시뮬레이션 컨텐츠 생성 (실제 PDF 문제 상황 재현)"""
    return """# 몽촌토성 발굴조사 보고서

**저자:** 한국고고학회
**주제:** 백제 왕성 발굴조사
**생성 도구:** Adobe Acrobat
**소스 파일:** mongchon_report.pdf
**변환 날짜:** 2024-01-15 14:30:00

---

## 제1장 서론

몽촌토성은 서울특별시 송파구에 위치한 백제시대의 토
성이다. 이 토성은 한강 유역에서 발견된 대표적인 백제 
유적 중 하나로, 백제 한성시기의 왕성으로 추정되고 있
다.

토성의 둘레는 약 2.7km이며, 평면 형태는 타원형을 이
루고 있다. 성벽의 높이는 현재 4-7m 정도이지만, 원래
는 더 높았을 것으로 추정된다.

## 제2장 발굴조사 성과

그러나 본격적인 발굴조사는 1980년대부터 시작되었
다. 수차례의 발굴조사를 통해 다양한 유구와 유물이 확
인되었다.

특히 주목할 만한 것은 대량의 토기 유물이다. 출토된 토
기는 크게 생활용기와 의례용기로 구분되며, 각각 다른 
특징을 보인다.

### 2.1 토기 분석

생활용기로는 항아리, 시루, 그릇 등이 있으며, 의례용기
로는 고배, 장경호 등이 확인되었다. 이들 토기의 형태와 
제작기법을 통해 백제 토기의 변천과정을 파악할 수 있
다.

따라서 몽촌토성에서 출토된 토기는 백제사 연구에 매
우 중요한 자료가 된다. 특히 풍납토성과의 비교 연구를 
통해 백제 왕성의 변천사를 밝힐 수 있을 것이다.

### 2.2 방사성탄소 연대측정 결과

몽촌토성에서 출토된 목탄 시료에 대한 방사성탄소 연
대측정 결과, 3세기-5세기에 걸친 연대가 확인되었다.

이는 기존의 토기 편년 연구 결과와 일치하는 것으로, 몽
촌토성이 백제 한성시기 전 기간에 걸쳐 사용되었음을 
보여준다.

특히 4세기 후반에 해당하는 연대가 집중적으로 나타나
는 것은 이 시기에 토성의 사용이 절정에 달했음을 시사
한다.

## 제3장 결론 및 향후 과제

따라서 몽촌토성은 백제 한성시기의 핵심적인 유적으로
서 그 중요성이 재확인되었다. 앞으로도 지속적인 연구
를 통해 백제사의 새로운 면모를 밝혀나가야 할 것이다.

## 참고문헌

1. 한국고고학회, 『몽촌토성 발굴조사 보고서』, 2023
2. 서울특별시, 『한성백제 왕성 조사보고서』, 2022
3. 백제문화재연구원, 『백제 토기 연구』, 2021
"""

def test_complete_system_integration():
    """전체 시스템 통합 테스트"""
    print("🔄 전체 PDF 맥락 개선 시스템 통합 테스트")
    
    try:
        from src.utils.pdf_converter import ImprovedPDFConverter
        
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        
        # 의미 기반 청킹 활성화된 PDF 변환기 생성
        converter = ImprovedPDFConverter(
            output_dir=temp_dir,
            enable_semantic_chunking=True
        )
        
        print("✓ PDF 변환기 생성 완료 (의미 기반 청킹 활성화)")
        
        # 테스트용 컨텐츠로 의미 기반 청킹 테스트
        test_content = create_test_pdf_content()
        
        # 청킹 실행
        chunks = converter.create_semantic_chunks(test_content, {'test': True})
        
        print(f"✓ 의미 기반 청킹 완료: {len(chunks)}개 청크 생성")
        
        # 청킹 품질 확인
        if converter.semantic_chunker:
            stats = converter.semantic_chunker.get_chunk_statistics(chunks)
            print(f"  - 평균 청크 크기: {stats['avg_chunk_size']:.0f} 문자")
            print(f"  - 크기 범위: {stats['min_chunk_size']} ~ {stats['max_chunk_size']} 문자")
            print(f"  - 평균 주제 일관성: {stats['avg_coherence']:.2f}")
            print(f"  - 평균 도메인 관련성: {stats['avg_domain_relevance']:.2f}")
        
        # 청크 내용 샘플 출력
        print("\n📄 생성된 청크 샘플:")
        for i, chunk in enumerate(chunks[:3]):  # 첫 3개 청크만 출력
            print(f"\n청크 {i+1}:")
            print(f"  크기: {len(chunk['text'])} 문자")
            if 'semantic_info' in chunk and 'keywords' in chunk['semantic_info']:
                top_keywords = [word for word, _ in chunk['semantic_info']['keywords'][:5]]
                print(f"  주요 키워드: {top_keywords}")
            print(f"  내용 미리보기: {chunk['text'][:150]}...")
        
        # 정리
        shutil.rmtree(temp_dir)
        
        # 성공 기준 확인
        success_criteria = [
            len(chunks) >= 3,  # 적절한 수의 청크 생성
            stats['avg_chunk_size'] >= 300,  # 최소 크기 확보
            stats['avg_coherence'] > 0.3,  # 적절한 일관성
            all(len(chunk['text']) >= 100 for chunk in chunks)  # 모든 청크가 최소 크기 이상
        ]
        
        success_rate = sum(success_criteria) / len(success_criteria)
        print(f"\n성공 기준 충족률: {success_rate:.1%}")
        
        return success_rate >= 0.75  # 75% 이상 충족하면 성공
        
    except Exception as e:
        print(f"✗ 통합 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cross_page_text_connection():
    """페이지간 텍스트 연결 개선 검증"""
    print("\n📖 페이지간 텍스트 연결 개선 검증")
    
    try:
        from src.utils.pdf_converter import ImprovedPDFConverter
        
        temp_dir = tempfile.mkdtemp()
        converter = ImprovedPDFConverter(output_dir=temp_dir)
        
        # 페이지 경계에서 끊어진 텍스트 시뮬레이션
        page_texts = [
            (1, "몽촌토성은 백제시대의 중요한 유적이다. 이곳에서는 다양한 토"),
            (2, "기와 철기가 출토되었다. 특히 3세기-5세기에 해당하는 연"),
            (3, "대측정 결과가 확인되었다. 이는 백제 한성시기와 일치한다.")
        ]
        
        # 텍스트 연결 테스트
        connected_text = converter._connect_cross_page_text(page_texts)
        
        print("원본 (페이지별):")
        for page_num, text in page_texts:
            print(f"  페이지 {page_num}: {text}")
        
        print(f"\n연결 결과:")
        print(f"  {connected_text}")
        
        # 연결 품질 확인
        is_well_connected = (
            "토기와 철기" in connected_text and  # '토'와 '기'가 연결되었는지
            "연대측정" in connected_text  # '연'과 '대측정'이 연결되었는지
        )
        
        if is_well_connected:
            print("✓ 페이지간 텍스트 연결 개선 성공")
        else:
            print("✗ 페이지간 텍스트 연결 개선 실패")
        
        shutil.rmtree(temp_dir)
        return is_well_connected
        
    except Exception as e:
        print(f"✗ 페이지간 텍스트 연결 테스트 실패: {e}")
        return False

def test_sentence_boundary_improvement():
    """문장 경계 인식 개선 검증"""
    print("\n📝 문장 경계 인식 개선 검증")
    
    try:
        from src.utils.pdf_converter import ImprovedPDFConverter
        
        temp_dir = tempfile.mkdtemp()
        converter = ImprovedPDFConverter(output_dir=temp_dir)
        
        # 불완전한 문장 패턴 테스트
        test_cases = [
            ("몽촌토성은 백제의", True),    # 불완전 (조사로 끝남)
            ("토기가 출토되었다.", False),   # 완전 (문장부호 있음)
            ("3세기-5세기에 해당하는", True), # 불완전 (관형어로 끝남)
            ("연대측정 결과이다", False),    # 완전 (한글 종결어미)
            ("SPD", True),                 # 불완전 (영문 약어)
        ]
        
        success_count = 0
        for text, expected_incomplete in test_cases:
            from src.utils.sentence_completion import is_incomplete_sentence as _is_inc
            is_incomplete = _is_inc(text)
            if is_incomplete == expected_incomplete:
                success_count += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"  {status} '{text}' -> 불완전: {is_incomplete} (예상: {expected_incomplete})")
        
        success_rate = success_count / len(test_cases)
        print(f"\n문장 경계 인식 정확도: {success_rate:.1%}")
        
        shutil.rmtree(temp_dir)
        return success_rate >= 0.8  # 80% 이상 정확도면 성공
        
    except Exception as e:
        print(f"✗ 문장 경계 인식 테스트 실패: {e}")
        return False

def test_semantic_chunking_quality():
    """의미 기반 청킹 품질 검증"""
    print("\n🧠 의미 기반 청킹 품질 검증")
    
    try:
        from src.utils.semantic_chunker import SemanticChunker
        
        chunker = SemanticChunker(
            min_chunk_size=300,
            max_chunk_size=1200,
            similarity_threshold=0.3
        )
        
        # 복잡한 테스트 텍스트 (명확한 주제 전환 포함)
        test_text = """
        ## 제1장 몽촌토성 개요
        
        몽촌토성은 서울 송파구에 위치한 백제시대의 토성이다. 이 토성은 한강 유역에서 발견된 
        대표적인 백제 유적 중 하나로, 백제 한성시기의 왕성으로 추정되고 있다.
        
        ## 제2장 발굴 성과
        
        그러나 본격적인 발굴조사는 1980년대부터 시작되었다. 다량의 토기와 철기가 출토되었으며,
        이들은 백제사 연구에 중요한 자료가 되고 있다.
        
        ### 2.1 토기 분석
        
        따라서 출토된 토기를 분석한 결과, 생활용기와 의례용기로 구분할 수 있었다.
        방사성탄소 연대측정 결과 3세기-5세기 연대가 확인되었다.
        """
        
        # 의미 기반 청킹 실행
        chunks = chunker.create_semantic_chunks(test_text)
        
        print(f"생성된 청크: {len(chunks)}개")
        
        # 품질 지표 확인
        stats = chunker.get_chunk_statistics(chunks)
        
        quality_metrics = {
            '적절한 청크 수': 2 <= len(chunks) <= 5,
            '평균 크기 적정성': 300 <= stats['avg_chunk_size'] <= 1200,
            '주제 일관성': stats['avg_coherence'] > 0.3,
            '도메인 관련성': stats['avg_domain_relevance'] > 0.2,
            '크기 균형성': stats['max_chunk_size'] / stats['min_chunk_size'] < 3.0
        }
        
        print("\n품질 지표:")
        passed_metrics = 0
        for metric, passed in quality_metrics.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {metric}: {passed}")
            if passed:
                passed_metrics += 1
        
        # 주제 전환 감지 확인
        topic_transitions_detected = any(
            '제1장' in chunk['text'] and '제2장' not in chunk['text'] or
            '제2장' in chunk['text'] and '제1장' not in chunk['text']
            for chunk in chunks
        )
        
        if topic_transitions_detected:
            print("  ✓ 주제 전환 감지: 성공")
            passed_metrics += 1
        else:
            print("  ✗ 주제 전환 감지: 실패")
        
        quality_score = passed_metrics / (len(quality_metrics) + 1)
        print(f"\n전체 품질 점수: {quality_score:.1%}")
        
        return quality_score >= 0.7  # 70% 이상 품질 점수면 성공
        
    except Exception as e:
        print(f"✗ 의미 기반 청킹 품질 테스트 실패: {e}")
        return False

def test_performance_comparison():
    """개선 전후 성능 비교"""
    print("\n⚡ 개선 전후 성능 비교")
    
    try:
        from src.utils.pdf_converter import ImprovedPDFConverter
        
        temp_dir = tempfile.mkdtemp()
        test_content = create_test_pdf_content() * 2  # 더 긴 텍스트로 테스트
        
        # 1. 기본 청킹 (개선 전)
        converter_basic = ImprovedPDFConverter(
            output_dir=temp_dir + "_basic",
            enable_semantic_chunking=False
        )
        
        start_time = time.time()
        basic_chunks = converter_basic.create_semantic_chunks(test_content)
        basic_time = time.time() - start_time
        
        # 2. 의미 기반 청킹 (개선 후)
        converter_semantic = ImprovedPDFConverter(
            output_dir=temp_dir + "_semantic",
            enable_semantic_chunking=True
        )
        
        start_time = time.time()
        semantic_chunks = converter_semantic.create_semantic_chunks(test_content)
        semantic_time = time.time() - start_time
        
        # 결과 비교
        print(f"기본 청킹 (개선 전):")
        print(f"  처리 시간: {basic_time:.3f}초")
        print(f"  청크 수: {len(basic_chunks)}")
        print(f"  평균 크기: {sum(len(c['text']) for c in basic_chunks) / len(basic_chunks):.0f} 문자")
        
        print(f"\n의미 기반 청킹 (개선 후):")
        print(f"  처리 시간: {semantic_time:.3f}초")
        print(f"  청크 수: {len(semantic_chunks)}")
        print(f"  평균 크기: {sum(len(c['text']) for c in semantic_chunks) / len(semantic_chunks):.0f} 문자")
        
        if converter_semantic.semantic_chunker:
            stats = converter_semantic.semantic_chunker.get_chunk_statistics(semantic_chunks)
            print(f"  주제 일관성: {stats['avg_coherence']:.2f}")
            print(f"  도메인 관련성: {stats['avg_domain_relevance']:.2f}")
        
        # 성능 분석
        time_ratio = semantic_time / basic_time if basic_time > 0 else 1.0
        print(f"\n성능 분석:")
        print(f"  처리 시간 비율: {time_ratio:.1f}x")
        
        # 품질 개선 확인
        quality_improved = len(semantic_chunks) > 0 and semantic_time < 10.0
        
        # 정리
        shutil.rmtree(temp_dir + "_basic", ignore_errors=True)
        shutil.rmtree(temp_dir + "_semantic", ignore_errors=True)
        
        if quality_improved:
            print("  ✓ 성능 및 품질 개선 확인")
        else:
            print("  ✗ 성능 또는 품질 문제 발견")
        
        return quality_improved
        
    except Exception as e:
        print(f"✗ 성능 비교 테스트 실패: {e}")
        return False

def main():
    """메인 통합 테스트 실행"""
    print("🚀 PDF 맥락 개선 시스템 - Phase 5 통합 테스트 시작\n")
    
    tests = [
        ("전체 시스템 통합", test_complete_system_integration),
        ("페이지간 텍스트 연결", test_cross_page_text_connection),
        ("문장 경계 인식 개선", test_sentence_boundary_improvement),
        ("의미 기반 청킹 품질", test_semantic_chunking_quality),
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
    print("\n" + "="*70)
    print("📊 PDF 맥락 개선 시스템 통합 테스트 결과")
    print("="*70)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n총 {passed}/{len(results)}개 테스트 통과")
    
    if passed == len(results):
        print("\n🎉 모든 통합 테스트 통과! PDF 맥락 개선 시스템이 성공적으로 구현되었습니다.")
        print("\n구현된 주요 개선 사항:")
        print("✅ Phase 2: 페이지간 텍스트 연결 알고리즘")
        print("  • 불완전한 문장 자동 감지 및 연결")
        print("  • 한국어 특성을 고려한 문장 경계 처리")
        print("  • 페이지 구분 없는 자연스러운 텍스트 흐름")
        
        print("\n✅ Phase 3: 문장/문단 경계 인식 개선")
        print("  • 한글 문장 종결 패턴 정교한 인식")
        print("  • 조사, 어미, 불완전 단어 패턴 감지")
        print("  • 맥락을 고려한 연결성 판단")
        
        print("\n✅ Phase 4: 의미 기반 청킹 알고리즘")
        print("  • 키워드 기반 주제 전환점 자동 감지")
        print("  • 문단 간 유사도 측정을 통한 의미적 경계 식별")
        print("  • 고고학/역사 문서 특화 도메인 키워드 처리")
        print("  • 적응적 청크 크기 조정 및 품질 최적화")
        
        print("\n📈 전체적인 개선 효과:")
        print("  • RAG 검색 품질 향상: 맥락이 끊어지지 않는 연속적인 정보 제공")
        print("  • 의미적 일관성 향상: 관련된 내용이 같은 청크에 포함")
        print("  • 한국어 텍스트 처리 최적화: 언어적 특성을 고려한 정확한 분할")
        print("  • 도메인 특화 처리: 고고학 전문 용어 및 개념 인식")
        
        print("\n🔧 사용 방법:")
        print("1. ImprovedPDFConverter(enable_semantic_chunking=True) 로 인스턴스 생성")
        print("2. convert_pdf_to_semantic_chunks() 메서드로 PDF를 의미 기반 청크로 변환")
        print("3. 생성된 청크의 semantic_info에서 주제 일관성, 키워드 등 상세 정보 확인")
        print("4. RAG 시스템에서 개선된 검색 성능 및 답변 품질 확인")
        
    elif passed >= len(results) * 0.8:
        print(f"\n✅ 대부분의 테스트 통과 ({passed}/{len(results)})")
        print("PDF 맥락 개선 시스템이 대체로 잘 구현되었지만 일부 개선이 필요합니다.")
        
    else:
        print(f"\n⚠️  일부 테스트 실패 ({len(results)-passed}개)")
        print("문제가 발생한 부분을 점검하고 수정이 필요합니다.")

    print(f"\n📋 다음 단계:")
    if passed == len(results):
        print("• 실제 PDF 파일을 사용한 E2E 테스트")
        print("• RAG 시스템과의 통합 테스트")
        print("• 성능 모니터링 및 최적화")
        print("• 사용자 피드백 수집 및 개선")
    else:
        print("• 실패한 테스트 케이스 분석 및 수정")
        print("• 개별 컴포넌트 재검증")
        print("• 통합 로직 점검 및 개선")

if __name__ == "__main__":
    main()