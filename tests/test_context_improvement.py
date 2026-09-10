#!/usr/bin/env python3
"""
PDF 페이지 경계 맥락 끊어짐 개선 테스트 스크립트
개선된 ImprovedPDFConverter가 페이지 경계에서 문장을 올바르게 연결하는지 검증합니다.
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_pdf_converter_improvements():
    """개선된 PDF 변환기 기능 테스트"""
    print("🔍 개선된 PDF 변환기 테스트")
    
    try:
        from src.utils.pdf_converter import ImprovedPDFConverter
        
        # ImprovedPDFConverter 인스턴스 생성
        temp_dir = tempfile.mkdtemp()
        converter = ImprovedPDFConverter(output_dir=temp_dir)
        
        # 새로 추가된 메서드들 확인
        required_methods = [
            '_connect_cross_page_text',
            '_can_connect_to_next',
            '_extract_and_process_images'
        ]
        
        all_exist = True
        for method in required_methods:
            if hasattr(converter, method):
                print(f"✓ {method} 메서드 존재")
            else:
                print(f"✗ {method} 메서드 누락")
                all_exist = False
        
        # 임시 디렉토리 정리
        shutil.rmtree(temp_dir)
        
        return all_exist
        
    except Exception as e:
        print(f"✗ PDF 변환기 테스트 실패: {e}")
        return False

def test_sentence_completion_logic():
    """문장 완성 로직 테스트"""
    print("\\n📝 문장 완성 로직 테스트")
    
    try:
        from src.utils.pdf_converter import ImprovedPDFConverter
        
        temp_dir = tempfile.mkdtemp()
        converter = ImprovedPDFConverter(output_dir=temp_dir)
        
        # 테스트 케이스: 불완전한 문장 감지
        test_cases = [
            ("그 결과, 몽", True),   # 불완전한 문장 (원본 문제 사례)
            ("풍납동 토", True),     # 불완전한 문장 (원본 문제 사례)
            ("이것은 완전한 문장이다.", False),  # 완전한 문장
            ("연구 결과를 확인했다", False),    # 완전한 문장 (문장부호 없지만 완전)
            ("따라서", True),        # 불완전한 문장
            ("123", True),          # 숫자만 (불완전)
            ("백제", True),         # 단일 단어 (불완전)
        ]
        
        all_passed = True
        for text, expected in test_cases:
            from src.utils.sentence_completion import is_incomplete_sentence as _is_inc
            result = _is_inc(text)
            if result == expected:
                print(f"✓ '{text}' -> {result} (예상: {expected})")
            else:
                print(f"✗ '{text}' -> {result} (예상: {expected})")
                all_passed = False
        
        # 임시 디렉토리 정리
        shutil.rmtree(temp_dir)
        
        return all_passed
        
    except Exception as e:
        print(f"✗ 문장 완성 로직 테스트 실패: {e}")
        return False

def test_connection_feasibility():
    """문장 연결 가능성 테스트"""
    print("\\n🔗 문장 연결 가능성 테스트")
    
    try:
        from src.utils.pdf_converter import ImprovedPDFConverter
        
        temp_dir = tempfile.mkdtemp()
        converter = ImprovedPDFConverter(output_dir=temp_dir)
        
        # 테스트 케이스: 문장 연결 가능성
        test_cases = [
            ("그 결과, 몽", "촌토성이 풍납동", True),    # 연결 가능
            ("풍납동 토", "성에서 원삼국시대", True),     # 연결 가능
            ("따라서 본 연구에서는", "1. 서론", False),   # 새로운 섹션 시작 (연결 불가)
            ("이것은 완전한 문장이다.", "그러나 다른", False),  # 새로운 문장 시작 (연결 불가)
            ("연구", "결과를 분석하면", True),           # 연결 가능
        ]
        
        all_passed = True
        for current, next_text, expected in test_cases:
            result = converter._can_connect_to_next(current, next_text)
            if result == expected:
                print(f"✓ '{current}' + '{next_text}' -> {result} (예상: {expected})")
            else:
                print(f"✗ '{current}' + '{next_text}' -> {result} (예상: {expected})")
                all_passed = False
        
        # 임시 디렉토리 정리
        shutil.rmtree(temp_dir)
        
        return all_passed
        
    except Exception as e:
        print(f"✗ 문장 연결 가능성 테스트 실패: {e}")
        return False

def test_cross_page_text_connection():
    """페이지 간 텍스트 연결 테스트"""
    print("\\n🔄 페이지 간 텍스트 연결 테스트")
    
    try:
        from src.utils.pdf_converter import ImprovedPDFConverter
        
        temp_dir = tempfile.mkdtemp()
        converter = ImprovedPDFConverter(output_dir=temp_dir)
        
        # 시뮬레이션된 페이지 텍스트 (원본 문제 상황 재현)
        page_texts = [
            (1, "이것은 첫 번째 페이지입니다.\\n연구 결과를 분석하면 몽"),
            (2, "촌토성이 풍납동 토성보다 늦게 축조되었다.\\n이는 중요한 발견이다."),
            (3, "따라서 우리는 백제의 왕성 변천사에 대해\\n새로운 해석을 할 수 있다.")
        ]
        
        # 텍스트 연결 테스트
        connected_text = converter._connect_cross_page_text(page_texts)
        
        print("연결된 텍스트:")
        print("-" * 50)
        print(connected_text)
        print("-" * 50)
        
        # 디버깅: 각 단계별 확인
        print("\\n디버깅 정보:")
        for i, (page_num, text) in enumerate(page_texts):
            lines = [line.strip() for line in text.split('\\n') if line.strip()]
            if lines:
                last_line = lines[-1]
                from src.utils.sentence_completion import is_incomplete_sentence as _is_inc2
                is_incomplete = _is_inc2(last_line)
                print(f"페이지 {page_num} 마지막 줄: '{last_line}' -> 불완전: {is_incomplete}")
                
                if i < len(page_texts) - 1:
                    next_page_num, next_text = page_texts[i + 1]
                    next_lines = [line.strip() for line in next_text.split('\\n') if line.strip()]
                    if next_lines:
                        first_next_line = next_lines[0]
                        can_connect = converter._can_connect_to_next(last_line, first_next_line)
                        print(f"  다음 페이지 첫 줄: '{first_next_line}' -> 연결 가능: {can_connect}")
                        print(f"  연결 결과: '{last_line + first_next_line}'")
        
        # 연결이 제대로 되었는지 확인
        success_indicators = [
            "몽촌토성이 풍납동" in connected_text,  # 페이지 경계 연결 확인
            "첫 번째 페이지" in connected_text,     # 첫 페이지 내용 포함
            "새로운 해석" in connected_text,       # 마지막 페이지 내용 포함
        ]
        
        print(f"\\n연결 확인 결과:")
        print(f"'몽촌토성이 풍납동' 포함: {success_indicators[0]}")
        print(f"'첫 번째 페이지' 포함: {success_indicators[1]}")
        print(f"'새로운 해석' 포함: {success_indicators[2]}")
        
        all_success = all(success_indicators)
        
        if all_success:
            print("✓ 페이지 간 텍스트 연결 성공")
        else:
            print("✗ 페이지 간 텍스트 연결 실패")
            print(f"성공 지표: {success_indicators}")
        
        # 임시 디렉토리 정리
        shutil.rmtree(temp_dir)
        
        return all_success
        
    except Exception as e:
        print(f"✗ 페이지 간 텍스트 연결 테스트 실패: {e}")
        return False

def test_existing_pdf_reprocessing():
    """기존 PDF 파일 재처리 테스트"""
    print("\\n📄 기존 PDF 파일 재처리 테스트")
    
    try:
        from src.utils.pdf_converter import ImprovedPDFConverter
        
        # 기존 PDF 파일 찾기
        pdf_files = list(Path(project_root).rglob("*.pdf"))
        
        if not pdf_files:
            print("ℹ️ 테스트할 PDF 파일이 없습니다.")
            return True
        
        # 첫 번째 PDF 파일로 테스트
        test_pdf = pdf_files[0]
        print(f"테스트 대상: {test_pdf.name}")
        
        temp_dir = tempfile.mkdtemp()
        converter = ImprovedPDFConverter(output_dir=temp_dir)
        
        # PDF 변환 시도
        def progress_callback(progress, message):
            print(f"진행률 {progress*100:.1f}%: {message}")
        
        markdown_content, image_count = converter.convert_pdf_to_markdown(
            str(test_pdf), 
            progress_callback=progress_callback
        )
        
        if markdown_content:
            print(f"✓ PDF 변환 성공: {len(markdown_content)} 문자, {image_count}개 이미지")
            
            # 생성된 MD 파일 확인
            md_files = list(Path(temp_dir).glob("*.md"))
            if md_files:
                md_file = md_files[0]
                print(f"✓ MD 파일 생성됨: {md_file.name}")
                
                # 내용 일부 확인
                with open(md_file, 'r', encoding='utf-8') as f:
                    content_preview = f.read()[:500]
                    print("\\n생성된 내용 미리보기:")
                    print("-" * 30)
                    print(content_preview + "...")
                    print("-" * 30)
            
            # 임시 디렉토리 정리
            shutil.rmtree(temp_dir)
            return True
        else:
            print("✗ PDF 변환 실패")
            shutil.rmtree(temp_dir)
            return False
        
    except Exception as e:
        print(f"✗ 기존 PDF 파일 재처리 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("🚀 PDF 페이지 경계 맥락 개선 테스트 시작\\n")
    
    tests = [
        ("PDF 변환기 개선사항", test_pdf_converter_improvements),
        ("문장 완성 로직", test_sentence_completion_logic),
        ("문장 연결 가능성", test_connection_feasibility),
        ("페이지 간 텍스트 연결", test_cross_page_text_connection),
        ("기존 PDF 재처리", test_existing_pdf_reprocessing),
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
    print("\\n" + "="*50)
    print("📊 테스트 결과 요약")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\\n총 {passed}/{len(results)}개 테스트 통과")
    
    if passed == len(results):
        print("\\n🎉 모든 테스트 통과! PDF 페이지 경계 맥락 개선이 완료되었습니다.")
        print("\\n구현된 개선사항:")
        print("• 페이지 경계에서 끊어진 문장 자동 연결")
        print("• 불완전한 문장 패턴 감지 알고리즘")
        print("• 문맥상 연결 가능한 텍스트 블록 식별")
        print("• 새로운 문장/단락 시작 감지 및 구분")
        print("• 이미지 정보와 페이지 정보 유지")
        
        print("\\n사용 방법:")
        print("1. 기존 PDF 파일을 다시 업로드하여 테스트")
        print("2. converted_docs/ 디렉토리에서 개선된 MD 파일 확인")
        print("3. 페이지 경계 문장 연결 개선 확인")
        print("4. RAG 시스템에서 더 나은 검색 결과 확인")
    else:
        print(f"\\n⚠️ {len(results)-passed}개 테스트 실패. 추가 확인이 필요합니다.")

if __name__ == "__main__":
    main()