#!/usr/bin/env python3
"""
단일 PDF 파일로 마크다운 변환을 테스트하는 스크립트

가장 작은 파일부터 테스트하여 변환기가 정상 동작하는지 확인합니다.
"""

import sys
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from src.converters import PDFToMarkdownConverter
from src.utils.logging_config import setup_logging, get_logger

# 로깅 설정
setup_logging(logging.INFO)
logger = get_logger(__name__)


def test_small_file():
    """가장 작은 파일로 변환 테스트"""
    
    # 테스트할 파일 (가장 작은 파일)
    test_file = "data/documents/0712 백제 왕궁·왕성(백제왕도 핵심유적, 풍납·몽촌토성 등) 연구 성과 발표(붙임).pdf"
    
    print("🧪 PDF → 마크다운 변환 테스트")
    print("=" * 50)
    print(f"📄 테스트 파일: {Path(test_file).name}")
    
    try:
        # 변환기 초기화 (OCR 사용)
        converter = PDFToMarkdownConverter(
            use_ocr=True,
            output_dir="data/processed/test_markdown"
        )
        
        # 변환 실행
        result = converter.convert_pdf(
            pdf_path=test_file,
            output_filename="test_백제왕궁_연구성과.md",
            remove_toc=True
        )
        
        if result.success:
            print(f"\n✅ 변환 성공!")
            print(f"📊 원본 페이지: {result.original_pages}")
            print(f"📊 처리된 페이지: {result.processed_pages}")
            print(f"🗑️  제거된 목차: {result.toc_pages_removed}페이지")
            print(f"📁 출력 파일: {result.output_file_path}")
            
            # 마크다운 파일 크기 확인
            output_path = Path(result.output_file_path)
            if output_path.exists():
                size_kb = output_path.stat().st_size / 1024
                print(f"📏 마크다운 파일 크기: {size_kb:.1f}KB")
                
                # 첫 몇 줄 미리보기
                with open(output_path, 'r', encoding='utf-8') as f:
                    preview = f.read(500)  # 처음 500자
                    print(f"\n📖 마크다운 미리보기:")
                    print("-" * 30)
                    print(preview)
                    if len(result.markdown_content) > 500:
                        print("...")
                    print("-" * 30)
            
            return True
        else:
            print(f"\n❌ 변환 실패: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {str(e)}")
        logger.error(f"테스트 실패: {str(e)}")
        return False


def test_without_ocr():
    """OCR 없이 변환 테스트 (속도 비교용)"""
    
    test_file = "data/documents/0712 백제 왕궁·왕성(백제왕도 핵심유적, 풍납·몽촌토성 등) 연구 성과 발표(붙임).pdf"
    
    print(f"\n🧪 OCR 없이 변환 테스트")
    print("-" * 30)
    
    try:
        # 변환기 초기화 (OCR 미사용)
        converter = PDFToMarkdownConverter(
            use_ocr=False,
            output_dir="data/processed/test_markdown"
        )
        
        # 변환 실행
        result = converter.convert_pdf(
            pdf_path=test_file,
            output_filename="test_백제왕궁_연구성과_no_ocr.md",
            remove_toc=True
        )
        
        if result.success:
            print(f"✅ OCR 없이 변환 성공!")
            print(f"📊 처리된 페이지: {result.processed_pages}")
            
            # 텍스트 길이 비교
            text_length = len(result.markdown_content)
            print(f"📏 텍스트 길이: {text_length:,}자")
            
            return True
        else:
            print(f"❌ OCR 없이 변환 실패: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ OCR 없이 테스트 실패: {str(e)}")
        return False


if __name__ == "__main__":
    try:
        # 1. OCR 사용 테스트
        success_ocr = test_small_file()
        
        # 2. OCR 미사용 테스트
        success_no_ocr = test_without_ocr()
        
        # 결과 요약
        print(f"\n🎯 테스트 결과 요약:")
        print(f"   OCR 사용: {'✅ 성공' if success_ocr else '❌ 실패'}")
        print(f"   OCR 미사용: {'✅ 성공' if success_no_ocr else '❌ 실패'}")
        
        if success_ocr or success_no_ocr:
            print(f"\n🎉 기본 변환 기능이 정상 동작합니다!")
            print(f"📁 테스트 결과 확인: data/processed/test_markdown/")
        else:
            print(f"\n⚠️  변환 기능에 문제가 있습니다. 로그를 확인해주세요.")
            
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {str(e)}") 