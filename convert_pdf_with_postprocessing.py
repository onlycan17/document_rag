#!/usr/bin/env python3
"""
PDF to Markdown 변환 및 2단계 품질 개선 통합 도구

이 스크립트는 PDF 파일을 마크다운으로 변환하고 자동으로 2단계 품질 개선을 수행합니다.

사용법:
    python convert_pdf_with_postprocessing.py <pdf_file_or_directory> [options]

예시:
    # 단일 PDF 파일 처리
    python convert_pdf_with_postprocessing.py document.pdf
    
    # 디렉토리 내 모든 PDF 처리
    python convert_pdf_with_postprocessing.py ./pdf_files/
    
    # 품질 목표 지정
    python convert_pdf_with_postprocessing.py document.pdf --target-quality 95
    
    # 2단계 처리 비활성화
    python convert_pdf_with_postprocessing.py document.pdf --no-postprocessing
"""

import argparse
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import time
from typing import Tuple, List

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.pdf_converter import ImprovedPDFConverter
from src.utils.md_postprocessor import MDPostProcessor
from src.utils.quality_checker import QualityChecker
from src.utils.parallel_processor import ParallelProcessor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class PDFWithPostprocessing:
    """PDF 변환 및 2단계 품질 개선 통합 처리기"""
    
    def __init__(self, enable_postprocessing: bool = True, target_quality: int = 90):
        self.enable_postprocessing = enable_postprocessing
        self.target_quality = target_quality
        self.pdf_converter = ImprovedPDFConverter(enable_postprocessing=False)  # 수동으로 후처리 제어
        self.postprocessor = MDPostProcessor() if enable_postprocessing else None
        self.quality_checker = QualityChecker() if enable_postprocessing else None
        
    def process_single_pdf(self, pdf_path: Path) -> Tuple[bool, str]:
        """단일 PDF 파일 처리"""
        try:
            logger.info(f"처리 시작: {pdf_path.name}")
            
            # 1단계: PDF to Markdown 변환
            logger.info("1단계: PDF → Markdown 변환 중...")
            start_time = time.time()
            
            markdown_content, image_count = self.pdf_converter.convert_pdf_to_markdown(str(pdf_path))
            if not markdown_content:
                return False, "PDF 변환 실패"
            
            convert_time = time.time() - start_time
            logger.info(f"  ✓ 변환 완료 ({convert_time:.1f}초, {image_count}개 이미지 추출)")
            
            # converted_docs에서 생성된 MD 파일 경로
            md_path = Path("converted_docs") / f"{pdf_path.stem}.md"
            
            # 2단계: 품질 개선 (옵션)
            if self.enable_postprocessing and self.postprocessor:
                logger.info(f"2단계: 품질 개선 처리 중... (목표: {self.target_quality}점)")
                start_time = time.time()
                
                processed_md_path = Path("processed_docs") / f"{pdf_path.stem}.md"
                
                # process_md_file 메서드 호출 (process_file이 아님)
                try:
                    processed_content = self.postprocessor.process_md_file(str(md_path))
                    
                    # 처리된 내용을 파일로 저장
                    with open(processed_md_path, 'w', encoding='utf-8') as f:
                        f.write(processed_content)
                    
                    # 품질 검사
                    quality_result = self.quality_checker.analyze_quality(processed_content)
                    quality_score = quality_result.get('total_score', 0)
                    success = quality_result.get('passed', False)
                    result = quality_result
                    
                except Exception as e:
                    success = False
                    result = str(e)
                
                postprocess_time = time.time() - start_time
                
                if success:
                    # 품질 점수 추출
                    if isinstance(result, dict) and 'total_score' in result:
                        quality_score = result['total_score']
                        logger.info(f"  ✓ 품질 개선 완료 ({postprocess_time:.1f}초, 최종 품질: {quality_score}점)")
                    else:
                        logger.info(f"  ✓ 품질 개선 완료 ({postprocess_time:.1f}초)")
                    
                    total_time = convert_time + postprocess_time
                    return True, f"처리 완료 (총 {total_time:.1f}초)"
                else:
                    logger.warning(f"  ⚠ 품질 개선 실패: {result}")
                    return True, f"1단계만 완료 ({convert_time:.1f}초)"
            else:
                return True, f"변환 완료 ({convert_time:.1f}초)"
                
        except Exception as e:
            logger.error(f"처리 중 오류 발생: {str(e)}")
            return False, str(e)
    
    def process_directory(self, directory: Path, parallel: bool = True) -> Tuple[int, int]:
        """디렉토리 내 모든 PDF 파일 처리"""
        pdf_files = list(directory.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"{directory}에서 PDF 파일을 찾을 수 없습니다.")
            return 0, 0
        
        logger.info(f"{len(pdf_files)}개의 PDF 파일 발견")
        
        if parallel and len(pdf_files) > 1:
            # 병렬 처리
            logger.info("병렬 처리 모드로 실행...")
            success_count = 0
            fail_count = 0
            
            # ParallelProcessor를 사용하여 병렬 처리
            def process_wrapper(pdf_path: str) -> Tuple[bool, str]:
                return self.process_single_pdf(Path(pdf_path))
            
            # 파일 경로 리스트 생성
            file_paths = [str(pdf) for pdf in pdf_files]
            
            # 병렬 처리 실행
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_file = {executor.submit(process_wrapper, path): path for path in file_paths}
                
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        success, message = future.result()
                        if success:
                            success_count += 1
                            logger.info(f"✓ {Path(file_path).name}: {message}")
                        else:
                            fail_count += 1
                            logger.error(f"✗ {Path(file_path).name}: {message}")
                    except Exception as e:
                        fail_count += 1
                        logger.error(f"✗ {Path(file_path).name}: {str(e)}")
        else:
            # 순차 처리
            success_count = 0
            fail_count = 0
            
            for pdf_file in pdf_files:
                success, message = self.process_single_pdf(pdf_file)
                if success:
                    success_count += 1
                else:
                    fail_count += 1
        
        return success_count, fail_count

def main():
    parser = argparse.ArgumentParser(
        description="PDF to Markdown 변환 및 2단계 품질 개선 통합 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "input",
        help="처리할 PDF 파일 또는 디렉토리 경로"
    )
    
    parser.add_argument(
        "--no-postprocessing",
        action="store_true",
        help="2단계 품질 개선을 건너뜁니다 (1단계 변환만 수행)"
    )
    
    parser.add_argument(
        "--target-quality",
        type=int,
        default=90,
        help="목표 품질 점수 (기본값: 90)"
    )
    
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="디렉토리 처리 시 순차 처리 모드 사용 (기본값: 병렬 처리)"
    )
    
    args = parser.parse_args()
    
    # 입력 경로 확인
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"경로를 찾을 수 없습니다: {input_path}")
        sys.exit(1)
    
    # 출력 디렉토리 생성
    Path("converted_docs").mkdir(exist_ok=True)
    Path("converted_docs/images").mkdir(exist_ok=True)
    if not args.no_postprocessing:
        Path("processed_docs").mkdir(exist_ok=True)
    
    # 처리기 생성
    processor = PDFWithPostprocessing(
        enable_postprocessing=not args.no_postprocessing,
        target_quality=args.target_quality
    )
    
    # 처리 시작
    start_time = time.time()
    
    if input_path.is_file():
        # 단일 파일 처리
        if input_path.suffix.lower() != '.pdf':
            logger.error("PDF 파일만 처리할 수 있습니다.")
            sys.exit(1)
        
        success, message = processor.process_single_pdf(input_path)
        if success:
            logger.info(f"\n✅ 처리 완료: {message}")
        else:
            logger.error(f"\n❌ 처리 실패: {message}")
            sys.exit(1)
    else:
        # 디렉토리 처리
        success_count, fail_count = processor.process_directory(
            input_path, 
            parallel=not args.sequential
        )
        
        total_time = time.time() - start_time
        logger.info(f"\n" + "="*50)
        logger.info(f"처리 완료 - 성공: {success_count}개, 실패: {fail_count}개")
        logger.info(f"총 처리 시간: {total_time:.1f}초")
        
        if fail_count > 0:
            sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()