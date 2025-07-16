#!/usr/bin/env python3
"""
몽촌토성 관련 PDF 문서를 마크다운으로 일괄 변환하는 스크립트

주요 기능:
- 몽촌토성 관련 PDF 파일 자동 식별
- 목차 제거 후 마크다운 변환
- 진행 상황 표시 및 상세 로깅
- 변환 결과 요약 보고서 생성
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import List, Dict

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from src.converters import PDFToMarkdownConverter
from src.utils.logging_config import setup_logging, get_logger

# 로깅 설정
setup_logging(logging.INFO)
logger = get_logger(__name__)


class MongchonDocumentConverter:
    """몽촌토성 관련 문서 변환을 담당하는 클래스"""
    
    # 몽촌토성 관련 파일 패턴 (파일명에 포함되어야 하는 키워드)
    MONGCHON_KEYWORDS = [
        "몽촌토성",
        "몽촌",
        "백제",
        "한성",
        "발굴조사",
        "토성"
    ]
    
    # 추가로 확실히 포함할 파일들 (파일명 패턴)
    EXPLICIT_FILES = [
        "2021년+몽촌토성+북문지+일원+발굴조사+자료집.pdf",
        "몽촌토성+북묵지+내측+발굴조사+보고서+1.pdf", 
        "몽촌토성4+하.pdf",
        "몽촌토성4+상.pdf",
        "0712 백제 왕궁·왕성(백제왕도 핵심유적, 풍납·몽촌토성 등) 연구 성과 발표(붙임).pdf",
        "000000169391.pdf"  # 사용자가 지정한 파일
    ]
    
    def __init__(self, documents_dir: str = "data/documents", use_ocr: bool = True):
        """
        몽촌토성 문서 변환기 초기화
        
        Args:
            documents_dir: PDF 문서가 있는 디렉토리
            use_ocr: OCR 사용 여부
        """
        self.documents_dir = Path(documents_dir)
        self.use_ocr = use_ocr
        
        # PDF to Markdown 변환기 초기화
        self.converter = PDFToMarkdownConverter(
            use_ocr=use_ocr,
            output_dir="data/processed/markdown"
        )
        
        logger.info(f"몽촌토성 문서 변환기 초기화 완료 (OCR: {use_ocr})")
    
    def find_mongchon_files(self) -> List[Path]:
        """
        몽촌토성 관련 PDF 파일들을 찾아서 반환
        
        Returns:
            몽촌토성 관련 PDF 파일 경로 리스트
        """
        if not self.documents_dir.exists():
            logger.error(f"문서 디렉토리를 찾을 수 없습니다: {self.documents_dir}")
            return []
        
        all_pdf_files = list(self.documents_dir.glob("*.pdf"))
        mongchon_files = []
        
        logger.info(f"📁 문서 디렉토리에서 PDF 파일 검색 중: {self.documents_dir}")
        logger.info(f"   총 {len(all_pdf_files)}개 PDF 파일 발견")
        
        # 1. 명시적으로 지정된 파일들 우선 추가
        for explicit_file in self.EXPLICIT_FILES:
            file_path = self.documents_dir / explicit_file
            if file_path.exists() and file_path not in mongchon_files:
                mongchon_files.append(file_path)
                logger.info(f"   ✅ 명시적 파일 발견: {explicit_file}")
        
        # 2. 키워드 기반으로 추가 파일 찾기
        for pdf_file in all_pdf_files:
            if pdf_file in mongchon_files:
                continue  # 이미 추가된 파일 스킵
                
            file_name_lower = pdf_file.name.lower()
            
            # 키워드 매칭 확인
            if any(keyword.lower() in file_name_lower for keyword in self.MONGCHON_KEYWORDS):
                mongchon_files.append(pdf_file)
                logger.info(f"   ✅ 키워드 매칭 파일: {pdf_file.name}")
        
        # 파일 크기 기준으로 정렬 (작은 파일부터 처리)
        mongchon_files.sort(key=lambda f: f.stat().st_size)
        
        logger.info(f"🎯 몽촌토성 관련 파일 {len(mongchon_files)}개 식별 완료")
        
        # 파일 목록과 크기 정보 출력
        for file_path in mongchon_files:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"   📄 {file_path.name} ({size_mb:.1f}MB)")
        
        return mongchon_files
    
    def convert_all_documents(self, remove_toc: bool = True, 
                            dry_run: bool = False) -> Dict[str, any]:
        """
        몽촌토성 관련 모든 문서를 변환
        
        Args:
            remove_toc: 목차 제거 여부
            dry_run: 실제 변환 없이 계획만 출력
            
        Returns:
            변환 결과 요약
        """
        mongchon_files = self.find_mongchon_files()
        
        if not mongchon_files:
            logger.warning("⚠️  변환할 몽촌토성 관련 파일을 찾을 수 없습니다")
            return {"success": False, "message": "변환할 파일 없음"}
        
        if dry_run:
            logger.info("🔍 DRY RUN 모드 - 실제 변환 없이 계획만 출력")
            total_size = sum(f.stat().st_size for f in mongchon_files) / (1024 * 1024)
            logger.info(f"📊 총 {len(mongchon_files)}개 파일, {total_size:.1f}MB 처리 예정")
            return {"success": True, "message": "계획 확인 완료", "files": len(mongchon_files)}
        
        # 실제 변환 시작
        start_time = time.time()
        results = {}
        success_count = 0
        error_count = 0
        total_original_pages = 0
        total_processed_pages = 0
        total_toc_removed = 0
        
        logger.info(f"🚀 몽촌토성 문서 변환 시작 - {len(mongchon_files)}개 파일")
        
        for i, pdf_file in enumerate(mongchon_files, 1):
            file_start_time = time.time()
            file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
            
            logger.info(f"\n📄 [{i}/{len(mongchon_files)}] 변환 중: {pdf_file.name}")
            logger.info(f"   📊 파일 크기: {file_size_mb:.1f}MB")
            
            try:
                # PDF 변환 실행
                result = self.converter.convert_pdf(
                    pdf_path=str(pdf_file),
                    remove_toc=remove_toc
                )
                
                results[pdf_file.name] = result
                file_time = time.time() - file_start_time
                
                if result.success:
                    success_count += 1
                    total_original_pages += result.original_pages
                    total_processed_pages += result.processed_pages
                    total_toc_removed += result.toc_pages_removed
                    
                    logger.info(f"   ✅ 변환 성공!")
                    logger.info(f"      📖 페이지: {result.original_pages} → {result.processed_pages}")
                    logger.info(f"      🗑️  목차 제거: {result.toc_pages_removed}페이지")
                    logger.info(f"      📁 출력: {result.output_file_path}")
                    logger.info(f"      ⏱️  소요시간: {file_time:.1f}초")
                else:
                    error_count += 1
                    logger.error(f"   ❌ 변환 실패: {result.error_message}")
                    logger.error(f"      ⏱️  실패까지 소요시간: {file_time:.1f}초")
                    
            except Exception as e:
                error_count += 1
                file_time = time.time() - file_start_time
                error_msg = f"예외 발생: {str(e)}"
                
                results[pdf_file.name] = {
                    "success": False,
                    "error_message": error_msg
                }
                
                logger.error(f"   ❌ 변환 중 예외 발생: {error_msg}")
                logger.error(f"      ⏱️  실패까지 소요시간: {file_time:.1f}초")
        
        # 전체 결과 요약
        total_time = time.time() - start_time
        
        logger.info(f"\n🎉 몽촌토성 문서 변환 완료!")
        logger.info(f"📊 처리 결과:")
        logger.info(f"   ✅ 성공: {success_count}개 파일")
        logger.info(f"   ❌ 실패: {error_count}개 파일")
        logger.info(f"   📖 총 페이지: {total_original_pages} → {total_processed_pages}")
        logger.info(f"   🗑️  목차 제거: {total_toc_removed}페이지")
        logger.info(f"   ⏱️  총 소요시간: {total_time:.1f}초")
        
        # 개별 파일 결과 상세 출력
        if results:
            logger.info(f"\n📋 개별 파일 결과:")
            for filename, result in results.items():
                if hasattr(result, 'success') and result.success:
                    logger.info(f"   ✅ {filename}: {result.processed_pages}페이지 처리완료")
                else:
                    error_msg = getattr(result, 'error_message', str(result.get('error_message', '알 수 없는 오류')))
                    logger.info(f"   ❌ {filename}: {error_msg}")
        
        return {
            "success": success_count > 0,
            "total_files": len(mongchon_files),
            "success_count": success_count,
            "error_count": error_count,
            "total_original_pages": total_original_pages,
            "total_processed_pages": total_processed_pages,
            "total_toc_removed": total_toc_removed,
            "total_time": total_time,
            "results": results
        }
    
    def generate_summary_report(self, conversion_results: Dict[str, any], 
                              output_file: str = "data/processed/conversion_summary.md") -> None:
        """
        변환 결과 요약 보고서를 마크다운으로 생성
        
        Args:
            conversion_results: 변환 결과 데이터
            output_file: 보고서 파일 경로
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 현재 시간
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 마크다운 보고서 생성
        report_content = f"""# 몽촌토성 관련 문서 마크다운 변환 보고서

생성일시: {current_time}

## 📊 변환 결과 요약

- **총 처리 파일**: {conversion_results.get('total_files', 0)}개
- **변환 성공**: {conversion_results.get('success_count', 0)}개  
- **변환 실패**: {conversion_results.get('error_count', 0)}개
- **원본 총 페이지**: {conversion_results.get('total_original_pages', 0)}페이지
- **처리된 페이지**: {conversion_results.get('total_processed_pages', 0)}페이지  
- **제거된 목차 페이지**: {conversion_results.get('total_toc_removed', 0)}페이지
- **총 소요 시간**: {conversion_results.get('total_time', 0):.1f}초

## 📋 개별 파일 처리 결과

"""
        
        if 'results' in conversion_results:
            for filename, result in conversion_results['results'].items():
                report_content += f"### {filename}\n\n"
                
                if hasattr(result, 'success') and result.success:
                    report_content += f"- **상태**: ✅ 변환 성공\n"
                    report_content += f"- **원본 페이지**: {result.original_pages}페이지\n"
                    report_content += f"- **처리된 페이지**: {result.processed_pages}페이지\n"
                    report_content += f"- **제거된 목차**: {result.toc_pages_removed}페이지\n"
                    report_content += f"- **출력 파일**: `{result.output_file_path}`\n"
                else:
                    error_msg = getattr(result, 'error_message', str(result.get('error_message', '알 수 없는 오류')))
                    report_content += f"- **상태**: ❌ 변환 실패\n"
                    report_content += f"- **오류 메시지**: {error_msg}\n"
                
                report_content += "\n"
        
        report_content += f"""
## 📁 출력 디렉토리

변환된 마크다운 파일들은 다음 디렉토리에 저장되었습니다:
```
data/processed/markdown/
```

## 🔍 변환 설정

- **OCR 사용**: {self.use_ocr}
- **목차 자동 제거**: 활성화
- **마크다운 구조화**: 활성화
- **한국어 최적화**: 활성화

---
*이 보고서는 몽촌토성 문서 변환 스크립트에 의해 자동 생성되었습니다.*
"""
        
        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📄 변환 요약 보고서 생성: {output_path}")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="몽촌토성 관련 PDF 문서를 마크다운으로 변환")
    parser.add_argument("--dry-run", action="store_true", 
                       help="실제 변환 없이 처리 계획만 출력")
    parser.add_argument("--no-ocr", action="store_true", 
                       help="OCR 사용하지 않음 (기본: OCR 사용)")
    parser.add_argument("--keep-toc", action="store_true", 
                       help="목차 제거하지 않음 (기본: 목차 제거)")
    parser.add_argument("--documents-dir", default="data/documents",
                       help="PDF 문서 디렉토리 (기본: data/documents)")
    
    args = parser.parse_args()
    
    # 시작 메시지
    print("🏛️ 몽촌토성 관련 문서 마크다운 변환 도구")
    print("=" * 50)
    
    try:
        # 변환기 초기화
        converter = MongchonDocumentConverter(
            documents_dir=args.documents_dir,
            use_ocr=not args.no_ocr
        )
        
        # 변환 실행
        results = converter.convert_all_documents(
            remove_toc=not args.keep_toc,
            dry_run=args.dry_run
        )
        
        if not args.dry_run and results.get("success"):
            # 요약 보고서 생성
            converter.generate_summary_report(results)
            
            print(f"\n🎉 변환 완료!")
            print(f"📊 성공: {results['success_count']}개, 실패: {results['error_count']}개")
            print(f"📁 출력 디렉토리: data/processed/markdown/")
            print(f"📄 상세 보고서: data/processed/conversion_summary.md")
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다")
    except Exception as e:
        logger.error(f"❌ 처리 중 오류 발생: {str(e)}")
        print(f"❌ 오류 발생: {str(e)}")


if __name__ == "__main__":
    main() 