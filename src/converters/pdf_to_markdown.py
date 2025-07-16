"""
PDF를 마크다운으로 변환하는 모듈

주요 기능:
- PDF에서 텍스트 추출 (OCR 포함)
- 목차 섹션 자동 감지 및 제거
- 텍스트 구조 분석 (제목, 문단, 목록 등)
- 마크다운 형식으로 변환
- 한국어 문서 최적화
"""

import re
import os
import sys
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

# 프로젝트 루트 경로를 sys.path에 추가 (직접 실행 시 필요)
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 프로젝트 내부 모듈 (절대 임포트로 변경)
try:
    from src.loaders.pdf_loader_advanced import AdvancedPDFLoader
    from src.utils.text_processing import TextProcessor
except ImportError:
    # 상대 임포트 폴백 (패키지로 실행될 때)
    from ..loaders.pdf_loader_advanced import AdvancedPDFLoader
    from ..utils.text_processing import TextProcessor

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """변환 결과를 담는 데이터 클래스"""
    success: bool
    markdown_content: str
    original_pages: int
    processed_pages: int
    toc_pages_removed: int
    output_file_path: str
    error_message: str = ""


class TOCDetector:
    """목차 감지 및 제거를 담당하는 클래스"""
    
    # 목차 관련 키워드 패턴
    TOC_KEYWORDS = [
        r'목\s*차',
        r'차\s*례', 
        r'contents',
        r'table\s*of\s*contents',
        r'index',
        r'색\s*인',
        r'찾\s*아\s*보\s*기'
    ]
    
    # 목차 구조 패턴
    TOC_PATTERNS = [
        r'^\s*\d+\.\s+.+?\s+\d+\s*$',  # 1. 제목 ... 페이지번호
        r'^\s*\d+\.\d+\s+.+?\s+\d+\s*$',  # 1.1 제목 ... 페이지번호
        r'^\s*[가-힣]+\.\s+.+?\s+\d+\s*$',  # 가. 제목 ... 페이지번호
        r'^\s*[①-⑳]\s+.+?\s+\d+\s*$',  # ① 제목 ... 페이지번호
        r'^\s*[ⅰ-ⅹ]+\.\s+.+?\s+\d+\s*$',  # ⅰ. 제목 ... 페이지번호
        r'^\s*제\s*\d+\s*[장절편]\s+.+?\s+\d+\s*$',  # 제1장 제목 ... 페이지번호
    ]
    
    # 점선 패턴 (목차에서 제목과 페이지 번호를 연결)
    DOTTED_PATTERNS = [
        r'\.{3,}',  # 연속된 점들 ...
        r'－{2,}',  # 연속된 대시들 --
        r'─{2,}',  # 연속된 긴 대시들 ──
        r'……+',   # 연속된 타원 부호들
    ]
    
    @classmethod
    def detect_toc_pages(cls, pages_content: List[str]) -> List[int]:
        """
        목차 페이지들을 감지하여 인덱스 리스트 반환
        
        Args:
            pages_content: 페이지별 텍스트 내용 리스트
            
        Returns:
            목차 페이지 인덱스 리스트 (0부터 시작)
        """
        toc_pages = []
        
        for page_idx, content in enumerate(pages_content):
            if cls._is_toc_page(content):
                toc_pages.append(page_idx)
                logger.debug(f"목차 페이지 감지: {page_idx + 1}")
        
        # 연속된 목차 페이지들을 확장하여 찾기
        expanded_toc_pages = cls._expand_toc_pages(pages_content, toc_pages)
        
        logger.info(f"총 {len(expanded_toc_pages)}개 목차 페이지 감지: {[p+1 for p in expanded_toc_pages]}")
        return expanded_toc_pages
    
    @classmethod
    def _is_toc_page(cls, content: str) -> bool:
        """단일 페이지가 목차인지 판단"""
        content_lower = content.lower()
        lines = content.split('\n')
        
        # 1. 목차 키워드 존재 확인
        has_toc_keyword = any(
            re.search(keyword, content_lower, re.IGNORECASE) 
            for keyword in cls.TOC_KEYWORDS
        )
        
        # 2. 목차 구조 패턴 개수 확인
        toc_pattern_count = 0
        dotted_line_count = 0
        
        for line in lines:
            # 목차 구조 패턴 체크
            for pattern in cls.TOC_PATTERNS:
                if re.search(pattern, line.strip()):
                    toc_pattern_count += 1
                    break
            
            # 점선 패턴 체크
            for dotted_pattern in cls.DOTTED_PATTERNS:
                if re.search(dotted_pattern, line):
                    dotted_line_count += 1
                    break
        
        # 3. 페이지 번호 패턴 확인
        page_number_pattern = r'\b\d{1,3}\b\s*$'
        page_number_lines = len([
            line for line in lines 
            if re.search(page_number_pattern, line.strip())
        ])
        
        # 목차 판단 로직
        total_lines = len([line for line in lines if line.strip()])
        
        # 조건 1: 목차 키워드가 있고 구조 패턴이 3개 이상
        if has_toc_keyword and toc_pattern_count >= 3:
            return True
        
        # 조건 2: 구조 패턴이 5개 이상이고 점선이나 페이지 번호가 많음
        if (toc_pattern_count >= 5 and 
            (dotted_line_count >= 3 or page_number_lines >= 3)):
            return True
        
        # 조건 3: 전체 라인의 50% 이상이 목차 패턴
        if (total_lines > 0 and 
            toc_pattern_count / total_lines >= 0.5 and 
            toc_pattern_count >= 4):
            return True
        
        return False
    
    @classmethod
    def _expand_toc_pages(cls, pages_content: List[str], initial_toc_pages: List[int]) -> List[int]:
        """
        감지된 목차 페이지 주변의 연관 페이지들도 목차로 확장
        (목차가 여러 페이지에 걸쳐 있을 수 있음)
        """
        if not initial_toc_pages:
            return initial_toc_pages
        
        expanded_pages = set(initial_toc_pages)
        
        # 각 목차 페이지의 앞뒤 페이지 확인
        for toc_page in initial_toc_pages:
            # 이전 페이지 확인
            if toc_page > 0:
                prev_content = pages_content[toc_page - 1]
                if cls._is_likely_toc_continuation(prev_content):
                    expanded_pages.add(toc_page - 1)
            
            # 다음 페이지 확인
            if toc_page < len(pages_content) - 1:
                next_content = pages_content[toc_page + 1]
                if cls._is_likely_toc_continuation(next_content):
                    expanded_pages.add(toc_page + 1)
        
        return sorted(list(expanded_pages))
    
    @classmethod
    def _is_likely_toc_continuation(cls, content: str) -> bool:
        """페이지가 목차의 연속인지 판단 (더 관대한 기준)"""
        lines = content.split('\n')
        
        # 목차 패턴이 적어도 2개 이상 있으면 목차 연속으로 판단
        toc_pattern_count = 0
        for line in lines:
            for pattern in cls.TOC_PATTERNS:
                if re.search(pattern, line.strip()):
                    toc_pattern_count += 1
                    break
        
        return toc_pattern_count >= 2


class MarkdownFormatter:
    """텍스트를 마크다운 형식으로 변환하는 클래스"""
    
    # 제목 패턴들 (우선순위 순)
    TITLE_PATTERNS = [
        (r'^제\s*(\d+)\s*[장편부]\s*(.+?)$', 1),  # 제1장, 제1편, 제1부
        (r'^(\d+)\.\s*(.+?)$', 2),  # 1. 제목
        (r'^(\d+)\.(\d+)\s*(.+?)$', 3),  # 1.1 제목
        (r'^(\d+)\.(\d+)\.(\d+)\s*(.+?)$', 4),  # 1.1.1 제목
        (r'^([가-힣])\.\s*(.+?)$', 3),  # 가. 제목
        (r'^([①-⑳])\s*(.+?)$', 4),  # ① 제목
        (r'^([ⅰ-ⅹ]+)\.\s*(.+?)$', 4),  # ⅰ. 제목
        (r'^([A-Z])\.\s*(.+?)$', 3),  # A. 제목
        (r'^◦\s*(.+?)$', 5),  # ◦ 제목
        (r'^○\s*(.+?)$', 5),  # ○ 제목
        (r'^●\s*(.+?)$', 5),  # ● 제목
    ]
    
    # 목록 항목 패턴들
    LIST_PATTERNS = [
        r'^[-\-‐‑‒–—]\s+(.+?)$',  # - 항목, – 항목, — 항목
        r'^[•▪▫]\s+(.+?)$',  # • 항목, ▪ 항목
        r'^[＊*]\s+(.+?)$',  # * 항목, ＊ 항목
        r'^\d+[)）]\s+(.+?)$',  # 1) 항목, 1） 항목
        r'^[가-힣][)）]\s+(.+?)$',  # 가) 항목
        r'^[①-⑳]\s+(.+?)$',  # ① 항목
    ]
    
    @classmethod
    def convert_to_markdown(cls, text: str) -> str:
        """
        텍스트를 마크다운으로 변환
        
        Args:
            text: 변환할 텍스트
            
        Returns:
            마크다운 형식의 텍스트
        """
        lines = text.split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                markdown_lines.append('')
                continue
            
            # 1. 제목 패턴 확인
            title_result = cls._convert_title(line)
            if title_result:
                markdown_lines.append(title_result)
                continue
            
            # 2. 목록 항목 확인
            list_result = cls._convert_list_item(line)
            if list_result:
                markdown_lines.append(list_result)
                continue
            
            # 3. 표 형태 확인 (간단한 경우만)
            table_result = cls._convert_table_row(line)
            if table_result:
                markdown_lines.append(table_result)
                continue
            
            # 4. 일반 문단
            markdown_lines.append(line)
        
        # 연속된 빈 줄 정리 (3개 이상을 2개로)
        result = '\n'.join(markdown_lines)
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
        
        return result.strip()
    
    @classmethod
    def _convert_title(cls, line: str) -> Optional[str]:
        """제목 패턴을 마크다운 헤더로 변환"""
        for pattern, level in cls.TITLE_PATTERNS:
            match = re.match(pattern, line.strip())
            if match:
                # 제목 텍스트 추출
                if pattern.startswith(r'^제\s*'):
                    # 제1장 형태
                    chapter_num = match.group(1)
                    title_text = match.group(2).strip()
                    header = '#' * min(level, 6)
                    return f"{header} 제{chapter_num}장 {title_text}"
                else:
                    # 다른 형태들
                    title_text = match.groups()[-1].strip()  # 마지막 그룹이 제목
                    header = '#' * min(level, 6)
                    return f"{header} {title_text}"
        
        # 길이와 위치로 제목 추정 (추가 로직)
        if (len(line) < 50 and 
            not line.endswith('.') and 
            not line.endswith(',') and
            not any(char in line for char in ['(', ')', '[', ']'])):
            # 짧고 문장 부호로 끝나지 않으면 제목일 가능성
            return f"## {line}"
        
        return None
    
    @classmethod
    def _convert_list_item(cls, line: str) -> Optional[str]:
        """목록 항목을 마크다운 목록으로 변환"""
        for pattern in cls.LIST_PATTERNS:
            match = re.match(pattern, line.strip())
            if match:
                item_text = match.group(1).strip()
                return f"- {item_text}"
        
        return None
    
    @classmethod
    def _convert_table_row(cls, line: str) -> Optional[str]:
        """표 형태의 줄을 마크다운 테이블로 변환 (기본적인 경우만)"""
        # 탭이나 다중 공백으로 구분된 경우
        if '\t' in line or '  ' in line:
            # 탭이나 다중 공백을 |로 변환
            cells = re.split(r'\t+|\s{3,}', line.strip())
            if len(cells) >= 2:
                return '| ' + ' | '.join(cell.strip() for cell in cells) + ' |'
        
        return None


class PDFToMarkdownConverter:
    """
    PDF 문서를 마크다운으로 변환하는 메인 클래스
    
    주요 기능:
    - PDF에서 텍스트 추출
    - 목차 자동 감지 및 제거
    - 마크다운 형식으로 변환
    - 한국어 문서 최적화
    """
    
    def __init__(self, use_ocr: bool = True, output_dir: str = "data/processed/markdown"):
        """
        PDF to Markdown 변환기 초기화
        
        Args:
            use_ocr: OCR 사용 여부
            output_dir: 출력 디렉토리 경로
        """
        self.use_ocr = use_ocr
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # PDF 로더 초기화
        self.pdf_loader = AdvancedPDFLoader(use_ocr=use_ocr)
        
        # 텍스트 프로세서 초기화
        self.text_processor = TextProcessor()
        
        logger.info(f"PDF to Markdown 변환기 초기화 완료 (OCR: {use_ocr})")
    
    def convert_pdf(self, pdf_path: str, output_filename: str = None, 
                   remove_toc: bool = True) -> ConversionResult:
        """
        PDF 파일을 마크다운으로 변환
        
        Args:
            pdf_path: 변환할 PDF 파일 경로
            output_filename: 출력 파일명 (None이면 자동 생성)
            remove_toc: 목차 제거 여부
            
        Returns:
            ConversionResult: 변환 결과
        """
        pdf_path = Path(pdf_path)
        
        try:
            logger.info(f"📄 PDF 변환 시작: {pdf_path.name}")
            
            # 1. PDF에서 텍스트 추출
            documents = self.pdf_loader.load_pdf(str(pdf_path))
            
            if not documents:
                return ConversionResult(
                    success=False,
                    markdown_content="",
                    original_pages=0,
                    processed_pages=0,
                    toc_pages_removed=0,
                    output_file_path="",
                    error_message="PDF에서 텍스트를 추출할 수 없습니다"
                )
            
            # 2. 페이지별 텍스트 분할
            pages_content = self._split_into_pages(documents[0].page_content)
            original_pages = len(pages_content)
            
            logger.info(f"   📖 텍스트 추출 완료: {original_pages}페이지")
            
            # 3. 목차 제거 (선택적)
            toc_pages_removed = 0
            if remove_toc:
                toc_pages = TOCDetector.detect_toc_pages(pages_content)
                if toc_pages:
                    pages_content = [
                        content for i, content in enumerate(pages_content)
                        if i not in toc_pages
                    ]
                    toc_pages_removed = len(toc_pages)
                    logger.info(f"   🗑️  목차 제거 완료: {toc_pages_removed}페이지 제거")
            
            # 4. 텍스트 정제 및 통합
            combined_text = self._combine_and_clean_pages(pages_content)
            
            # 5. 마크다운 변환
            markdown_content = MarkdownFormatter.convert_to_markdown(combined_text)
            
            # 6. 출력 파일 저장
            if not output_filename:
                output_filename = f"{pdf_path.stem}.md"
            
            output_path = self.output_dir / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            processed_pages = len(pages_content)
            
            logger.info(f"✅ 변환 완료: {output_path}")
            logger.info(f"   📊 원본: {original_pages}페이지 → 처리: {processed_pages}페이지")
            
            return ConversionResult(
                success=True,
                markdown_content=markdown_content,
                original_pages=original_pages,
                processed_pages=processed_pages,
                toc_pages_removed=toc_pages_removed,
                output_file_path=str(output_path)
            )
            
        except Exception as e:
            error_msg = f"PDF 변환 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            
            return ConversionResult(
                success=False,
                markdown_content="",
                original_pages=0,
                processed_pages=0,
                toc_pages_removed=0,
                output_file_path="",
                error_message=error_msg
            )
    
    def _split_into_pages(self, full_text: str) -> List[str]:
        """전체 텍스트를 페이지별로 분할"""
        # "--- 페이지 N ---" 패턴으로 분할
        page_pattern = r'\n?---\s*페이지\s*\d+\s*---\n?'
        pages = re.split(page_pattern, full_text)
        
        # 빈 페이지 제거
        pages = [page.strip() for page in pages if page.strip()]
        
        return pages
    
    def _combine_and_clean_pages(self, pages_content: List[str]) -> str:
        """페이지들을 통합하고 정제"""
        combined_text = ""
        
        for page_content in pages_content:
            # 각 페이지 정제
            cleaned_page = self.text_processor.clean_text(page_content)
            
            if cleaned_page:
                combined_text += cleaned_page + "\n\n"
        
        # 전체 텍스트 후처리
        combined_text = self._post_process_text(combined_text)
        
        return combined_text.strip()
    
    def _post_process_text(self, text: str) -> str:
        """텍스트 후처리 (한국어 최적화)"""
        # 연속된 공백 정리
        text = re.sub(r' +', ' ', text)
        
        # 연속된 줄바꿈 정리 (3개 이상을 2개로)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # 페이지 번호나 헤더/푸터 패턴 제거
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*-\s*\d+\s*-\s*$', '', text, flags=re.MULTILINE)
        
        # 한국어 문장 부호 정규화
        text = text.replace('．', '.')
        text = text.replace('，', ',')
        text = text.replace('；', ';')
        text = text.replace('：', ':')
        
        return text
    
    def batch_convert(self, pdf_files: List[str], remove_toc: bool = True) -> Dict[str, ConversionResult]:
        """
        여러 PDF 파일을 일괄 변환
        
        Args:
            pdf_files: PDF 파일 경로 리스트
            remove_toc: 목차 제거 여부
            
        Returns:
            파일별 변환 결과 딕셔너리
        """
        results = {}
        
        for i, pdf_file in enumerate(pdf_files, 1):
            file_path = Path(pdf_file)
            
            logger.info(f"📄 [{i}/{len(pdf_files)}] 변환 중: {file_path.name}")
            
            result = self.convert_pdf(
                pdf_path=str(file_path),
                remove_toc=remove_toc
            )
            
            results[file_path.name] = result
            
            if result.success:
                logger.info(f"   ✅ 완료: {result.processed_pages}페이지 처리")
            else:
                logger.error(f"   ❌ 실패: {result.error_message}")
        
        return results 


def main():
    """
    메인 실행 함수 - 명령행에서 직접 실행할 때 사용
    
    사용법:
        python3 pdf_to_markdown.py
        
    Example:
        python3 pdf_to_markdown.py
    """
    import argparse
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='PDF를 마크다운으로 변환합니다')
    parser.add_argument('pdf_file', nargs='?', help='변환할 PDF 파일 경로')
    parser.add_argument('--output', '-o', help='출력 파일명')
    parser.add_argument('--no-toc', action='store_true', help='목차 제거 비활성화')
    parser.add_argument('--no-ocr', action='store_true', help='OCR 비활성화')
    parser.add_argument('--batch', '-b', help='여러 PDF 파일이 있는 디렉토리 경로')
    
    args = parser.parse_args()
    
    # PDF 파일 또는 디렉토리가 지정되지 않은 경우 예시 실행
    if not args.pdf_file and not args.batch:
        # 테스트용 PDF 파일이 있는지 확인
        test_files = []
        data_dir = Path("data/documents")
        if data_dir.exists():
            test_files = list(data_dir.glob("*.pdf"))
        
        if test_files:
            print(f"🔍 발견된 PDF 파일들:")
            for i, file in enumerate(test_files[:5], 1):  # 최대 5개만 표시
                print(f"   {i}. {file.name}")
            
            choice = input(f"\n변환할 파일 번호를 선택하세요 (1-{min(5, len(test_files))}) 또는 Enter로 첫 번째 파일: ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(test_files):
                selected_file = test_files[int(choice) - 1]
            else:
                selected_file = test_files[0]
            
            args.pdf_file = str(selected_file)
            print(f"✅ 선택된 파일: {args.pdf_file}")
        else:
            print("❌ 변환할 PDF 파일을 지정해주세요.")
            print("\n사용법:")
            print("  python3 pdf_to_markdown.py <PDF파일경로>")
            print("  python3 pdf_to_markdown.py --batch <디렉토리경로>")
            print("\n예시:")
            print("  python3 pdf_to_markdown.py document.pdf")
            print("  python3 pdf_to_markdown.py --batch data/documents/")
            return
    
    # 변환기 초기화
    converter = PDFToMarkdownConverter(
        use_ocr=not args.no_ocr,
        output_dir="data/processed/markdown"
    )
    
    try:
        if args.batch:
            # 배치 변환
            batch_dir = Path(args.batch)
            if not batch_dir.exists():
                print(f"❌ 디렉토리를 찾을 수 없습니다: {batch_dir}")
                return
            
            pdf_files = list(batch_dir.glob("*.pdf"))
            if not pdf_files:
                print(f"❌ PDF 파일을 찾을 수 없습니다: {batch_dir}")
                return
            
            print(f"📁 배치 변환 시작: {len(pdf_files)}개 파일")
            results = converter.batch_convert(
                pdf_files=[str(f) for f in pdf_files],
                remove_toc=not args.no_toc
            )
            
            # 결과 요약
            success_count = sum(1 for r in results.values() if r.success)
            print(f"\n📊 배치 변환 완료:")
            print(f"   ✅ 성공: {success_count}개")
            print(f"   ❌ 실패: {len(results) - success_count}개")
            
        else:
            # 단일 파일 변환
            pdf_path = Path(args.pdf_file)
            if not pdf_path.exists():
                print(f"❌ 파일을 찾을 수 없습니다: {pdf_path}")
                return
            
            print(f"📄 PDF 변환 시작: {pdf_path.name}")
            
            result = converter.convert_pdf(
                pdf_path=str(pdf_path),
                output_filename=args.output,
                remove_toc=not args.no_toc
            )
            
            if result.success:
                print(f"✅ 변환 완료!")
                print(f"   📊 원본: {result.original_pages}페이지")
                print(f"   📋 처리: {result.processed_pages}페이지")
                if result.toc_pages_removed > 0:
                    print(f"   🗑️  목차 제거: {result.toc_pages_removed}페이지")
                print(f"   📁 출력: {result.output_file_path}")
            else:
                print(f"❌ 변환 실패: {result.error_message}")
                
    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {str(e)}")
        logger.exception("변환 중 오류 발생")


if __name__ == "__main__":
    main()