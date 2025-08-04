#!/usr/bin/env python3
"""
개선된 PDF to Markdown Converter 모듈

주요 기능:
- 문장 연결 문제 해결
- 헤딩 인식 로직 정교화
- 한글 텍스트 완벽 지원
- 이미지 추출 및 별도 저장
"""

import os
import fitz  # PyMuPDF
from pathlib import Path
import re
from datetime import datetime
from typing import Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)

class ImprovedPDFConverter:
    """개선된 PDF to Markdown 변환기"""
    
    def __init__(self, output_dir: str = "converted_docs"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        
    def convert_pdf_to_markdown(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> Tuple[Optional[str], int]:
        """
        PDF 파일을 마크다운으로 변환
        
        Args:
            pdf_path: PDF 파일 경로
            progress_callback: 진행 상황 콜백 함수
        
        Returns:
            Tuple[마크다운 내용, 이미지 개수]
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            logger.error(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            return None, 0
            
        try:
            if progress_callback:
                progress_callback(0.1, "PDF 파일 열기...")
            
            doc = fitz.open(pdf_path)
            
            if progress_callback:
                progress_callback(0.2, "텍스트 및 이미지 추출 중...")
            
            markdown_content, image_count = self._extract_text_and_images(doc, pdf_path, progress_callback)
            doc.close()
            
            if progress_callback:
                progress_callback(0.9, "마크다운 파일 생성 중...")
            
            # 마크다운 파일 저장
            md_filename = f"{pdf_path.stem}.md"
            md_path = self.output_dir / md_filename
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            if progress_callback:
                progress_callback(1.0, f"변환 완료: {image_count}개 이미지 추출")
            
            logger.info(f"PDF 변환 완료: {pdf_path.name} -> {md_path}")
            return markdown_content, image_count
            
        except Exception as e:
            logger.error(f"PDF 변환 실패: {pdf_path} - {str(e)}")
            if progress_callback:
                progress_callback(1.0, f"변환 실패: {str(e)}")
            return None, 0
    
    def _extract_text_and_images(self, doc, pdf_path: Path, progress_callback: Optional[Callable] = None) -> Tuple[str, int]:
        """텍스트와 이미지 추출 (개선된 버전)"""
        markdown_content = []
        image_count = 0
        total_pages = len(doc)
        
        # PDF 메타데이터 추가
        metadata = doc.metadata
        if metadata.get('title'):
            markdown_content.append(f"# {metadata['title']}")
        else:
            markdown_content.append(f"# {pdf_path.stem}")
            
        if metadata.get('author'):
            markdown_content.append(f"\n**저자:** {metadata['author']}")
        if metadata.get('subject'):
            markdown_content.append(f"**주제:** {metadata['subject']}")
        if metadata.get('creator'):
            markdown_content.append(f"**생성 도구:** {metadata['creator']}")
            
        markdown_content.append(f"\n**소스 파일:** {pdf_path.name}")
        markdown_content.append(f"**변환 날짜:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_content.append("\n---\n")
        
        # 각 페이지 처리
        for page_num in range(total_pages):
            page = doc[page_num]
            
            # 진행률 업데이트
            if progress_callback:
                progress = 0.2 + (page_num / total_pages) * 0.6
                progress_callback(progress, f"페이지 {page_num + 1}/{total_pages} 처리 중...")
            
            markdown_content.append(f"\n## 페이지 {page_num + 1}\n")
            
            # 텍스트 추출 (개선된 방법)
            text = page.get_text()
            if text.strip():
                # 개선된 텍스트 정리 및 포맷팅
                cleaned_text = self._clean_and_format_text(text)
                if cleaned_text.strip():
                    markdown_content.append(cleaned_text)
            
            # 이미지 추출
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    # 이미지 데이터 추출
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # PNG로 변환
                    image_filename = f"{pdf_path.stem}_page{page_num+1}_img{img_index+1}.png"
                    image_path = self.images_dir / image_filename
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        pix.save(str(image_path))
                    else:  # CMYK: convert to RGB first
                        pix1 = fitz.Pixmap(fitz.csRGB, pix)
                        pix1.save(str(image_path))
                        pix1 = None
                    
                    pix = None
                    
                    # 마크다운에 이미지 참조 추가
                    markdown_content.append(f"\n![이미지 {image_count + 1}](images/{image_filename})\n")
                    image_count += 1
                    
                except Exception as e:
                    logger.warning(f"이미지 추출 실패 (페이지 {page_num + 1}, 이미지 {img_index + 1}): {e}")
        
        return "\n".join(markdown_content), image_count
    
    def _clean_and_format_text(self, text: str) -> str:
        """개선된 텍스트 정리 및 마크다운 포맷팅"""
        if not text.strip():
            return ""
        
        # 1단계: 기본 정리
        # 여러 개의 개행을 하나로 줄이기
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # 페이지 번호 제거 (숫자만 있는 라인)
        text = re.sub(r'^[Pp]age\s*\d+.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*-\s*\d+\s*-\s*$', '', text, flags=re.MULTILINE)
        
        # 2단계: 줄 단위로 처리
        lines = text.split('\n')
        processed_lines = []
        current_paragraph = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                # 빈 줄 처리
                if current_paragraph:
                    # 현재 문단을 완성하여 추가
                    paragraph_text = self._join_paragraph_lines(current_paragraph)
                    if paragraph_text:
                        processed_lines.append(paragraph_text)
                    current_paragraph = []
                processed_lines.append('')
                continue
                
            # 3단계: 실제 헤딩인지 판단 (더 엄격한 기준)
            if self._is_real_heading(line, i, lines):
                # 현재 문단을 먼저 완성
                if current_paragraph:
                    paragraph_text = self._join_paragraph_lines(current_paragraph)
                    if paragraph_text:
                        processed_lines.append(paragraph_text)
                    current_paragraph = []
                
                # 헤딩 추가
                level = self._get_heading_level(line)
                processed_lines.append(f"{'#' * level} {line}")
            else:
                # 일반 텍스트는 문단에 추가
                current_paragraph.append(line)
        
        # 마지막 문단 처리
        if current_paragraph:
            paragraph_text = self._join_paragraph_lines(current_paragraph)
            if paragraph_text:
                processed_lines.append(paragraph_text)
        
        # 4단계: 최종 정리
        result = '\n'.join(processed_lines)
        
        # 연속된 빈 줄 정리
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
        
        return result.strip()
    
    def _is_real_heading(self, line: str, line_index: int, all_lines: list) -> bool:
        """실제 헤딩인지 더 정교하게 판단"""
        # 너무 긴 텍스트는 헤딩이 아님
        if len(line) > 100:
            return False
        
        # 문장 부호로 끝나는 경우 일반적으로 헤딩이 아님
        if line.endswith(('.', '다', '음', '었다', '였다', '한다', '된다', '이다', '않다')):
            return False
        
        # 명확한 헤딩 패턴들
        heading_patterns = [
            r'^제\s*\d+\s*장',  # 제1장, 제 2 장
            r'^제\s*\d+\s*절',  # 제1절, 제 2 절
            r'^\d+\.\s*[가-힣]',  # 1. 서론
            r'^[가-힣]\.\s*[가-힣]',  # 가. 개요
            r'^\([가-힣]\)',  # (가)
            r'^\d+\)\s*[가-힣]',  # 1) 목적
            r'^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.',  # I. II. III.
            r'^【[^】]+】',  # 【제목】
            r'^◎|^○|^●|^■|^□',  # 기호로 시작
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, line):
                return True
        
        # 전체가 대문자인 경우 (영어)
        if line.isupper() and len(line) < 50 and re.search(r'[A-Z]', line):
            return True
        
        # 특정 키워드로 시작하는 제목들
        heading_keywords = [
            '서론', '결론', '요약', '개요', '배경', '목적', '방법', '결과', '고찰', '참고문헌',
            '조사', '발굴', '유적', '유물', '분석', '검토', '연구', '현황'
        ]
        
        for keyword in heading_keywords:
            if line.startswith(keyword) and len(line) < 80:
                # 다음 줄이 내용인지 확인
                if line_index + 1 < len(all_lines):
                    next_line = all_lines[line_index + 1].strip()
                    if next_line and len(next_line) > 20:  # 다음 줄이 충분히 긴 내용이면
                        return True
        
        return False
    
    def _join_paragraph_lines(self, lines: list) -> str:
        """문단의 줄들을 자연스럽게 연결"""
        if not lines:
            return ""
        
        # 각 줄의 끝을 확인하여 연결 방식 결정
        result_parts = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            if i == 0:
                result_parts.append(line)
            else:
                prev_line = lines[i-1].strip()
                
                # 이전 줄이 완전한 문장으로 끝나는지 확인
                if self._ends_complete_sentence(prev_line):
                    # 새로운 문장 시작
                    result_parts.append(" " + line)
                else:
                    # 문장 계속 (줄바꿈으로 인해 끊어진 경우)
                    result_parts.append(line)
        
        return "".join(result_parts)
    
    def _ends_complete_sentence(self, line: str) -> bool:
        """줄이 완전한 문장으로 끝나는지 판단"""
        if not line:
            return True
            
        # 한글 문장 종결 패턴
        korean_endings = ['다', '음', '였다', '었다', '한다', '된다', '이다', '않다', '있다', '없다']
        
        for ending in korean_endings:
            if line.endswith(ending + '.') or line.endswith(ending):
                return True
        
        # 영어/숫자 문장 종결
        if line.endswith(('.', '!', '?', ':', ';')):
            return True
            
        # 닫는 괄호나 따옴표로 끝나는 경우
        if line.endswith((')', '"', "'", '』', '】')):
            return True
            
        return False
    
    def _get_heading_level(self, line: str) -> int:
        """헤딩 레벨 결정 (개선된 버전)"""
        # 장/절 구조
        if re.match(r'^제\s*\d+\s*장', line):
            return 1
        elif re.match(r'^제\s*\d+\s*절', line):
            return 2
        
        # 숫자 패턴으로 레벨 결정
        if re.match(r'^\d+\.', line):  # 1., 2., 3.
            return 2
        elif re.match(r'^\d+\.\d+', line):  # 1.1, 1.2
            return 3
        elif re.match(r'^\d+\.\d+\.\d+', line):  # 1.1.1
            return 4
        elif re.match(r'^[가-힣]\.', line):  # 가., 나., 다.
            return 3
        elif re.match(r'^\([가-힣]\)', line):  # (가), (나)
            return 4
        elif re.match(r'^\d+\)', line):  # 1), 2)
            return 4
        elif re.match(r'^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.', line):  # I., II.
            return 2
        else:
            return 2