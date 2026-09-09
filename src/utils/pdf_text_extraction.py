#!/usr/bin/env python3
"""
PDF 텍스트 추출·정제 믹스인

ImprovedPDFConverter에서 텍스트/이미지 추출과 마크다운 정제 책임만 분리한 믹스인.
인스턴스 속성(images_dir 등)은 본체(ImprovedPDFConverter)에서 제공된다.
"""

import fitz  # PyMuPDF
from pathlib import Path
import re
from datetime import datetime
from typing import Optional, Callable, Tuple
import logging

from .pdf_heading_utils import is_real_heading, join_paragraph_lines, get_heading_level
from .sentence_completion import is_incomplete_sentence
from .text_processing import TextProcessor

logger = logging.getLogger(__name__)


class PdfTextExtractionMixin:
    """PDF 텍스트 추출 및 마크다운 정제 기능"""

    def _extract_text_and_images(
        self, doc, pdf_path: Path, progress_callback: Optional[Callable] = None
    ) -> Tuple[str, int]:
        """텍스트와 이미지 추출 (맥락 연결 개선 버전)"""
        markdown_content = []
        image_count = 0
        total_pages = len(doc)

        # PDF 메타데이터 추가
        metadata = doc.metadata
        if metadata.get("title"):
            markdown_content.append(f"# {metadata['title']}")
        else:
            markdown_content.append(f"# {pdf_path.stem}")

        if metadata.get("author"):
            markdown_content.append(f"\n**저자:** {metadata['author']}")
        if metadata.get("subject"):
            markdown_content.append(f"**주제:** {metadata['subject']}")
        if metadata.get("creator"):
            markdown_content.append(f"**생성 도구:** {metadata['creator']}")

        markdown_content.append(f"\n**소스 파일:** {pdf_path.name}")
        markdown_content.append(f"**변환 날짜:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_content.append("\n---\n")

        # 1단계: 모든 페이지에서 텍스트와 이미지 정보 수집
        if progress_callback:
            progress_callback(0.2, "모든 페이지 텍스트 수집 중...")

        page_texts = []
        page_images = []

        for page_num in range(total_pages):
            page = doc[page_num]

            # 진행률 업데이트
            if progress_callback:
                progress = 0.2 + (page_num / total_pages) * 0.3
                progress_callback(progress, f"페이지 {page_num + 1}/{total_pages} 텍스트 수집 중...")
            # 콜백이 없을 때도 대용량 문서에서 진행 상황을 가시화하기 위한 하트비트 로그
            elif (page_num + 1) % 25 == 0 or page_num == 0:
                try:
                    logger.info(f"⏳ 페이지 진행: {page_num + 1}/{total_pages} 텍스트 수집 중")
                except Exception as err:
                    logger.debug(f"페이지 진행 로그 출력 실패(무시): {err}")

            # 텍스트 수집
            text = page.get_text()
            page_texts.append((page_num + 1, text.strip() if text else ""))

            # 이미지 정보 수집
            image_list = page.get_images()
            page_images.append((page_num + 1, image_list))

        # 2단계: 페이지 경계 텍스트 연결 처리
        if progress_callback:
            progress_callback(0.5, "페이지 경계 텍스트 연결 처리 중...")

        connected_text = self._connect_cross_page_text(page_texts)

        # 3단계: 연결된 텍스트를 마크다운으로 변환 (페이지별 이미지 정보와 함께)
        if progress_callback:
            progress_callback(0.6, "마크다운 텍스트 변환 중...")
        else:
            logger.info("🧩 텍스트 연결 및 마크다운 변환 단계 진입")

        if connected_text.strip():
            # 이미지 처리를 먼저 수행
            if progress_callback:
                progress_callback(0.65, "이미지 추출 및 매핑 중...")
            else:
                logger.info("🖼️ 페이지 이미지 매핑 및 추출 시작")

            image_references = self._extract_and_process_images(doc, pdf_path, page_images, progress_callback)
            image_count = len(image_references)

            # 텍스트 정리 및 이미지 삽입을 함께 처리
            cleaned_text = self._clean_and_format_text_with_images(connected_text, page_images, image_references)
            if cleaned_text.strip():
                markdown_content.append(cleaned_text)
        else:
            # 텍스트가 없는 경우에도 이미지는 추출
            if progress_callback:
                progress_callback(0.7, "이미지 추출 중...")

            image_references = self._extract_and_process_images(doc, pdf_path, page_images, progress_callback)
            image_count = len(image_references)

            # 이미지만 있는 경우
            if image_references:
                markdown_content.append("\n\n## 추출된 이미지\n")
                for img_ref in image_references:
                    markdown_content.append(img_ref)

        return "\n".join(markdown_content), image_count

    def _connect_cross_page_text(self, page_texts: list) -> str:
        """페이지 경계에서 끊어진 텍스트를 자연스럽게 연결 (개선된 버전)"""
        if not page_texts:
            return ""

        # 모든 페이지의 텍스트를 먼저 정리
        all_lines = []
        page_boundaries = []  # 페이지 경계 정보 저장

        for i, (page_num, text) in enumerate(page_texts):
            if not text.strip():
                continue

            text_lines = [line.strip() for line in text.split("\n") if line.strip()]
            if not text_lines:
                continue

            # 페이지 시작 위치 기록
            page_boundaries.append((page_num, len(all_lines)))
            all_lines.extend(text_lines)

        if not all_lines:
            return ""

        # 줄별로 연결 가능성 검토 및 연결 처리
        connected_lines = []
        i = 0

        while i < len(all_lines):
            current_line = all_lines[i]

            # 다음 줄이 있는지 확인
            if i < len(all_lines) - 1:
                next_line = all_lines[i + 1]

                # 현재 줄이 불완전한 문장이고 다음 줄과 연결 가능한지 확인
                if is_incomplete_sentence(current_line) and self._can_connect_to_next(current_line, next_line):
                    # 연결 처리
                    connected_line = current_line + next_line
                    connected_lines.append(connected_line)

                    # 로그 출력 (디버깅용)
                    logger.debug(f"문장 연결: '{current_line}' + '{next_line}' = '{connected_line}'")

                    # 다음 줄은 이미 연결했으므로 건너뛰기
                    i += 2
                    continue

            # 연결하지 않는 경우 그대로 추가
            connected_lines.append(current_line)
            i += 1

        return "\n".join(connected_lines)

    def _can_connect_to_next(self, current_line: str, next_line: str) -> bool:
        """현재 줄과 다음 줄이 연결 가능한지 판단"""
        if not current_line or not next_line:
            return False

        current_line = current_line.strip()
        next_line = next_line.strip()

        # 다음 줄이 명백한 새로운 문장/단락으로 시작하는 경우 연결하지 않음
        new_sentence_starters = [
            "그러나",
            "하지만",
            "따라서",
            "그런데",
            "또한",
            "그리고",
            "한편",
            "첫째",
            "둘째",
            "셋째",
            "다음",
            "마지막으로",
            "제",
            "장",
            "절",  # 제1장, 제2절 등
            "그 결과",
            "이에 따라",
            "결론적으로",
        ]

        for starter in new_sentence_starters:
            if next_line.startswith(starter):
                return False

        # 숫자나 기호로 시작하는 새로운 항목들
        if re.match(r"^\d+[\.\)]\s", next_line):  # 1. 또는 1)
            return False
        if re.match(r"^[가-힣][\.\)]\s", next_line):  # 가. 또는 가)
            return False

        # 대문자로 시작하는 새로운 영어 문장
        if re.match(r"^[A-Z][a-z]", next_line):
            return False

        # 현재 줄이 숫자나 짧은 단어로 끝나고, 다음 줄이 자연스럽게 이어질 수 있는 경우
        if len(current_line.split()[-1]) <= 3:  # 마지막 단어가 3글자 이하
            return True

        return True

    def _extract_and_process_images(
        self, doc, pdf_path: Path, page_images: list, progress_callback: Optional[Callable] = None
    ) -> list:
        """모든 페이지의 이미지를 추출하고 마크다운 참조 생성"""
        image_references = []
        total_images = sum(len(images) for _, images in page_images)
        processed_images = 0
        MIN_IMAGE_SIZE = 50  # 최소 이미지 크기 (픽셀)

        for page_num, image_list in page_images:
            for img_index, img in enumerate(image_list):
                try:
                    # 진행률 업데이트
                    if progress_callback and total_images > 0:
                        progress = 0.7 + (processed_images / total_images) * 0.2
                        progress_callback(progress, f"이미지 {processed_images + 1}/{total_images} 추출 중...")

                    # 이미지 데이터 추출
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)

                    # 이미지 크기 확인 - 너무 작은 이미지는 건너뛰기
                    if pix.width < MIN_IMAGE_SIZE or pix.height < MIN_IMAGE_SIZE:
                        logger.info(f"   ⚠️  너무 작은 이미지 건너뛰기: {pix.width}x{pix.height} (페이지 {page_num})")
                        pix = None
                        processed_images += 1
                        continue

                    # PNG로 변환
                    # 파일명 NFC 정규화 및 안전화
                    safe_stem = TextProcessor.sanitize_filename(pdf_path.stem)
                    image_filename = f"{safe_stem}_page{page_num:03d}_img{img_index+1:03d}.png"
                    image_path = self.images_dir / image_filename

                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        pix.save(str(image_path))
                    else:  # CMYK: convert to RGB first
                        pix1 = fitz.Pixmap(fitz.csRGB, pix)
                        pix1.save(str(image_path))
                        pix1 = None

                    logger.info(f"   ✅ 이미지 추출 성공: {image_filename} ({pix.width}x{pix.height})")
                    pix = None

                    # 마크다운 참조 생성
                    image_ref = f"![이미지 {len(image_references) + 1} (페이지 {page_num})](images/{image_filename})"
                    image_references.append(image_ref)
                    processed_images += 1

                except Exception as e:
                    logger.warning(f"이미지 추출 실패 (페이지 {page_num}, 이미지 {img_index + 1}): {e}")
                    processed_images += 1

        return image_references

    def _clean_and_format_text_with_images(self, text: str, page_images: list, image_references: list) -> str:
        """텍스트 정리 및 이미지 삽입 처리 (페이지 구분 없이)"""
        if not text.strip():
            return ""

        # 페이지 구분자 제거 및 이미지 위치 정보 저장
        page_image_map = {}

        for page_num, images in page_images:
            if images:
                page_image_map[page_num] = len(images)

        # 페이지 구분자 제거하면서 이미지 삽입 위치 기억
        lines = text.split("\n")
        processed_lines = []
        image_ref_index = 0

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # 페이지 구분자 감지 및 처리
            if line.startswith("<!-- 페이지 ") and line.endswith(" 시작 -->"):
                # 페이지 번호 추출
                page_match = re.search(r"페이지 (\d+)", line)
                if page_match:
                    page_num = int(page_match.group(1))

                    # 해당 페이지의 이미지가 있으면 삽입
                    if page_num in page_image_map:
                        images_in_page = page_image_map[page_num]
                        for _ in range(images_in_page):
                            if image_ref_index < len(image_references):
                                processed_lines.append(f"\n{image_references[image_ref_index]}\n")
                                image_ref_index += 1
                i += 1
                continue

            # 빈 줄 처리
            if not line:
                processed_lines.append("")
                i += 1
                continue

            # 일반 텍스트 처리
            processed_lines.append(line)
            i += 1

        # 남은 이미지 참조가 있으면 마지막에 추가
        if image_ref_index < len(image_references):
            processed_lines.append("\n\n## 추가 이미지\n")
            while image_ref_index < len(image_references):
                processed_lines.append(image_references[image_ref_index])
                image_ref_index += 1

        # 최종 텍스트 정리
        result_text = "\n".join(processed_lines)

        # 기존 텍스트 정리 로직 적용 (헤딩 감지, 문단 정리 등)
        return self._clean_and_format_text(result_text)

    def _clean_and_format_text(self, text: str) -> str:
        """개선된 텍스트 정리 및 마크다운 포맷팅"""
        if not text.strip():
            return ""

        # 1단계: 기본 정리
        # 여러 개의 개행을 하나로 줄이기
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)

        # 페이지 번호 제거 (숫자만 있는 라인)
        text = re.sub(r"^[Pp]age\s*\d+.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", text, flags=re.MULTILINE)

        # 2단계: 줄 단위로 처리
        lines = text.split("\n")
        processed_lines = []
        current_paragraph = []

        for i, line in enumerate(lines):
            line = line.strip()

            if not line:
                # 빈 줄 처리
                if current_paragraph:
                    # 현재 문단을 완성하여 추가
                    paragraph_text = join_paragraph_lines(current_paragraph)
                    if paragraph_text:
                        processed_lines.append(paragraph_text)
                    current_paragraph = []
                processed_lines.append("")
                continue

            # 3단계: 실제 헤딩인지 판단 (더 엄격한 기준)
            if is_real_heading(line, i, lines):
                # 현재 문단을 먼저 완성
                if current_paragraph:
                    paragraph_text = join_paragraph_lines(current_paragraph)
                    if paragraph_text:
                        processed_lines.append(paragraph_text)
                    current_paragraph = []

                # 헤딩 추가
                level = get_heading_level(line)
                processed_lines.append(f"{'#' * level} {line}")
            else:
                # 일반 텍스트는 문단에 추가
                current_paragraph.append(line)

        # 마지막 문단 처리
        if current_paragraph:
            paragraph_text = join_paragraph_lines(current_paragraph)
            if paragraph_text:
                processed_lines.append(paragraph_text)

        # 4단계: 최종 정리
        result = "\n".join(processed_lines)

        # 연속된 빈 줄 정리
        result = re.sub(r"\n\s*\n\s*\n+", "\n\n", result)

        return result.strip()
