#!/usr/bin/env python3
"""
개선된 PDF to Markdown Converter 모듈

주요 기능:
- 문장 연결 문제 해결
- 헤딩 인식 로직 정교화
- 한글 텍스트 완벽 지원
- 이미지 추출 및 별도 저장
"""

import fitz  # PyMuPDF
from pathlib import Path
import re
from datetime import datetime
from typing import Tuple, Optional, Callable, List, Dict
import logging
from .text_processing import TextProcessor
from .sentence_completion import is_incomplete_sentence

# 의미 기반 청킹 모듈 import
try:
    from .semantic_chunker import SemanticChunker

    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False
    logging.warning("SemanticChunker를 사용할 수 없습니다. 기본 청킹을 사용합니다.")

logger = logging.getLogger(__name__)


class ImprovedPDFConverter:
    """개선된 PDF to Markdown 변환기"""

    def __init__(
        self,
        output_dir: str = "converted_docs",
        enable_semantic_chunking: bool = True,
        enable_postprocessing: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.enable_semantic_chunking = enable_semantic_chunking and SEMANTIC_CHUNKING_AVAILABLE
        self.enable_postprocessing = enable_postprocessing
        self.processed_dir = Path("processed_docs")

        # 출력 디렉토리 생성
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        if self.enable_postprocessing:
            self.processed_dir.mkdir(exist_ok=True)

        # 의미 기반 청킹 초기화
        if self.enable_semantic_chunking:
            self.semantic_chunker = SemanticChunker(min_chunk_size=300, max_chunk_size=1500, similarity_threshold=0.3)
        else:
            self.semantic_chunker = None

    def convert_pdf_to_markdown(
        self, pdf_path: str, progress_callback: Optional[Callable] = None
    ) -> Tuple[Optional[str], int]:
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

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            # 2단계 후처리 수행
            if self.enable_postprocessing:
                if progress_callback:
                    progress_callback(0.95, "2단계 품질 개선 처리 중...")

                try:
                    from .md_postprocessor import MDPostProcessor
                    from .quality_checker import QualityChecker

                    # 현재 선택된 제공자/모델을 후처리에 전달
                    _prov = None
                    _model = None
                    try:
                        from config import settings as _settings

                        try:
                            import streamlit as st  # type: ignore

                            _prov = st.session_state.get("current_provider", None)
                            _model = st.session_state.get("current_model", None)
                        except Exception as err:
                            logger.debug(f"세션 provider/model 조회 실패(무시): {err}")
                        _prov = _prov or getattr(_settings, "llm_provider", "local")
                        if not _model:
                            if _prov == "openai":
                                _model = getattr(_settings, "openai_model", None)
                            elif _prov == "google":
                                _model = getattr(_settings, "google_model", None)
                            elif _prov == "anthropic":
                                _model = getattr(_settings, "anthropic_model", None)
                            else:
                                _model = None
                    except Exception as err:
                        logger.debug(f"세션 전처리 모델 조회 실패(무시): {err}")
                    postprocessor = MDPostProcessor(provider=_prov, model_name=_model)
                    quality_checker = QualityChecker(provider=_prov, model_name=_model)

                    # MD 파일 후처리
                    processed_md_path = self.processed_dir / md_filename

                    # process_md_file 메서드 호출 (process_file이 아님)
                    try:
                        processed_content = postprocessor.process_md_file(str(md_path))

                        # 처리된 내용을 파일로 저장
                        with open(processed_md_path, "w", encoding="utf-8") as f:
                            f.write(processed_content)

                        # 품질 검사
                        quality_result = quality_checker.analyze_quality(processed_content)
                        quality_result.get("total_score", 0)
                        success = quality_result.get("passed", False)
                        result = quality_result

                    except Exception as e:
                        success = False
                        result = str(e)

                    if success:
                        logger.info(f"2단계 품질 개선 완료: {processed_md_path}")
                        if progress_callback:
                            progress_callback(1.0, f"변환 및 품질 개선 완료: {image_count}개 이미지 추출")
                    else:
                        logger.warning(f"2단계 품질 개선 실패: {result}")
                        if progress_callback:
                            progress_callback(1.0, f"1단계 변환만 완료: {image_count}개 이미지 추출")

                except Exception as e:
                    logger.error(f"2단계 후처리 중 오류 발생: {str(e)}")
                    if progress_callback:
                        progress_callback(1.0, f"1단계 변환만 완료: {image_count}개 이미지 추출")
            else:
                if progress_callback:
                    progress_callback(1.0, f"변환 완료: {image_count}개 이미지 추출")

            logger.info(f"PDF 변환 완료: {pdf_path.name} -> {md_path}")
            return markdown_content, image_count

        except Exception as e:
            logger.error(f"PDF 변환 실패: {pdf_path} - {str(e)}")
            if progress_callback:
                progress_callback(1.0, f"변환 실패: {str(e)}")
            return None, 0

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
                    paragraph_text = self._join_paragraph_lines(current_paragraph)
                    if paragraph_text:
                        processed_lines.append(paragraph_text)
                    current_paragraph = []
                processed_lines.append("")
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
        result = "\n".join(processed_lines)

        # 연속된 빈 줄 정리
        result = re.sub(r"\n\s*\n\s*\n+", "\n\n", result)

        return result.strip()

    def create_semantic_chunks(self, markdown_content: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        마크다운 내용을 의미 기반으로 청크 분할

        Args:
            markdown_content: 분할할 마크다운 텍스트
            metadata: 추가 메타데이터

        Returns:
            List[Dict]: 의미 기반 청크 리스트
        """
        if not self.enable_semantic_chunking or not self.semantic_chunker:
            # 의미 기반 청킹을 사용할 수 없는 경우 기본 청킹
            return self._create_basic_chunks(markdown_content, metadata)

        logger.info("의미 기반 청킹 시작")
        chunks = self.semantic_chunker.create_semantic_chunks(markdown_content, metadata)

        # 청킹 통계 로깅
        if chunks:
            stats = self.semantic_chunker.get_chunk_statistics(chunks)
            logger.info(
                f"의미 기반 청킹 완료: {stats['total_chunks']}개 청크, "
                f"평균 크기: {stats['avg_chunk_size']:.0f}자, "
                f"평균 일관성: {stats['avg_coherence']:.2f}, "
                f"평균 도메인 관련성: {stats['avg_domain_relevance']:.2f}"
            )

        return chunks

    def _create_basic_chunks(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """기본 청킹 방식 (의미 기반 청킹을 사용할 수 없는 경우)"""
        if not text.strip():
            return []

        chunks = []
        max_chunk_size = 1200
        min_chunk_size = 300

        # 문단별로 분할
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        current_chunk = ""
        chunk_id = 0

        for paragraph in paragraphs:
            # 현재 청크에 문단을 추가했을 때 크기 확인
            potential_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph

            if len(potential_chunk) > max_chunk_size and current_chunk:
                # 현재 청크가 최소 크기를 만족하면 저장
                if len(current_chunk) >= min_chunk_size:
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata.update(
                        {"chunk_method": "basic", "chunk_id": chunk_id, "chunk_size": len(current_chunk)}
                    )

                    chunks.append(
                        {
                            "text": current_chunk,
                            "metadata": chunk_metadata,
                            "semantic_info": {"chunk_id": chunk_id, "keywords": []},
                        }
                    )

                    chunk_id += 1
                    current_chunk = paragraph
                else:
                    current_chunk = potential_chunk
            else:
                current_chunk = potential_chunk

        # 마지막 청크 처리
        if current_chunk:
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({"chunk_method": "basic", "chunk_id": chunk_id, "chunk_size": len(current_chunk)})

            chunks.append(
                {
                    "text": current_chunk,
                    "metadata": chunk_metadata,
                    "semantic_info": {"chunk_id": chunk_id, "keywords": []},
                }
            )

        return chunks

    def convert_pdf_to_semantic_chunks(
        self, pdf_path: str, progress_callback: Optional[Callable] = None
    ) -> Tuple[List[Dict], int]:
        """
        PDF를 의미 기반 청크로 변환

        Args:
            pdf_path: PDF 파일 경로
            progress_callback: 진행 상황 콜백 함수

        Returns:
            Tuple[청크 리스트, 이미지 개수]
        """
        # 1단계: 일반 마크다운 변환
        markdown_content, image_count = self.convert_pdf_to_markdown(pdf_path, progress_callback)

        if not markdown_content:
            return [], image_count

        # 2단계: 의미 기반 청킹 적용
        if progress_callback:
            progress_callback(0.95, "의미 기반 청킹 처리 중...")

        metadata = {
            "source_file": Path(pdf_path).name,
            "conversion_date": datetime.now().isoformat(),
            "image_count": image_count,
            "processing_method": "improved_semantic",
        }

        chunks = self.create_semantic_chunks(markdown_content, metadata)

        if progress_callback:
            progress_callback(1.0, f"의미 기반 청킹 완료: {len(chunks)}개 청크 생성")

        return chunks, image_count

    def _is_real_heading(self, line: str, line_index: int, all_lines: list) -> bool:
        """실제 헤딩인지 더 정교하게 판단"""
        # 너무 긴 텍스트는 헤딩이 아님
        if len(line) > 100:
            return False

        # 문장 부호로 끝나는 경우 일반적으로 헤딩이 아님
        if line.endswith((".", "다", "음", "었다", "였다", "한다", "된다", "이다", "않다")):
            return False

        # 명확한 헤딩 패턴들
        heading_patterns = [
            r"^제\s*\d+\s*장",  # 제1장, 제 2 장
            r"^제\s*\d+\s*절",  # 제1절, 제 2 절
            r"^\d+\.\s*[가-힣]",  # 1. 서론
            r"^[가-힣]\.\s*[가-힣]",  # 가. 개요
            r"^\([가-힣]\)",  # (가)
            r"^\d+\)\s*[가-힣]",  # 1) 목적
            r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.",  # I. II. III.
            r"^【[^】]+】",  # 【제목】
            r"^◎|^○|^●|^■|^□",  # 기호로 시작
        ]

        for pattern in heading_patterns:
            if re.match(pattern, line):
                return True

        # 전체가 대문자인 경우 (영어)
        if line.isupper() and len(line) < 50 and re.search(r"[A-Z]", line):
            return True

        # 특정 키워드로 시작하는 제목들
        heading_keywords = [
            "서론",
            "결론",
            "요약",
            "개요",
            "배경",
            "목적",
            "방법",
            "결과",
            "고찰",
            "참고문헌",
            "조사",
            "발굴",
            "유적",
            "유물",
            "분석",
            "검토",
            "연구",
            "현황",
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
                prev_line = lines[i - 1].strip()

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
        korean_endings = ["다", "음", "였다", "었다", "한다", "된다", "이다", "않다", "있다", "없다"]

        for ending in korean_endings:
            if line.endswith(ending + ".") or line.endswith(ending):
                return True

        # 영어/숫자 문장 종결
        if line.endswith((".", "!", "?", ":", ";")):
            return True

        # 닫는 괄호나 따옴표로 끝나는 경우
        if line.endswith((")", '"', "'", "』", "】")):
            return True

        return False

    def _get_heading_level(self, line: str) -> int:
        """헤딩 레벨 결정 (개선된 버전)"""
        # 장/절 구조
        if re.match(r"^제\s*\d+\s*장", line):
            return 1
        elif re.match(r"^제\s*\d+\s*절", line):
            return 2

        # 숫자 패턴으로 레벨 결정
        if re.match(r"^\d+\.", line):  # 1., 2., 3.
            return 2
        elif re.match(r"^\d+\.\d+", line):  # 1.1, 1.2
            return 3
        elif re.match(r"^\d+\.\d+\.\d+", line):  # 1.1.1
            return 4
        elif re.match(r"^[가-힣]\.", line):  # 가., 나., 다.
            return 3
        elif re.match(r"^\([가-힣]\)", line):  # (가), (나)
            return 4
        elif re.match(r"^\d+\)", line):  # 1), 2)
            return 4
        elif re.match(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.", line):  # I., II.
            return 2
        else:
            return 2
