#!/usr/bin/env python3
"""
개선된 PDF to Markdown Converter 모듈

주요 기능:
- 문장 연결 문제 해결
- 헤딩 인식 로직 정교화
- 한글 텍스트 완벽 지원
- 이미지 추출 및 별도 저장

책임 분할:
- 텍스트/이미지 추출·정제: pdf_text_extraction.PdfTextExtractionMixin
- 마크다운 청킹: pdf_semantic_chunking.SemanticChunkingMixin
- 헤딩/문장 판정 순수 헬퍼: pdf_heading_utils
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Tuple, Optional, Callable
import logging

from .pdf_text_extraction import PdfTextExtractionMixin
from .pdf_semantic_chunking import SemanticChunkingMixin

# 의미 기반 청킹 모듈 import
try:
    from .semantic_chunker import SemanticChunker

    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False
    logging.warning("SemanticChunker를 사용할 수 없습니다. 기본 청킹을 사용합니다.")

logger = logging.getLogger(__name__)


class ImprovedPDFConverter(PdfTextExtractionMixin, SemanticChunkingMixin):
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

            md_path = self._save_markdown(pdf_path, markdown_content)
            self._postprocess_if_enabled(md_path, progress_callback, image_count)

            logger.info(f"PDF 변환 완료: {pdf_path.name} -> {md_path}")
            return markdown_content, image_count

        except Exception as e:
            logger.error(f"PDF 변환 실패: {pdf_path} - {str(e)}")
            if progress_callback:
                progress_callback(1.0, f"변환 실패: {str(e)}")
            return None, 0

    def _save_markdown(self, pdf_path: Path, markdown_content: str) -> Path:
        """변환된 마크다운을 출력 디렉토리에 저장"""
        md_path = self.output_dir / f"{pdf_path.stem}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        return md_path

    def _postprocess_if_enabled(self, md_path: Path, progress_callback: Optional[Callable], image_count: int) -> None:
        """2단계 품질 개선(후처리) 수행. 실패해도 1단계 변환 결과는 유지"""
        if not self.enable_postprocessing:
            if progress_callback:
                progress_callback(1.0, f"변환 완료: {image_count}개 이미지 추출")
            return

        if progress_callback:
            progress_callback(0.95, "2단계 품질 개선 처리 중...")

        try:
            self._run_postprocessing(md_path, progress_callback, image_count)
        except Exception as e:
            logger.error(f"2단계 후처리 중 오류 발생: {str(e)}")
            if progress_callback:
                progress_callback(1.0, f"1단계 변환만 완료: {image_count}개 이미지 추출")

    def _run_postprocessing(self, md_path: Path, progress_callback: Optional[Callable], image_count: int) -> None:
        """2단계 품질 개선(후처리) 실행 및 결과 콜백 보고"""
        from .md_postprocessor import MDPostProcessor
        from .quality_checker import QualityChecker
        from config import settings as _settings

        provider, model_name = self._resolve_postprocess_models(_settings)

        postprocessor = MDPostProcessor(provider=provider, model_name=model_name)
        quality_checker = QualityChecker(provider=provider, model_name=model_name)
        processed_md_path = self.processed_dir / md_path.name

        success, result = self._postprocess_and_check_quality(
            postprocessor, quality_checker, md_path, processed_md_path
        )

        if success:
            logger.info(f"2단계 품질 개선 완료: {processed_md_path}")
            if progress_callback:
                progress_callback(1.0, f"변환 및 품질 개선 완료: {image_count}개 이미지 추출")
        else:
            logger.warning(f"2단계 품질 개선 실패: {result}")
            if progress_callback:
                progress_callback(1.0, f"1단계 변환만 완료: {image_count}개 이미지 추출")

    @staticmethod
    def _postprocess_and_check_quality(postprocessor, quality_checker, md_path: Path, processed_md_path: Path):
        """MD 후처리 → 저장 → 품질 검사. (성공 여부, 결과) 반환"""
        try:
            processed_content = postprocessor.process_md_file(str(md_path))

            # 처리된 내용을 파일로 저장
            with open(processed_md_path, "w", encoding="utf-8") as f:
                f.write(processed_content)

            # 품질 검사
            quality_result = quality_checker.analyze_quality(str(processed_md_path))
            quality_result.get("total_score", 0)
            return quality_result.get("passed", False), quality_result

        except Exception as e:
            return False, str(e)

    @staticmethod
    def _resolve_postprocess_models(settings) -> Tuple[Optional[str], Optional[str]]:
        """현재 선택된 제공자/모델을 후처리에 전달하기 위한 값 조회"""
        provider = None
        model_name = None

        try:
            import streamlit as st  # type: ignore

            provider = st.session_state.get("current_provider", None)
            model_name = st.session_state.get("current_model", None)
        except Exception as err:
            logger.debug(f"세션 provider/model 조회 실패(무시): {err}")

        provider = provider or getattr(settings, "llm_provider", "local")
        if not model_name:
            if provider == "openai":
                model_name = getattr(settings, "openai_model", None)
            elif provider == "google":
                model_name = getattr(settings, "google_model", None)
            elif provider == "anthropic":
                model_name = getattr(settings, "anthropic_model", None)
            else:
                model_name = None

        return provider, model_name
