import logging
from typing import List, Tuple

import fitz  # PyMuPDF
import PyPDF2
from langchain.schema import Document

from config import settings

from .ocr_engines import OCR_ENGINES, available_engines

logger = logging.getLogger(__name__)

# 페이지당 이 글자 수 미만이면 텍스트 레이어가 없는 스캔·이미지 PDF로 본다
MIN_TEXT_CHARS_PER_PAGE = 50


def has_text_layer(file_path: str) -> bool:
    """PDF에 추출 가능한 텍스트 레이어가 충분한지 확인 (스캔·이미지 PDF 판별)"""
    # ponytail: 문서 전체 기준 판정 — 일부 페이지만 스캔된 PDF는 페이지별 OCR이 필요하면 확장
    with fitz.open(file_path) as doc:
        text_chars = sum(len(page.get_text().strip()) for page in doc)
        return text_chars >= MIN_TEXT_CHARS_PER_PAGE * max(doc.page_count, 1)


class AdvancedPDFLoader:
    """OCR 기능이 포함된 고급 PDF 로더 (설정한 OCR 엔진 우선, 실패 시 로컬 엔진으로 폴백)"""

    def __init__(self, use_ocr: bool = True):
        """
        Args:
            use_ocr: OCR 사용 여부 (엔진 우선순위는 settings.ocr_engine)
        """
        self.use_ocr = use_ocr

    def load_pdf(self, file_path: str, progress_callback=None) -> List[Document]:
        """PDF 파일을 로드하고 텍스트 추출 (텍스트 레이어가 부족하면 OCR)"""
        page_count = self._get_page_count(file_path)

        # 1. 텍스트 레이어가 충분하면 일반 텍스트 추출
        if has_text_layer(file_path):
            text_content = self._extract_text_pypdf(file_path)
            metadata = {"source": file_path, "extraction_method": "pypdf", "page_count": page_count}
            return [Document(page_content=text_content, metadata=metadata)]

        # 2. 스캔·이미지 PDF는 OCR
        if not self.use_ocr:
            raise ValueError(f"텍스트 레이어가 없는 PDF이며 OCR이 비활성화되어 있습니다: {file_path}")

        engine, page_texts = self._ocr_pages(file_path, progress_callback)
        ocr_text = "".join(f"\n--- 페이지 {n} ---\n{text}" for n, text in enumerate(page_texts, 1) if text.strip())
        if not ocr_text.strip():
            raise ValueError(f"PDF에서 텍스트를 추출할 수 없습니다: {file_path}")

        metadata = {"source": file_path, "extraction_method": "ocr", "ocr_engine": engine, "page_count": page_count}
        return [Document(page_content=ocr_text.strip(), metadata=metadata)]

    def _ocr_pages(self, file_path: str, progress_callback=None) -> Tuple[str, List[str]]:
        """사용 가능한 엔진을 우선순위대로 시도해 (엔진명, 페이지별 텍스트)를 반환"""
        engines = available_engines(settings.ocr_engine)
        for engine in engines:
            try:
                logger.info(f"텍스트 레이어 부족 — OCR({engine})로 텍스트 추출: {file_path}")
                return engine, OCR_ENGINES[engine](file_path, progress_callback)
            except Exception as e:
                logger.warning(f"{engine} OCR 실패, 다음 엔진으로 폴백: {type(e).__name__}: {e}")
        raise ValueError(f"사용 가능한 OCR 엔진이 모두 실패했습니다 (시도: {engines}): {file_path}")

    def _extract_text_pypdf(self, file_path: str) -> str:
        """PyPDF2를 사용한 텍스트 추출"""
        text = ""
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- 페이지 {page_num + 1} ---\n"
                    text += page_text

        return text.strip()

    def _get_page_count(self, file_path: str) -> int:
        """PDF 페이지 수 반환"""
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return len(pdf_reader.pages)
        except (OSError, PyPDF2.errors.PdfReadError) as e:
            logger.debug(f"페이지 수 확인 실패({type(e).__name__}), 0으로 처리: {file_path}")
            return 0

    @staticmethod
    def check_ocr_availability() -> bool:
        """사용 가능한 OCR 엔진(Upstage 키·macOS Vision·한국어 Tesseract)이 하나라도 있는지 확인"""
        return bool(available_engines(settings.ocr_engine))
