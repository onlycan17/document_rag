import logging
from typing import List

import fitz  # PyMuPDF
import PyPDF2
import pytesseract
from langchain.schema import Document
from PIL import Image

try:  # macOS Vision OCR (pyobjc). 다른 OS에서는 Tesseract만 사용
    import objc
    import Vision
    from Foundation import NSData
except ImportError:
    Vision = None

logger = logging.getLogger(__name__)

# 페이지당 이 글자 수 미만이면 텍스트 레이어가 없는 스캔·이미지 PDF로 본다
MIN_TEXT_CHARS_PER_PAGE = 50
OCR_DPI = 300
VISION_LANGUAGES = ["ko-KR", "en-US"]


def has_text_layer(file_path: str) -> bool:
    """PDF에 추출 가능한 텍스트 레이어가 충분한지 확인 (스캔·이미지 PDF 판별)"""
    # ponytail: 문서 전체 기준 판정 — 일부 페이지만 스캔된 PDF는 페이지별 OCR이 필요하면 확장
    with fitz.open(file_path) as doc:
        text_chars = sum(len(page.get_text().strip()) for page in doc)
        return text_chars >= MIN_TEXT_CHARS_PER_PAGE * max(doc.page_count, 1)


def vision_ocr_available() -> bool:
    return Vision is not None


def recognize_text_vision(png_bytes: bytes) -> str:
    """macOS Vision으로 이미지의 텍스트를 인식해 줄 단위로 반환"""
    with objc.autorelease_pool():
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setRecognitionLanguages_(VISION_LANGUAGES)
        request.setUsesLanguageCorrection_(True)
        image_data = NSData.dataWithBytes_length_(png_bytes, len(png_bytes))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(image_data, None)
        success, error = handler.performRequests_error_([request], None)
        if not success:
            raise RuntimeError(f"Vision OCR 실패: {error}")
        return "\n".join(obs.topCandidates_(1)[0].string() for obs in (request.results() or []))


class AdvancedPDFLoader:
    """OCR 기능이 포함된 고급 PDF 로더 (macOS Vision 우선, 없으면 Tesseract)"""

    def __init__(self, use_ocr: bool = True, ocr_language: str = "kor+eng"):
        """
        Args:
            use_ocr: OCR 사용 여부
            ocr_language: Tesseract OCR 언어 설정 (kor: 한국어, eng: 영어, kor+eng: 한국어+영어)
        """
        self.use_ocr = use_ocr
        self.ocr_language = ocr_language
        self.ocr_engine = "vision" if vision_ocr_available() else "tesseract"

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

        logger.info(f"텍스트 레이어 부족 — OCR({self.ocr_engine})로 텍스트 추출: {file_path}")
        ocr_text = self._extract_text_ocr(file_path, progress_callback)
        if not ocr_text:
            raise ValueError(f"PDF에서 텍스트를 추출할 수 없습니다: {file_path}")

        metadata = {
            "source": file_path,
            "extraction_method": "ocr",
            "ocr_engine": self.ocr_engine,
            "page_count": page_count,
        }
        return [Document(page_content=ocr_text, metadata=metadata)]

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

    def _extract_text_ocr(self, file_path: str, progress_callback=None) -> str:
        """페이지를 하나씩 렌더링해 OCR (전체 페이지를 메모리에 올리지 않음)"""
        parts = []
        with fitz.open(file_path) as doc:
            for index, page in enumerate(doc):
                logger.info(f"페이지 {index + 1}/{doc.page_count} OCR 처리 중...")
                if progress_callback:
                    progress = 0.3 + (0.4 * (index / doc.page_count))
                    progress_callback(progress, f"OCR 처리 중... (페이지 {index + 1}/{doc.page_count})")

                page_text = self._ocr_page(page)
                if page_text.strip():
                    parts.append(f"\n--- 페이지 {index + 1} ---\n{page_text}")

        return "".join(parts).strip()

    def _ocr_page(self, page: fitz.Page) -> str:
        """단일 페이지를 그레이스케일로 렌더링해 OCR"""
        pixmap = page.get_pixmap(dpi=OCR_DPI, colorspace=fitz.csGRAY)
        if self.ocr_engine == "vision":
            return recognize_text_vision(pixmap.tobytes("png"))

        image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
        return pytesseract.image_to_string(image, lang=self.ocr_language, config="--psm 3")

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
        """OCR 사용 가능 여부 확인 (macOS Vision 또는 한국어 언어팩이 있는 Tesseract)"""
        if vision_ocr_available():
            return True
        try:
            # Tesseract 설치 확인
            import subprocess

            result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                # 한국어 언어팩 확인
                result = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True)
                if "kor" in result.stdout:
                    return True
                else:
                    logger.warning("Tesseract 한국어 언어팩이 설치되지 않았습니다.")
                    return False
            return False
        except OSError:
            # tesseract 미설치 시 FileNotFoundError 등
            return False
