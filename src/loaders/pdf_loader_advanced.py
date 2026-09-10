import tempfile
from typing import List
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)


class AdvancedPDFLoader:
    """OCR 기능이 포함된 고급 PDF 로더"""

    def __init__(self, use_ocr: bool = True, ocr_language: str = "kor+eng"):
        """
        Args:
            use_ocr: OCR 사용 여부
            ocr_language: OCR 언어 설정 (kor: 한국어, eng: 영어, kor+eng: 한국어+영어)
        """
        self.use_ocr = use_ocr
        self.ocr_language = ocr_language

    def load_pdf(self, file_path: str, progress_callback=None) -> List[Document]:
        """PDF 파일을 로드하고 텍스트 추출"""
        documents = []

        # 1. 먼저 일반적인 텍스트 추출 시도
        try:
            text_content = self._extract_text_pypdf(file_path)
            if text_content and len(text_content.strip()) > 50:  # 의미있는 텍스트가 있는 경우
                documents.append(
                    Document(
                        page_content=text_content,
                        metadata={
                            "source": file_path,
                            "extraction_method": "pypdf",
                            "page_count": self._get_page_count(file_path),
                        },
                    )
                )
                return documents
        except Exception as e:
            logger.warning(f"PyPDF 텍스트 추출 실패: {str(e)}")

        # 2. 텍스트 추출이 실패하거나 내용이 부족한 경우 OCR 시도
        if self.use_ocr:
            try:
                logger.info(f"OCR을 사용하여 텍스트 추출 시도: {file_path}")
                ocr_text = self._extract_text_ocr(file_path, progress_callback)
                if ocr_text:
                    documents.append(
                        Document(
                            page_content=ocr_text,
                            metadata={
                                "source": file_path,
                                "extraction_method": "ocr",
                                "ocr_language": self.ocr_language,
                                "page_count": self._get_page_count(file_path),
                            },
                        )
                    )
            except Exception as e:
                logger.error(f"OCR 추출 실패: {str(e)}")
                raise ValueError(f"PDF에서 텍스트를 추출할 수 없습니다: {file_path}")

        if not documents:
            raise ValueError(f"PDF에서 텍스트를 추출할 수 없습니다: {file_path}")

        return documents

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
        """OCR을 사용한 텍스트 추출"""
        text = ""

        # PDF를 이미지로 변환
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # DPI를 높이면 품질은 좋아지지만 처리 시간이 오래 걸림
                # thread_count 제거하여 pickle 오류 방지
                images = convert_from_path(file_path, dpi=200, output_folder=temp_dir, fmt="png")

                # 각 페이지에서 OCR 수행
                for i, image in enumerate(images):
                    logger.info(f"페이지 {i+1}/{len(images)} OCR 처리 중...")

                    if progress_callback:
                        ocr_progress = 0.3 + (0.4 * (i / len(images)))
                        progress_callback(ocr_progress, f"OCR 처리 중... (페이지 {i+1}/{len(images)})")

                    # 이미지 전처리 (선택사항)
                    image = self._preprocess_image(image)

                    # OCR 수행
                    page_text = pytesseract.image_to_string(
                        image,
                        lang=self.ocr_language,
                        config="--psm 3",  # 페이지 분할 모드: 자동
                    )

                    if page_text.strip():
                        text += f"\n--- 페이지 {i + 1} ---\n"
                        text += page_text

            except Exception as e:
                logger.error(f"PDF to Image 변환 실패: {str(e)}")
                raise

        return text.strip()

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """OCR 정확도 향상을 위한 이미지 전처리"""
        # 그레이스케일 변환
        if image.mode != "L":
            image = image.convert("L")

        # 추가적인 이미지 처리가 필요한 경우 여기에 구현
        # 예: 대비 향상, 노이즈 제거 등

        return image

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
        """OCR 사용 가능 여부 확인"""
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
