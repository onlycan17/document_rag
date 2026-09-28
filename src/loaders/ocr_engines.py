"""스캔 PDF OCR 엔진 — 모든 엔진은 PDF 경로를 받아 페이지별 텍스트 목록을 반환한다

- upstage: Upstage Document Parse (표 구조 보존, 머리말·쪽번호 제거, 유료 $0.01/쪽)
- vision: macOS Vision (로컬·무료, 표는 줄 단위로 풀림)
- tesseract: 그 외 OS 폴백 (기울임꼴 한글에 취약)
"""

import logging
import subprocess
from typing import Callable, Dict, List, Optional

import fitz  # PyMuPDF
import pytesseract
import requests
from PIL import Image

from config import settings
from src.embeddings.embedding_model import api_retry_with_backoff

try:  # macOS Vision OCR (pyobjc). 다른 OS에서는 사용 불가
    import objc
    import Vision
    from Foundation import NSData
except ImportError:
    Vision = None

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[float, str], None]]
OcrEngine = Callable[[str, ProgressCallback], List[str]]

OCR_DPI = 300
VISION_LANGUAGES = ["ko-KR", "en-US"]
TESSERACT_LANGUAGE = "kor+eng"
UPSTAGE_URL = "https://api.upstage.ai/v1/document-digitization"
UPSTAGE_PAGES_PER_REQUEST = 50  # 동기 API 한도 100쪽, 요청당 처리 시간을 고려해 절반으로 나눔
UPSTAGE_SKIP_CATEGORIES = {"header", "footer"}  # 머리말·쪽번호는 페이지 경계에서 문맥을 끊으므로 제외


def _report(progress_callback: ProgressCallback, done: int, total: int, engine: str) -> None:
    logger.info(f"{engine} OCR {done}/{total}쪽")
    if progress_callback:
        progress_callback(0.3 + 0.4 * done / total, f"OCR 처리 중... ({engine}, {done}/{total}쪽)")


@api_retry_with_backoff()
def _post_upstage(pdf_bytes: bytes) -> dict:
    """Document Parse 동기 요청 (429·네트워크 오류는 지수 백오프로 재시도)"""
    response = requests.post(
        UPSTAGE_URL,
        headers={"Authorization": f"Bearer {settings.upstage_api_key}"},
        files={"document": ("pages.pdf", pdf_bytes, "application/pdf")},
        data={
            "model": "document-parse",
            "ocr": "force",
            "mode": "standard",
            "output_formats": '["markdown"]',
            "coordinates": "false",
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def ocr_upstage(file_path: str, progress_callback: ProgressCallback = None) -> List[str]:
    """Upstage Document Parse로 페이지별 마크다운을 얻는다 (요청 단위로 PDF를 나눠 전송)"""
    with fitz.open(file_path) as doc:
        page_count = doc.page_count
        pages: List[List[str]] = [[] for _ in range(page_count)]
        for start in range(0, page_count, UPSTAGE_PAGES_PER_REQUEST):
            end = min(start + UPSTAGE_PAGES_PER_REQUEST, page_count)
            with fitz.open() as part:
                part.insert_pdf(doc, from_page=start, to_page=end - 1)
                result = _post_upstage(part.tobytes())
            # 응답의 page는 전송한 부분 PDF 기준 1부터 시작
            for element in result.get("elements", []):
                if element.get("category") not in UPSTAGE_SKIP_CATEGORIES:
                    pages[start + element["page"] - 1].append(element["content"]["markdown"])
            logger.info(f"Upstage 과금 페이지: {result.get('usage', {}).get('pages')}")
            _report(progress_callback, end, page_count, "upstage")
    return ["\n\n".join(parts) for parts in pages]


def _ocr_each_page(
    file_path: str, recognize: Callable[[fitz.Pixmap], str], engine: str, progress_callback
) -> List[str]:
    """페이지를 하나씩 그레이스케일로 렌더링해 인식 (전체 페이지를 메모리에 올리지 않음)"""
    texts = []
    with fitz.open(file_path) as doc:
        for index, page in enumerate(doc):
            texts.append(recognize(page.get_pixmap(dpi=OCR_DPI, colorspace=fitz.csGRAY)))
            _report(progress_callback, index + 1, doc.page_count, engine)
    return texts


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


def ocr_vision(file_path: str, progress_callback: ProgressCallback = None) -> List[str]:
    return _ocr_each_page(
        file_path, lambda pixmap: recognize_text_vision(pixmap.tobytes("png")), "vision", progress_callback
    )


def _recognize_text_tesseract(pixmap: fitz.Pixmap) -> str:
    image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    return pytesseract.image_to_string(image, lang=TESSERACT_LANGUAGE, config="--psm 3")


def ocr_tesseract(file_path: str, progress_callback: ProgressCallback = None) -> List[str]:
    return _ocr_each_page(file_path, _recognize_text_tesseract, "tesseract", progress_callback)


def _tesseract_available() -> bool:
    try:
        result = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True)
    except OSError:  # tesseract 미설치
        return False
    return result.returncode == 0 and "kor" in result.stdout


OCR_ENGINES: Dict[str, OcrEngine] = {"upstage": ocr_upstage, "vision": ocr_vision, "tesseract": ocr_tesseract}
_AVAILABILITY: Dict[str, Callable[[], bool]] = {
    "upstage": lambda: bool(settings.upstage_api_key),
    "vision": lambda: Vision is not None,
    "tesseract": _tesseract_available,
}


def available_engines(preferred: str) -> List[str]:
    """선호 엔진을 먼저, 이어서 무료 로컬 엔진 순으로 사용 가능한 엔진 목록"""
    order = [preferred] + [name for name in ("vision", "tesseract") if name != preferred]
    return [name for name in order if name in OCR_ENGINES and _AVAILABILITY[name]()]
