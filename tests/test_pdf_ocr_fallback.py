"""스캔(이미지) PDF 판별과 OCR 폴백 테스트 — 외부 API 없이 로컬 OCR 엔진만 사용"""

from pathlib import Path

import fitz
import pytest

from src.loaders.pdf_loader_advanced import AdvancedPDFLoader, has_text_layer

SAMPLE_TEXT = "몽촌토성 북문지 발굴조사 보고서\n백제 한성기 토기와 기와가 출토되었다."


def _make_text_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), SAMPLE_TEXT * 3, fontname="korea", fontsize=14)
    doc.save(path)
    return path


def _make_scanned_pdf(path: Path) -> Path:
    """텍스트 PDF를 이미지로 렌더링해 텍스트 레이어 없는 PDF로 만든다"""
    source = fitz.open(_make_text_pdf(path.with_name("source.pdf")))
    pixmap = source[0].get_pixmap(dpi=200)
    doc = fitz.open()
    page = doc.new_page(width=source[0].rect.width, height=source[0].rect.height)
    page.insert_image(page.rect, stream=pixmap.tobytes("png"))
    doc.save(path)
    return path


def test_has_text_layer_distinguishes_text_and_scanned_pdf(tmp_path):
    assert has_text_layer(str(_make_text_pdf(tmp_path / "text.pdf")))
    assert not has_text_layer(str(_make_scanned_pdf(tmp_path / "scanned.pdf")))


@pytest.mark.skipif(not AdvancedPDFLoader.check_ocr_availability(), reason="OCR 엔진(Vision/Tesseract) 없음")
def test_scanned_pdf_is_extracted_with_ocr(tmp_path):
    documents = AdvancedPDFLoader().load_pdf(str(_make_scanned_pdf(tmp_path / "scanned.pdf")))

    assert documents[0].metadata["extraction_method"] == "ocr"
    assert "몽촌토성" in documents[0].page_content
    assert "출토" in documents[0].page_content


def _routing_loader(calls: list):
    from src.loaders.document_loader import EnhancedDocumentLoader

    loader = object.__new__(EnhancedDocumentLoader)
    loader.use_intelligent_image_extraction = True
    loader.use_agent_preprocessing = True
    loader._cleanup_existing_images_for_pdf = lambda path: None
    loader._extract_intelligent_images = lambda path, cb: calls.append("image_analysis")
    loader._load_with_agent_converter = lambda path, cb: calls.append("agent")
    loader._load_with_improved_converter = lambda path, cb: calls.append("improved") or ["improved"]
    loader._load_with_ocr = lambda path, cb: calls.append("ocr") or ["ocr"]
    return loader


def test_scanned_pdf_skips_image_analysis_and_converters_and_goes_to_ocr(tmp_path):
    calls: list = []
    result = _routing_loader(calls)._load_pdf_file(str(_make_scanned_pdf(tmp_path / "scanned.pdf")))

    assert calls == ["ocr"] and result == ["ocr"]


def test_text_pdf_uses_improved_converter(tmp_path):
    calls: list = []
    result = _routing_loader(calls)._load_pdf_file(str(_make_text_pdf(tmp_path / "text.pdf")))

    assert calls == ["image_analysis", "agent", "improved"] and result == ["improved"]
