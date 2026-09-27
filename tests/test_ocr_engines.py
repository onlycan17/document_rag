"""OCR 엔진 선택·폴백과 Upstage 응답 → 페이지 매핑 테스트 (외부 API 호출 없음)"""

from pathlib import Path

import fitz

import src.loaders.ocr_engines as ocr_engines
import src.loaders.pdf_loader_advanced as loader_module
from config import settings
from src.loaders.pdf_loader_advanced import AdvancedPDFLoader


def _make_blank_pdf(path: Path, pages: int) -> str:
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    doc.save(path)
    return str(path)


def _element(page: int, category: str, markdown: str) -> dict:
    return {"page": page, "category": category, "content": {"markdown": markdown}}


def test_upstage_maps_batched_pages_and_drops_headers_footers(tmp_path, monkeypatch):
    """요청을 나눠 보내도 응답의 상대 페이지 번호가 원본 페이지로 맞춰지고, 머리말·쪽번호는 빠진다"""
    responses = [
        {
            "elements": [
                _element(1, "header", "보고서 제목"),
                _element(1, "paragraph", "1쪽 본문"),
                _element(2, "table", "| 표 |"),
                _element(2, "footer", "- 2 -"),
            ],
            "usage": {"pages": 2},
        },
        {"elements": [_element(1, "paragraph", "3쪽 본문")], "usage": {"pages": 1}},
    ]
    sent_page_counts = []

    def fake_post(pdf_bytes):
        with fitz.open(stream=pdf_bytes, filetype="pdf") as part:
            sent_page_counts.append(part.page_count)
        return responses.pop(0)

    monkeypatch.setattr(ocr_engines, "UPSTAGE_PAGES_PER_REQUEST", 2)
    monkeypatch.setattr(ocr_engines, "_post_upstage", fake_post)

    pages = ocr_engines.ocr_upstage(_make_blank_pdf(tmp_path / "scan.pdf", 3))

    assert sent_page_counts == [2, 1]
    assert pages == ["1쪽 본문", "| 표 |", "3쪽 본문"]


def test_upstage_is_skipped_without_api_key(monkeypatch):
    monkeypatch.setattr(settings, "upstage_api_key", None)
    assert "upstage" not in ocr_engines.available_engines("upstage")


def test_preferred_engine_comes_first_then_local_engines(monkeypatch):
    monkeypatch.setitem(ocr_engines._AVAILABILITY, "upstage", lambda: True)
    monkeypatch.setitem(ocr_engines._AVAILABILITY, "vision", lambda: True)
    monkeypatch.setitem(ocr_engines._AVAILABILITY, "tesseract", lambda: True)

    assert ocr_engines.available_engines("upstage") == ["upstage", "vision", "tesseract"]
    assert ocr_engines.available_engines("tesseract") == ["tesseract", "vision"]


def test_loader_falls_back_to_next_engine_when_preferred_fails(tmp_path, monkeypatch):
    def failing_upstage(path, progress_callback=None):
        raise RuntimeError("503 Service Unavailable")

    monkeypatch.setattr(loader_module, "available_engines", lambda preferred: ["upstage", "vision"])
    monkeypatch.setitem(ocr_engines.OCR_ENGINES, "upstage", failing_upstage)
    monkeypatch.setitem(ocr_engines.OCR_ENGINES, "vision", lambda path, progress_callback=None: ["", "2쪽 본문"])

    documents = AdvancedPDFLoader().load_pdf(_make_blank_pdf(tmp_path / "scan.pdf", 2))

    assert documents[0].metadata["ocr_engine"] == "vision"
    assert documents[0].page_content == "--- 페이지 2 ---\n2쪽 본문"
