"""pdf_loading 모듈 헬퍼 함수 단위 테스트.

_large_ 메서드에서 분리된 순수 변환·해석 로직의 동작 고정 회귀망이다.
"""

from pathlib import Path

import pytest

from config import settings
from src.loaders.pdf_loading import (
    build_extraction_metadata,
    convert_agent_images_to_metadata,
    copy_images_to_permanent,
    resolve_provider_model,
    scan_images_dir,
)


def test_명시적모델이면_그대로통과(monkeypatch):
    provider, model = resolve_provider_model("OpenAI", "gpt-4o")
    assert (provider, model) == ("openai", "gpt-4o")


@pytest.mark.parametrize(
    "provider_attr,model_attr,expected",
    [
        ("openai", "openai_model", "o-model"),
        ("google", "google_model", "g-model"),
        ("anthropic", "anthropic_model", "a-model"),
    ],
)
def test_제공자별기본모델(monkeypatch, provider_attr, model_attr, expected):
    monkeypatch.setattr(settings, model_attr, expected, raising=False)
    provider, model = resolve_provider_model(provider_attr, None)
    assert (provider, model) == (provider_attr, expected)


def test_openrouter는_텍스트모델우선_없으면멀티모달(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_model", None, raising=False)
    monkeypatch.setattr(settings, "openrouter_mm_model", "mm-model", raising=False)
    assert resolve_provider_model("openrouter", None) == ("openrouter", "mm-model")

    monkeypatch.setattr(settings, "openrouter_model", "text-model", raising=False)
    assert resolve_provider_model("openrouter", None) == ("openrouter", "text-model")


def test_미지정provider는전역기본_알수없는제공자는모델없음(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openrouter", raising=False)
    provider, _ = resolve_provider_model(None, None)
    assert provider == "openrouter"

    provider, model = resolve_provider_model("unknown-ai", None)
    assert (provider, model) == ("unknown-ai", None)


def test_이미지디렉터리스캔_페이지파싱(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "doc_page3_img1.png").write_bytes(b"x")
    (images / "doc_page10_img2.png").write_bytes(b"yy")
    (images / "unrelated.txt").write_text("x")

    found = scan_images_dir(tmp_path)

    assert len(found) == 2
    by_name = {info["filename"]: info for info in found}
    assert by_name["doc_page3_img1.png"]["page"] == 3
    assert by_name["doc_page10_img2.png"]["description"] == "페이지 10 이미지"
    assert by_name["doc_page3_img1.png"]["size"] == 1


def test_이미지디렉터리없으면빈목록(tmp_path):
    assert scan_images_dir(tmp_path) == []


def test_이미지를영구위치로복사_경로갱신(tmp_path):
    src = tmp_path / "src" / "a_page1_img1.png"
    src.parent.mkdir()
    src.write_bytes(b"data")
    info = {"filename": src.name, "path": str(src), "relative_path": "old"}

    copy_images_to_permanent([info], dest_dir=tmp_path / "dest")

    assert (tmp_path / "dest" / src.name).read_bytes() == b"data"
    assert info["path"] == str(tmp_path / "dest" / src.name)
    assert info["relative_path"] == f"static/images/pdf/{src.name}"


def test_지능형추출결과를메타데이터로변환():
    results = {
        "document_topic": {"main_topic": "성곽 조사"},
        "statistics": {"total_images_found": 5, "relevant_images_saved": 2, "text_images_converted": 1},
        "images": [
            {
                "saved": True,
                "image_file": "/out/a.png",
                "page": 2,
                "type": "photo",
                "relevance_score": 0.85,
                "description": "성벽",
            },
            {"saved": False, "image_file": "/out/b.png", "page": 3},
            {"saved": True, "image_file": "/out/c.png", "page": 4, "extracted_text": "OCR 결과"},
        ],
    }

    meta = build_extraction_metadata(results, "/data/report.pdf", Path("/out"))

    assert meta["intelligent_extraction_completed"] is True
    assert meta["total_images"] == 5 and meta["relevant_images"] == 2
    assert [img["filename"] for img in meta["extracted_images"]] == ["a.png", "c.png"]
    md = meta["image_markdown_content"]
    assert "# report.pdf" in md and "**문서 주제**: 성곽 조사" in md
    assert "관련도: 0.85" in md and "OCR 결과" in md


def test_에이전트이미지형식을메타데이터로변환(tmp_path):
    images_info = {1: [("images/rel.png", "상대"), (str(tmp_path / "abs.png"), "절대")]}

    meta = convert_agent_images_to_metadata(images_info, str(tmp_path))

    assert meta["total_images"] == 2 and meta["relevant_images"] == 2
    first = meta["extracted_images"][0]
    assert Path(first["path"]) == tmp_path / "images" / "rel.png"
    assert first["page"] == 1 and first["source"] == "agent_converter"
