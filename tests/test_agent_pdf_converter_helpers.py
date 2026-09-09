"""agent_pdf_converter 블록 조립 순수 함수 단위 테스트."""

from src.utils.agent_pdf_converter import build_text_blocks


def test_이미지없으면페이지헤더와본문만():
    blocks = build_text_blocks([(1, "본문"), (2, "두 번째")], {})

    assert blocks == ["[페이지 1]\n본문", "[페이지 2]\n두 번째"]


def test_설명있는이미지포함():
    images = {1: [("./images/a.png", "성벽 전경")]}

    block = build_text_blocks([(1, "본문")], images)[0]

    assert "### 페이지 내 이미지" in block
    assert "![이미지](./images/a.png)" in block and "**이미지 설명**: 성벽 전경" in block


def test_설명없는이미지는폴백문구():
    images = {3: [("./images/b.png", None)]}

    block = build_text_blocks([(3, "본문")], images)[0]

    assert "**이미지**: 페이지 3의 이미지 ./images/b.png" in block
    assert "이미지 설명" not in block


def test_빈텍스트목록이면빈결과():
    assert build_text_blocks([], {}) == []
