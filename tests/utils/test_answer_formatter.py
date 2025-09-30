from src.utils.answer_formatter import AnswerFormatter


def test_format_adds_sections_and_bullets():
    formatter = AnswerFormatter()
    raw_text = (
        "몽촌토성은 백제 시기의 토성으로 생활과 방어를 함께 책임졌습니다.\n\n"
        "1. 토기는 당시 사람들의 생활상을 보여주는 중요한 자료입니다.\n"
        "2. 성곽은 외침을 막는 방어 기지 역할을 했습니다."
    )
    sources = [{"file_name": "history.md", "page": "12", "relevance_score": 93.5}]

    formatted = formatter.format(raw_text, sources)

    assert formatted.startswith("### 핵심 요약")
    assert "### 상세 설명" in formatted
    assert "### 참고 자료" in formatted
    assert "- 토기는 당시" in formatted
    assert "history.md" in formatted


def test_format_respects_existing_headings():
    formatter = AnswerFormatter()
    raw_text = (
        "### 핵심 요약\n- 핵심만 짚어 드립니다.\n\n"
        "### 상세 설명\n- 세부 내용도 이미 정리되어 있습니다.\n\n"
        "### 참고 자료\n- 기존 출처 안내"
    )

    formatted = formatter.format(raw_text, [])

    assert formatted.count("### 핵심 요약") == 1
    assert formatted.count("### 상세 설명") == 1
    assert formatted.count("### 참고 자료") == 1
    assert "기존 출처 안내" in formatted


def test_format_fills_reference_placeholder_when_missing():
    formatter = AnswerFormatter()
    raw_text = (
        "몽촌토성은 백제 왕성으로 추정되는 유적입니다.\n\n"
        "발굴 과정에서 확인된 건물터는 궁궐 공간을 보여줍니다."
    )

    formatted = formatter.format(raw_text, None)

    assert "### 참고 자료" in formatted
    assert "관련 문서 정보를 자동으로 수집하지 못했습니다" in formatted
