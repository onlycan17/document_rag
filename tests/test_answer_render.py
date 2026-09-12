from src.utils.answer_render import (
    format_inline_markup,
    normalize_answer_markdown,
    split_answer_sections,
)


def test_normalizes_glued_heading_and_bullet():
    """실제 화면에서 깨졌던 입력: 헤딩/불릿이 앞 텍스트에 붙은 경우."""
    broken = "관련 내용입니다. [출처 1, 2, 8, 9]### 참고 자료- 문서 1, 3, 5: 몽촌토성"

    normalized = normalize_answer_markdown(broken)

    assert "9]### 참고 자료" not in normalized
    assert "### 참고 자료\n- 문서 1" in normalized


def test_glued_heading_is_recognized_as_section():
    """붙어 있던 헤딩도 섹션으로 분해되어야 한다."""
    broken = "핵심입니다.### 핵심 요약\n요약 본문입니다."

    sections = split_answer_sections(broken)

    assert sections[0].kind == "summary"
    assert "### 핵심 요약" not in sections[0].body
    assert "요약 본문입니다." in sections[0].body


def test_inline_bullets_are_split():
    text = "체계적이지 못했습니다. [출처 3]   - 2013년부터 조사했습니다."

    normalized = normalize_answer_markdown(text)

    assert "[출처 3]\n- 2013년부터" in normalized


def test_inline_star_bullets_keep_bold_text():
    normalized = normalize_answer_markdown("**위치**입니다.    *   **역사:** 백제 유적")
    assert normalized == "**위치**입니다.\n- **역사:** 백제 유적"


def test_glued_table_rows_are_restored():
    text = "정리한 표입니다.\n| 구분 | 내용 || :--- | :--- || 위치 | 서울 |"
    assert "| 구분 | 내용 |\n| :--- | :--- |\n| 위치 | 서울 |" in normalize_answer_markdown(text)
    assert normalize_answer_markdown("a || b") == "a || b"
    code = "```\n| a || --- |\n```"
    assert normalize_answer_markdown(code) == code


def test_bold_is_converted_to_strong():
    out = format_inline_markup("용기로 **옻(漆)**을 담았습니다")

    assert "<strong>옻(漆)</strong>" in out
    assert "**" not in out


def test_citation_badges_single_and_multi():
    out = format_inline_markup("근거 [출처 8] 와 [출처 1, 2, 8, 9] 입니다")

    assert out.count('<span class="cite">출처 8</span>') == 2  # 단일 [출처 8] + 목록
    assert out.count('<span class="cite">출처 1</span>') == 1
    assert out.count('<span class="cite">출처 9</span>') == 1
    assert "[출처" not in out


def test_inline_markup_escapes_html():
    out = format_inline_markup("<script>alert(1)</script> [출처 3]")

    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert '<span class="cite">출처 3</span>' in out


def test_split_sections_returns_expected_kinds():
    text = (
        "### 핵심 요약\n한 줄 요약입니다.\n\n" "### 상세 설명\n- 자세한 내용\n\n" "### 참고 자료\n- [출처 1] history.md"
    )

    sections = split_answer_sections(text)

    assert [section.kind for section in sections] == ["summary", "details", "references"]
    assert sections[0].body == "한 줄 요약입니다."


def test_split_sections_without_headings_returns_other():
    sections = split_answer_sections("그냥 문장입니다.")

    assert len(sections) == 1
    assert sections[0].kind == "other"


def test_split_sections_attaches_preamble_to_first_section():
    text = "서두 문장입니다.\n\n### 핵심 요약\n요약입니다."

    sections = split_answer_sections(text)

    assert sections[0].kind == "summary"
    assert "서두 문장입니다." in sections[0].body
    assert "요약입니다." in sections[0].body


def test_duplicate_summary_renders_one_heading(monkeypatch):
    """모델이 요약을 반복해도 제목은 하나이며 본문은 빠짐없이 표시한다."""
    from ui.components import chat_interface

    rendered = []
    monkeypatch.setattr(chat_interface.st, "markdown", lambda text, **kwargs: rendered.append(text))
    chat_interface.render_assistant_content("### 핵심 요약\n소개입니다.\n### 핵심 요약\n실제 요약입니다.")
    assert rendered.count("#### 핵심 요약") == 1
    assert "소개입니다." in rendered
    assert "실제 요약입니다." in rendered
