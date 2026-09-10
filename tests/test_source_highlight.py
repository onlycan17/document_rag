#!/usr/bin/env python3
"""출처 하이라이트·문단 점프 헬퍼 단위테스트."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.source_highlight import extract_question_terms, highlight_keywords, render_source_preview


CONTENT = """몽촌토성은 서울 송파구에 위치한 백제 시대의 토성이다.

발굴조사 결과 2024년에 다양한 토기편이 출토되었다.

축조 방법은 판축 기법을 사용한 것으로 추정된다."""

QUESTION = "몽촌토성에서 발굴조사로 확인된 토기편은 무엇인가요?"


def test_extract_question_terms_strips_particles():
    terms = extract_question_terms(QUESTION)
    assert "몽촌토성" in terms
    assert "발굴조사" in terms
    assert "토기편" in terms
    assert "무엇인가요" not in terms


def test_extract_question_terms_deduplicates_and_limits():
    terms = extract_question_terms("토기 토기 토기 배 배 " + " ".join(f"단어{i}" for i in range(20)))
    assert terms.count("토기") == 1
    assert len(terms) <= 8


def test_highlight_keywords_bolds_matches():
    highlighted = highlight_keywords("몽촌토성은 백제의 토성이다.", "몽촌토성이 뭐야?")
    assert "**몽촌토성**은 백제의 토성이다." == highlighted


def test_highlight_keywords_without_terms_returns_text():
    assert highlight_keywords("특별한 키워드 없음", "알려줘") == "특별한 키워드 없음"


def test_render_source_preview_jumps_to_relevant_paragraph():
    preview = render_source_preview(CONTENT, QUESTION)
    # 발굴조사·토기편 문단이 중심이 되어야 한다 (문단 3개면 ±1이 전체를 덮어 생략 없음)
    assert "**토기편**이 출토되었다" in preview
    assert "생략" not in preview


def test_render_source_preview_shows_skipped_paragraph_counts():
    long_content = CONTENT + "\n\n보존 처리 계획은 추후 발표된다.\n\n전시 일정은 박물관 공지를 확인한다."
    preview = render_source_preview(long_content, QUESTION)
    assert "**토기편**이 출토되었다" in preview
    assert "뒤 문단 2개 생략" in preview


def test_render_source_preview_marks_matched_keywords():
    preview = render_source_preview(CONTENT, QUESTION)
    assert "**발굴조사** 결과 2024년에 다양한 **토기편**이 출토되었다" in preview


def test_render_source_preview_empty_content():
    assert "비어 있습니다" in render_source_preview("", QUESTION)


def test_render_source_preview_single_paragraph_no_ellipsis():
    preview = render_source_preview("한 문단뿐인 내용이다.", "문단이 뭐야")
    assert "생략" not in preview
    assert "한 **문단**뿐인 내용이다." in preview
