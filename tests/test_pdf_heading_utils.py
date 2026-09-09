#!/usr/bin/env python3
"""
pdf_heading_utils 순수 헬퍼 회귀 테스트

pdf_converter의 헤딩/문단 판정 로직을 모듈 분리하면서 동작 동일성을 검증한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.pdf_heading_utils import is_real_heading, ends_complete_sentence, join_paragraph_lines, get_heading_level


def test_is_real_heading():
    assert is_real_heading("제1장 총칙", 0, ["제1장 총칙", "이 조례는 문화재 보존에 관한 사항을 규정한다."]) is True
    assert is_real_heading("가. 개요", 0, ["가. 개요", "이 연구는 조사 목적으로 시작된다."]) is True
    assert is_real_heading("이 조치는 문화재 보호를 위한 것이다.", 0, ["x"]) is False
    assert is_real_heading("a" * 150, 0, ["x"]) is False


def test_ends_complete_sentence():
    assert ends_complete_sentence("설치되어 있다.") is True
    assert ends_complete_sentence("완료되었다") is True
    assert ends_complete_sentence("조사 결과를") is False
    assert ends_complete_sentence("") is True


def test_join_paragraph_lines():
    assert join_paragraph_lines(["문화재를 보", "호한다"]) == "문화재를 보호한다"
    assert join_paragraph_lines(["첫 문장이다.", "둘째 문장이다."]) == "첫 문장이다. 둘째 문장이다."
    assert join_paragraph_lines([]) == ""


def test_get_heading_level():
    assert get_heading_level("제1장 총칙") == 1
    assert get_heading_level("제2절 적용범위") == 2
    assert get_heading_level("1. 개요") == 2
    assert get_heading_level("1.1 배경") == 2
    assert get_heading_level("가. 목적") == 3
    assert get_heading_level("(가) 세부") == 4


if __name__ == "__main__":
    test_is_real_heading()
    test_ends_complete_sentence()
    test_join_paragraph_lines()
    test_get_heading_level()
    print("✅ pdf_heading_utils 회귀 테스트 통과")
