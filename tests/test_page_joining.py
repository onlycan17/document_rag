"""페이지 경계 문장 연결(join_page_boundaries) 테스트"""

import pytest

from src.utils.sentence_completion import can_join_lines, join_page_boundaries


def test_sentence_cut_at_page_end_is_completed_from_next_page():
    pages = [
        "첫 문단이다.\n특히 NEIS와 K-에듀파인 시스템의 기능 개선 및 추가",
        "도입의 필요성이 제기되었다.\n다음 문단이다.",
    ]

    joined = join_page_boundaries(pages)

    assert joined[0].endswith("기능 개선 및 추가 도입의 필요성이 제기되었다.")
    assert joined[1] == "다음 문단이다."


def test_sentence_spanning_several_lines_is_carried_until_it_ends():
    pages = ["교사들은 학습, 관리, 행정 등", "다양한 업무 영역에서\n경감 필요성을\n제기하였다.\n다음 문단이다."]

    joined = join_page_boundaries(pages)

    assert joined[0] == "교사들은 학습, 관리, 행정 등 다양한 업무 영역에서 경감 필요성을 제기하였다."
    assert joined[1] == "다음 문단이다."


def test_complete_sentence_and_page_count_are_kept():
    pages = ["완결된 문장이다.", "새 페이지 문장이다.", ""]
    assert join_page_boundaries(pages) == pages


def test_toc_lines_ending_with_page_numbers_are_not_joined():
    pages = ["2. 평가 기록 누적 및 공유 114", "3. 결과 분석 120", "4. 향후 과제 131"]
    assert join_page_boundaries(pages) == pages


@pytest.mark.parametrize(
    "current,following",
    [
        ("# (1) 교사 의견", "교사들은 다양한 업무를"),  # 제목
        ("본문이 이어지는", "| 구분 | 값 |"),  # 표
        ("본문이 이어지는", "<표 IV-18> 검사결과"),  # 표 캡션
        ("본문이 이어지는", "1. 서론"),  # 번호 항목
        ("본문이 이어지는", "그러나 다른 견해도"),  # 새 문장
        ("본문이 이어지는", "제2장 조사 결과"),  # 장 제목
    ],
)
def test_structure_lines_and_new_sentences_are_not_joined(current, following):
    assert not can_join_lines(current, following)


def test_only_the_cut_sentence_moves_and_rest_of_paragraph_stays_on_next_page():
    pages = [
        "다만, 학교 현장에서는 교사가 스스로 느끼는 업무경감",
        "효과는 미흡한 실정이다. 따라서 새로운 방안이 필요하다.\n\n다음 문단이다.",
    ]

    joined = join_page_boundaries(pages)

    assert joined[0] == "다만, 학교 현장에서는 교사가 스스로 느끼는 업무경감 효과는 미흡한 실정이다."
    assert joined[1] == "따라서 새로운 방안이 필요하다.\n\n다음 문단이다."


def test_trailing_blank_lines_do_not_block_joining():
    pages = ["기능 개선 및 추가\n\n", "도입이 필요하다."]
    assert join_page_boundaries(pages)[0] == "기능 개선 및 추가 도입이 필요하다."


def test_decimal_point_is_not_a_sentence_end():
    pages = ["한강은 서울을 강남과 강북으로 나누며 한강의 총 연장은", "497.5㎞이며 서울 구간은 42.9㎞이다. 다음 문장."]
    assert (
        join_page_boundaries(pages)[0]
        == "한강은 서울을 강남과 강북으로 나누며 한강의 총 연장은 497.5㎞이며 서울 구간은 42.9㎞이다."
    )
