#!/usr/bin/env python3
"""
PDF 변환용 헤딩/문단 판정 순수 헬퍼

pdf_converter에서 무상태 로직만 순수 분리한 모듈.
모든 함수는 인스턴스 상태에 의존하지 않는다(단위 테스트 대상).
"""

import re


def is_real_heading(line: str, line_index: int, all_lines: list) -> bool:
    """실제 헤딩인지 더 정교하게 판단"""
    # 너무 긴 텍스트는 헤딩이 아님
    if len(line) > 100:
        return False

    # 문장 부호로 끝나는 경우 일반적으로 헤딩이 아님
    if line.endswith((".", "다", "음", "었다", "였다", "한다", "된다", "이다", "않다")):
        return False

    # 명확한 헤딩 패턴들
    heading_patterns = [
        r"^제\s*\d+\s*장",  # 제1장, 제 2 장
        r"^제\s*\d+\s*절",  # 제1절, 제 2 절
        r"^\d+\.\s*[가-힣]",  # 1. 서론
        r"^[가-힣]\.\s*[가-힣]",  # 가. 개요
        r"^\([가-힣]\)",  # (가)
        r"^\d+\)\s*[가-힣]",  # 1) 목적
        r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.",  # I. II. III.
        r"^【[^】]+】",  # 【제목】
        r"^◎|^○|^●|^■|^□",  # 기호로 시작
    ]

    for pattern in heading_patterns:
        if re.match(pattern, line):
            return True

    # 전체가 대문자인 경우 (영어)
    if line.isupper() and len(line) < 50 and re.search(r"[A-Z]", line):
        return True

    # 특정 키워드로 시작하는 제목들
    heading_keywords = [
        "서론",
        "결론",
        "요약",
        "개요",
        "배경",
        "목적",
        "방법",
        "결과",
        "고찰",
        "참고문헌",
        "조사",
        "발굴",
        "유적",
        "유물",
        "분석",
        "검토",
        "연구",
        "현황",
    ]

    for keyword in heading_keywords:
        if line.startswith(keyword) and len(line) < 80:
            # 다음 줄이 내용인지 확인
            if line_index + 1 < len(all_lines):
                next_line = all_lines[line_index + 1].strip()
                if next_line and len(next_line) > 20:  # 다음 줄이 충분히 긴 내용이면
                    return True

    return False


def ends_complete_sentence(line: str) -> bool:
    """줄이 완전한 문장으로 끝나는지 판단"""
    if not line:
        return True

    # 한글 문장 종결 패턴
    korean_endings = ["다", "음", "였다", "었다", "한다", "된다", "이다", "않다", "있다", "없다"]

    for ending in korean_endings:
        if line.endswith(ending + ".") or line.endswith(ending):
            return True

    # 영어/숫자 문장 종결
    if line.endswith((".", "!", "?", ":", ";")):
        return True

    # 닫는 괄호나 따옴표로 끝나는 경우
    if line.endswith((")", '"', "'", "』", "】")):
        return True

    return False


def join_paragraph_lines(lines: list) -> str:
    """문단의 줄들을 자연스럽게 연결"""
    if not lines:
        return ""

    # 각 줄의 끝을 확인하여 연결 방식 결정
    result_parts = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        if i == 0:
            result_parts.append(line)
        else:
            prev_line = lines[i - 1].strip()

            # 이전 줄이 완전한 문장으로 끝나는지 확인
            if ends_complete_sentence(prev_line):
                # 새로운 문장 시작
                result_parts.append(" " + line)
            else:
                # 문장 계속 (줄바꿈으로 인해 끊어진 경우)
                result_parts.append(line)

    return "".join(result_parts)


def get_heading_level(line: str) -> int:
    """헤딩 레벨 결정 (개선된 버전)"""
    # 장/절 구조
    if re.match(r"^제\s*\d+\s*장", line):
        return 1
    elif re.match(r"^제\s*\d+\s*절", line):
        return 2

    # 숫자 패턴으로 레벨 결정
    if re.match(r"^\d+\.", line):  # 1., 2., 3.
        return 2
    elif re.match(r"^\d+\.\d+", line):  # 1.1, 1.2
        return 3
    elif re.match(r"^\d+\.\d+\.\d+", line):  # 1.1.1
        return 4
    elif re.match(r"^[가-힣]\.", line):  # 가., 나., 다.
        return 3
    elif re.match(r"^\([가-힣]\)", line):  # (가), (나)
        return 4
    elif re.match(r"^\d+\)", line):  # 1), 2)
        return 4
    elif re.match(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.", line):  # I., II.
        return 2
    else:
        return 2
