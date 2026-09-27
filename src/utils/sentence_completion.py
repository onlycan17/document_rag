"""문장 완성도 판단 유틸 — 페이지 경계에서 끊어진 문장을 감지한다.

pdf_converter의 375줄 메서드에서 순수 분리했다. 동작 변경 시
tests/test_sentence_completion.py 골든 케이스(61개)가 실패하도록 설계되어 있다.
"""

import re

_COMPLETE_ENDINGS = [
    ".",
    "!",
    "?",
    "다.",
    "음.",
    "였다.",
    "었다.",
    "한다.",
    "된다.",
    "이다.",
    "않다.",
    "습니다.",
    "니다.",
]

_KOREAN_ENDINGS = ["다", "음", "였다", "었다", "한다", "된다", "이다", "않다", "있다", "없다", "습니다", "니다"]

_COMPLETE_VERBS = [
    "다",
    "음",
    "였다",
    "었다",
    "한다",
    "된다",
    "이다",
    "않다",
    "있다",
    "없다",
    "습니다",
    "니다",
    "하였다",
    "하였음",
    "되었다",
    "있었다",
]

_VERB_STEM_PATTERNS = [
    # 진행형/상태 동사 어간
    r".*하고\s*있$",  # "구분하고 있" → "구분하고 있다"
    r".*되고\s*있$",  # "설치되고 있" → "설치되고 있다"
    r".*고\s*있$",  # "놓고 있" → "놓고 있다"
    r".*어\s*있$",  # "만들어 있" → "만들어 있다"
    r".*아\s*있$",  # "자라 있" → "자라 있다"
    # 연결어미로 끝나는 경우
    r".*하고$",  # "확인하고" → "확인하고 있다/한다"
    r".*되고$",  # "설치되고" → "설치되고 있다"
    r".*이고$",  # "중요하이고" → "중요하다"
    r".*며$",  # "하며" → "하면서"
    r".*면서$",  # "하면서" → "하면서 이어진다"
    r".*하여$",  # "설치하여" → "설치하여 있다"
    r".*되어$",  # "건설되어" → "건설되어 있다"
    # 관형형 어미
    r".*하는$",  # "만드는" → "만드는 것이다"
    r".*되는$",  # "설치되는" → "설치되는 것이다"
    r".*인$",  # "중요한" → "중요한 것이다"
    r".*는$",  # "오는" → "오는 것이다"
    r".*을$",  # "만들" → "만들 것이다"
    r".*ㄹ$",  # "할" → "할 것이다"
]

_PROPER_NOUN_PATTERNS = [
    # 역사서명 + 첫 글자
    r".*삼국사기\s*백$",  # "삼국사기 백" → "삼국사기 백제본기"
    r".*삼국유사\s*고$",  # "삼국유사 고" → "삼국유사 고구려"
    r".*조선왕조실록\s*태$",  # "조선왕조실록 태" → "조선왕조실록 태조"
    r".*고려사\s*세$",  # "고려사 세" → "고려사 세가"
    r".*한국사\s*고$",  # "한국사 고" → "한국사 고대편"
    # 지명 + 첫 글자
    r".*서울특별시\s*[가-힣]$",  # "서울특별시 송" → "서울특별시 송파구"
    r".*경기도\s*[가-힣]$",  # "경기도 하" → "경기도 하남시"
    r".*충청북도\s*[가-힣]$",  # "충청북도 청" → "충청북도 청주시"
    # 기관명 + 첫 글자
    r".*한국문화재보호재단\s*[가-힣]$",
    r".*국립중앙박물관\s*[가-힣]$",
    r".*서울역사박물관\s*[가-힣]$",
    # 인명 패턴
    r".*[가-힣]{2,3}왕\s*[가-힣]$",  # "온조왕 1" → "온조왕 14년"
    r".*[가-힣]{2,3}제\s*[가-힣]$",  # "백제 온" → "백제 온조"
    # 숫자와 연결되는 경우
    r".*년\s*[가-힣]$",  # "475년 고" → "475년 고구려"
    r".*세기\s*[가-힣]$",  # "6세기 신" → "6세기 신라"
    r".*대\s*[가-힣]$",  # "조선시대 전" → "조선시대 전기"
]

_SINGLE_CHAR_AFTER_NOUN = [
    r".*[가-힣]{2,}\s*[가나다라마바사아자차카타파하]$",  # 명사 + 자음
    r".*[가-힣]{2,}\s*[의를을에서는이가와과로]$",  # 명사 + 조사 일부
]

_PROBLEMATIC_ENDINGS = [
    # 기본 불완전 패턴 (단일 글자 - 대부분 불완전)
    "몽",
    "토",
    "백",
    "성",
    "왕",
    "제",
    "풍",
    "납",
    "동",
    "촌",
    "결",
    "과",
    "SPD",
    "가",
    "나",
    "다",
    "라",
    "마",
    "바",
    "사",
    "아",
    "자",
    "차",
    "카",
    "타",
    "파",
    "하",
    "갑",
    "을",
    "병",
    "정",
    "무",
    "기",
    "경",
    "신",
    "임",
    "계",
    # 조사류
    "를",
    "을",
    "의",
    "에",
    "서",
    "는",
    "이",
    "가",
    "한",
    "된",
    "할",
    "될",
    "와",
    "과",
    "로",
    "으로",
    # 명사로 끝나는 경우 (문서에서 자주 끊어지는 패턴)
    "측정치를",
    "분포를",
    "연대",
    "시대",
    "토기",
    "기종의",
    "형태를",
    "비교한",
    "후",
    "주요",
    "결과",
    "분기로",
    "나눌",
    "수",
    "있었다",
    "확인되는",
    "시기를",
    # 실제 발견된 문제 패턴들 추가
    "개보수흔적",
    "축조흔적",
    "보수흔적",
    "수리흔적",
    "건축흔적",
    "시설흔적",
    "발굴흔적",
    "사용흔적",
    "폐기흔적",
    "매몰흔적",
    "조성흔적",
    # 일반적인 명사들 (문장 중간에서 끊어질 수 있는)
    "유구",
    "유물",
    "토층",
    "구조",
    "형태",
    "양상",
    "특징",
    "현상",
    "상황",
    "조건",
    "방법",
    "과정",
    "절차",
    "단계",
    "순서",
    "계획",
    "목적",
    "의도",
    "취지",
    "분석",
    "검토",
    "조사",
    "연구",
    "관찰",
    "확인",
    "파악",
    "추정",
    "판단",
    # 수식어나 관형어
    "주요한",
    "중요한",
    "특별한",
    "일반적인",
    "전체적인",
    "부분적인",
    "개별적인",
    "다양한",
    "여러",
    "많은",
    "적은",
    "큰",
    "작은",
    "높은",
    "낮은",
    "깊은",
    "얕은",
    # 접속 표현
    "그리고",
    "또한",
    "하지만",
    "그러나",
    "따라서",
    "그런데",
    "한편",
    "첫째",
    "둘째",
    "셋째",
    "마지막으로",
    # 책명, 문서명에서 자주 끊어지는 패턴
    "보고서",
    "연구서",
    "논문집",
    "학술지",
    "잡지",
    "신문",
    "기록",
    "자료",
    "편찬위원회",
    "연구소",
    "박물관",
    "재단",
    "협회",
    "학회",
]

_PARTICLES = [
    "를",
    "을",
    "의",
    "에",
    "서",
    "는",
    "이",
    "가",
    "와",
    "과",
    "로",
    "으로",
    "부터",
    "까지",
    "에서",
    "에게",
    "께서",
    "께",
    "으로서",
    "으로써",
    "처럼",
    "같이",
    "보다",
    "만큼",
    "마다",
    "조차",
    "까지도",
    "밖에",
    "만",
    "든지",
    "거나",
    "든가",
    "던지",
]

_THREE_CHAR_NOUN_PATTERNS = [
    r".*[조사연관분해석토기유물구조시설]$",  # 명사적 어미
    r".*[건축발굴수리보수개선공사작업]$",  # 행위 관련 명사
    r".*[계획방법과정절차단계순서]$",  # 과정 관련 명사
]

_INCOMPLETE_MORPHEMES = ["ㄴ", "ㄹ", "ㅁ", "ㅂ", "ㄷ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

_CONNECTIVE_ENDINGS = (",", "하여", "하며", "이며", "되며", "되어", "이고", "되고", "로서", "로써")


def _ends_with_complete_punctuation(line: str) -> bool:
    """문장부호·확실한 종결형으로 끝나면 완전 문장."""
    return any(line.endswith(ending) for ending in _COMPLETE_ENDINGS)


def _korean_ending_verdict(line: str) -> bool | None:
    """문장부호 없는 한글 어미 판정. True=불완전, False=완전, None=판단 유보."""
    for ending in _KOREAN_ENDINGS:
        if not line.endswith(ending):
            continue
        if len(ending) == 1 and line == ending:
            return True
        if len(line) > 10:
            for verb in _COMPLETE_VERBS:
                if line.endswith(verb) and len(line) > len(verb) + 5:
                    return False
    return None


def _matches_any(patterns: tuple[str, ...], line: str) -> bool:
    return any(re.search(p, line) for p in patterns)


def _ends_with_short_korean_word(line: str) -> bool:
    """마지막 단어가 1~2글자 순수 한글이면 불완전."""
    words = line.split()
    if not (words and len(words[-1]) <= 2):
        return False
    return re.match(r"^[가-힣]{1,2}$", words[-1]) is not None


def _ends_with_three_char_noun(line: str) -> bool:
    """3글자 순수 한글이 명사형 어미 글자로 끝나면 불완전."""
    words = line.split()
    if not (words and len(words[-1]) == 3):
        return False
    last_word = words[-1]
    if re.match(r"^[가-힣]{3}$", last_word) is None:
        return False
    return any(re.match(p, last_word) for p in _THREE_CHAR_NOUN_PATTERNS)


def is_incomplete_sentence(line: str) -> bool:
    """문장이 불완전한지 판단 (페이지 경계에서 끊어진 문장 감지)."""
    if not line:
        return False

    line = line.strip()
    if _ends_with_complete_punctuation(line):
        return False

    verdict = _korean_ending_verdict(line)
    if verdict is not None:
        return verdict

    if _matches_any(_VERB_STEM_PATTERNS, line):
        return True
    if _matches_any(_PROPER_NOUN_PATTERNS, line):
        return True
    if _matches_any(_SINGLE_CHAR_AFTER_NOUN, line):
        return True
    if any(line.endswith(ending) for ending in _PROBLEMATIC_ENDINGS):
        return True
    if any(line.endswith(particle) for particle in _PARTICLES):
        return True
    if re.search(r"[0-9A-Za-z]$", line):
        return True
    if _ends_with_short_korean_word(line):
        return True
    if _ends_with_three_char_noun(line):
        return True
    if line.endswith(_CONNECTIVE_ENDINGS):
        return True
    return any(line.endswith(morpheme) for morpheme in _INCOMPLETE_MORPHEMES)


# 새 문장·항목을 시작하는 말 — 앞 줄이 미완성이어도 이어 붙이지 않는다
_NEW_SENTENCE_STARTERS = (
    "그러나",
    "하지만",
    "따라서",
    "그런데",
    "또한",
    "그리고",
    "한편",
    "첫째",
    "둘째",
    "셋째",
    "다음",
    "마지막으로",
    "그 결과",
    "이에 따라",
    "결론적으로",
)
# 제목·표·목록·페이지 표시·표/그림 캡션 같은 구조 요소 줄
_STRUCTURE_LINE = re.compile(r"^(#|\||\[페이지 \d+\]|--- 페이지 \d+ ---|[<\[](표|그림)|[-*•·]\s)")
_MAX_CARRIED_LINES = 3


def can_join_lines(current_line: str, next_line: str) -> bool:
    """미완성 문장 줄 뒤에 다음 줄을 이어 붙여도 되는지 (새 문장·항목·구조 요소면 False)"""
    current, following = current_line.strip(), next_line.strip()
    if not current or not following:
        return False
    if _STRUCTURE_LINE.match(current) or _STRUCTURE_LINE.match(following):
        return False
    if following.startswith(_NEW_SENTENCE_STARTERS):
        return False
    if re.match(r"^\d+[\.\)]\s", following) or re.match(r"^[가-힣][\.\)]\s", following):  # 1. / 가) 항목
        return False
    if re.match(r"^제\s*\d+\s*[장절편부]", following):  # 제1장·제2절 (제기·장소·절차 같은 단어는 허용)
        return False
    return not re.match(r"^[A-Z][a-z]", following)  # 새 영어 문장


# 문장 끝: 마침표·물음표·느낌표 뒤 공백/줄끝 (소수점 "3.5"는 제외)
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def _take_sentence_remainder(line: str) -> tuple[str, str]:
    """줄에서 첫 문장 끝까지(앞 페이지로 옮길 부분)와 나머지를 나눈다. 문장 끝이 없으면 줄 전체를 옮긴다."""
    match = _SENTENCE_END.search(line)
    if not match:
        return line.strip(), ""
    return line[: match.end()].strip(), line[match.end() :].strip()


_MIN_PROSE_LINE_CHARS = 20


def _ends_mid_sentence(line: str) -> bool:
    """페이지 마지막 줄이 끊긴 문장인지 (명사로 끝난 긴 본문 줄도 포함)

    is_incomplete_sentence는 제목 보호를 위해 명사로 끝나는 줄을 완결로 본다. 페이지 경계에서는
    20자 이상 본문 줄이 문장부호·쪽번호 없이 끝나면 다음 페이지로 이어지는 문장으로 본다.
    """
    # ponytail: 규칙 판정이라 페이지 끝의 제목("…방안 연구")은 끊긴 문장과 구분 못 함 — 이후 LLM 정리가 문맥으로 보정
    text = line.strip()
    if re.search(r"\s\d{1,4}$", text):  # "3. 결과 분석 120" 같은 목차·쪽 참조 줄
        return False
    if is_incomplete_sentence(text):
        return True
    return (
        len(text) >= _MIN_PROSE_LINE_CHARS
        and not _STRUCTURE_LINE.match(text)
        and not _SENTENCE_END.search(text[-1:])
        and not text[-1].isdigit()
    )


def join_page_boundaries(page_texts: list[str]) -> list[str]:
    """페이지 끝에서 끊긴 문장을 다음 페이지에서 문장이 끝나는 곳까지 끌어와 잇는다 (페이지 수는 유지)

    문단 전체가 아니라 끊긴 문장의 나머지만 옮기므로 다음 페이지의 나머지 본문과 [페이지 N] 위치는 유지된다.
    한글은 대부분 띄어쓰기 자리에서 줄이 바뀌므로 공백으로 잇는다. 단어 중간이 끊긴 경우의
    잘못된 공백은 이후 LLM 정리 단계의 띄어쓰기 교정이 바로잡는다.
    """
    pages = [text.strip("\n").rstrip() for text in page_texts]
    for index in range(len(pages) - 1):
        lines = pages[index].split("\n")
        following = pages[index + 1].split("\n")
        carried = 0
        while (
            carried < _MAX_CARRIED_LINES
            and lines[-1].strip()
            and following
            and _ends_mid_sentence(lines[-1])
            and can_join_lines(lines[-1], following[0])
        ):
            moved, rest = _take_sentence_remainder(following.pop(0))
            lines[-1] = f"{lines[-1].rstrip()} {moved}"
            if rest:
                following.insert(0, rest)
            carried += 1
        pages[index] = "\n".join(lines)
        pages[index + 1] = "\n".join(following).strip("\n")
    return pages
