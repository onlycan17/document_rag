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
