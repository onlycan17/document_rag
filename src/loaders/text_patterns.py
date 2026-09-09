"""
텍스트 처리 정규식 패턴 사전

TextProcessor가 사용하는 마크다운/정제/한국어/구조 패턴 정의.
로직이 아닌 패턴 데이터만 모은 모듈이다.
"""

import re
from typing import Any, Dict

# 마크다운 관련 패턴
MARKDOWN_PATTERNS: Dict[str, Any] = {
    # 헤더 패턴
    "headers": re.compile(r"^(#{1,6})\s*(.+?)$", re.MULTILINE),
    # 코드 블록 패턴
    "code_blocks": re.compile(r"```[\s\S]*?```", re.MULTILINE),
    "inline_code": re.compile(r"`([^`]+)`"),
    # 링크 패턴
    "links": re.compile(r"\[([^\]]+)\]\(([^)]+)\)"),
    # 이미지 패턴
    "images": re.compile(r"!\[([^\]]*)\]\(([^)]+)\)"),
    # 리스트 패턴
    "unordered_lists": re.compile(r"^\s*[-*+]\s+(.+)$", re.MULTILINE),
    "ordered_lists": re.compile(r"^\s*\d+\.\s+(.+)$", re.MULTILINE),
    # 테이블 패턴
    "tables": re.compile(r"^\|.*\|$", re.MULTILINE),
    # 인용구 패턴
    "blockquotes": re.compile(r"^>\s*(.+)$", re.MULTILINE),
    # 강조 패턴
    "bold": re.compile(r"\*\*([^*]+)\*\*"),
    "italic": re.compile(r"\*([^*]+)\*"),
    # 구분선 패턴
    "horizontal_rules": re.compile(r"^[-*_]{3,}$", re.MULTILINE),
}


# 텍스트 정제 패턴
CLEANING_PATTERNS: Dict[str, Any] = {
    # 불필요한 공백 패턴
    "multiple_spaces": re.compile(r"\s{2,}"),
    "multiple_newlines": re.compile(r"\n{3,}"),
    "trailing_spaces": re.compile(r"[ \t]+$", re.MULTILINE),
    "leading_spaces": re.compile(r"^[ \t]+", re.MULTILINE),
    # 특수 문자 패턴
    "special_chars": re.compile(r'[^\w\s가-힣.,!?;:()\-"\'\n]'),
    "multiple_punctuation": re.compile(r"([.,!?;:]){2,}"),
    # HTML 잔여물 패턴
    "html_tags": re.compile(r"<[^>]+>"),
    "html_entities": re.compile(r"&[a-zA-Z0-9#]+;"),
    # 기타 불필요한 패턴
    "page_numbers": re.compile(r"\b\d+\s*페이지\b|\bPage\s*\d+\b", re.IGNORECASE),
    "footnotes": re.compile(r"\[\d+\]|\(\d+\)"),
    "empty_lines": re.compile(r"^\s*$", re.MULTILINE),
}


# 한국어 처리 관련 패턴
KOREAN_PATTERNS: Dict[str, Any] = {
    # 한국어 문장 패턴
    "korean_sentence": re.compile(r"[가-힣][^.!?]*[.!?]"),
    "korean_words": re.compile(r"[가-힣]+"),
    # 한국어 특수 패턴
    "korean_numbers": re.compile(r"[영일이삼사오육칠팔구십백천만억조]+"),
    "korean_units": re.compile(r"[개명마리권장점번째회차단계부분]"),
    # 한국어 조사 패턴
    "particles": re.compile(r"[은는이가을를에서의로으로와과]"),
    # 한국어 문장 부호 정규화
    "korean_punctuation": {
        "．": ".",
        "，": ",",
        "：": ":",
        "；": ";",
        "？": "?",
        "！": "!",
        "（": "(",
        "）": ")",
        "「": '"',
        "」": '"',
        "『": '"',
        "』": '"',
    },
}


# 구조 보존 관련 패턴
STRUCTURE_PATTERNS: Dict[str, Any] = {
    # 섹션 패턴
    "sections": re.compile(r"^(제?\s*\d+\s*[장절편부].*?)$", re.MULTILINE),
    "subsections": re.compile(r"^(\d+\.\d+.*?)$", re.MULTILINE),
    # 목록 구조
    "numbered_items": re.compile(r"^\s*(\d+)\.\s*(.+)$", re.MULTILINE),
    "bulleted_items": re.compile(r"^\s*[•·▪▫-]\s*(.+)$", re.MULTILINE),
    # 테이블 구조
    "table_headers": re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE),
    "table_separators": re.compile(r"^\s*\|[\s\-:]+\|\s*$", re.MULTILINE),
    # 인용 및 참조
    "citations": re.compile(r"\[[^\]]+\]\s*$", re.MULTILINE),
    "references": re.compile(r"^참고\s*[:：]|^출처\s*[:：]|^Reference\s*:", re.MULTILINE | re.IGNORECASE),
}
