"""검색 품질을 떨어뜨리는 정크 텍스트 제거 (이미지 참조, 모델 토큰, 목차 점선, 숫자 나열, 판권면)"""

import re

# 경로에 괄호가 한 단계 중첩된 경우까지 처리하는 마크다운 이미지 링크 (예: 연구(KERIS) F_page001.png)
MARKDOWN_IMAGE = re.compile(r"!\[([^\]\n]*)\]\((?:[^()\n]|\([^()\n]*\))*\)")
# 변환기가 붙인 의미 없는 대체 텍스트: "이미지", "이미지 3 (페이지 1)" 등
PLACEHOLDER_ALT = re.compile(r"^\s*(이미지|이メージ|image)?\s*\d*\s*(\(페이지\s*\d+\))?\s*$", re.IGNORECASE)
# 멀티모달 변환 모델(GLM)이 남기는 박스 토큰 (밑줄이 지워진 형태 포함)
MODEL_BOX_TOKEN = re.compile(r"<\|(begin|end)_?of_?box\|>")

JUNK_LINES = (
    re.compile(r"[·.…•∙]{5,}\s*\d*\s*$"),  # 목차 점선 리더: "발간사·······5", "모델 •••••"
    re.compile(r"^[\d\s.,%~\-]+$"),  # 숫자만 있는 줄: 도표 수치, 쪽번호
    # 기존 정제 로직이 깨뜨린 이미지 참조: "이미지 1 (페이지 1) 연구(KERIS) Fpage001img001.png)"
    re.compile(r"^(이미지|이メージ)(\s+\d+\s+\(페이지\s*\d+\))?(\s.*\.(png|jpe?g|gif|webp)\)?)?$", re.IGNORECASE),
    re.compile(
        r"페이지\s*\d+의 이미지\s+\S+\.(png|jpe?g|gif|webp)$", re.IGNORECASE
    ),  # "**이미지**: 페이지 2의 이미지 ./a.png"
)
COLOPHON = re.compile(r"ISBN\s*[\d\-]{10,}")


def _replace_image(match: re.Match) -> str:
    alt = match.group(1)
    return "" if PLACEHOLDER_ALT.match(alt) else alt


def remove_junk_lines(text: str) -> str:
    """이미지 링크·모델 토큰을 지우고 정크 줄을 제거한다. 나머지 줄은 그대로 유지한다."""
    text = MODEL_BOX_TOKEN.sub("", MARKDOWN_IMAGE.sub(_replace_image, text))
    kept = [line for line in text.split("\n") if not any(p.search(line.strip()) for p in JUNK_LINES)]
    return "\n".join(kept)


def is_colophon(text: str) -> bool:
    """판권면(ISBN·발행 정보) 청크 여부"""
    return bool(COLOPHON.search(text))
