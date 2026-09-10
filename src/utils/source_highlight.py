"""출처 문서에서 질문과 관련된 문단을 찾아 하이라이트 미리보기를 만드는 도구."""

from __future__ import annotations

import re
from typing import List

# 흔한 조사·어미 (질문 키워드에서 떼어내기 위한 접미사)
_PARTICLE_SUFFIXES = (
    "에서는",
    "에서의",
    "에는",
    "으로는",
    "으로",
    "로는",
    "에게서",
    "에게",
    "한테",
    "에서",
    "의",
    "를",
    "을",
    "는",
    "은",
    "가",
    "이",
    "와",
    "과",
    "로",
    "도",
    "만",
    "까지",
    "부터",
)

# 질문에서 빼도 되는 기능어
_STOPWORDS = {
    "뭐",
    "무엇",
    "어떤",
    "어떻게",
    "왜",
    "언제",
    "어디",
    "누구",
    "얼마나",
    "몇",
    "어느",
    "알려줘",
    "알려주세요",
    "설명해줘",
    "설명해주세요",
    "말해줘",
    "말해주세요",
    "궁금",
    "궁금해요",
    "입니다",
    "하세요",
    "해줘",
}


def extract_question_terms(question: str) -> List[str]:
    """질문에서 문서와 대조할 핵심 단어를 뽑는다 (길이 2 이상, 중복 제거)."""
    terms: List[str] = []
    for token in re.split(r"\s+", question.strip()):
        word = token.strip()
        for suffix in _PARTICLE_SUFFIXES:
            if len(word) > len(suffix) and word.endswith(suffix):
                word = word[: -len(suffix)]
                break
        if len(word) < 2 or word in _STOPWORDS:
            continue
        if word not in terms:
            terms.append(word)
    return terms[:8]


def highlight_keywords(text: str, question: str) -> str:
    """텍스트 안에서 질문 키워드와 일치하는 부분을 마크다운 굵게 표시한다."""
    terms = extract_question_terms(question)
    if not terms:
        return text
    pattern = re.compile("(" + "|".join(re.escape(term) for term in terms) + ")")
    return pattern.sub(r"**\1**", text)


def _split_paragraphs(content: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]


def render_source_preview(content: str, question: str, context: int = 1) -> str:
    """출처 본문에서 질문과 가장 관련 높은 문단을 중심으로 한 미리보기 마크다운을 만든다.

    Streamlit은 채팅 메시지 내부 앵커 이동을 지원하지 않으므로,
    '문단 점프'는 해당 문단(±context)을 먼저 보여주는 방식으로 구현한다.
    """
    paragraphs = _split_paragraphs(content)
    if not paragraphs:
        return "- 출처 본문이 비어 있습니다."

    terms = extract_question_terms(question)
    best_index = 0
    if terms:
        best_score = -1
        for index, paragraph in enumerate(paragraphs):
            score = sum(paragraph.count(term) for term in terms)
            if score > best_score:
                best_score = score
                best_index = index

    start = max(0, best_index - context)
    end = min(len(paragraphs), best_index + context + 1)
    skipped_before, skipped_after = start, len(paragraphs) - end

    lines: List[str] = []
    if skipped_before:
        lines.append(f"*… 앞 문단 {skipped_before}개 생략 …*")
    for paragraph in paragraphs[start:end]:
        lines.append(highlight_keywords(paragraph, question))
        lines.append("")
    if skipped_after:
        lines.append(f"*… 뒤 문단 {skipped_after}개 생략 …*")
    return "\n".join(lines).strip()
