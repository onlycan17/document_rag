"""RAG 응답 마크다운을 화면에 안전하게 그리기 위한 순수 헬퍼."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from src.utils.answer_formatter import AnswerFormatter

# 섹션 헤딩/별칭은 AnswerFormatter 정의를 그대로 재사용한다(중복 방지).
_KNOWN_HEADINGS: List[str] = sorted(
    {alias for aliases in AnswerFormatter.SECTION_ALIASES.values() for alias in aliases},
    key=len,
    reverse=True,
)
_HEADING_ALTERNATION = "|".join(re.escape(heading) for heading in _KNOWN_HEADINGS)

# 앞 텍스트에 붙어 나온 ATX 헤딩을 줄 시작으로 분리한다 (예: "내용### 참고 자료").
# 직전 문자가 공백/`#`이면 이미 정상 위치이므로 건드리지 않는다.
_GLUED_HEADING_RE = re.compile(r"(?<=[^\s#])[ \t]*(#{1,6})[ \t]+(?=\S)")
# 헤딩 줄에 바로 붙은 본문/불릿을 다음 줄로 분리한다 (예: "### 참고 자료- 문서 1").
_HEADING_BODY_RE = re.compile(
    rf"^(#{{1,6}}[ \t]*)({_HEADING_ALTERNATION}):?[ \t]*(?=\S)",
    re.MULTILINE,
)
_HEADER_PREFIX_RE = re.compile(r"^[#*>\-\d\.\s]+")
_CITATION_RE = re.compile(r"\[출처\s*([0-9,\s]+)\]")
# 한 줄에 공백으로 뭉쳐 나온 불릿을 줄바꿈으로 분리한다 (LLM 출력에서 흔함).
_INLINE_BULLET_RE = re.compile(r"(?<=\S)[ \t]{2,}[-*][ \t]+")
# CJK 조사가 뒤따르면 CommonMark가 **강조**를 닫지 못하므로 직접 <strong>으로 변환한다.
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")

_SECTION_TITLES = {
    "summary": "핵심 요약",
    "details": "상세 설명",
    "references": "참고 자료",
    "other": "기타",
}


@dataclass
class AnswerSection:
    """답변 한 구역(요약/상세/참고 등)."""

    kind: str
    title: str
    body: str


def normalize_answer_markdown(text: str) -> str:
    """헤딩이 앞뒤 텍스트에 붙어 나온 응답을 정상 마크다운으로 되돌린다."""
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _GLUED_HEADING_RE.sub(r"\n\n\1 ", normalized)
    normalized = _HEADING_BODY_RE.sub(r"\1\2\n", normalized)
    normalized = _INLINE_BULLET_RE.sub("\n- ", normalized)
    normalized = _restore_table_rows(normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _restore_table_rows(text: str) -> str:
    """구분 행이 있는 마크다운 표에 한해서 붙은 행 경계를 복구한다."""
    lines = []
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
        if not in_code and "||" in line and re.search(r"\|\s*:?-{3,}:?\s*\|", line):
            start = line.find("|")
            prefix = line[:start].rstrip()
            table = line[start:].replace("||", "|\n|")
            line = f"{prefix}\n\n{table}" if prefix else table
        lines.append(line)
    return "\n".join(lines)


def _match_section_kind(line: str) -> Optional[str]:
    """한 줄이 섹션 헤딩이면 그 kind를 돌려준다."""
    stripped = line.strip()
    if not stripped:
        return None
    header = _HEADER_PREFIX_RE.sub("", stripped).strip().rstrip(":")
    header_lower = header.lower()
    for key, aliases in AnswerFormatter.LOWER_ALIASES.items():
        if header_lower in aliases:
            return key
    return None


def split_answer_sections(text: str) -> List[AnswerSection]:
    """정규화한 응답을 섹션 목록으로 분해한다."""
    normalized = normalize_answer_markdown(text)
    if not normalized:
        return []

    sections: List[AnswerSection] = []
    preamble: List[str] = []
    buffer: List[str] = []
    current_kind: Optional[str] = None

    def flush() -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        buffer = []
        if current_kind is None:
            if body:
                preamble.append(body)
            return
        sections.append(AnswerSection(current_kind, _SECTION_TITLES[current_kind], body))

    for line in normalized.splitlines():
        kind = _match_section_kind(line)
        if kind:
            flush()
            current_kind = kind
            continue
        buffer.append(line)
    flush()

    if not sections:
        return [AnswerSection("other", _SECTION_TITLES["other"], normalized)]
    if preamble:
        first = sections[0]
        merged = "\n\n".join([*preamble, first.body]).strip()
        sections[0] = AnswerSection(first.kind, first.title, merged)
    return sections


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_inline_markup(text: str) -> str:
    """인라인 마크다운(강조/출처)을 HTML로 바꾼다. HTML은 먼저 이스케이프한다."""
    escaped = _escape_html(text)
    escaped = _BOLD_RE.sub(r"<strong>\1</strong>", escaped)

    def _replace(match: re.Match[str]) -> str:
        numbers = re.findall(r"\d+", match.group(1))
        return " ".join(f'<span class="cite">출처 {number}</span>' for number in numbers)

    return _CITATION_RE.sub(_replace, escaped)
