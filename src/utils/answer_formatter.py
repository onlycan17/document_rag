"""RAG 응답을 보기 좋은 구조로 다듬는 도구."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class AnswerFormatter:
    """LLM 응답을 `핵심 요약 → 상세 설명 → 참고 자료` 구조로 정리한다."""

    SECTION_ORDER: List[tuple[str, str]] = [
        ("summary", "### 핵심 요약"),
        ("details", "### 상세 설명"),
        ("references", "### 참고 자료"),
    ]

    SECTION_ALIASES: Dict[str, set[str]] = {
        "summary": {"핵심 요약", "핵심정리", "요약", "요점 정리", "한줄 요약"},
        "details": {"상세 설명", "자세한 설명", "세부 내용", "본문", "상세"},
        "references": {"참고 자료", "출처", "근거", "참고문헌", "참고"},
    }
    LOWER_ALIASES: Dict[str, set[str]] = {
        key: {alias.lower() for alias in value} for key, value in SECTION_ALIASES.items()
    }

    ERROR_KEYWORDS = ("오류", "에러", "키", "다시 시도", "업로드", "문서를", "없습니다")

    BULLET_PATTERN = re.compile(r"^\s*(\d+)[.)]\s+")

    def format(self, raw_text: str, sources: Optional[List[Dict[str, Any]]] = None) -> str:
        """주어진 텍스트를 구조화된 마크다운으로 변환한다."""
        if not raw_text or not raw_text.strip():
            return raw_text
        text = raw_text.strip()
        if self._is_error_message(text):
            return text
        extracted, fallback = self._extract_sections(text)
        sections = self._fill_sections(extracted, fallback, sources)
        return self._render_sections(sections)

    def _is_error_message(self, text: str) -> bool:
        """오류 안내 문장은 가공하지 않고 그대로 반환한다."""
        if len(text) > 280:
            return False
        lowered = text.lower()
        return any(keyword in lowered for keyword in self.ERROR_KEYWORDS)

    def _extract_sections(self, text: str) -> tuple[Dict[str, str], str]:
        """기존에 작성된 섹션을 추출하고 나머지 텍스트를 돌려준다."""
        sections: Dict[str, str] = {"summary": "", "details": "", "references": ""}
        current_key: Optional[str] = None
        buffer: List[str] = []
        fallback_chunks: List[str] = []

        for line in text.splitlines():
            candidate = self._match_section(line)
            if candidate:
                if current_key:
                    sections[current_key] = self._clean_block(buffer)
                    buffer.clear()
                elif buffer:
                    fallback_chunks.append("\n".join(buffer))
                    buffer.clear()
                current_key = candidate
                continue
            buffer.append(line)

        if current_key:
            sections[current_key] = self._clean_block(buffer)
        elif buffer:
            fallback_chunks.append("\n".join(buffer))

        fallback = "\n".join(chunk for chunk in fallback_chunks if chunk.strip())
        return sections, fallback

    def _match_section(self, line: str) -> Optional[str]:
        """헤딩/레이블을 섹션 키로 변환한다."""
        stripped = line.strip()
        if not stripped:
            return None
        header = re.sub(r"^[#*>\-\d\.\s]+", "", stripped).strip().rstrip(":")
        header_lower = header.lower()
        for key, aliases in self.LOWER_ALIASES.items():
            if header_lower in aliases:
                return key
        return None

    def _fill_sections(
        self,
        sections: Dict[str, str],
        fallback_text: str,
        sources: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, str]:
        """누락된 섹션을 보충한다."""
        paragraphs = self._split_paragraphs(fallback_text)
        if not sections["summary"] and paragraphs:
            sections["summary"] = paragraphs.pop(0)
        if not sections["details"] and paragraphs:
            sections["details"] = "\n\n".join(paragraphs)
            paragraphs.clear()
        if not sections["references"]:
            sections["references"] = self._build_reference_block(sources)
        elif sources:
            references = sections["references"].strip()
            merged = references + "\n" + self._build_reference_block(sources)
            sections["references"] = merged.strip()

        sections["summary"] = self._normalize_lists(sections["summary"])
        sections["details"] = self._normalize_lists(sections["details"])
        sections["references"] = self._normalize_lists(sections["references"])

        if not sections["summary"]:
            sections["summary"] = "- 답변이 충분히 생성되지 않았습니다. 다시 질문해보세요."
        if not sections["details"]:
            sections["details"] = "- 관련 문서에서 추가 정보를 찾지 못했습니다."
        if not sections["references"]:
            sections["references"] = "- 관련 문서 정보를 자동으로 수집하지 못했습니다."
        return sections

    def _render_sections(self, sections: Dict[str, str]) -> str:
        """섹션을 마크다운 문자열로 조합한다."""
        blocks: List[str] = []
        for key, heading in self.SECTION_ORDER:
            body = sections.get(key, "").strip()
            blocks.append(f"{heading}\n{body}")
        return "\n\n".join(blocks).strip()

    def _clean_block(self, lines: List[str]) -> str:
        """여러 줄을 단락 문자열로 정리한다."""
        joined = "\n".join(lines).strip()
        return re.sub(r"\n{3,}", "\n\n", joined)

    def _split_paragraphs(self, text: str) -> List[str]:
        """빈 줄 기준으로 단락을 분리한다."""
        if not text.strip():
            return []
        chunks = re.split(r"\n\s*\n", text.strip())
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _normalize_lists(self, text: str) -> str:
        """번호 목록 등을 불릿 목록으로 맞춘다."""
        if not text.strip():
            return text.strip()
        lines = text.splitlines()
        normalized: List[str] = []
        for line in lines:
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            if self.BULLET_PATTERN.match(stripped):
                item = self.BULLET_PATTERN.sub("", stripped)
                normalized.append(f"{indent}- {item.strip()}")
                continue
            if stripped.startswith("•"):
                normalized.append(f"{indent}- {stripped.lstrip('•').strip()}")
                continue
            normalized.append(line)
        return "\n".join(normalized).strip()

    def _build_reference_block(self, sources: Optional[List[Dict[str, Any]]]) -> str:
        """출처 정보를 불릿 목록으로 만든다."""
        if not sources:
            return ""
        lines: List[str] = []
        for index, source in enumerate(sources, start=1):
            name = source.get("file_name") or source.get("source") or "출처 미상"
            details: List[str] = []
            relevance = source.get("relevance_score")
            if isinstance(relevance, (int, float)) and relevance:
                details.append(f"관련도 {float(relevance):.1f}%")
            page = source.get("page")
            if page and page != "N/A":
                details.append(f"p.{page}")
            chunk_id = source.get("chunk_id")
            if chunk_id:
                details.append(chunk_id)
            detail_str = f" ({', '.join(details)})" if details else ""
            lines.append(f"- [출처 {index}] {name}{detail_str}")
        return "\n".join(lines).strip()
