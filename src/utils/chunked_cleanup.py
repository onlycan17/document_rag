"""긴 문서를 문장 경계에서 나눠 LLM으로 교정한다

- 분할: 빈 줄로 나뉜 문단 단위로 모으고, 문장이 끝난 문단에서만 자른다 (표·제목은 쪼개지 않음)
- 문맥: 각 청크에 앞뒤 청크의 원문 일부를 "참고용"으로 함께 보낸다 → 청크끼리 독립이라 병렬 처리 가능
- 검증: 교정은 띄어쓰기·문장 연결만 하므로 출력 글자 수가 입력과 거의 같아야 하고,
  [페이지 N] 표시(출처 페이지)와 표 행 수가 그대로여야 한다. 어긋나면 그 청크는 원문을 쓴다
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List

from src.utils.sentence_completion import ends_mid_sentence

logger = logging.getLogger(__name__)

TARGET_CHUNK_CHARS = 3000
MAX_CHUNK_CHARS = 4500
CONTEXT_CHARS = 300
MIN_LENGTH_RATIO = 0.85
MAX_LENGTH_RATIO = 1.15
# 청크 출력이 입력과 비슷한 길이여야 하므로 한국어 4,500자를 담을 여유
MAX_OUTPUT_TOKENS = 8192

_PROMPT = """PDF에서 추출한 문서의 [본문]을 교정하세요.

규칙:
1. 줄바꿈으로 끊긴 문장을 한 문장으로 잇고 띄어쓰기를 교정한다 (예: "운영에홍승연은" → "운영에 홍승연은")
2. 끊어진 단어·고유명사를 복원한다 (예: "삼국사기 백 제본기" → "삼국사기 백제본기")
3. 내용 추가·삭제·요약 금지. 숫자·단위·고유명사·인용 표시는 원문 그대로 둔다
4. 제목(#), 목록, 마크다운 표는 형태를 유지한다. [페이지 N] 줄은 하나도 빠짐없이 원래 위치에 그대로 둔다
5. [앞 문맥]과 [뒤 문맥]은 문장이 어떻게 이어지는지 판단하는 참고용이다. 교정하거나 출력하지 않는다
6. 설명·머리말 없이 교정한 [본문]만 출력한다

[앞 문맥]
{previous}

[본문]
{chunk}

[뒤 문맥]
{following}"""

# LLM이 덧붙이는 역할 설명·머리말·구분선
_ARTIFACT_PATTERNS = [
    r"^\s*\[본문\]\s*\n?",
    r"한국어 문서 전문가로서[^\n]*\n?",
    r"한국문화사 문서에서[^\n]*\n?",
    r"다음은[^\n]*(교정된|텍스트입니다)[^\n]*:\n?",
    r"교정된 텍스트[^\n]*:\n?",
    r"문맥을[^\n]*연결(하겠습니다|하여[^\n]*제공합니다)[^\n]*\n?",
    r"텍스트의 문맥을[^\n]*연결[^\n]*\n?",
    r"PDF에서 추출된[^\n]*연결하여[^\n]*\n?",
    r"^#+\s*한국어 문서[^\n]*제공합니다[^\n]*\n?",
    r"^\s*```[a-z]*\s*$",
    r"^\s*---+\s*$",
]


@dataclass
class CleanupResult:
    text: str
    total_chunks: int
    fallback_chunks: List[int] = field(default_factory=list)  # 원문으로 대체된 청크 번호 (1부터)


def strip_llm_artifacts(text: str) -> str:
    cleaned = text
    for pattern in _ARTIFACT_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _ends_at_safe_boundary(paragraph: str) -> bool:
    lines = [line for line in paragraph.split("\n") if line.strip()]
    return bool(lines) and not ends_mid_sentence(lines[-1])


def split_into_chunks(text: str) -> List[str]:
    """문단을 모아 약 TARGET_CHUNK_CHARS에서, 문장이 끝난 문단 뒤에서만 자른다 (최대 MAX_CHUNK_CHARS)"""
    chunks: List[str] = []
    current: List[str] = []
    size = 0
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        current.append(paragraph)
        size += len(paragraph) + 2
        if size >= MAX_CHUNK_CHARS or (size >= TARGET_CHUNK_CHARS and _ends_at_safe_boundary(paragraph)):
            chunks.append("\n\n".join(current))
            current, size = [], 0
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _visible_length(text: str) -> int:
    return len(re.sub(r"\s", "", text))


_PAGE_MARKER = re.compile(r"^\[페이지 \d+\]$", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\|", re.MULTILINE)
_LEADING_PAGE_MARKERS = re.compile(r"(?:\[페이지 \d+\]\n)+")


def _structure_signature(text: str) -> tuple[list[str], int]:
    return _PAGE_MARKER.findall(text), len(_TABLE_ROW.findall(text))


def _is_faithful(original: str, corrected: str) -> bool:
    ratio = _visible_length(corrected) / max(_visible_length(original), 1)
    same_structure = _structure_signature(original) == _structure_signature(corrected)
    return MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO and same_structure


def _context(snippet: str) -> str:
    """참고 문맥에서 [페이지 N]을 뺀다 — LLM이 문맥의 표시를 본문으로 베껴 넣는 문제 방지"""
    return _PAGE_MARKER.sub("", snippet).strip()


def clean_document(text: str, call_llm: Callable[[str], str], max_workers: int = 3) -> CleanupResult:
    """문서를 분할해 병렬로 교정하고, 검증에 실패한 청크는 원문으로 되돌려 합친다"""
    chunks = split_into_chunks(text)

    def clean_chunk(index: int) -> tuple[str, bool]:
        # 청크 맨 앞의 [페이지 N]은 LLM이 반복적으로 빠뜨리므로 떼어 두었다가 결과 앞에 다시 붙인다
        marker = _LEADING_PAGE_MARKERS.match(chunks[index])
        head = marker.group(0) if marker else ""
        prompt = _PROMPT.format(
            previous=_context(chunks[index - 1][-CONTEXT_CHARS:]) if index > 0 else "(문서 시작)",
            chunk=chunks[index][len(head) :],
            following=_context(chunks[index + 1][:CONTEXT_CHARS]) if index + 1 < len(chunks) else "(문서 끝)",
        )
        try:
            corrected = head + strip_llm_artifacts(call_llm(prompt) or "")
        except Exception as e:
            logger.warning(f"청크 {index + 1}/{len(chunks)} 교정 실패, 원문 사용: {type(e).__name__}: {e}")
            return chunks[index], False
        if not _is_faithful(chunks[index], corrected):
            logger.warning(
                f"청크 {index + 1}/{len(chunks)} 검증 실패(글자 {_visible_length(corrected)}/"
                f"{_visible_length(chunks[index])}, 페이지 표시·표 행 {_structure_signature(corrected)} ≠ "
                f"{_structure_signature(chunks[index])}), 원문 사용"
            )
            return chunks[index], False
        return corrected, True

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        results = list(pool.map(clean_chunk, range(len(chunks))))

    # 병렬 호출이 몰려 난 요청 한도 초과(429)나 LLM의 비결정적 구조 훼손은 한 번 더 순차로 시도하면 대부분 통과
    retry_indices = [index for index, (_, ok) in enumerate(results) if not ok]
    if retry_indices:
        logger.info(f"실패 청크 {[index + 1 for index in retry_indices]} 순차 재시도")
        for index in retry_indices:
            results[index] = clean_chunk(index)

    fallback = [index + 1 for index, (_, ok) in enumerate(results) if not ok]
    logger.info(f"분할 교정 완료: {len(chunks)}개 청크, 원문 대체 {len(fallback)}개 {fallback}")
    return CleanupResult("\n\n".join(text for text, _ in results), len(chunks), fallback)
