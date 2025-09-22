"""스트리밍 응답 헬퍼"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class StreamingResult:
    """스트리밍 수행 결과"""

    answer: str = ""
    context_documents: list[Any] = field(default_factory=list)
    search_info: Dict[str, Any] = field(default_factory=dict)
    status: str = "unknown"
    error: Optional[str] = None


def process_streaming_response(
    stream_generator: Iterable[Dict[str, Any]],
    response_placeholder: Any,
    status_placeholder: Any,
    *,
    is_local_model: bool = False,
    logger: Optional[logging.Logger] = None,
) -> StreamingResult:
    log = logger or LOGGER
    result = StreamingResult()
    for chunk in stream_generator:
        _apply_chunk(chunk, result, response_placeholder, status_placeholder, is_local_model)
        if result.status in {"error", "success"}:
            break
    status_placeholder.empty()
    if result.status == "error" and result.error:
        log.error("스트리밍 응답 오류: %s", result.error)
    return result


def _apply_chunk(
    chunk: Dict[str, Any],
    result: StreamingResult,
    response_placeholder: Any,
    status_placeholder: Any,
    is_local_model: bool,
) -> None:
    chunk_type = chunk.get("type", "")
    if chunk_type == "status":
        _update_status(status_placeholder, chunk.get("content", ""), is_local_model)
    elif chunk_type in {"response", "content"}:
        _append_text(result, chunk, response_placeholder)
    elif chunk_type == "sources":
        result.context_documents = chunk.get("content", result.context_documents)
    elif chunk_type == "search_info":
        result.search_info = chunk.get("content", result.search_info)
    elif chunk_type == "final":
        _handle_final_chunk(chunk, result, response_placeholder)
    elif chunk_type == "error":
        result.status = "error"
        result.error = chunk.get("content", "스트리밍 중 오류가 발생했습니다.")
        status_placeholder.error(result.error)


def _update_status(status_placeholder: Any, message: str, is_local_model: bool) -> None:
    if not message:
        return
    if is_local_model and "생성" in message:
        status_placeholder.warning("⏳ 로컬 모델이 답변을 생성 중입니다. 시간이 걸릴 수 있습니다...")
        return
    status_placeholder.info(message)


def _append_text(result: StreamingResult, chunk: Dict[str, Any], response_placeholder: Any) -> None:
    text = chunk.get("full_content") or chunk.get("content") or ""
    if not text:
        return
    result.answer = text
    response_placeholder.markdown(text)


def _handle_final_chunk(chunk: Dict[str, Any], result: StreamingResult, response_placeholder: Any) -> None:
    final_data = chunk.get("content") or chunk.get("full_content") or {}
    if isinstance(final_data, dict):
        result.answer = final_data.get("answer", result.answer)
        result.context_documents = final_data.get("context_documents", result.context_documents)
        result.search_info = final_data.get("search_info", result.search_info)
    elif isinstance(final_data, str):
        result.answer = final_data or result.answer
    response_placeholder.markdown(result.answer)
    result.status = "success"
