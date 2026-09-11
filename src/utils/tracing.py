"""Langfuse 관측 헬퍼.

LLM 호출 경로에 트레이싱을 붙이는 유일한 진입점. Langfuse가 꺼져 있거나
SDK가 설치되지 않은 환경(오프라인·내부망 미구축)에서도 앱 동작에 영향이
없도록 항상 no-op으로 폴백한다.

활성화 조건(모두 충족):
- LANGFUSE_ENABLED=true
- LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 설정 (셀프호스팅 스택에서 발급)

계측 규약 (langfuse/skills instrumentation 참조):
- 함수 인자를 통째로 기록하지 않는다(capture_input=False) — 프롬프트 등
  필요한 값만 record_generation으로 명시 설정한다(설정값·시크릿 유출 방지).
- 관측 타입을 구체적으로 지정한다: LLM 호출 generation, 검색 retriever.
"""

import logging
import os
from contextlib import nullcontext
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def langfuse_enabled() -> bool:
    """환경변수 기준으로 Langfuse 트레이싱 활성 여부를 판정한다."""
    if os.getenv("LANGFUSE_ENABLED", "false").lower() != "true":
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))


def observe_if_enabled(
    name: str,
    as_type: str = "span",
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[Callable], Callable]:
    """langfuse.observe 데코레이터를 조건부로 적용한다.

    Args:
        name: Langfuse에 기록될 observation 이름
        as_type: observation 타입 (span/generation/retriever/agent/tool/event)
        capture_input: 함수 인자 자동 수집 여부. 시크릿·설정 객체가 인자로
            들어올 수 있는 메서드는 False로 두고 record_generation 등으로 명시 설정
    """
    if not langfuse_enabled():
        return lambda fn: fn
    try:
        from langfuse import observe
    except ImportError:
        logger.debug("langfuse SDK 미설치 — 트레이싱 비활성화")
        return lambda fn: fn
    return observe(name=name, as_type=as_type, capture_input=capture_input, capture_output=capture_output)


def _patch_current_observation(kind: str, **kwargs) -> None:
    """현재 활성 observation을 갱신한다. 비활성/미설치/활성 observation 없음 모두 무시.

    langfuse 4.x 클라이언트는 update_current_generation/update_current_span만
    제공하므로 kind로 메서드를 결정한다(미래 버전의 update_current_observation도 폴백 지원).
    """
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    if not kwargs or not langfuse_enabled():
        return
    try:
        from langfuse import get_client

        client = get_client()
        method = getattr(client, f"update_current_{kind}", None) or getattr(client, "update_current_observation", None)
        if method is not None:
            method(**kwargs)
    except Exception as e:
        logger.debug(f"Langfuse observation 갱신 실패(무시): {type(e).__name__}: {e}")


def record_generation(
    model: str,
    input: Optional[dict] = None,
    usage_details: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> None:
    """현재 generation observation에 모델명·입력·토큰 사용량을 기록한다.

    provider별 응답 파싱 위치에서 호출한다. Langfuse가 자동 비용 계산을 하려면
    model과 usage_details(input/output 토큰 수)가 필요하다.
    """
    _patch_current_observation("generation", model=model, input=input, usage_details=usage_details, metadata=metadata)


def record_retriever_output(sources: list[dict]) -> None:
    """현재 retriever observation에 검색 결과 요약(출처·점수)을 기록한다."""
    _patch_current_observation("span", output=sources)


def record_input(input: dict) -> None:
    """현재 observation의 입력을 명시 설정한다 (인자 자동 수집 대신 필요한 값만)."""
    _patch_current_observation("span", input=input)


def langfuse_callbacks() -> list:
    """LangChain 체인용 Langfuse CallbackHandler 목록을 반환한다. 비활성 시 빈 목록.

    LangChain LLM 체인(prompt | llm) 호출을 자동으로 generation 관측(모델명·토큰
    사용량 포함)으로 기록하며, 현재 트레이스 컨텍스트에 중첩된다.
    """
    if not langfuse_enabled():
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except ImportError:
        logger.debug("langfuse SDK 미설치 — LangChain 콜백 비활성화")
        return []


def propagate_trace_attributes(session_id: Optional[str] = None, user_id: Optional[str] = None):
    """현재 트레이스에 세션·사용자 속성을 전파하는 컨텍스트 매니저.

    비활성 환경에서는 아무 일도 하지 않는 null 컨텍스트를 반환한다.
    """
    if not langfuse_enabled() or (session_id is None and user_id is None):
        return nullcontext()
    try:
        from langfuse import propagate_attributes

        attrs = {}
        if session_id is not None:
            attrs["session_id"] = session_id
        if user_id is not None:
            attrs["user_id"] = user_id
        return propagate_attributes(**attrs)
    except ImportError:
        logger.debug("langfuse SDK 미설치 — 트레이싱 비활성화")
        return nullcontext()
