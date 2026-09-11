"""Langfuse 관측 헬퍼.

LLM 호출 경로에 트레이싱을 붙이는 유일한 진입점. Langfuse가 꺼져 있거나
SDK가 설치되지 않은 환경(오프라인·내부망 미구축)에서도 앱 동작에 영향이
없도록 항상 no-op으로 폴백한다.

활성화 조건(모두 충족):
- LANGFUSE_ENABLED=true
- LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 설정 (셀프호스팅 스택에서 발급)
"""

import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)


def langfuse_enabled() -> bool:
    """환경변수 기준으로 Langfuse 트레이싱 활성 여부를 판정한다."""
    if os.getenv("LANGFUSE_ENABLED", "false").lower() != "true":
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))


def observe_if_enabled(name: str) -> Callable[[Callable], Callable]:
    """langfuse.observe 데코레이터를 조건부로 적용한다.

    Args:
        name: Langfuse에 기록될 span 이름
    """
    if not langfuse_enabled():
        return lambda fn: fn
    try:
        from langfuse import observe
    except ImportError:
        logger.debug("langfuse SDK 미설치 — 트레이싱 비활성화")
        return lambda fn: fn
    return observe(name=name)
