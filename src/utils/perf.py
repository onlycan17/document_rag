from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def time_block(name: str) -> Iterator[float]:
    """
    간단한 성능 측정 컨텍스트 매니저.
    용어(설명: 코드 구간의 시작~끝 시간을 재서 경과 시간을 구함).
    사용 예:
        with time_block("search") as t:
            ...
        # 종료 시점에 t 값이 경과초로 업데이트됨
    """
    start = time.perf_counter()
    elapsed = 0.0
    try:
        yield elapsed
    finally:
        elapsed = time.perf_counter() - start


def now() -> float:
    """고해상도 현재 시간 반환(초)."""
    return time.perf_counter()

