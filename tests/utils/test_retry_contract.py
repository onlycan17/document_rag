#!/usr/bin/env python3
"""llm_retry_with_backoff 재시도 계약 검증 — 설정 오류는 재시도하지 않는다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from src.agents.base_agent import llm_retry_with_backoff


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr("src.agents.base_agent.time.sleep", lambda _seconds: None)


def test_value_error_is_not_retried(no_sleep, monkeypatch):
    monkeypatch.setattr("src.agents.base_agent.settings.api_max_retries", 3, raising=False)
    calls = []

    @llm_retry_with_backoff()
    def missing_key():
        calls.append(1)
        raise ValueError("API 키가 설정되지 않았습니다.")

    with pytest.raises(ValueError):
        missing_key()

    assert len(calls) == 1


def test_runtime_error_is_retried_until_exhausted(no_sleep, monkeypatch):
    monkeypatch.setattr("src.agents.base_agent.settings.api_max_retries", 2, raising=False)
    monkeypatch.setattr("src.agents.base_agent.settings.api_base_delay", 0.0, raising=False)
    calls = []

    @llm_retry_with_backoff()
    def api_down():
        calls.append(1)
        raise RuntimeError("네트워크 오류")

    with pytest.raises(RuntimeError):
        api_down()

    assert len(calls) == 3
