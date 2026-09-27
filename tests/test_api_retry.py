"""api_retry_with_backoff 재시도 대상 검증 — 일시적 서버 오류(5xx)는 재시도, 요청 오류(4xx)는 즉시 실패"""

import pytest
import requests

import src.embeddings.embedding_model as embedding_module
from src.embeddings.embedding_model import api_retry_with_backoff


def _http_error(status: int, reason: str) -> requests.HTTPError:
    kind = "Server" if status >= 500 else "Client"
    return requests.HTTPError(f"{status} {kind} Error: {reason} for url: https://api.example.com")


def _call_with_failures(error: Exception, failures: int) -> list:
    calls = []

    @api_retry_with_backoff(max_retries=3, base_delay=0, max_delay=0)
    def flaky():
        calls.append(1)
        if len(calls) <= failures:
            raise error
        return "ok"

    flaky()
    return calls


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(embedding_module.time, "sleep", lambda seconds: None)


@pytest.mark.parametrize(
    "status,reason", [(500, "Internal Server Error"), (502, "Bad Gateway"), (503, "Service Unavailable")]
)
def test_server_errors_are_retried(status, reason):
    assert len(_call_with_failures(_http_error(status, reason), failures=2)) == 3


def test_client_error_is_not_retried():
    with pytest.raises(requests.HTTPError):
        _call_with_failures(_http_error(400, "Bad Request"), failures=1)
