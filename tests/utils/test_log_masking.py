#!/usr/bin/env python3
"""로그 민감정보 마스킹 검증."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import SecretMaskingFormatter, mask_secrets


def test_masks_known_env_key_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-1234567890abcdef")
    masked = mask_secrets("호출 실패: 키 sk-test-1234567890abcdef 사용 중")
    assert "sk-test-1234567890abcdef" not in masked
    assert "***" in masked


def test_masks_bearer_token_pattern(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    masked = mask_secrets("Authorization: Bearer abc123def456ghi789 전송")
    assert "abc123def456ghi789" not in masked


def test_masks_google_key_pattern(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    google_key = "AIza" + "SyA1234567890abcdefghijklmnopqrstuv"
    masked = mask_secrets(f"구글 키 {google_key} 노출")
    assert google_key not in masked


def test_normal_text_untouched(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    text = "문서 12개 처리 완료, 검색 0.32초"
    assert mask_secrets(text) == text


def test_short_values_not_masked(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "short")
    assert mask_secrets("값은 short 입니다") == "값은 short 입니다"


def test_formatter_masks_formatted_output(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-abcdefgh12345678")
    formatter = SecretMaskingFormatter("%(levelname)s - %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="OpenRouter 오류: 401 - 키 sk-or-v1-abcdefgh12345678",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert "sk-or-v1-abcdefgh12345678" not in output
    assert "***" in output
