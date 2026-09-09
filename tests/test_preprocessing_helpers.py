"""preprocessing_model 순수 헬퍼 단위 테스트.

LLM 호출 없이 프롬프트 조립·응답 파싱 계약만 고정한다.
"""

from types import SimpleNamespace

import pytest

from config import settings
from src.processing.preprocessing_model import (
    build_openrouter_body,
    build_openrouter_url,
    build_preprocessing_prompt,
    extract_chat_completion_text,
    extract_responses_text,
    openrouter_headers,
    resolve_openrouter_model,
)


def test_프롬프트에원문과지시포함():
    prompt = build_preprocessing_prompt("원본 텍스트입니다")

    assert "원본 텍스트입니다" in prompt
    assert "한국어 문장의 연결성" in prompt and "전처리된 텍스트만 반환해주세요" in prompt


def test_responses응답추출_세경로():
    assert extract_responses_text(SimpleNamespace(output_text="편의속성")) == "편의속성"

    nested = SimpleNamespace(output=[SimpleNamespace(content=[SimpleNamespace(text="구조추출")])])
    assert extract_responses_text(nested) == "구조추출"

    assert extract_responses_text(SimpleNamespace()) is None


def test_openrouter모델해석_우선순위(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_model", "text-model", raising=False)
    assert resolve_openrouter_model() == "text-model"

    monkeypatch.setattr(settings, "openrouter_model", None, raising=False)
    monkeypatch.setattr(settings, "openrouter_mm_model", "mm-model", raising=False)
    assert resolve_openrouter_model() == "mm-model"


def test_openrouter요청구성_슬래시정리(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_base", "https://example.com/api/", raising=False)
    assert build_openrouter_url() == "https://example.com/api/v1/chat/completions"

    headers = openrouter_headers("secret-key")
    assert headers["Authorization"] == "Bearer secret-key" and headers["X-Title"] == "RAG-Preprocessor"


def test_요청본_온도실수변환(monkeypatch):
    monkeypatch.setattr(settings, "preprocessing_temperature", "0.7", raising=False)

    body = build_openrouter_body("z-ai/glm-4.5v", "프롬프트")

    assert body["model"] == "z-ai/glm-4.5v" and body["max_tokens"] == 4000
    assert body["temperature"] == pytest.approx(0.7)


def test_챗응답파싱_정상과결손():
    data = {"choices": [{"message": {"content": "  답변  "}}]}
    assert extract_chat_completion_text(data) == "답변"

    assert extract_chat_completion_text({}) == ""
    assert extract_chat_completion_text({"choices": [{}]}) == ""
