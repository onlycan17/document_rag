"""기본 모델 설정·레지스트리 일관성과 OpenRouter 비전 요청 형식 테스트 (외부 API 호출 없음)"""

import pytest

from config import settings
from src.models.model_registry import ModelRegistry


@pytest.mark.parametrize("provider", ["openai", "google", "anthropic", "openrouter"])
def test_default_model_is_registered(provider):
    """기본 모델이 레지스트리에 없으면 토큰 한도가 엉뚱한 기본값으로 계산되고 UI 목록에서도 빠진다"""
    assert ModelRegistry.get_model_config(settings.model_for(provider)) is not None


def test_model_for_falls_back_to_multimodal_model_when_text_model_empty(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_model", "")
    monkeypatch.setattr(settings, "openrouter_mm_model", "mm-model")
    assert settings.model_for("openrouter") == "mm-model"


def test_available_models_come_from_registry():
    from src.rag.llm_manager import LLMManager

    models = LLMManager.get_available_models(object.__new__(LLMManager))
    assert models == ModelRegistry.get_all_models()
    assert settings.openai_model in [m["model"] for m in models["openai"]]


def test_image_analysis_sends_image_as_image_url_with_reasoning_off(monkeypatch, tmp_path):
    from PIL import Image

    from src.utils.openrouter_image_service import OpenRouterImageService

    image_path = tmp_path / "figure.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    captured = {}

    def fake_post(self, body):
        captured.update(body)
        return {"choices": [{"message": {"content": "RELEVANCE: 0.9\nDESCRIPTION: 도면\nTEXT: 1호"}}]}

    monkeypatch.setattr(OpenRouterImageService, "_post_chat_completions", fake_post)
    service = object.__new__(OpenRouterImageService)
    result = service.analyze_image(str(image_path))

    image_parts = [p for p in captured["messages"][1]["content"] if p["type"] != "text"]
    assert image_parts and all(p["type"] == "image_url" for p in image_parts)
    assert captured["reasoning"] == {"enabled": False}
    assert result["relevance"] == 0.9


def test_multimodal_preprocessing_retries_on_rate_limit(monkeypatch):
    import src.embeddings.embedding_model as embedding_module
    import src.processing.multimodal_preprocessing_model as module

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code, self._payload, self.text = status_code, payload, str(payload)

        def json(self):
            return self._payload

    responses = [FakeResponse(429, {"error": "rate limit"}), FakeResponse(200, {"choices": []})]
    sent = []
    monkeypatch.setattr(
        module.requests, "post", lambda url, json, headers, timeout: sent.append(json) or responses.pop(0)
    )
    monkeypatch.setattr(embedding_module.time, "sleep", lambda seconds: None)

    assert module._post_openrouter("https://example.com", {"model": "m"}, {}) == {"choices": []}
    assert len(sent) == 2
