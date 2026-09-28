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


def test_record_input_merges_inputs_without_add_inputs_method(monkeypatch):
    """langsmith 0.2.x RunTree에는 add_inputs가 없다 — 입력 병합과 이후 출력 기록이 모두 되어야 한다"""
    import src.utils.tracing as tracing

    class RunTreeWithoutAddInputs:
        def __init__(self):
            self.inputs, self.outputs, self.metadata = {"question": "q"}, {}, {}

        def add_metadata(self, metadata):
            self.metadata.update(metadata)

        def add_outputs(self, outputs):
            self.outputs.update(outputs)

    run = RunTreeWithoutAddInputs()
    monkeypatch.setattr(tracing, "_current_run_tree", lambda: run)

    tracing._patch_current_run(input={"top_k": 12}, output={"answer": "a"}, metadata={"cache_hit": False})

    assert run.inputs == {"question": "q", "top_k": 12}
    assert run.outputs == {"answer": "a"} and run.metadata == {"cache_hit": False}


def test_unregistered_openrouter_model_uses_openrouter_context_length(monkeypatch):
    import src.rag.llm_manager as llm_module
    from src.rag.llm_manager import LLMManager

    manager = object.__new__(LLMManager)
    monkeypatch.setattr(
        llm_module, "get_openrouter_context_length", lambda model_id: {"upstage/solar-pro4": 524288}.get(model_id)
    )

    assert manager.get_model_context_window("upstage/solar-pro4") == 524288
    assert manager.get_model_context_window("vendor/unknown-model") == 8192  # 조회 실패 시 기본값
    assert manager.get_model_context_window("qwen/qwen3.8-flash") == ModelRegistry.get_context_window(
        "qwen/qwen3.8-flash"
    )


def test_openrouter_context_lengths_are_cached_from_model_list(monkeypatch):
    import src.utils.openrouter_models as openrouter_models

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"data": [{"id": "upstage/solar-pro4", "context_length": 524288}, {"id": "a/b"}]}

    calls = []
    monkeypatch.setattr(openrouter_models, "_cache", {"models": [], "contexts": {}, "ts": 0.0})
    monkeypatch.setattr(openrouter_models.requests, "get", lambda *a, **k: calls.append(1) or FakeResponse())

    assert openrouter_models.get_openrouter_context_length("upstage/solar-pro4") == 524288
    assert openrouter_models.get_openrouter_context_length("a/b") is None
    assert len(calls) == 1  # 캐시 재사용
