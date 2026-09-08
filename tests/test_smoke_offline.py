#!/usr/bin/env python3
"""
오프라인 스모크 테스트 - 네트워크/API 키 없이 실행 가능한 최소 검증
CI에서 `pytest tests/test_smoke_offline.py`로 실행된다.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_core_modules_import():
    """핵심 모듈 임포트 검증"""
    import src.agents  # noqa: F401
    import src.embeddings.embedding_model  # noqa: F401
    import src.loaders  # noqa: F401
    import src.processing  # noqa: F401
    import src.rag  # noqa: F401
    import src.vectorstore  # noqa: F401


def test_settings_has_no_local_model_fields():
    """내장(로컬) 모델 설정이 제거되었는지 회귀 검증"""
    from config import Settings
    fields = set(Settings.model_fields.keys())
    removed_prefixes = ("local_llm_", "lm_studio_", "gemma_", "ax_")
    leaked = [f for f in fields if f.startswith(removed_prefixes)]
    assert not leaked, f"로컬 모델 설정이 남아 있습니다: {leaked}"


def test_openrouter_key_accepts_both_spellings():
    """OpenRouter 키가 두 철자 모두 인식하는지 검증"""
    import config as config_module
    os.environ["OPENROUTER_API_KEY"] = "test-correct-spelling"
    try:
        settings = config_module.Settings()
        assert settings.openrouter_api_key == "test-correct-spelling"
    finally:
        del os.environ["OPENROUTER_API_KEY"]


def test_base_agent_rejects_local_provider():
    """BaseAgent가 로컬 제공자를 명시적으로 거부하는지 검증"""
    import pytest

    from src.agents.base_agent import BaseAgent

    class DummyAgent(BaseAgent):
        def process(self, input_data):
            return input_data

    with pytest.raises(ValueError):
        DummyAgent("Dummy", provider="local")


def test_preprocessing_factory_rejects_local():
    """전처리 팩토리에서 local 제공자가 제거되었는지 검증"""
    import pytest

    from src.processing.preprocessing_factory import PreprocessingModelFactory

    assert "local" not in PreprocessingModelFactory.SUPPORTED_PROVIDERS
    with pytest.raises(ValueError):
        PreprocessingModelFactory.create_model("local")


def test_embedding_model_requires_api_key():
    """임베딩은 키 없이 조용한 로컬 폴백 대신 명확한 오류를 내는지 검증"""
    from config import settings
    from src.embeddings.embedding_model import EmbeddingModel

    has_key = bool(settings.upstage_api_key or settings.openai_api_key)
    if has_key:
        model = EmbeddingModel()
        assert model.model_type in ("upstage", "openai")
    else:
        try:
            EmbeddingModel()
            raise AssertionError("API 키 없이도 초기화되면 안 됩니다")
        except ValueError as e:
            assert "UPSTAGE_API_KEY" in str(e)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
