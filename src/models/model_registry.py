"""LLM 모델 정보 중앙 관리"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """모델 설정 정보"""
    name: str
    provider: str
    max_tokens: int
    context_window: int
    description: str
    model_id: str


class ModelRegistry:
    """LLM 모델 정보를 중앙에서 관리하는 레지스트리"""
    
    # 모델별 설정 정보
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        # OpenAI 모델       
        "gpt-4o": {
            "provider": "openai",
            "max_tokens": 4096,
            "context_window": 128000,
            "description": "최신 옴니 모델"
        },
        "gpt-4o-mini": {
            "provider": "openai",
            "max_tokens": 16384,
            "context_window": 128000,
            "description": "가벼운 옴니 모델"
        },
        "gpt-4.1": {
            "provider": "openai",
            "max_tokens": 8192,
            "context_window": 1000000,
            "description": "코딩 특화 1M 토큰"
        },
        "gpt-4.1-mini": {
            "provider": "openai",
            "max_tokens": 8192,
            "context_window": 1000000,
            "description": "균형잡힌 1M 토큰"
        },
        "gpt-4.1-nano": {
            "provider": "openai",
            "max_tokens": 4096,
            "context_window": 1000000,
            "description": "빠르고 저렴한 1M 토큰"
        },
        "gpt-5-pro": {
            "provider": "openai",
            "max_tokens": 4096,
            "context_window": 128000,
            "description": "고성능 모델"
        },
        "gpt-5": {
            "provider": "openai",
            "max_tokens": 4096,
            "context_window": 128000,
            "description": "고성능 모델"
        },
        "gpt-5-lite": {
            "provider": "openai",
            "max_tokens": 8192,
            "context_window": 65536,
            "description": "경량 모델"
        },
        "gpt-5-mini": {
            "provider": "openai",
            "max_tokens": 16384,
            "context_window": 128000,
            "description": "가벼운 옴니 모델"
        },
        
        # Google Gemini 모델
        "gemini-1.5-flash": {
            "provider": "google",
            "max_tokens": 8192,
            "context_window": 1048576,
            "description": "빠른 응답"
        },
        "gemini-1.5-flash-8b": {
            "provider": "google",
            "max_tokens": 8192,
            "context_window": 1048576,
            "description": "더 빠른 경량 모델"
        },
        "gemini-1.5-pro": {
            "provider": "google",
            "max_tokens": 8192,
            "context_window": 2097152,
            "description": "고급 기능"
        },
        "gemini-2.0-flash": {
            "provider": "google",
            "max_tokens": 8192,
            "context_window": 1048576,
            "description": "최신 2.0 버전"
        },
        "gemini-2.5-flash": {
            "provider": "google",
            "max_tokens": 8192,
            "context_window": 1048576,
            "description": "최신 2.5 버전"
        },
        "gemini-2.5-pro": {
            "provider": "google",
            "max_tokens": 2048,
            "context_window": 32768,
            "description": "안정적인 버전"
        },
        "gemini-1.0-pro": {
            "provider": "google",
            "max_tokens": 2048,
            "context_window": 32768,
            "description": "안정적인 버전"
        },
        "gemini-2.0-ultra": {
            "provider": "google",
            "max_tokens": 8192,
            "context_window": 1048576,
            "description": "고성능 멀티모달"
        },
        
        # Anthropic Claude 모델
        "claude-3-5-sonnet-20241022": {
            "provider": "anthropic",
            "max_tokens": 8192,
            "context_window": 200000,
            "description": "최신 최고 성능"
        },
        "claude-3-haiku-20240307": {
            "provider": "anthropic",
            "max_tokens": 4096,
            "context_window": 200000,
            "description": "빠르고 효율적"
        },
        "claude-3-sonnet-20240229": {
            "provider": "anthropic",
            "max_tokens": 4096,
            "context_window": 200000,
            "description": "균형잡힌 성능"
        },
        "claude-3-opus-20240229": {
            "provider": "anthropic",
            "max_tokens": 4096,
            "context_window": 200000,
            "description": "최고 성능"
        },
        "claude-2.1": {
            "provider": "anthropic",
            "max_tokens": 4096,
            "context_window": 200000,
            "description": "Claude 2.1"
        },
        "claude-2.0": {
            "provider": "anthropic",
            "max_tokens": 4096,
            "context_window": 100000,
            "description": "Claude 2.0"
        },
        "claude-instant-1.2": {
            "provider": "anthropic",
            "max_tokens": 4096,
            "context_window": 100000,
            "description": "빠른 응답"
        },
        "claude-4-1-opus-20250901": {
            "provider": "anthropic",
            "max_tokens": 8192,
            "context_window": 200000,
            "description": "최신 Claude 4 Opus"
        },
        "claude-4-sonnet": {
            "provider": "anthropic",
            "max_tokens": 8192,
            "context_window": 200000,
            "description": "Claude 4 Sonnet"
        },
        "claude-4-1-haiku-20250901": {
            "provider": "anthropic",
            "max_tokens": 8192,
            "context_window": 200000,
            "description": "Claude 4 Haiku"
        },
        
        # 로컬 모델 (동적 설정)
        "local-model": {
            "provider": "local",
            "max_tokens": 512,  # 기본값, 설정에서 오버라이드 가능
            "context_window": 4096,  # 기본값, 설정에서 오버라이드 가능
            "description": "로컬 LLM"
        },
        # 로컬: Midm-2.0 GGUF (laama.cpp)
        "midm-2.0-gguf": {
            "provider": "local",
            "max_tokens": 512,
            "context_window": 4096,
            "description": "Midm-2.0-Base-Instruct (GGUF, llama.cpp)"
        },
        # 로컬: EXAONE 4.0 32B (Transformers)
        "exaone-4.0-32b": {
            "provider": "local",
            "max_tokens": 1024,
            "context_window": 32768,
            "description": "LGAI-EXAONE/EXAONE-4.0-32B (Transformers)"
        },
        # 로컬: Gemma 3n GGUF (llama.cpp)
        "gemma-3n-gguf": {
            "provider": "local",
            "max_tokens": 1024,
            "context_window": 4096,
            "description": "Gemma-3n-E4B-it (GGUF, llama.cpp)"
        }
    }
    
    @classmethod
    def get_model_config(cls, model_id: str) -> Optional[ModelConfig]:
        """
        모델 ID로 설정 정보 조회
        
        Args:
            model_id: 모델 ID
            
        Returns:
            ModelConfig 객체 또는 None
        """
        if model_id not in cls.MODEL_CONFIGS:
            logger.warning(f"Unknown model ID: {model_id}")
            return None
        
        config = cls.MODEL_CONFIGS[model_id]
        return ModelConfig(
            name=model_id.replace("-", " ").title(),
            provider=config["provider"],
            max_tokens=config["max_tokens"],
            context_window=config["context_window"],
            description=config["description"],
            model_id=model_id
        )
    
    @classmethod
    def get_models_by_provider(cls, provider: str) -> List[ModelConfig]:
        """
        제공자별 모델 목록 조회
        
        Args:
            provider: LLM 제공자 (openai, google, anthropic, local)
            
        Returns:
            ModelConfig 리스트
        """
        models = []
        for model_id, config in cls.MODEL_CONFIGS.items():
            if config["provider"] == provider:
                models.append(cls.get_model_config(model_id))
        return [m for m in models if m is not None]
    
    @classmethod
    def get_max_tokens(cls, model_id: str, default: int = 4096) -> int:
        """모델의 최대 토큰 수 조회"""
        config = cls.get_model_config(model_id)
        return config.max_tokens if config else default
    
    @classmethod
    def get_context_window(cls, model_id: str, default: int = 8192) -> int:
        """모델의 컨텍스트 윈도우 크기 조회"""
        config = cls.get_model_config(model_id)
        return config.context_window if config else default
    
    @classmethod
    def format_context_size(cls, tokens: int) -> str:
        """토큰 수를 읽기 쉬운 형태로 변환"""
        if tokens >= 1000000:
            return f"{tokens // 1000000}M"
        elif tokens >= 1000:
            return f"{tokens // 1000}K"
        else:
            return str(tokens)
    
    @classmethod
    def update_local_model_config(cls, max_tokens: int, context_window: int):
        """로컬 모델 설정 업데이트"""
        cls.MODEL_CONFIGS["local-model"]["max_tokens"] = max_tokens
        cls.MODEL_CONFIGS["local-model"]["context_window"] = context_window
        logger.info(f"Local model config updated: max_tokens={max_tokens}, context_window={context_window}")
    
    @classmethod
    def get_all_models(cls) -> Dict[str, List[Dict[str, str]]]:
        """UI용 전체 모델 목록 조회"""
        provider_names = {
            "openai": "OpenAI",
            "google": "Google Gemini",
            "anthropic": "Anthropic Claude",
            "local": "로컬 LLM"
        }
        
        result = {}
        for provider_key, provider_name in provider_names.items():
            models = cls.get_models_by_provider(provider_key)
            result[provider_key] = []
            
            for model in models:
                result[provider_key].append({
                    "name": model.name,
                    "model": model.model_id,
                    "description": f"{model.description} ({cls.format_context_size(model.context_window)} 컨텍스트, 최대 {model.max_tokens}토큰)"
                })
        # LM Studio(로컬 모델 탐색)가 활성화되어 있으면 해당 결과를 병합
        try:
            from src.models.lm_studio import list_lm_studio_models
            lm_models = list_lm_studio_models()
            local_models = lm_models.get('local', [])
            if local_models:
                # LM Studio에서 제공하는 모델을 우선적으로 'local'에 추가
                lm_entries = []
                for m in local_models:
                    lm_entries.append({
                        'name': m.get('name') or m.get('id'),
                        'model': m.get('id') or m.get('name'),
                        'description': m.get('description') or ''
                    })
                # 기존 로컬 모델 리스트 앞에 위치시키기
                result.setdefault('local', [])
                result['local'] = lm_entries + result.get('local', [])
        except Exception:
            # LM Studio 통합 실패 시 무시
            pass

        return result