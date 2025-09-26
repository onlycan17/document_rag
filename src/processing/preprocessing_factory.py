"""
전처리 모델 팩토리

사용자가 선택한 모델 유형에 따라 적절한 전처리 모델을 생성합니다.
"""

from typing import Optional, Dict, Any, List
import logging

from .preprocessing_model import PreprocessingModel, LocalPreprocessingModel, APIPreprocessingModel
from .multimodal_preprocessing_model import MultimodalPreprocessingModel
from config import settings

logger = logging.getLogger(__name__)


class PreprocessingModelFactory:
    """전처리 모델 생성 팩토리"""
    
    # 지원하는 모델 제공자
    SUPPORTED_PROVIDERS = {
        "local": "로컬 모델",
        "openai": "OpenAI GPT",
        "google": "Google Gemini", 
        "anthropic": "Anthropic Claude",
        "openrouter": "OpenRouter (OpenAI 호환)"
    }
    
    # 기본 모델 설정
    DEFAULT_MODELS = {
        "local": "local_default",
        # OpenAI: 저비용 멀티모달 기본값
        "openai": "gpt-4o-mini",
        # Google: 저비용 멀티모달 기본값
        "google": "gemini-1.5-flash-8b",
        # Anthropic: 저비용 멀티모달 기본값
        "anthropic": "claude-3-5-haiku-20241022",
        # OpenRouter: 텍스트 전처리용 모델(없으면 멀티모달 기본값으로 폴백)
        "openrouter": getattr(settings, 'openrouter_model', None) or getattr(settings, 'openrouter_mm_model', 'z-ai/glm-4.5v'),
    }
    
    @classmethod
    def create_model(cls, model_type: str, model_name: Optional[str] = None, **kwargs) -> PreprocessingModel:
        """
        전처리 모델을 생성합니다.
        
        Args:
            model_type: 모델 유형 ('local', 'openai', 'google', 'anthropic', 'openrouter')
            model_name: 특정 모델 이름 (선택사항)
            **kwargs: 추가 매개변수
            
        Returns:
            생성된 전처리 모델 인스턴스
            
        Raises:
            ValueError: 지원하지 않는 모델 유형인 경우
        """
        if model_type not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"지원하지 않는 모델 유형: {model_type}. "
                f"지원하는 유형: {list(cls.SUPPORTED_PROVIDERS.keys())}"
            )
        
        # 기본 모델 이름 사용
        if not model_name:
            model_name = cls.DEFAULT_MODELS.get(model_type, "default")
        
        logger.info(f"전처리 모델 생성: {model_type} - {model_name}")
        
        try:
            if model_type == "local":
                return LocalPreprocessingModel(model_name)
            else:
                # 멀티모달 모델인지 확인
                if cls._is_multimodal_model(model_name, model_type):
                    logger.info(f"멀티모달 모델 감지: {model_name}")
                    return MultimodalPreprocessingModel(model_name, provider=model_type)
                else:
                    return APIPreprocessingModel(model_name, provider=model_type)
                
        except Exception as e:
            logger.error(f"전처리 모델 생성 실패: {e}")
            raise
    
    @classmethod
    def create_from_config(cls, config_key: str = "preprocessing_model", **kwargs) -> PreprocessingModel:
        """
        설정에서 모델 유형을 읽어 전처리 모델을 생성합니다.
        
        Args:
            config_key: 설정에서 읽을 키 이름
            **kwargs: 추가 매개변수
            
        Returns:
            생성된 전처리 모델 인스턴스
        """
        # 설정에서 모델 유형 읽기
        model_type = getattr(settings, config_key, "local")
        
        # 추가 설정 읽기
        if model_type == "openai":
            model_name = settings.openai_model
        elif model_type == "google":
            model_name = settings.google_model
        elif model_type == "anthropic":
            model_name = settings.anthropic_model
        elif model_type == "openrouter":
            model_name = getattr(settings, 'openrouter_model', None) or getattr(settings, 'openrouter_mm_model', None)
        else:
            model_name = None
        
        return cls.create_model(model_type, model_name, **kwargs)
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        사용 가능한 모델 목록을 반환합니다.
        
        Returns:
            모델 제공자별 정보를 포함한 딕셔너리
        """
        available_models = {}
        
        for provider, description in cls.SUPPORTED_PROVIDERS.items():
            model_info = {
                "description": description,
                "default_model": cls.DEFAULT_MODELS.get(provider, "default"),
                "available": False,
                "api_key_required": provider != "local"
            }
            
            # API 키 확인
            if provider != "local":
                if provider == "openai" and settings.openai_api_key:
                    model_info["available"] = True
                elif provider == "google" and settings.google_api_key:
                    model_info["available"] = True
                elif provider == "anthropic" and settings.anthropic_api_key:
                    model_info["available"] = True
                elif provider == "openrouter" and getattr(settings, 'openrouter_api_key', None):
                    model_info["available"] = True
            else:
                model_info["available"] = True
            
            available_models[provider] = model_info
        
        return available_models
    
    @classmethod
    def check_model_availability(cls, model_type: str) -> bool:
        """
        특정 모델의 사용 가능 여부를 확인합니다.
        
        Args:
            model_type: 확인할 모델 유형
            
        Returns:
            사용 가능 여부
        """
        if model_type not in cls.SUPPORTED_PROVIDERS:
            return False
        
        if model_type == "local":
            return True
        
        # API 키 확인
        if model_type == "openai":
            return bool(settings.openai_api_key)
        elif model_type == "google":
            return bool(settings.google_api_key)
        elif model_type == "anthropic":
            return bool(settings.anthropic_api_key)
        elif model_type == "openrouter":
            return bool(getattr(settings, 'openrouter_api_key', None))
        
        return False
    
    @classmethod
    def get_model_requirements(cls, model_type: str) -> Dict[str, Any]:
        """
        모델 사용에 필요한 요구사항을 반환합니다.
        
        Args:
            model_type: 모델 유형
            
        Returns:
            요구사항 정보
        """
        if model_type not in cls.SUPPORTED_PROVIDERS:
            return {"error": "지원하지 않는 모델 유형"}
        
        requirements = {
            "model_type": model_type,
            "description": cls.SUPPORTED_PROVIDERS[model_type],
            "default_model": cls.DEFAULT_MODELS.get(model_type, "default"),
            "api_key_required": model_type != "local"
        }
        
        if model_type != "local":
            if model_type == "openai":
                requirements["api_key_name"] = "OPENAI_API_KEY"
                requirements["api_key_set"] = bool(settings.openai_api_key)
            elif model_type == "google":
                requirements["api_key_name"] = "GOOGLE_API_KEY"
                requirements["api_key_set"] = bool(settings.google_api_key)
            elif model_type == "anthropic":
                requirements["api_key_name"] = "ANTHROPIC_API_KEY"
                requirements["api_key_set"] = bool(settings.anthropic_api_key)
            elif model_type == "openrouter":
                # 주의: 사양상 철자 고정 (OPNEROUTER_API_KEY)
                requirements["api_key_name"] = "OPNEROUTER_API_KEY"
                requirements["api_key_set"] = bool(getattr(settings, 'openrouter_api_key', None))
        
        return requirements
    
    @classmethod
    def _is_multimodal_model(cls, model_name: str, provider: str) -> bool:
        """
        모델이 멀티모달을 지원하는지 확인합니다.
        
        Args:
            model_name: 모델 이름
            provider: 제공자
            
        Returns:
            멀티모달 지원 여부
        """
        # MultimodalPreprocessingModel의 지원 모델 목록 참조
        from .multimodal_preprocessing_model import MultimodalPreprocessingModel
        supported_models = MultimodalPreprocessingModel.SUPPORTED_MULTIMODAL_MODELS.get(provider, [])
        return model_name in supported_models
    
    @classmethod
    def get_multimodal_models(cls) -> Dict[str, List[str]]:
        """
        사용 가능한 멀티모달 모델 목록을 반환합니다.
        
        Returns:
            제공자별 멀티모달 모델 리스트
        """
        from .multimodal_preprocessing_model import MultimodalPreprocessingModel
        return MultimodalPreprocessingModel.get_supported_multimodal_models()
