"""
LLM 관리 모듈

이 모듈은 다양한 LLM 제공자(OpenAI, Google, Anthropic, Local)의 초기화와 관리를 담당합니다.
"""

from typing import Optional, Dict, List
from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp, HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnablePassthrough

from config import settings
from src.models import ModelRegistry
from src.constants import (
    MODEL_PROMPT_TOKENS, MODEL_MIN_CONTEXT, MODEL_MAX_CONTEXT,
    MODEL_SAFETY_MARGIN, LOCAL_MODEL_SAFETY_MARGIN, TOKEN_TO_CHAR_RATIO,
    LOCAL_MODEL_TIMEOUT
)
import logging

logger = logging.getLogger(__name__)


class LLMManager:
    """
    LLM 생성 및 관리를 담당하는 클래스
    
    주요 기능:
    - 다양한 LLM 제공자 지원 (OpenAI, Google, Anthropic, Local)
    - 스트리밍/비스트리밍 LLM 생성
    - 모델별 토큰 제한 관리
    - 프롬프트 템플릿 생성
    """
    
    def __init__(self):
        """LLM 매니저 초기화"""
        self.current_provider = None
        self.current_model = None
        self.is_qwen_gguf = False
        self.is_midm_gguf = False
        
    def initialize_llm(self, provider: Optional[str] = None, model: Optional[str] = None):
        """일반 LLM 초기화 (비스트리밍)"""
        return self.create_llm(provider, model, streaming=False)
    
    def initialize_streaming_llm(self, provider: Optional[str] = None, model: Optional[str] = None):
        """스트리밍 LLM 초기화"""
        return self.create_llm(provider, model, streaming=True)
    
    def create_llm(self, provider: Optional[str] = None, model: Optional[str] = None, streaming: bool = False):
        """LLM 생성 (스트리밍/비스트리밍 공통)"""
        # provider가 지정되지 않으면 설정에서 가져옴
        if provider is None:
            provider = settings.llm_provider
        
        # 실제 사용할 모델명 결정
        if provider == "local":
            actual_model = model or settings.local_llm_model
        else:
            actual_model = model or getattr(settings, f"{provider}_model", None)
        
        # 모델별 최대 토큰 수 가져오기 (로컬은 설정값 우선)
        max_tokens = self.get_max_tokens_for_model(provider, actual_model) if actual_model else settings.max_tokens
        if provider == "local":
            try:
                max_tokens = int(getattr(settings, 'local_llm_max_tokens', max_tokens))
            except Exception:
                pass
        
        logger.info(f"LLM 초기화: provider={provider}, model={actual_model}, max_tokens={max_tokens}, streaming={streaming}")
        
        # 스트리밍 콜백 설정
        callbacks = [StreamingStdOutCallbackHandler()] if streaming else []
        
        # 현재 모델 정보 저장
        self.current_provider = provider
        self.current_model = actual_model
            
        if provider == "openai":
            return self._create_openai_llm(actual_model, max_tokens, streaming, callbacks)
        elif provider == "google":
            return self._create_google_llm(actual_model, max_tokens, streaming, callbacks)
        elif provider == "anthropic":
            return self._create_anthropic_llm(actual_model, max_tokens, streaming, callbacks)
        elif provider == "local":
            return self._create_local_llm(actual_model, max_tokens, streaming, callbacks)
        else:
            raise ValueError(f"지원하지 않는 LLM 제공자입니다: {provider}")
    
    def _create_openai_llm(self, model: str, max_tokens: int, streaming: bool, callbacks: List):
        """OpenAI LLM 생성"""
        if not settings.openai_api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        
        return ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model_name=model,
            temperature=settings.temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            callbacks=callbacks
        )
    
    def _create_google_llm(self, model: str, max_tokens: int, streaming: bool, callbacks: List):
        """Google LLM 생성"""
        if not settings.google_api_key:
            raise ValueError("Google API 키가 설정되지 않았습니다.")
        
        return ChatGoogleGenerativeAI(
            google_api_key=settings.google_api_key,
            model=model,
            temperature=settings.temperature,
            max_output_tokens=max_tokens,
            streaming=streaming,
            callbacks=callbacks
        )
    
    def _create_anthropic_llm(self, model: str, max_tokens: int, streaming: bool, callbacks: List):
        """Anthropic LLM 생성"""
        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API 키가 설정되지 않았습니다.")
        
        return ChatAnthropic(
            anthropic_api_key=settings.anthropic_api_key,
            model_name=model,
            temperature=settings.temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            callbacks=callbacks
        )
    
    def _create_local_llm(self, model: str, max_tokens: int, streaming: bool, callbacks: List):
        """로컬 LLM 생성"""
        """로컬 모델: OpenAI 호환 HTTP 엔드포인트 사용.
        - 엔드포인트: {base}/v1
        - 헬스체크: GET {base}/health
        - 큐 상태: GET {base}/v1/queue/stats
        - 모델 목록: GET {base}/v1/models
        - 채팅: POST {base}/v1/chat/completions
        - 임베딩: POST {base}/v1/embeddings
        """
        # 모델에 엔드포인트가 포함된 경우(형식: "<model_id>|<base_url>") 우선 사용
        override_base = None
        if model and isinstance(model, str) and "|" in model:
            try:
                parts = model.split("|", 1)
                model = parts[0]
                override_base = parts[1]
            except Exception:
                override_base = None
        
        # 스킴 보강 및 프리플라이트 점검
        def ensure_scheme(u: str) -> str:
            if not u:
                return u
            u = u.strip()
            if not (u.startswith("http://") or u.startswith("https://")):
                return "http://" + u
            return u
        
        # 우선 순위: 모델에 포함된 override_base -> 설정의 LM Studio API URL -> 기존 local_llm_base_url
        candidate_base = override_base or getattr(settings, 'lm_studio_api_url', None) or settings.local_llm_base_url
        base = ensure_scheme(candidate_base).rstrip("/")
        
        try:
            from src.utils.http_probe import probe_local_server
            probe = probe_local_server(base)
            logger.info(f"로컬 LLM 프리플라이트: base={base} health={probe['health']} models={probe['models']} queue={probe['queue']}")
            
            # 모델 엔드포인트가 응답하지 않으면 1620으로 폴백 시도(멀티모달/기본 서버)
            if not probe.get('models'):
                host = base.split('://', 1)[-1].split(':')[0]
                cand = f"http://{host}:1620"
                probe2 = probe_local_server(cand)
                if probe2.get('models'):
                    logger.warning(f"/v1/models 미응답: {base} → {cand} 폴백")
                    base = cand
        except Exception:
            # 점검 실패 시 그대로 진행
            pass
        
        logger.info(f"로컬 LLM HTTP 엔드포인트 사용: {base} (model={model})")
        logger.info(f"ChatOpenAI(local) 설정: base_url={base}/v1, model={model}")
        
        return ChatOpenAI(
            api_key=settings.local_llm_api_key,
            base_url=f"{base}/v1",
            model=model,
            temperature=settings.temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            callbacks=callbacks,
            timeout=LOCAL_MODEL_TIMEOUT,
            max_retries=getattr(settings, 'local_llm_max_retries', 3),
        )
    
    def create_prompt_template(self) -> PromptTemplate:
        """프롬프트 템플릿 생성 (Qwen GGUF는 ChatML 형식 사용)"""
        if getattr(self, 'is_qwen_gguf', False):
            template = (
                "<|im_start|>system\n"
                "역할: 한국 역사 주제를 한국어로 쉽고 정확하게 설명하는 조력자.\n"
                "원칙: (1) 사실 근거 (2) 간결하고 쉬운 표현 (3) 번호/불릿으로 정리 (4) 한자/전문 용어는 괄호로 풀어쓰기.\n"
                "중요: 아래 원칙이나 지침 문구를 답변에 출력하지 말 것. '초등학생 수준' 등 메타 문구 금지.\n"
                "출력 형식: 질문에 대한 답변 본문만. 도입 멘트(예: '~설명해줄게요')와 예시/지침 제목 출력 금지.\n\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "다음은 검색된 관련 문서들입니다:\n{context}\n\n"
                "위 문서들을 참고하여 다음 질문에 답변하세요.\n질문: {question}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            template = """당신은 국사의 전문적인 지식을 초등학교 학생들도 알기쉽게 친절하게 전달하는 AI 어시스턴트입니다.
상세하고 정확한 정보를 제공하면서도, 초등학생들도 이해할 수 있도록 쉽게 설명하는 것이 당신의 역할입니다.

다음은 검색된 관련 문서들입니다:
{context}

위 문서들의 내용을 참고하여 다음 질문에 답변해주세요:
{question}

답변:"""

        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
    
    def get_model_max_tokens(self, model_id: str) -> int:
        """모델별 최대 토큰 수 반환"""
        return ModelRegistry.get_max_tokens(model_id, default=settings.max_tokens)
    
    def get_model_context_window(self, model_id: str) -> int:
        """모델별 전체 컨텍스트 윈도우 크기 반환 (토큰 단위)"""
        return ModelRegistry.get_context_window(model_id, default=8192)
    
    def get_max_tokens_for_model(self, provider: str, model: str) -> int:
        """특정 모델의 최대 토큰 수 반환"""
        return self.get_model_max_tokens(model)
    
    def get_max_context_length_for_model(self, provider: str = None, model: str = None) -> int:
        """현재 모델의 최대 컨텍스트 길이 계산 (문자 단위)"""
        # 현재 모델 정보 사용
        if not provider:
            provider = self.current_provider
        if not model:
            model = self.current_model or getattr(settings, f"{provider}_model", None)
        
        # 모델의 전체 컨텍스트 윈도우 크기 (토큰)
        total_context_tokens = self.get_model_context_window(model)
        
        # 출력용 토큰 예약
        output_tokens = self.get_max_tokens_for_model(provider, model)
        
        # 프롬프트 템플릿용 토큰 예약
        prompt_tokens = MODEL_PROMPT_TOKENS.get(provider, MODEL_PROMPT_TOKENS["default"])
        
        # 안전 마진
        safety_margin = LOCAL_MODEL_SAFETY_MARGIN if provider == "local" else MODEL_SAFETY_MARGIN
        
        # 사용 가능한 컨텍스트 토큰
        available_context_tokens = int((total_context_tokens - output_tokens - prompt_tokens) * safety_margin)
        
        # 토큰을 문자로 변환
        max_context_chars = available_context_tokens * TOKEN_TO_CHAR_RATIO
        
        # 최소/최대 제한
        min_context = MODEL_MIN_CONTEXT.get(provider, MODEL_MIN_CONTEXT["default"])
        max_context = MODEL_MAX_CONTEXT["default"]
        
        # 모델별 특별 제한
        # 로컬 모델은 설정/레지스트리 기반 계산값을 그대로 활용 (별도 상수 제한 제거)
        if provider == "openai" and model and "gpt-3.5" in model:
            max_context = MODEL_MAX_CONTEXT["gpt-3.5"]
        
        return max(min_context, min(max_context, max_context_chars))
    
    def get_available_models(self) -> Dict[str, List[Dict[str, str]]]:
        """사용 가능한 모델 목록 반환"""
        models = {
            # 2025년 9월 18일 기준 최신 모델 목록
            "openai": [
                {
                    "id": "gpt-5-pro",
                    "name": "GPT-5 Pro",
                    "description": f"최고 성능 모델 (출력: {self.get_model_max_tokens('gpt-5-pro'):,}, 컨텍스트: {self.get_model_context_window('gpt-5-pro'):,}K)"
                },
                {
                    "id": "gpt-5",
                    "name": "GPT-5",
                    "description": f"표준 모델 (출력: {self.get_model_max_tokens('gpt-5'):,}, 컨텍스트: {self.get_model_context_window('gpt-5'):,}K)"
                },
                {
                    "id": "gpt-5-lite",
                    "name": "GPT-5 Lite",
                    "description": f"경량 모델 (출력: {self.get_model_max_tokens('gpt-5-lite'):,}, 컨텍스트: {self.get_model_context_window('gpt-5-lite'):,}K)"
                },
                {
                    "id": "gpt-5-mini",
                    "name": "GPT-5 Mini",
                    "description": f"초경량 모델 (출력: {self.get_model_max_tokens('gpt-5-mini'):,}, 컨텍스트: {self.get_model_context_window('gpt-5-mini'):,}K)"
                }
            ],
            "google": [
                {
                    "id": "gemini-2.0-ultra",
                    "name": "Gemini 2.0 Ultra",
                    "description": f"최고 성능 모델 (출력: {self.get_model_max_tokens('gemini-2.0-ultra'):,}, 컨텍스트: {self.get_model_context_window('gemini-2.0-ultra'):,}K)"
                },
                {
                    "id": "gemini-2.5-pro",
                    "name": "Gemini 2.5 Pro",
                    "description": f"고성능 모델 (출력: {self.get_model_max_tokens('gemini-2.5-pro'):,}, 컨텍스트: {self.get_model_context_window('gemini-2.5-pro'):,}K)"
                },
                {
                    "id": "gemini-2.5-flash",
                    "name": "Gemini 2.5 Flash",
                    "description": f"경량 모델 (출력: {self.get_model_max_tokens('gemini-2.5-flash'):,}, 컨텍스트: {self.get_model_context_window('gemini-2.5-flash'):,}K)"
                }
            ],
            "anthropic": [
                {
                    "id": "claude-4-1-opus-20250901",
                    "name": "Claude 4.1 Opus",
                    "description": f"최고 성능 모델 (출력: {self.get_model_max_tokens('claude-4-1-opus-20250901'):,}, 컨텍스트: {self.get_model_context_window('claude-4-1-opus-20250901'):,}K)"
                },
                {
                    "id": "claude-4-sonnet",
                    "name": "Claude 4 Sonnet",
                    "description": f"균형 모델 (출력: {self.get_model_max_tokens('claude-4-sonnet'):,}, 컨텍스트: {self.get_model_context_window('claude-4-sonnet'):,}K)"
                },
                {
                    "id": "claude-4-1-haiku-20250901",
                    "name": "Claude 4.1 Haiku",
                    "description": f"경량 모델 (출력: {self.get_model_max_tokens('claude-4-1-haiku-20250901'):,}, 컨텍스트: {self.get_model_context_window('claude-4-1-haiku-20250901'):,}K)"
                }
            ],
            "local": self._get_local_models()
        }
        
        return models
    
    def _get_local_models(self) -> List[Dict[str, str]]:
        """로컬 모델 목록 반환"""
        models = [
            {
                "id": "local-model",
                "name": "로컬 모델",
                "description": f"로컬 GGUF 모델 (최대 토큰: {getattr(settings, 'local_llm_max_tokens', 2048)}, 컨텍스트: {getattr(settings, 'local_llm_context_window', 4096)})"
            }
        ]

        # LM Studio(로컬 모델 서버)에서 모델 목록을 가져와 병합 시도
        try:
            from src.models.lm_studio import list_lm_studio_models
            lm = list_lm_studio_models()
            local_models = lm.get('local', [])
            for m in local_models:
                models.append({
                    'id': m.get('id') or m.get('name'),
                    'name': m.get('name') or m.get('id'),
                    'description': m.get('description') or ''
                })
        except Exception:
            # 실패 시 기존 기본 모델만 반환
            pass

        return models
    
    def create_chain(self, llm, prompt_template: PromptTemplate):
        """LLM과 프롬프트 템플릿으로 체인 생성 (최신 Runnable 방식)"""
        return prompt_template | llm
    
    # 기존 호환성 메서드들
    def _get_model_max_tokens(self) -> Dict[str, int]:
        """모델별 최대 토큰 수 반환 (기존 호환성)"""
        models = {}
        for provider_models in self.get_available_models().values():
            for model_info in provider_models:
                model_id = model_info.get('id') or model_info.get('model')
                if model_id:
                    models[model_id] = self.get_model_max_tokens(model_id)
        return models
    
    def _get_model_context_window(self) -> Dict[str, int]:
        """모델별 컨텍스트 윈도우 크기 반환 (기존 호환성)"""
        # 2025년 9월 18일 기준 최신 모델 정보로 업데이트
        return {
            "gpt-5-pro": 1024, "gpt-5": 512, "gpt-5-lite": 256, "gpt-5-mini": 128,
            "gemini-2.0-ultra": 8192, "gemini-2.5-pro": 4096, "gemini-2.5-flash": 2048,
            "claude-4-1-opus-20250901": 2048, "claude-4-sonnet": 1024, "claude-4-1-haiku-20250901": 500,
        }

    def _get_model_context_window_full(self) -> Dict[str, int]:
        models = {}
        for provider_models in self.get_available_models().values():
            for model_info in provider_models:
                model_id = model_info.get('id')
                if model_id:
                    models[model_id] = self.get_model_context_window(model_id)
        return models
    
    def _get_max_tokens_for_model(self, provider: str, model: Optional[str] = None) -> int:
        """특정 모델의 최대 토큰 수 반환 (기존 호환성)"""
        return self.get_max_tokens_for_model(provider, model)