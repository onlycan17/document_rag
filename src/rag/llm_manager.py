"""
LLM 관리 모듈

이 모듈은 다양한 외부 LLM 제공자(OpenAI, Google, Anthropic, OpenRouter)의 초기화와 관리를 담당합니다.
"""

import logging
from typing import Dict, List, Optional

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from config import settings
from src.constants import (
    MODEL_MAX_CONTEXT,
    MODEL_MIN_CONTEXT,
    MODEL_PROMPT_TOKENS,
    MODEL_SAFETY_MARGIN,
    TOKEN_TO_CHAR_RATIO,
)
from src.models import ModelRegistry

logger = logging.getLogger(__name__)

UNKNOWN_OPENROUTER_MODEL_MAX_TOKENS = 16384


class LLMManager:
    """
    LLM 생성 및 관리를 담당하는 클래스

    주요 기능:
    - 다양한 외부 LLM 제공자 지원 (OpenAI, Google, Anthropic, OpenRouter)
    - 스트리밍/비스트리밍 LLM 생성
    - 모델별 토큰 제한 관리
    - 프롬프트 템플릿 생성
    """

    def __init__(self):
        """LLM 매니저 초기화"""
        self.current_provider = None
        self.current_model = None

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

        # 실제 사용할 모델명 결정 (+ 안전 보정)
        actual_model = model or getattr(settings, f"{provider}_model", None)

        # OpenRouter 방어로직: 비어있거나 잘못된 기본값이 들어오면 안전한 기본값으로 보정
        if provider == "openrouter":
            if not actual_model or str(actual_model).strip().lower() in ("", "local-model"):
                fallback = settings.model_for("openrouter")
                if actual_model != fallback:
                    logger.warning(f"OpenRouter 모델 자동 보정: '{actual_model}' → '{fallback}'")
                    actual_model = fallback

        # 모델별 최대 토큰 수 가져오기
        max_tokens = self.get_max_tokens_for_model(provider, actual_model) if actual_model else settings.max_tokens

        if provider == "openrouter" and actual_model and ModelRegistry.get_model_config(actual_model) is None:
            max_tokens = UNKNOWN_OPENROUTER_MODEL_MAX_TOKENS
            logger.info(
                f"미등록 OpenRouter 모델 '{actual_model}': 출력 토큰 {max_tokens} 사용 "
                f"(기본 4096은 reasoning 토큰 소진으로 빈 답변 위험)"
            )

        logger.info(
            f"LLM 초기화: provider={provider}, model={actual_model}, max_tokens={max_tokens}, streaming={streaming}"
        )

        # 스트리밍 콜백 설정
        # LangSmith는 랭체인 체인(prompt | llm)을 LANGSMITH_TRACING 환경변수로
        # 자동 관측하므로 별도 관측 CallbackHandler는 붙이지 않는다.
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
        elif provider == "openrouter":
            return self._create_openrouter_llm(actual_model, max_tokens, streaming, callbacks)
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
            callbacks=callbacks,
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
            callbacks=callbacks,
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
            callbacks=callbacks,
        )

    def _create_openrouter_llm(self, model: str, max_tokens: int, streaming: bool, callbacks: List):
        """OpenRouter(OpenAI 호환) LLM 생성"""
        api_key = getattr(settings, "openrouter_api_key", None)
        if not api_key:
            raise ValueError("OpenRouter API 키(OPNEROUTER_API_KEY)가 설정되지 않았습니다.")
        base = getattr(settings, "openrouter_api_base", "https://openrouter.ai/api").rstrip("/")
        logger.info(f"ChatOpenAI(openrouter) 설정: base_url={base}/v1, model={model}")
        return ChatOpenAI(
            api_key=api_key,
            base_url=f"{base}/v1",
            model=model,
            temperature=settings.temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            stream_usage=True,  # 스트리밍에서도 토큰 사용량 수신 (LangSmith 비용 기록용)
            callbacks=callbacks,
        )

    def create_prompt_template(self) -> PromptTemplate:
        """프롬프트 템플릿 생성"""
        template = """당신은 국사의 전문적인 지식을 누구든지 알기쉽게 친절하게 전달하는 AI 어시스턴트입니다.
상세하고 정확한 정보를 제공하면서도, 남녀노소 누구든 이해할 수 있도록 쉽게 설명하는 것이 당신의 역할입니다.

다음은 검색된 관련 문서들입니다:
{context}

위 문서들의 내용을 참고하여 다음 질문에 답변해주세요:
{question}

답변은 아래와 같은 명확한 구조로 작성해주세요. 각 섹션의 제목을 반드시 포함해야 합니다.

### 핵심 요약
(질문에 대한 핵심 답변을 1-2 문장으로 요약합니다.)

### 상세 설명
(핵심 요약에 대한 구체적인 내용을 번호나 글머리 기호를 사용하여 단계별로 설명합니다. 필요시 표를 사용할 수 있습니다.
사실을 서술할 때는 해당 내용이 나온 문서를 [출처 n] 형식으로 문장 끝에 표기합니다. n은 컨텍스트의 [문서 n] 번호와 같습니다.)

### 참고 자료
(답변을 생성하는 데 사용된 문서의 구체적인 내용을 간단히 언급합니다. 문서 제목이나 주요 내용을 기반으로 작성할 수 있습니다.)

답변:"""

        return PromptTemplate(input_variables=["context", "question"], template=template)

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
        safety_margin = MODEL_SAFETY_MARGIN

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
        """사용 가능한 모델 목록 반환 (ModelRegistry 단일 출처)"""
        return ModelRegistry.get_all_models()

    def create_chain(self, llm, prompt_template: PromptTemplate):
        """LLM과 프롬프트 템플릿으로 체인 생성 (최신 Runnable 방식)"""
        return prompt_template | llm
