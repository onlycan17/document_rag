"""
문서 전처리 모델 인터페이스 및 구현체

PDF 문서의 텍스트 추출 및 전처리를 위한 로컬 및 외부 API 모델을 제공합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

import requests

from config import settings

from .document_pipeline import build_metadata, concatenate_documents, load_documents

logger = logging.getLogger(__name__)


def build_preprocessing_prompt(text: str) -> str:
    """텍스트 전처리 지시 프롬프트를 만든다."""
    return f"""
    다음 텍스트를 전문적으로 전처리해주세요:

    1. 한국어 문장의 연결성을 개선하세요
    2. 적절한 띄어쓰기를 적용하세요  
    3. 문맥에 맞는 문장 구조를 만드세요
    4. 불필요한 공백과 줄바꿈을 정리하세요
    5. 전문적이고 읽기 쉬운 텍스트로 만드세요

    텍스트:
    {text}

    전처리된 텍스트만 반환해주세요.
    """


def extract_responses_text(resp: Any) -> str | None:
    """OpenAI Responses API 응답에서 텍스트를 추출한다 (편의 프로퍼티→구조 폴백)."""
    processed_text = getattr(resp, "output_text", None)
    if not processed_text:
        try:
            processed_text = resp.output[0].content[0].text  # type: ignore[attr-defined]
        except Exception:
            processed_text = None
    return processed_text


def resolve_openrouter_model() -> str:
    """설정에서 OpenRouter 모델을 해석한다 (텍스트→멀티모달→하드코딩 기본)."""
    return getattr(settings, "openrouter_model", None) or getattr(settings, "openrouter_mm_model", "z-ai/glm-4.5v")


def build_openrouter_url() -> str:
    api_base = getattr(settings, "openrouter_api_base", "https://openrouter.ai/api")
    return f"{api_base.rstrip('/')}/v1/chat/completions"


def openrouter_headers(api_key: str) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-Title": "RAG-Preprocessor",
    }


def build_openrouter_body(model: str, prompt: str) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(settings.preprocessing_temperature),
        "max_tokens": 4000,
    }


def extract_chat_completion_text(data: Dict[str, Any]) -> str:
    """OpenAI 호환 chat/completions JSON에서 답변 텍스트를 추출한다."""
    return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()


class PreprocessingModel(ABC):
    """문서 전처리 모델의 추상 기본 클래스"""

    def __init__(self, model_name: str = "default"):
        """
        전처리 모델 초기화

        Args:
            model_name: 사용할 모델 이름
        """
        self.model_name = model_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def preprocess_text(self, text: str, **kwargs) -> str:
        """
        텍스트를 전처리합니다.

        Args:
            text: 전처리할 텍스트
            **kwargs: 추가 매개변수

        Returns:
            전처리된 텍스트
        """
        pass

    @abstractmethod
    def extract_and_preprocess(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        파일에서 텍스트를 추출하고 전처리합니다.

        Args:
            file_path: 처리할 파일 경로
            **kwargs: 추가 매개변수

        Returns:
            전처리 결과를 포함한 딕셔너리
            - text: 전처리된 텍스트
            - metadata: 메타데이터
            - processing_method: 사용된 처리 방법
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """모델 사용 가능 여부를 확인합니다."""
        pass

    def get_model_info(self) -> Dict[str, str]:
        """모델 정보를 반환합니다."""
        return {
            "model_type": self.__class__.__name__,
            "model_name": self.model_name,
            "description": self.__doc__ or "문서 전처리 모델",
        }


class APIPreprocessingModel(PreprocessingModel):
    """외부 API 모델을 사용한 문서 전처리"""

    def __init__(self, model_name: str = "gpt-4o-mini", provider: str = "openai"):
        """
        API 전처리 모델 초기화

        Args:
            model_name: 사용할 API 모델 이름
            provider: API 제공자 (openai, google, anthropic)
        """
        super().__init__(model_name)
        self.provider = provider
        self._client = None
        self._initialized = False

    def _initialize_client(self):
        """API 클라이언트를 초기화합니다."""
        if self._initialized:
            return

        try:
            from config import settings

            if self.provider == "openai":
                if not settings.openai_api_key:
                    raise ValueError("OpenAI API 키가 설정되지 않았습니다")
                import openai

                self._client = openai.OpenAI(api_key=settings.openai_api_key)

            elif self.provider == "google":
                if not settings.google_api_key:
                    raise ValueError("Google API 키가 설정되지 않았습니다")
                import google.generativeai as genai

                genai.configure(api_key=settings.google_api_key)
                self._client = genai.GenerativeModel(self.model_name)

            elif self.provider == "anthropic":
                if not settings.anthropic_api_key:
                    raise ValueError("Anthropic API 키가 설정되지 않았습니다")
                import anthropic

                self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            elif self.provider == "openrouter":
                # OpenRouter는 OpenAI 호환 HTTP를 사용하므로, 간단한 플래그만 유지
                # 실제 호출은 preprocess_text에서 requests로 처리
                from types import SimpleNamespace

                self._client = SimpleNamespace(provider="openrouter")
            else:
                raise ValueError(f"지원하지 않는 제공자: {self.provider}")

            self._initialized = True
            self.logger.info(f"API 전처리 모델 초기화 완료: {self.provider} - {self.model_name}")

        except Exception as e:
            self.logger.error(f"API 클라이언트 초기화 실패: {e}")
            raise

    def preprocess_text(self, text: str, **kwargs) -> str:
        """API 모델을 사용하여 텍스트를 전처리합니다. 실패 시 원본을 반환한다."""
        self._initialize_client()

        try:
            prompt = build_preprocessing_prompt(text)
            processors = {
                "openai": self._call_openai,
                "google": self._call_google,
                "anthropic": self._call_anthropic,
                "openrouter": self._call_openrouter,
            }
            processor = processors.get(self.provider)
            if not processor:
                raise ValueError(f"지원하지 않는 제공자: {self.provider}")

            processed_text = processor(prompt)
            self.logger.info(f"API 전처리 완료: {self.provider} - {len(text)} -> {len(processed_text)} 문자")
            return processed_text.strip()

        except Exception as e:
            self.logger.error(f"API 텍스트 전처리 실패: {e}")
            return text  # 실패 시 원본 반환

    def _call_openai(self, prompt: str) -> str:
        """Responses API 우선, 미지원 모델이면 Chat Completions로 폴백."""
        processed_text = None
        try:
            if hasattr(self._client, "responses"):
                resp = self._client.responses.create(
                    model=self.model_name,
                    input=prompt,
                    temperature=settings.preprocessing_temperature,
                    max_output_tokens=4000,
                )
                processed_text = extract_responses_text(resp)
        except Exception:
            processed_text = None

        if not processed_text:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.3,
            )
            processed_text = response.choices[0].message.content
        return processed_text

    def _call_google(self, prompt: str) -> str:
        return self._client.generate_content(prompt).text

    def _call_anthropic(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model_name, max_tokens=4000, messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def _call_openrouter(self, prompt: str) -> str:
        """OpenRouter(OpenAI 호환) Chat Completions 호출."""
        import os

        api_key = getattr(settings, "openrouter_api_key", None) or os.getenv("OPNEROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API 키(OPNEROUTER_API_KEY)가 설정되지 않았습니다")

        model = self.model_name or resolve_openrouter_model()
        response = requests.post(
            build_openrouter_url(),
            json=build_openrouter_body(model, prompt),
            headers=openrouter_headers(api_key),
            timeout=120,
        )
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter 오류: {response.status_code} - {response.text}")

        return extract_chat_completion_text(response.json())

    def extract_and_preprocess(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        파일에서 텍스트를 추출하고 API 모델로 전처리합니다.

        Args:
            file_path: 처리할 파일 경로
            **kwargs: 추가 매개변수

        Returns:
            전처리 결과 딕셔너리
        """
        self._initialize_client()

        try:
            result = load_documents(
                file_path,
                use_ocr=kwargs.get("use_ocr", True),
                use_agent_preprocessing=False,
                enable_postprocessing=False,
            )
            if not result.success:
                return {
                    "text": "",
                    "metadata": {"error": result.error or "문서를 추출할 수 없습니다"},
                    "processing_method": f"{self.provider}_extraction_failed",
                }
            extracted_text = concatenate_documents(result.documents)
            processed_text = self.preprocess_text(extracted_text, **kwargs)
            metadata = build_metadata(
                original_length=len(extracted_text),
                processed_length=len(processed_text),
                document_count=len(result.documents),
                file_path=file_path,
                extra={
                    "api_provider": self.provider,
                    "api_model": self.model_name,
                    "processing_method": f"{self.provider}_api_preprocessing",
                },
            )
            return {
                "text": processed_text,
                "metadata": metadata,
                "processing_method": f"{self.provider}_api_preprocessing",
            }
        except Exception as e:
            self.logger.error(f"API 파일 전처리 실패: {e}")
            return {
                "text": "",
                "metadata": {"error": str(e)},
                "processing_method": f"{self.provider}_api_preprocessing_failed",
            }

    def is_available(self) -> bool:
        """API 모델 사용 가능 여부를 확인합니다."""
        try:
            self._initialize_client()
            return True
        except Exception as e:
            self.logger.warning(f"API 모델 사용 불가: {e}")
            return False
