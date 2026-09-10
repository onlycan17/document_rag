"""
외부 API LLM 에이전트 베이스 클래스

지원 제공자: openai / google / anthropic / openrouter (모두 외부 API)
내장(로컬) 모델 경로는 제거되었습니다 — 모든 추론은 외부 API 호출로 수행됩니다.
"""

import logging
import os
import random
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Dict, List, Optional

import requests

from config import settings

logger = logging.getLogger(__name__)


def llm_retry_with_backoff():
    """LLM API 호출 재시도 데코레이터 (지수 백오프). 설정 오류(ValueError)는 재시도하지 않는다."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            max_retries = max(0, int(getattr(settings, "api_max_retries", 3)))
            base_delay = float(getattr(settings, "api_base_delay", 3.0))
            max_delay = float(getattr(settings, "api_max_delay", 60.0))

            last_exception: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except ValueError:
                    # 키 미설정 등 설정 문제는 재시도해도 같은 결과이므로 즉시 실패
                    raise
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2**attempt) + random.uniform(0, 0.5), max_delay)
                        logger.warning(
                            f"🔄 LLM 호출 재시도 {attempt + 1}/{max_retries + 1} "
                            f"({type(e).__name__}): {delay:.1f}초 후"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"🚨 LLM 호출 최대 재시도 초과 ({type(e).__name__}): {e}")
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator


class BaseAgent(ABC):
    """
    외부 API LLM을 활용하는 에이전트 베이스 클래스

    제공자 결정 순서: 명시 인자 → settings.llm_provider → openrouter
    """

    SUPPORTED_PROVIDERS = ("openai", "google", "anthropic", "openrouter")

    def __init__(
        self,
        agent_name: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.agent_name = agent_name
        resolved = (provider or getattr(settings, "llm_provider", None) or "openrouter").lower()
        if resolved == "local":
            raise ValueError(
                f"{agent_name}: 로컬 모델은 지원되지 않습니다. "
                "외부 API 제공자(openai/google/anthropic/openrouter)를 지정해주세요."
            )
        self.provider = resolved

        self.model_name = model_name or self._default_model_for(self.provider)
        self.max_tokens = int(getattr(settings, "max_tokens", 4096))

        logger.info(f"🤖 {agent_name} 에이전트 초기화 완료")
        logger.info(f"📡 LLM: provider={self.provider} model={self.model_name}")

    @staticmethod
    def _default_model_for(provider: str) -> str:
        """제공자별 기본 모델명 반환"""
        if provider == "openai":
            return getattr(settings, "openai_model", None) or "gpt-4o-mini"
        if provider == "google":
            return getattr(settings, "google_model", None) or "gemini-1.5-flash-8b"
        if provider == "anthropic":
            return getattr(settings, "anthropic_model", None) or "claude-3-5-haiku-20241022"
        # openrouter: 텍스트용 모델 우선, 없으면 멀티모달 기본값
        return getattr(settings, "openrouter_model", None) or getattr(settings, "openrouter_mm_model", "z-ai/glm-4.5v")

    @llm_retry_with_backoff()
    def _call_llm(self, prompt: str, temperature: float = 0.1, max_tokens: Optional[int] = None) -> str:
        """현재 provider에 맞는 외부 LLM 호출"""
        if not max_tokens:
            max_tokens = self.max_tokens
        if self.provider == "openai":
            return self._call_openai(prompt, temperature=temperature, max_tokens=max_tokens)
        if self.provider == "google":
            return self._call_google(prompt, temperature=temperature, max_tokens=max_tokens)
        if self.provider == "anthropic":
            return self._call_anthropic(prompt, temperature=temperature, max_tokens=max_tokens)
        if self.provider == "openrouter":
            return self._call_openrouter(prompt, temperature=temperature, max_tokens=max_tokens)
        raise ValueError(f"지원하지 않는 provider: {self.provider}")

    # ===== 외부 API 호출 경로 =====
    def _call_openai(self, prompt: str, temperature: float, max_tokens: int) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        model = self.model_name or self._default_model_for("openai")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers, timeout=120)
        if resp.status_code != 200:
            raise Exception(f"OpenAI 오류: {resp.status_code} - {resp.text}")
        data = resp.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()

    def _call_google(self, prompt: str, temperature: float, max_tokens: int) -> str:
        try:
            genai = __import__("google.generativeai", fromlist=["generativeai"])
        except Exception:
            raise ValueError("google.generativeai 패키지가 설치되어 있지 않습니다.")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API 키가 설정되지 않았습니다.")
        genai.configure(api_key=api_key)
        model_name = self.model_name or self._default_model_for("google")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([prompt])
        text = getattr(response, "text", None)
        if not text:
            raise Exception("Google Generative AI 응답이 비어있습니다.")
        return text.strip()

    def _call_anthropic(self, prompt: str, temperature: float, max_tokens: int) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API 키가 설정되지 않았습니다.")
        model = self.model_name or self._default_model_for("anthropic")
        try:
            import anthropic  # type: ignore

            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=float(temperature),
                messages=[{"role": "user", "content": prompt}],
            )
            contents = getattr(msg, "content", [])
            if contents and hasattr(contents[0], "text"):
                return contents[0].text.strip()
            raise Exception("Anthropic 응답 파싱 실패")
        except ImportError:
            # SDK 미설치 시 HTTP 직접 호출
            import json as _json

            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {
                "model": model,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "messages": [{"role": "user", "content": prompt}],
            }
            r = requests.post(
                "https://api.anthropic.com/v1/messages", headers=headers, data=_json.dumps(body), timeout=120
            )
            if r.status_code != 200:
                raise Exception(f"Anthropic 오류: {r.status_code} - {r.text}")
            content = r.json().get("content", [])
            if content and isinstance(content, list) and content[0].get("type") == "text":
                return content[0].get("text", "").strip()
            raise Exception("Anthropic 응답 파싱 실패(HTTP)")

    def _call_openrouter(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """OpenRouter(OpenAI 호환) Chat Completions 호출"""
        api_key = getattr(settings, "openrouter_api_key", None) or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API 키(OPENROUTER_API_KEY)가 설정되지 않았습니다.")
        model = self.model_name or self._default_model_for("openrouter")
        api_base = getattr(settings, "openrouter_api_base", "https://openrouter.ai/api")
        url = f"{api_base.rstrip('/')}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Title": "RAG-Agent",
        }
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        resp = requests.post(url, json=body, headers=headers, timeout=120)
        if resp.status_code != 200:
            raise Exception(f"OpenRouter 오류: {resp.status_code} - {resp.text}")
        data = resp.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()

    def _create_korean_prompt(self, task_description: str, content: str, examples: Optional[List[str]] = None) -> str:
        """한국어 특화 프롬프트 생성"""
        prompt = f"""당신은 한국어 문서 전처리 전문 에이전트입니다.

작업: {task_description}

내용:
{content}

지침:
- 한국어의 조사, 어미, 어간 등 언어학적 특성을 고려하세요
- 고고학, 역사학 전문 용어의 특성을 이해하세요
- 문서의 학술적 맥락을 보존하세요
- 간결하고 정확한 답변을 제공하세요

"""
        if examples:
            prompt += "\n예시:\n"
            for i, example in enumerate(examples, 1):
                prompt += f"{i}. {example}\n"
        prompt += "\n답변:"
        return prompt

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """에이전트의 주요 처리 로직 (하위 클래스에서 구현)"""
        pass

    def get_agent_info(self) -> Dict[str, Any]:
        """에이전트 정보 반환"""
        return {
            "name": self.agent_name,
            "provider": self.provider,
            "model": self.model_name,
            "max_tokens": self.max_tokens,
        }
