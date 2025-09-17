"""
로컬 LLM 에이전트 베이스 클래스
"""

import json
import logging
import requests
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import time
import random
from functools import wraps

from config import settings
import os

try:
    # 옵셔널: 로컬 GGUF 모델 직접 로딩용 (없으면 HTTP 모드로 폴백)
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover
    Llama = None  # noqa: N816

logger = logging.getLogger(__name__)

def llm_retry_with_backoff(max_retries=2, base_delay=5.0, max_delay=60.0):
    """
    로컬 LLM API 호출 재시도 데코레이터
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_message = str(e)
                    
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 0.5), max_delay)
                        logger.warning(f"🔄 로컬 LLM 호출 재시도 {attempt + 1}/{max_retries + 1}: {delay:.1f}초 후")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"🚨 로컬 LLM 호출 최대 재시도 초과: {str(e)}")
                        break
            
            raise last_exception
        return wrapper
    return decorator


class LocalLLMAgent(ABC):
    """
    LLM을 활용한 에이전트 베이스 클래스

    - 기본은 로컬이지만, 화면/설정에서 선택한 제공자(openai/google/anthropic/local)를 따르도록 확장
    - 로컬 선택 시: 1620 포트 멀티모달 서버 우선(base URL 강제 정규화)
    """
    
    def __init__(
        self,
        agent_name: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.agent_name = agent_name
        # 현재 제공자 결정: 명시값 → 세션설정값(.env) 순
        self.provider = (provider or settings.llm_provider or "local").lower()

        # 모델명 결정(제공자별 기본값 사용)
        if model_name:
            self.model_name = model_name
        else:
            if self.provider == "openai":
                self.model_name = getattr(settings, 'openai_model', 'gpt-4o-mini')
            elif self.provider == "google":
                self.model_name = getattr(settings, 'google_model', 'gemini-1.5-flash-8b')
            elif self.provider == "anthropic":
                self.model_name = getattr(settings, 'anthropic_model', 'claude-3-5-haiku-20241022')
            else:
                self.model_name = getattr(settings, 'local_llm_model', 'local-model')

        # base_url 결정(로컬만 사용). 로컬이면 1620 멀티모달 선호 포트 적용
        if self.provider == "local":
            self.base_url = self._prefer_local_mm_port(base_url or getattr(settings, 'local_llm_base_url', 'http://localhost:3620'))
        else:
            self.base_url = base_url or ""

        # 토큰/컨텍스트 설정(로컬/외부 공통 기본)
        self.max_tokens = getattr(settings, 'local_llm_max_tokens', 2048) if self.provider == 'local' else getattr(settings, 'max_tokens', 4096)
        self.context_window = getattr(settings, 'local_llm_context_window', 4096)
        self._llama = None  # 로컬 GGUF 백엔드 (존재 시 사용)
        
        # 초기화 로그
        logger.info(f"🤖 {agent_name} 에이전트 초기화 완료")
        if self.provider == 'local':
            logger.info(f"📡 LLM: provider=local base={self.base_url} model={self.model_name}")
        else:
            logger.info(f"📡 LLM: provider={self.provider} model={self.model_name}")

        # 로컬 선택 시에만 GGUF 백엔드 시도
        if self.provider == 'local':
            self._initialize_local_llm_backend()

    def _prefer_local_mm_port(self, url: str) -> str:
        """로컬 base URL을 멀티모달 선호 포트(기본 1620)로 정규화"""
        try:
            from urllib.parse import urlparse
            prefer_port = int(getattr(settings, 'local_mm_prefer_port', '1620'))
            p = urlparse(url if url.startswith('http') else f"http://{url}")
            host = p.hostname or 'localhost'
            scheme = p.scheme or 'http'
            return f"{scheme}://{host}:{prefer_port}"
        except Exception:
            # 실패 시 원본 반환
            return url

    def _initialize_local_llm_backend(self) -> None:
        """로컬 GGUF 모델 백엔드 초기화 (가능할 경우).

        우선순위:
        1) 환경변수 LOCAL_LLM_GGUF_PATH
        2) 프로젝트 기본 경로 models/korean/Midm-2.0-Base-Instruct-Q4_K_S.gguf
        실패 시 HTTP 모드 유지
        """
        if Llama is None:
            return

        try:
            # 1) 환경변수/자동 탐색으로 GGUF 경로 해석
            from src.utils.model_bootstrap import get_gguf_path
            gguf = get_gguf_path()
            gguf_path = str(gguf)

            if not os.path.exists(gguf_path):
                logger.info(f"로컬 GGUF 모델을 찾지 못했습니다: {gguf_path}")
                return

            logger.info("🚀 로컬 GGUF 모델 백엔드 초기화 시도...")
            # 메모리 보호: 과도한 n_ctx는 상한으로 캡핑
            from config import settings as _settings
            n_ctx_used = min(self.context_window, getattr(_settings, 'local_llm_max_context_cap', self.context_window))
            if n_ctx_used < self.context_window:
                logger.warning(f"n_ctx {self.context_window} -> {n_ctx_used} (LOCAL_LLM_MAX_CONTEXT_CAP 적용)")
            self._llama = Llama(
                model_path=gguf_path,
                n_ctx=n_ctx_used,
                n_threads=getattr(_settings, 'local_llm_threads', 8),
                n_gpu_layers=getattr(_settings, 'local_llm_n_gpu_layers', 0),
                verbose=False,
            )
            logger.info("✅ 로컬 GGUF 모델 백엔드 활성화 완료")
        except Exception as e:  # pragma: no cover
            # 로컬 로드 실패 시 HTTP 모드 유지
            self._llama = None
            logger.warning(f"⚠️ 로컬 GGUF 백엔드 초기화 실패, HTTP 모드 유지: {e}")
    
    @llm_retry_with_backoff()
    def _call_local_llm(self, prompt: str, temperature: float = 0.1, max_tokens: Optional[int] = None) -> str:
        """로컬 LLM 호출 (GGUF→HTTP)."""
        if not max_tokens:
            max_tokens = self.max_tokens

        # 1) 로컬 GGUF 백엔드 우선
        if self._llama is not None:
            try:
                response = self._llama(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    stop=["</answer>", "\n\n\n"],
                )
                text = response.get("choices", [{}])[0].get("text", "").strip()
                if text:
                    return text
                # 비어있으면 설정에 따라 처리
                msg = "⚠️ GGUF 응답이 비어 있습니다."
                if settings.disable_http_fallback:
                    logger.warning(f"{msg} HTTP 폴백 비활성화로 중단")
                    raise Exception("GGUF empty response and HTTP fallback disabled")
                logger.warning(f"{msg} HTTP 모드로 폴백합니다.")
            except Exception as e:  # pragma: no cover
                if settings.disable_http_fallback:
                    logger.warning(f"⚠️ GGUF 호출 실패, HTTP 폴백 비활성화로 중단: {e}")
                    raise
                logger.warning(f"⚠️ GGUF 호출 실패, HTTP 모드 폴백: {e}")

        # 2) HTTP 폴백 (기존 동작)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if settings.disable_http_fallback:
            raise Exception("HTTP fallback disabled")

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                raise Exception(f"API 호출 실패: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            raise Exception(f"네트워크 오류: {str(e)}")
        except Exception as e:
            raise Exception(f"LLM 호출 오류: {str(e)}")

    # ===== 외부 API 호출 경로 =====
    def _call_openai(self, prompt: str, temperature: float, max_tokens: int) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("OpenAI API 키가 설정되지 않았습니다.")
        model = self.model_name or getattr(settings, 'openai_model', 'gpt-4o-mini')
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
            raise Exception("google.generativeai 패키지가 설치되어 있지 않습니다.")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise Exception("Google API 키가 설정되지 않았습니다.")
        genai.configure(api_key=api_key)
        model_name = self.model_name or getattr(settings, 'google_model', 'gemini-1.5-flash-8b')
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([prompt])
        text = getattr(response, 'text', None)
        if not text:
            raise Exception("Google Generative AI 응답이 비어있습니다.")
        return text.strip()

    def _call_anthropic(self, prompt: str, temperature: float, max_tokens: int) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise Exception("Anthropic API 키가 설정되지 않았습니다.")
        try:
            import anthropic  # type: ignore
            client = anthropic.Anthropic(api_key=api_key)
            model = self.model_name or getattr(settings, 'anthropic_model', 'claude-3-5-haiku-20241022')
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=float(temperature),
                messages=[{"role": "user", "content": prompt}],
            )
            contents = getattr(msg, 'content', [])
            if contents and hasattr(contents[0], 'text'):
                return contents[0].text.strip()
            # HTTP 폴백 불가 시 에러
            raise Exception("Anthropic 응답 파싱 실패")
        except Exception as e:
            # HTTP 직접 호출 폴백
            try:
                import json as _json
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                body = {
                    "model": self.model_name or getattr(settings, 'anthropic_model', 'claude-3-5-haiku-20241022'),
                    "max_tokens": int(max_tokens),
                    "temperature": float(temperature),
                    "messages": [{"role": "user", "content": prompt}],
                }
                r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, data=_json.dumps(body), timeout=120)
                if r.status_code != 200:
                    raise Exception(f"Anthropic 오류: {r.status_code} - {r.text}")
                data = r.json()
                content = data.get("content", [])
                # content는 [{type:'text', text:'...'}]
                if content and isinstance(content, list) and content[0].get('type') == 'text':
                    return content[0].get('text', '').strip()
                raise Exception("Anthropic 응답 파싱 실패(HTTP)")
            except Exception as e2:
                raise Exception(f"Anthropic 호출 실패: {e2}")

    def _call_llm(self, prompt: str, temperature: float = 0.1, max_tokens: Optional[int] = None) -> str:
        """현재 provider에 맞는 LLM 호출"""
        if not max_tokens:
            max_tokens = self.max_tokens
        if self.provider == 'local':
            return self._call_local_llm(prompt, temperature=temperature, max_tokens=max_tokens)
        if self.provider == 'openai':
            return self._call_openai(prompt, temperature=temperature, max_tokens=max_tokens)
        if self.provider == 'google':
            return self._call_google(prompt, temperature=temperature, max_tokens=max_tokens)
        if self.provider == 'anthropic':
            return self._call_anthropic(prompt, temperature=temperature, max_tokens=max_tokens)
        raise Exception(f"지원하지 않는 provider: {self.provider}")
    
    def _create_korean_prompt(self, task_description: str, content: str, examples: List[str] = None) -> str:
        """
        한국어 특화 프롬프트 생성
        """
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
        """
        에이전트의 주요 처리 로직 (하위 클래스에서 구현)
        """
        pass
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        에이전트 정보 반환
        """
        return {
            "name": self.agent_name,
            "model": self.model_name,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "context_window": self.context_window
        }
