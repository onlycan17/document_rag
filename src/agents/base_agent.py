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
    로컬 LLM을 활용한 에이전트 베이스 클래스
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.base_url = settings.local_llm_base_url
        self.model_name = settings.local_llm_model
        self.max_tokens = getattr(settings, 'local_llm_max_tokens', 2048)
        self.context_window = getattr(settings, 'local_llm_context_window', 4096)
        self._llama = None  # 로컬 GGUF 백엔드 (존재 시 사용)
        
        logger.info(f"🤖 {agent_name} 에이전트 초기화 완료")
        logger.info(f"📡 로컬 LLM: {self.base_url} - {self.model_name}")

        # 로컬 GGUF 백엔드 시도 (모델 파일이 있으면 우선 사용)
        self._initialize_local_llm_backend()

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
            # 1) 환경변수 경로 우선
            gguf_path = os.getenv("LOCAL_LLM_GGUF_PATH")
            if not gguf_path:
                # 2) 로컬 모델 디렉토리 자동 탐색 (사용자 구성 경로)
                from src.utils.model_bootstrap import get_midm_path
                gguf_path = str(get_midm_path())

            if not os.path.exists(gguf_path):
                logger.info(f"로컬 GGUF 모델을 찾지 못했습니다: {gguf_path}")
                return

            logger.info("🚀 로컬 GGUF 모델 백엔드 초기화 시도...")
            self._llama = Llama(
                model_path=gguf_path,
                n_ctx=self.context_window,
                n_threads=int(os.getenv("LOCAL_LLM_THREADS", "4")),
                verbose=False,
            )
            logger.info("✅ 로컬 GGUF 모델 백엔드 활성화 완료")
        except Exception as e:  # pragma: no cover
            # 로컬 로드 실패 시 HTTP 모드 유지
            self._llama = None
            logger.warning(f"⚠️ 로컬 GGUF 백엔드 초기화 실패, HTTP 모드 유지: {e}")
    
    @llm_retry_with_backoff()
    def _call_local_llm(self, prompt: str, temperature: float = 0.1, max_tokens: Optional[int] = None) -> str:
        """로컬 LLM 호출.

        - GGUF 백엔드가 활성화되어 있으면 llama.cpp로 직접 추론
        - 아니면 OpenAI 호환 HTTP 엔드포인트로 폴백
        """
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
                # 비어있으면 안전하게 폴백
                logger.warning("⚠️ GGUF 응답이 비어 있습니다. HTTP 모드로 폴백합니다.")
            except Exception as e:  # pragma: no cover
                logger.warning(f"⚠️ GGUF 호출 실패, HTTP 모드 폴백: {e}")

        # 2) HTTP 폴백 (기존 동작)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

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