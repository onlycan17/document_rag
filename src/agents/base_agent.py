"""
로컬 LLM 에이전트 베이스 클래스
"""

import json
import logging
import requests
from typing import Optional, Dict, Any, List, Tuple
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


class LocalLLMCircuitOpen(Exception):
    """로컬 LLM 서킷 브레이커가 열린 상태 예외"""
    pass

def llm_retry_with_backoff():
    """
    로컬 LLM API 호출 재시도 데코레이터
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 설정값과 동기화(시점 평가)
            try:
                from config import settings as _s  # 지연 임포트로 최신값 반영
                _max_retries = max(0, int(getattr(_s, 'local_llm_max_retries', 3)))
                _base_delay = float(getattr(_s, 'api_base_delay', 3.0))
                _max_delay = float(getattr(_s, 'api_max_delay', 60.0))
            except Exception:
                _max_retries, _base_delay, _max_delay = 2, 5.0, 60.0
            last_exception = None
            
            for attempt in range(_max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # 서킷 오픈 예외는 즉시 중단(재시도 금지)
                    if isinstance(e, LocalLLMCircuitOpen):
                        raise
                    last_exception = e
                    error_message = str(e)
                    
                    if attempt < _max_retries:
                        delay = min(_base_delay * (2 ** attempt) + random.uniform(0, 0.5), _max_delay)
                        logger.warning(f"🔄 로컬 LLM 호출 재시도 {attempt + 1}/{_max_retries + 1}: {delay:.1f}초 후")
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

    - 기본은 로컬이지만, 화면/설정에서 선택한 제공자(openai/google/anthropic/local/openrouter)를 따르도록 확장
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
        self.provider = (provider or settings.llm_provider or "openrouter").lower()

        # 정책 강제: 에이전트 경로는 무조건 OpenRouter 사용
        try:
            if getattr(settings, 'enforce_openrouter_for_agents', True):
                if self.provider != 'openrouter':
                    logger.warning(f"에이전트 LLM 제공자 강제 적용: {self.provider} → openrouter")
                self.provider = 'openrouter'
        except Exception:
            # 설정 접근 실패 시에도 안전하게 OpenRouter로 고정
            self.provider = 'openrouter'

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
            elif self.provider == "openrouter":
                # 텍스트용 openrouter_model 우선, 없으면 멀티모달 기본으로 폴백
                try:
                    self.model_name = getattr(settings, 'openrouter_model', None) or getattr(settings, 'openrouter_mm_model', 'z-ai/glm-4.5v')
                except Exception:
                    self.model_name = 'z-ai/glm-4.5v'
            else:
                self.model_name = getattr(settings, 'local_llm_model', 'local-model')

        # 방어 로직: OpenRouter인데 잘못된 로컬 기본값이 들어온 경우 자동 보정
        if self.provider == 'openrouter' and (not self.model_name or self.model_name.strip().lower() == 'local-model'):
            corrected = None
            try:
                corrected = getattr(settings, 'openrouter_model', None) or getattr(settings, 'openrouter_mm_model', 'z-ai/glm-4.5v')
            except Exception:
                corrected = 'z-ai/glm-4.5v'
            if corrected != self.model_name:
                logger.warning(f"OpenRouter에 잘못된 모델명이 감지되어 자동 보정: '{self.model_name}' → '{corrected}'")
                self.model_name = corrected

        # base_url 결정
        if self.provider == "local":
            # 로컬은 LM Studio HTTP 서버를 우선 사용 (디렉토리 GGUF 대신)
            def _ensure_scheme(u: str) -> str:
                if not u:
                    return u
                u = u.strip()
                if not (u.startswith("http://") or u.startswith("https://")):
                    return "http://" + u
                return u
            lm_base = getattr(settings, 'lm_studio_api_url', None) or getattr(settings, 'local_llm_base_url', 'http://localhost:3620')
            base = _ensure_scheme(lm_base).rstrip('/')
            # 멀티모달 선호 포트 정규화 적용
            self.base_url = self._prefer_local_mm_port(base)
        else:
            self.base_url = (base_url or "").rstrip('/')

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

        # 로컬 GGUF 백엔드는 기본 비활성화 (LM Studio HTTP 사용)
        if self.provider == 'local' and not getattr(settings, 'disable_local_gguf', True):
            self._initialize_local_llm_backend()

        # 로컬 모델명 유효성 점검: LM Studio에서 사용 가능한 모델을 조회하여 자동 보정
        if self.provider == 'local':
            try:
                import requests as _req
                r = _req.get(f"{self.base_url}/v1/models", timeout=3.0)
                if r.status_code == 200:
                    data = r.json()
                    names = [m.get('id') or m.get('name') for m in data.get('data', [])]
                    if names and self.model_name not in names:
                        logger.warning(f"로컬 서버에 모델 '{self.model_name}'이 없습니다. 사용 가능: {names[:5]}{'...' if len(names)>5 else ''}")
                        # 같은 계열 모델 자동 선택(간단 휴리스틱) 또는 첫 번째 모델
                        fallback = None
                        for n in names:
                            if isinstance(n, str) and n.lower().split(':')[0] in (self.model_name or '').lower():
                                fallback = n
                                break
                        self.model_name = fallback or names[0]
                        logger.info(f"로컬 모델 자동 선택: {self.model_name}")
            except Exception:
                pass

        # 서킷 브레이커 상태: 인스턴스 단위로 유지(엔드포인트별)
        self._circuit_state: Dict[str, Dict[str, Any]] = {}

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

        # 2) HTTP 호출 (LM Studio 등 OpenAI 호환 서버)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }

        # 멀티 엔드포인트 후보 구성: 설정의 목록 → 기본(base_url) → 1620 선호 포트
        candidate_bases: List[str] = []
        try:
            urls = getattr(settings, 'local_llm_base_urls', None)
            if urls:
                for u in str(urls).split(','):
                    u = u.strip()
                    if not u:
                        continue
                    candidate_bases.append(u.rstrip('/'))
        except Exception:
            pass
        if self.base_url:
            candidate_bases.append(self.base_url.rstrip('/'))
        # 1620 선호 포트 보정 추가
        try:
            preferred = self._prefer_local_mm_port(self.base_url)
            if preferred and preferred not in candidate_bases:
                candidate_bases.append(preferred)
        except Exception:
            pass

        # 중복 제거(입력 순서 유지)
        seen = set()
        unique_bases = []
        for b in candidate_bases:
            if b not in seen:
                unique_bases.append(b)
                seen.add(b)

        api_key = getattr(settings, 'local_llm_api_key', None)
        headers = {"Content-Type": "application/json"}
        if api_key and api_key not in ("", "not-needed"):
            headers["Authorization"] = f"Bearer {api_key}"

        # 서킷 브레이커 파라미터
        now = time.time()
        cb_threshold = 3  # 연속 실패 허용 한계
        cb_window = 120.0  # 초
        cb_cooldown = 60.0  # 초

        last_errors: List[Tuple[str, str]] = []

        for base in unique_bases:
            # 서킷 상태 확인
            st = self._circuit_state.get(base)
            if st and st.get('cooldown_until', 0) > now:
                # 서킷 오픈 상태면 건너뛴다
                logger.warning(f"⛔ 로컬 LLM 서킷 열림: {base} (남은 {int(st['cooldown_until']-now)}초)")
                continue

            try:
                response = requests.post(
                    f"{base}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=getattr(settings, 'local_llm_timeout', 60),
                )
                if response.status_code == 200:
                    # 성공 시 서킷 상태 초기화
                    if base in self._circuit_state:
                        self._circuit_state.pop(base, None)
                    result = response.json()
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    msg = f"HTTP {response.status_code}: {response.text[:200]}"
                    last_errors.append((base, msg))
                    # 실패 누적 및 서킷 관리
                    rec = self._circuit_state.setdefault(base, {"fails": 0, "first_ts": now, "cooldown_until": 0})
                    rec["fails"] += 1
                    # 윈도우 외면 리셋
                    if now - rec.get("first_ts", now) > cb_window:
                        rec["fails"], rec["first_ts"] = 1, now
                    if rec["fails"] >= cb_threshold:
                        rec["cooldown_until"] = now + cb_cooldown
                        logger.error(f"🚧 로컬 LLM 서킷 열림: {base} (연속 실패 {rec['fails']}회)")
            except requests.exceptions.RequestException as e:
                msg = f"네트워크 오류: {str(e)}"
                last_errors.append((base, msg))
                rec = self._circuit_state.setdefault(base, {"fails": 0, "first_ts": now, "cooldown_until": 0})
                rec["fails"] += 1
                if now - rec.get("first_ts", now) > cb_window:
                    rec["fails"], rec["first_ts"] = 1, now
                if rec["fails"] >= cb_threshold:
                    rec["cooldown_until"] = now + cb_cooldown
                    logger.error(f"🚧 로컬 LLM 서킷 열림: {base} (연속 실패 {rec['fails']}회)")
            except Exception as e:
                msg = f"LLM 호출 오류: {str(e)}"
                last_errors.append((base, msg))

        # 모든 시도가 실패한 경우: 최근 오류 요약 포함
        if last_errors:
            details = "; ".join([f"{b} -> {m}" for b, m in last_errors[:3]])
        else:
            details = "no endpoints tried"
        raise Exception(f"로컬 HTTP 호출 실패: {details}")

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
        if self.provider == 'openrouter':
            return self._call_openrouter(prompt, temperature=temperature, max_tokens=max_tokens)
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

    # ===== OpenRouter 호출 경로 =====
    def _call_openrouter(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """OpenRouter(OpenAI 호환) Chat Completions 호출"""
        api_key = getattr(settings, 'openrouter_api_key', None) or os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise Exception("OpenRouter API 키(OPENROUTER_API_KEY)가 설정되지 않았습니다.")
        model = self.model_name or getattr(settings, 'openrouter_model', getattr(settings, 'openrouter_mm_model', 'z-ai/glm-4.5v'))
        api_base = getattr(settings, 'openrouter_api_base', 'https://openrouter.ai/api')
        url = f"{api_base.rstrip('/')}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # 선택 헤더(권장): 서비스 명시
            "X-Title": "RAG-Postprocessor"
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
