"""LangSmith 관측 헬퍼.

LLM 호출 경로에 트레이싱을 붙이는 유일한 진입점. LangSmith가 꺼져 있거나
SDK가 설치되지 않은 환경(오프라인·내부망 미구축)에서도 앱 동작에 영향이
없도록 항상 no-op으로 폴백한다.

활성화 조건(모두 충족):
- LANGSMITH_TRACING=true
- LANGSMITH_API_KEY 설정 (SaaS 키. 셀프호스팅 시 LANGSMITH_ENDPOINT 필요)

계측 규약:
- 함수 인자를 통째로 기록하지 않는다(process_inputs 재정의) — 프롬프트 등
  필요한 값만 record_* 로 명시 설정한다(설정값·시크릿 유출 방지).
- 관측 타입을 구체적으로 지정한다: LLM 호출 llm, 검색 retriever.
- LangSmith는 랭체인 체인(prompt | llm)을 환경변수만으로 자동 관측하므로
  별도 CallbackHandler를 붙이지 않는다. 이 모듈은 비랭체인 직접 호출
  (BaseAgent._call_llm)과 세부 단계(검색·재정렬·생성)를 명시적으로 기록한다.
"""

import logging
import os
from contextlib import nullcontext
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# LangSmith run_type: chain / llm / retriever / tool / agent / prompt
_RUN_TYPE_MAP = {
    "span": "chain",
    "chain": "chain",
    "generation": "llm",
    "retriever": "retriever",
    "tool": "tool",
    "agent": "agent",
}


def langsmith_enabled() -> bool:
    """환경변수 기준으로 LangSmith 트레이싱 활성 여부를 판정한다."""
    if os.getenv("LANGSMITH_TRACING", "false").lower() != "true":
        return False
    return bool(os.getenv("LANGSMITH_API_KEY"))


def ensure_langsmith_env() -> None:
    """config에서 읽은 LangSmith 설정을 실제 환경변수로 반영한다.

    LangSmith SDK는 process 시작 시점의 환경변수를 읽으므로, config.settings
    로드 후 이 함수를 호출해 랭체인 자동 관측과 @traceable이 같은 설정을
    인식하도록 한다. 이미 환경변수가 설정되어 있으면(config와 무관하게
    우선) 덮어쓰지 않는다.
    """
    if os.getenv("LANGSMITH_TRACING"):
        return
    try:
        from config import settings  # 지연 import: 순환 피함
    except Exception as e:
        logger.debug(f"LangSmith env 반영 실패(config 로드 불가): {type(e).__name__}: {e}")
        return
    if getattr(settings, "langsmith_tracing", "false").lower() == "true":
        os.environ["LANGSMITH_TRACING"] = "true"
        if getattr(settings, "langsmith_api_key", None):
            os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        if getattr(settings, "langsmith_endpoint", None):
            os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        if getattr(settings, "langsmith_project", None):
            os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project


def observe_if_enabled(
    name: str,
    as_type: str = "span",
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[Callable], Callable]:
    """langsmith.traceable 데코레이터를 조건부로 적용한다.

    Args:
        name: LangSmith에 기록될 run 이름
        as_type: 관측 타입 (span/chain/generation/retriever/tool/agent)
        capture_input: 입력 자동 수집 여부. 시크릿·설정 객체가 인자로
            들어올 수 있는 메서드는 False로 두고 record_* 로 명시 설정
        capture_output: 출력 자동 수집 여부
    """
    if not langsmith_enabled():
        return lambda fn: fn
    try:
        from langsmith import traceable
    except ImportError:
        logger.debug("langsmith SDK 미설치 — 트레이싱 비활성화")
        return lambda fn: fn

    run_type = _RUN_TYPE_MAP.get(as_type, "chain")

    # capture 파라미터를 traceable의 process_inputs/process_outputs로 매핑
    # (프롬프트·시크릿 대량 노출 방지, 성능 절약)
    kwargs: dict = {"name": name, "run_type": run_type}
    if not capture_input:
        kwargs["process_inputs"] = lambda _inputs: {}
    if not capture_output:
        kwargs["process_outputs"] = lambda _outputs: {}

    return traceable(**kwargs)


def _current_run_tree():
    """현재 활성 래닝 tree를 반환한다. 활성 없으면 None."""
    if not langsmith_enabled():
        return None
    try:
        from langsmith.run_helpers import get_current_run_tree

        return get_current_run_tree()
    except Exception:
        return None


def _patch_current_run(**kwargs) -> None:
    """현재 활성 run에 메타데이터·입출력을 병합한다. 비활성/활성 없음은 무시.

    LangSmith의 RunTree는 add_metadata/add_inputs/add_outputs로 갱신한다.
    """
    if not kwargs:
        return
    run = _current_run_tree()
    if run is None:
        return
    try:
        metadata = kwargs.pop("metadata", None)
        inputs = kwargs.pop("input", None)
        outputs = kwargs.pop("output", None)
        if metadata:
            run.add_metadata(metadata)
        if inputs:
            run.add_inputs(inputs)
        if outputs:
            run.add_outputs(outputs)
        # metadata가 아니고 input/output도 아닌 남은 키는 metadata로 전달
        if kwargs:
            run.add_metadata(kwargs)
    except Exception as e:
        logger.debug(f"LangSmith run 갱신 실패(무시): {type(e).__name__}: {e}")


def record_generation(
    model: str,
    input: Optional[dict] = None,
    usage_details: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> None:
    """현재 generation(llm) run에 모델명·입력·토큰 사용량을 기록한다.

    LangSmith는 랭체인 경로에서 usage를 자동 기록하지만, BaseAgent._call_llm
    같은 직접 호출은 별도로 사용량을 기록한다. usage는 metadata로 넣는다.
    """
    patch_metadata = dict(metadata or {})
    if model:
        patch_metadata["model"] = model
    if usage_details:
        patch_metadata["usage"] = usage_details
    _patch_current_run(metadata=patch_metadata, input=input)


def record_retriever_output(sources: list[dict]) -> None:
    """현재 retriever run에 검색 결과 요약(출처·점수)을 기록한다."""
    _patch_current_run(output=sources)


def record_input(input: dict) -> None:
    """현재 run의 입력을 명시 설정한다 (인자 자동 수집 대신 필요한 값만)."""
    _patch_current_run(input=input)


def record_output(output: dict) -> None:
    """현재 run의 출력을 명시 설정한다 (record_input과 대칭)."""
    _patch_current_run(output=output)


def record_metadata(metadata: dict) -> None:
    """현재 run에 메타데이터(점수·소요시간·문서수 등)를 병합한다."""
    _patch_current_run(metadata=metadata)


def propagate_trace_attributes(session_id: Optional[str] = None, user_id: Optional[str] = None):
    """현재 트레이스에 세션·사용자 속성을 전파하는 컨텍스트 매니저.

    LangSmith는 `tracing_context`로 해당 블록의 run에 metadata·tags를 전파한다.
    비활성 환경에서는 아무 일도 하지 않는 null 컨텍스트를 반환한다.
    """
    if not langsmith_enabled() or (session_id is None and user_id is None):
        return nullcontext()
    try:
        from langsmith.run_helpers import tracing_context

        metadata = {}
        if session_id is not None:
            metadata["session_id"] = session_id
        if user_id is not None:
            metadata["user_id"] = user_id
        return tracing_context(metadata=metadata)
    except ImportError:
        logger.debug("langsmith SDK 미설치 — 트레이싱 비활성화")
        return nullcontext()
