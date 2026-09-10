"""pytest 공유 설정.

tests/에는 오프라인 회귀 테스트와 실제 외부 API를 호출하는 수동 테스트가 섞여 있다.
실제 API 호출 테스트는 `network` 마커로 자동 표시되며, pyproject의
`addopts = -m "not network"` 설정 때문에 기본 실행에서 제외된다.

- 자동 표시 대상: debug/, integration/, legacy/, processing/, utils/ 디렉토리
  (tests/README.md 기준 - 실제 문서 처리/LLM 호출을 검증하는 수동 테스트) 및
  아래 NETWORK_ROOT_FILES에 명시된 루트 레벨 파일.
- 수동 실행: `pytest tests/ -m network` (느리고 API 비용이 발생함에 유의)
"""

from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent

# 실제 외부 API(임베딩/LLM)를 호출하는 루트 레벨 테스트 파일
NETWORK_ROOT_FILES = {
    "test_simple.py",  # RAGChain E2E 질의 (LLM 호출)
    "test_large_context.py",  # ParallelRAGProcessor (LLM 호출)
    "test_faiss_upstage_compatibility.py",  # 실제 임베딩 API 호출
    "test_rate_limit_handling.py",  # 실제 임베딩 API 호출
    "test_upstage_embedding.py",  # 실제 임베딩 API 호출
}

# 실제 처리 파이프라인을 실행하는 수동/디버그 테스트 디렉토리
NETWORK_DIRS = {"debug", "integration", "legacy", "processing", "utils"}

# 위 network 디렉토리 안에 있지만 순수 오프라인 회귀 테스트인 파일 (기본 실행에 포함)
OFFLINE_EXCEPTIONS = {
    Path("utils") / "test_log_masking.py",
    Path("utils") / "test_retry_contract.py",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """수집된 테스트에 network 마커를 자동 부여한다."""
    for item in items:
        rel = item.path.relative_to(BASE_DIR)
        if rel in OFFLINE_EXCEPTIONS:
            continue
        in_network_dir = any(part in NETWORK_DIRS for part in rel.parts[:-1])
        if rel.name in NETWORK_ROOT_FILES or in_network_dir:
            item.add_marker(pytest.mark.network)
