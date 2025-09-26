# 테스트 가이드

## 1. 스모크 테스트
- `python tests/test_simple.py` : 로더→DB→RAG 한 사이클 확인.

## 2. 개별 테스트 예시
- `python tests/debug/test_rag_query.py` : 질의 응답 흐름 점검
- `python tests/debug/test_rag_context.py` : 컨텍스트 길이/분할 확인

## 3. 권장 품질 도구(제안)
- 린트/포맷: `ruff`, `black`
- 타입 검사: `mypy`

## 4. CI 훅(제안)
- 포맷(`make fmt`), 테스트(`make test`), 린트(`make lint`) 3종 통과를 기준으로 머지.

