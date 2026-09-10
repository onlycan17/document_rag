# 신뢰성·테스트·출처인용 개선 계획 (2026-09-10)

> **상태: 완료(2026-09-10)** — 1~3단계 전부 수행, 회귀망 131 passed. 실제 결과는 `docs/CHANGELOG.md` 2026-09-10 항목 참조.
> **2차 회기 완료(2026-09-10)** — 레거시 테스트 수집 오류 4건 정리 + rag_chain/rag_parallel_processor/base_agent/preprocessing_model 예외 정비. 상세는 `docs/CHANGELOG.md` "신뢰성 개선 2차" 항목 참조.

## 1단계: 예외 처리 정비 (`src/rag/query_engine.py` 중심)

| 위치 | 현재 | 개선 |
|---|---|---|
| 16-22행 | `src.utils.perf` try-import (perf.py 실존 → 죽은 코드) | 직접 import로 정리 |
| `query()` 122행 경계 | `except Exception` + 사용자 메시지에 원시 오류 문자열 노출 | `exc_info=True` 스택 로깅 + 사용자 메시지는 오류 유형만 표기 |
| `stream_query()` 217행 경계 | 동일 | 동일 |
| 전처리 확장 3곳 (263/287/297) | `logger.warning(f"...{e}")` | 로그에 예외 클래스명 포함 (원인 분류) |
| `fallback_search` 312행 / `search_documents` 476행 | 동일 | 동일 |
| `process_standard_context` 413 / `process_large_context` 437 | re-raise | 유지 (상위 경계가 처리) |

범위 제한: query_engine에 집중. rag_chain/base_agent 등은 2차 회기에서 처리 완료.

## 2단계: 무테스트 핵심 모듈 최소 단위테스트

- `tests/test_context_chunker.py` — 분할/병합 핵심 로직 (외부 API 불필요 케이스)
- `tests/test_document_processor.py` — `src/rag/document_processor.py` 포맷·출처 생성
- `tests/test_summarizer.py` — `src/rag/summarizer.py` 순수 로직
- 원칙: 외부 API 모킹 최소화, 순수 함수 중심, 기존 tests/ 관례(pytest) 준수

## 3단계: 출처 인라인 인용 (TODO_TASKS.md 항목)

- AnswerFormatter는 이미 구현·연동 완료 → 본 항목만 구현.
- 답변 본문에 `[출처 n]` 표기 + 참고 자료 섹션과 번호 대응.
- 프롬프트 템플릿(rag_chain)에 인용 규칙 추가, 포매터는 기존 구조 유지.

## 검증

- `ruff check` + `python -m pytest tests/` 전체 회귀 (기존 101개 통과 유지)
- 완료 후 CHANGELOG / TODO_TASKS 동기화
