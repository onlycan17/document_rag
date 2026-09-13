# 개선 계획 (2026-09-13)

관측 백엔드 전환(Langfuse→LangSmith)이 절반만 이행되어 코드·문서·설정이 불일치하는
상태를 정리한다. 프라이어티 순서대로 진행하며, 각 단계마다 오프라인 회귀망
(`pytest tests/`, 202 passed)을 유지한다.

## 배경 요약 (조사 결과)

| # | 문제 | 현재 영향 | 근거 |
|---|------|-----------|------|
| ① | `scripts/eval/retrieval_eval.py`가 삭제된 `langfuse_enabled`를 import | **실행 즉시 ImportError (스크립트 완전 파손)** | `tracing.py`는 `langsmith_enabled`로 교체됨 |
| ② | CI·Makefile이 `tests/test_smoke_offline.py` 단일 파일만 실행 | 회귀망(202개)이 CI에서 절대 안 돌아감 | `ci.yml`, `Makefile` |
| ③ | 삭제된 로컬 모델/Langfuse가 문서에 생존 | 신규 개발자 혼란(존재하지 않는 설정 안내) | `docs/*`, `.env` |
| ④ | ruff 린트 최소 규칙 + CI에 format 미강제 | 스타일 비일관성 통과 | `pyproject.toml`, `ci.yml` |

## 작업 분해

### ① 평가 스크립트 Langfuse → LangSmith 이행
- **파일**: `scripts/eval/retrieval_eval.py`
- **진단**: `from src.utils.tracing import langfuse_enabled` → `langfuse_enabled`는 tracing.py에 없음.
  module 로드 시 ImportError → `--upload` 없이도 실행 불가.
- **수정**:
  - `langfuse_enabled` → `langsmith_enabled`
  - `upload_to_langfuse` → `upload_to_langsmith`: Langfuse Dataset·`create_score` 대신
    기존 `tracing.py` 헬퍼(`observe_if_enabled`, `record_input`, `record_output`,
    `record_metadata`, `propagate_trace_attributes`)로 **사례별·요약 트레이스를 기록하고
    지표(Hit@k·best_rank·MRR)를 run 메타데이터로 병합**.
  - docstring·`--upload` 도움말·안내 문구를 LangSmith 기준으로 갱신.
  - **신규 의존성 없음** (langsmith는 이미 선택 의존성·`tracing.py`에서 조건부 import).
- **검증**: `venv/bin/python scripts/eval/retrieval_eval.py --help` 로 import 정상 확인.

### ② CI·Makefile이 전체 오프라인 회귀망 실행
- **파일**: `.github/workflows/ci.yml`, `Makefile`
- **수정**: 스모크 단일 파일 → `pytest tests/ -q` (addopts `-m 'not network'` 가 이미
  네트워크 테스트 제외, 로컬 202 passed 확인).
- **검증**: 로컬 `pytest tests/` 202 passed.

### ③ 문서·설정 정리 (삭제된 로컬 모델/Langfuse 참조 제거)
- **파일**: `docs/API_SPEC.md`, `docs/ARCHITECTURE.md`, `docs/INSTALL.md`,
  `docs/SECURITY.md`, `docs/TECH_STACK.md`, 루트 `CLAUDE.md`,
  `AGENT_SYSTEM_REPORT.md`, `.env.example`
- **수정**: `LOCAL_LLM_*`·`LM Studio`·`GEMMA`·`레거시 Langfuse` 참조를 외부 API 전용/현행
  LangSmith 기준으로 갱신. `.env.example`에서 사장 키 제거(있다면).
  `.env`(개인 파일)는 커밋 대상이 아니므로 안내만.
- **주의**: `docs/CHANGELOG.md`는 이력이므로 삭제하지 않고 유지(과거 기록).

### ④ 린트·포맷 강제
- **파일**: `pyproject.toml`, `.github/workflows/ci.yml`
- **수정**:
  - ruff `select` 확장: `["E4","E7","E9","F"]` → `["E","W","F","I"]` (기본 오류·경고·isort).
    심화 규칙(bugbear 등)은 기존 코드 대량 위반 유발 위험 있으므로 이번엔 도입하지 않음.
  - `per-file-ignores`에 `I` 허용 여부 확인(E402 상황과 충돌 최소화).
  - CI에 `ruff format --check .` 스텝 추가.
- **검증**: `ruff check .` + `ruff format --check .` 통과. 기존 코드가 새 규칙으로 깨지면
  해당 파일만 정리(포맷은 `ruff format .` 적용).

## 완료 기준
- `ruff check .` 0건 / `ruff format --check .` 통과
- `pytest tests/ -q` 202 passed
- `retrieval_eval.py --help` 정상 실행
- 위 4개 파일군의 코드·문서·설정이 LangSmith/외부 API 전용 기준으로 일치

## 완료 결과 (2026-09-13)
| 항목 | 상태 | 핵심 변경 |
|------|------|-----------|
| ① 평가 스크립트 | ✅ | `langfuse_enabled`→`langsmith_enabled`, `upload_to_langfuse`→`upload_to_langsmith`(LangSmith run + metadata 기록), 역사 `--help` 정상 |
| ② CI·Makefile | ✅ | CI: `pytest tests/` 전체 회귀망 + `ruff format --check`. Makefile: `pytest tests/` |
| ③ 문서·설정 | ✅ | 로컬 모델/Langfuse 참조 8개 문서 제거·현행화, AGENT_SYSTEM_REPORT 배너, TODO 반영 |
| ④ 린트·포맷 | ✅ | ruff select에 `I` 추가 + 64건 자동수정, CI format 검사 추가 |

**검증**: `ruff check .` 통과 · `ruff format --check .` 107개 형식화 · `pytest tests/` **202 passed**

## 후속 (⑤·⑥) — 2026-09-13
| 항목 | 상태 | 비고 |
|------|------|------|
| ⑤ vector_db 이중 클래스 | ✅ | `EnhancedVectorDatabase`+빈 호환 서브클래스 → **단일 `VectorDatabase`** 통합. 테스트·문서 import 갱신 |
| ⑥ E501 정리 | ✅ | 의도적 제외 결정 — 위반 27건은 모델명·URL·JS/CSS·프롬프트 등 정당한 긴 단일 문자열. 강제 재래핑은 인위적 줄바꿈 유발. 사유를 `pyproject.toml`에 명시 |

**검증(후속)**: `ruff check .` · `ruff format --check .` · `pytest tests/` **202 passed**

## 후속 2 (TODO.md 잔여 3항목) — 2026-09-13
| 항목 | 상태 | 비고 |
|------|------|------|
| `app.py` 모듈 분할 | ✅ | 이미 297줄(<600)이며 `ui/` 모듈에 위임 — 추가 분할 불필요, TODO 해소 |
| OpenRouter 이미지 분석 재시도/스로틀 | ✅ | 기존 `api_retry_with_backoff` 재사용(429·타임아웃·일부 5xx 지수 백오프, config `API_MAX_RETRIES` 등 준수). `_post_chat_completions` 분리, docstring 모델 기본값 현행화 |
| 대용량 PDF 진행률 UI | ✅ | `process_pdf`에 `progress_callback` 연결 — 이미지 분석 단계 진행률이 Streamlit 바 실시간 반영. 에러 시 중단(폴백 금지) 정책은 유지 |

**검증(후속 2)**: `ruff check .` · `ruff format --check .` · `pytest tests/` **202 passed**

**참고**: `TODO.md`의 "아키텍처 개선 작업"(캐시·비동기·마이크로서비스·분산 벡터DB 등)은 규모가 커서 별도 리서치·설계가 필요한 항목으로 남겨둠.
