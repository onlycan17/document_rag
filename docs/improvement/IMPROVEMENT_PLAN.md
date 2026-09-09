# 개선 계획서 (IMPROVEMENT_PLAN)

- 작성일: 2026-09-08
- 근거: 프로젝트 분석 결과 + TODO.md 대기 항목 + 실제 코드 확인(호출 관계, git 추적 상태)
- 원칙: 동작에 지장 없는 정리 → 구조 통일 → 자동화 → 증상 기반 튜닝 순. 규모 문제의 증거가 없는 인프라(캐시/마이크로서비스)는 보류.

## 우선순위 총괄

| 순위 | フェーズ | 공수 | 리스크 | 핵심 근거 |
|---|---|---|---|---|
| 1 | 저장소 청소 | 0.5~1일 | 낮음 | 레거시 사본·디버그 스크립트가 git 추적 중 |
| 2 | 로더 이중구조 통일 | 1~2일 | 중간 | 동일 클래스명 2개가 컨트롤러 안에서 혼용 (실제 bug 온상) |
| 3 | 설정(config) 안정화 | 0.5일 | 낮음 | 중복 정의·오타 키명·하드코딩 IP |
| 4 | CI/자동화 | 1일 | 낮음 | Makefile이 도구 없으면 pass 처리 →CI에서 무력화 |
| 5 | 신뢰성 튜닝 | 증상 발생 시 | - | 실제 실패 로그 기준으로 1건씩 |

---

## Phase 1 — 저장소 청소 (완료 2026-09-09)

### 1.1 레거시 사본 삭제
- [x] `app_original.py` (102KB) git rm — 리팩토링 전 Streamlit 사본. git 히스토리에 보존되므로 복구 가능
- 검증: `make ci` 통과, `streamlit run app.py` 정상 기동

### 1.2 디버그 스크립트 정리 (모두 git 추적 중 → git mv/rm)
- [x] `run_postprocess*.py` 3종 삭제(디버그 전용, 참조 0건) + 루트 로그 파일 정리
- [x] `debug_metadata_check.py`, `monitor_progress.py` 삭제(참조 0건)
- [x] 루트 로그 2종 삭제
- 검증: 삭제한 스크립트를 참조하는 문서/코드 없는지 grep

### 1.3 문서 위치 정리 (사용자 결정 필요)
- [x] `domain.md` → `docs/DOMAIN.md` 이동(git mv), README 구조 갱신

---

## Phase 2 — 로더 이중구조 통일 (완료 2026-09-09, 생존자 정책 변경)

**현황 (실제 확인 결과)**
- 레거시: `src/loaders/document_loader.py`의 `DocumentLoader`(=EnhancedDocumentLoader 상속, 1,549줄+) — PDF→MD 변환·에이전트 파이프라인 통합. 사용처: `document_pipeline.py`, `multimodal_preprocessing_model.py`, `file_uploader.py`, `main_controller.py:243`, `scripts/`(패키지 __init__ 경유)
- 리팩터드: `src/loaders/document_loader_refactored.py`의 `DocumentLoader` — 사용처: `main_controller.py:97`, 테스트 4곳
- **문제**: 같은 이름 클래스가 경로에 따라 다르게 동작 → 한쪽만 고치면 조용히 어긋나는 구조

> **결정 번복 기록(2026-09-09)**: 실사 결과 refactored 로더는 `load_document()` 미구현으로 UI 업로드 경로와 호환되지 않았고,
> 레거시(1,549줄)가 PDF 변환·에이전트 모드·이미지 메타데이터 등 프로덕션 기능의 사실상 단일 구현체였다.
> refactored에 기능을 이식하는 것은 1,200줄 상당의 검증 없는 재작성 리스크라, **레거시 생존 + refactored 삭제**로 전환.
> 발견된 잠재 버그: 세션 초기화 시 refactored 인스턴스가 만들어져 첫 사이드바 변경 전 업로드 시 AttributeError 발생 가능했음 — 통일로 해소.
> 레거시 600줄 초과 문제는 별도 후속 과제(모듈 분할)로 이연.

### 2.1 기능 차이 실사
- [x] 대조 완료: refactored는 load_document 미구현 등 레거시 대비 부분 구현

### 2.2 생존자 확정 및 전환
- [~~refactored 보강~~] → 레거시 생존으로 정책 변경(위 기록 참조)
- [x] main_controller 초기화/설정갱신 2곳을 레거시로 통일, preprocessing_model='local' 잔여 기본값 제거(레거시 기본값도 openrouter로)
- [x] `document_loader_refactored.py` 삭제 + 전용 테스트 2종 정리, debug 스크립트 레거시 API로 재연결
- 검증: `tests/integration/test_refactored_loaders.py`, `python tests/test_simple.py` 스모크, 실제 PDF 업로드 → 채팅 답변 E2E 1회

---

## Phase 3 — 설정 안정화 (config.py)

### 3.1 중복 정의 제거
- [x] openrouter 설정 중복 제거 — 단일 블록 유지, 키는 OPENROUTER/OPNEROUTER 양쪽 인식

### 3.2 API 키 오타 처리 (사용자 결정 필요)
- [x] `OPNEROUTER_API_KEY`(오타) → 두 이름 모두 인식하도록 처리. 기존 .env 그대로 동작
- 주의: 프로젝트 사양에 "철자 고정" 주석 존재 → 변경 여부 확인 후 진행

### 3.3 하드코딩 IP 제거
- [x] 하드코딩 IP 포함 local_llm_* 설정 전체 삭제(로컬 LLM 기능 자체가 제거됨)

- 검증: 설정 import 스모크 + 앱 기동 + 로컬 LLM 질의 1회

---

## Phase 4 — CI/자동화 (완료 2026-09-09)

### 4.1 툴 체인 정의
- [x] `pyproject.toml` 생성: ruff lint(E4,E7,E9,F)+line-length 120, per-file-ignore(app/scripts/tests의 E402). 잔여 lint 153건 → 자동픽스+F841/E722/F401 수동 수정으로 0건

### 4.2 오프라인 스모크 테스트
- [x] `tests/test_smoke_offline.py` 6종: 임포트/설정 회귀(로컬 필드 제거·키 양쪽 철자)/local 제공자 거부/임베딩 키 요구. 인메모리 DB 검색은 가짜 임베딩 목이 필요해 이연(필요 시 추가)

### 4.3 GitHub Actions
- [x] `.github/workflows/ci.yml`: ruff check + 오프라인 스모크. **format --check는 보류** — 63개 파일이 미포맷 상태, 전체 포매팅은 로직 변경과 섞이지 않게 별도 커밋으로 진행해야 함(pre-commit 훅은 유지)
- [x] Makefile test 타깃을 오프라인 스모크로 교체(전체 스위트는 키/네트워크 의존이라 CI 부적합), CI는 ruff/pytest 직접 호출

- 검증: 로컬 `make ci` 통과(fmt→lint 0건→스모크 6 passed). 워크플로우 자체는 푸시 후 첫 실행으로 최종 확인(요구사항 설치 시간 소요)

---

## Phase 5 — 신뢰성 튜닝 (증상 기반, 착수 조건부)

실제 실패 로그/사용자 신고가 있을 때 1건씩만 진행. 선제 구현 금지.
- [ ] OpenRouter 이미지 분석 429/타임아웃 스로틀·재시도 튜닝 (429 로그 축적 시)
- [ ] 대용량 PDF(>150MB) 진행률/에러 UI 개선 (실제 불만 발생 시)
- [ ] 로컬 LLM 장애 시 클라우드 폴백 토글 (서킷 브레이커 개방 사례 확인 시)

---

## 명시적 보류 (증거 생길 때까지 하지 않음)

TODO.md의 아래 항목들은 현재 규모(단일 사용자/사내 챗봇)에서 문제를 일으킨 증거가 없어 **보류**. 발생 조건만 기록:
- Redis 검색 캐시 → 동일 질의 반복이 체감 지연의 주원인일 때
- 마이크로서비스/API 게이트웨이 분리와 분산 벡터DB → 동시 사용자·문서량이 FAISS 단일 인덱스 한계를 넘을 때

---

## Phase 0 (완료 2026-09-08) — 내장(로컬) 모델 → 외부 API 전면 전환

사용자 지시: 모든 내장 모델 호출을 외부 API 호출로 변경. `.env` 실측 결과 기본값은 이미 외부 API(openrouter/openai)로 전환되어 있었으므로, **호출 가능한 코드 경로 자체를 제거**하는 것이 작업 범위.

### 삭제 대상 (로컬 전용·고아 파일)
- [x] `src/models/lm_studio.py`, `src/utils/model_bootstrap.py` — LM Studio/GGUF 로딩·부트스트랩
- [x] `src/utils/gemma_multimodal.py`, `src/utils/ax_multimodal.py`, `src/utils/korean_text_model.py` — 로컬 멀티모달/LLM 추론
- [x] `src/utils/local_image_service.py`(고아), `src/utils/intelligent_image_extractor_korean.py`(인스턴스화 0건·고아)
- [x] `src/utils/http_probe.py` — 로컬 서버 프리플라이트 전용
- [x] `scripts/download_models.py` — GGUF 모델 다운로드
- [x] `LocalPreprocessingModel` 클래스, `LLMManager._create_local_llm`/`_get_local_models`, `base_agent`의 GGUF·LM Studio·서킷브레이커 경로

### 수정 대상
- [x] `base_agent.py`: `LocalLLMAgent`→`BaseAgent`(외부 API 전용: openai/google/anthropic/openrouter)
- [x] `llm_manager.py`: local 프로바이더 제거, ModelRegistry 의존 제거(자체 테이블 사용), GGUF 플래그 제거
- [x] `preprocessing_factory.py`·`embedding_model.py`: local 분기 제거. 임베딩 키 미설정 시 조용한 로컬 폴백 대신 명확한 한국어 오류
- [x] `image_analyzer.py`·`agent_pdf_converter.py`·`rag_chain.py`·`sidebar.py`: local 경로/노출 제거
- [x] `config.py`: local_*/lm_studio_*/gemma/ax 설정 삭제, openrouter 중복 정의 정리(Phase 3 병행), API 키 양쪽 철자 인식
- [x] 테스트: 로컬 전용 기능 테스트 삭제, import 갱신

### 범위 제외 (1줄 기록)
- `SentenceTransformersTokenTextSplitter`(document_loader): 추론이 아닌 토큰카운트용이라 유지 — 제거 시 청킹·벡터DB 재구축 필요

## 진행 규칙

1. 한 기능 완료 → `make ci` + 해당 검증 항목 통과 확인 → 다음 항목
2. 후크/검사 실패 시 즉시 중단 → 해결 → 재개
3. 각 Phase 완료 시 본 문서 체크박스 갱신 + TODO.md 동기화


## Phase 5 — 죽은 코드 제거 및 관측성 (완료 2026-09-09)

### 5.1 참조 없는 죽은 모듈 삭제 (약 2,500줄)
- [x] 전수 참조 검사(document_splitter·file_loader_manager·content_processor·tag_processor·md_parallel_processor 모두 프로덕션 참조 0건 확인)
- [x] 5종 git rm + 이들을 테스트하는 미추적 디버그 스크립트 2종 삭제 (test_md_postprocessing.py는 생존 parallel_processor 사용이라 보존)

### 5.2 에러 삼킴(except→pass) 로그화
- [x] 17곳에 debug/warning 로그 추가 (세션 조회 폴백·임시파일 정리·모델 병합 등). 핫패스 파싱 탐색 3곳(structure_parser·openrouter_image_service·logging_config 부트스트랩)은 의도적 흐름으로 제외

- 검증: ruff 0건, compileall, 오프라인 스모크 6 passed, AppTest 예외 없음
