# 변경 이력(Changelog)

## 2026-09-11 (관측성 — 검색 평가 결과 Langfuse Dataset 업로드)
- feat(eval): `retrieval_eval.py`에 `--upload` 플래그 신설 — 골든 셋을 `retrieval-golden` Dataset으로 등록(질의 기준 중복 생성 방지), 실행마다 사례별 트레이스(`hit@k`·`best_rank`, 미검색=0)와 요약 트레이스(`hit@k`·`mrr`) 점수를 NUMERIC으로 기록
- feat(tracing): `record_output` 헬퍼 추가(record_input과 대칭), 평가 실행은 `retrieval-eval-<타임스탬프>` 세션으로 묶어 UI 회기 비교 지원
- 검증: 실제 실행(Hit@5 100% / MRR 0.875) 후 API 감사 — Dataset 아이템 10건, 세션 내 트레이스 11건(사례 10+요약 1), 점수 22건(hit@5×11·best_rank×10·mrr×1) 연결 확인. 오프라인 회귀 190 passed, ruff 통과

## 2026-09-11 (관측성 고도화 — langfuse/skills 계측 규약 적용)
- feat(tracing): 트레이스 계층 구조 완성 — `query_engine.query/stream_query`를 rag-query 트레이스 루트로 만들고(본문 `_query_impl` 분리), `vector_db.search`에 retriever 관측(출처·점수 기록), LangChain 메인 답변 체인에 `langfuse.langchain.CallbackHandler` 부착(`llm_manager.create_llm`). 메인 RAG 흐름은 `prompt_template | ChatOpenAI` 경로라 `BaseAgent._call_llm` 데코레이션만으로는 generation이 기록되지 않던 문제 해결
- fix(tracing): langfuse 4.15.2에는 `update_current_observation`/`update_current_trace` 메서드가 없어 모든 관측 갱신이 무음 실패(model=null·usage=0·trace input 누락) — `_patch_current_observation` 헬퍼로 `update_current_generation`/`update_current_span`을 올바르게 호출하도록 수정. 트레이스 루트 input 기록 API는 해당 버전에 존재하지 않아 제거(YAGNI)
- feat(tracing): 세션 추적 — UI에서 `langfuse_session_id`(uuid4) 생성, `query/stream_query/invoke`에 session_id 전달 경로 추가(`rag_chain`·`query_engine`), `propagate_trace_attributes`로 trace에 session 연결. 대화 단위 트레이스 그룹핑 가능
- fix(rag): `rag_parallel_processor`의 `run_in_executor` 워커에 `contextvars.copy_context().run` 래핑 — 스레드 경계에서 Langfuse 관측이 트레이스에서 분리(고아 generation)되는 문제 예방
- 설치: `github.com/langfuse/skills` 공식 AI 스킬(`~/.agents/skills/langfuse/`) — 베이스라인 감사(모델명·토큰·계층·session) 기반 반복 개선 워크플로우 적용
- 검증: E2E 질의 1회 → Langfuse API 감사에서 rag-query span 하위에 generation(qwen3-vl-235b, input 8,760/output 772/total 9,532 토큰) + retriever(출처·점수 12건) 중첩, sessionId 연결 확인. 오프라인 회귀 190 passed, ruff 통과

## 2026-09-11 (관측성 — Langfuse 셀프호스팅 도입)
- feat(observability): Langfuse v3 셀프호스팅 스택 신설 — LangSmith(SaaS)는 내부망 미대응이므로 자체 호스팅 가능한 오픈소스 대안 채택. `infra/langfuse/docker-compose.yml` (web·worker·postgres·clickhouse·minio·redis, 로컬 개발용 시크릿 내장 — 운영 시 재생성 안내 포함)
- feat(tracing): `src/utils/tracing.py` 헬퍼 — `observe_if_enabled` 데코레이터, 비활성화·SDK 미설치 시 no-op 폴백. `BaseAgent._call_llm`에 `llm_generation` span 연결로 전 LLM 호출 경로(OpenRouter/OpenAI/Google/Anthropic) 트레이싱
- 부트스트랩: `LANGFUSE_INIT_*` 환경변수로 조직·프로젝트·관리자 계정·API 키 자동 프로비저닝 (UI 가입 불필요). 키는 프로젝트 `.env`(커밋 제외)에만 저장
- 가이드: `docs/LANGFUSE_GUIDE.md` 신설 (기동·UI·내부망 배포 체크리스트·문제 해결)
- 검증: 실제 LLM 호출 1회 → Langfuse API에서 `llm_generation` 트레이스 1건 확인, 오프라인 회귀 190 passed, ruff 통과
- 참고: v3 스택은 `LANGFUSE_S3_*`·`REDIS_HOST/PORT` 등 최신 env 명명 필요, 단일 노드는 `CLICKHOUSE_CLUSTER_ENABLED=false` — 삽질 기록은 compose 주석과 가이드 참조

## 2026-09-11 (RAG 검색 품질 개선 — 임베딩 분리·평가 하네스)
- feat(embeddings): 문서·질의 임베딩 모델 분리 — 문서는 `solar-embedding-1-large-passage`(신규 `UPSTAGE_EMBEDDING_DOC_MODEL`), 질의는 기존 `-query` 모델 사용(업스테이지 권장 구성). OpenAI는 구분 모델이 없어 동일 모델 사용. 벡터 DB 전체 재구축 필요(완료: converted_docs 마크다운 3종, 935 청크)
- feat(eval): 검색 평가 하네스 신설 — 골든 셋(`tests/eval/golden_retrieval.json` 10질의), 순수 지표 함수 `src/utils/retrieval_metrics.py`(best_relevant_rank·Hit@k·MRR, 오프라인 테스트 6건), 실행 스크립트 `scripts/eval/retrieval_eval.py`. 측정 없던 튜닝에서 숫자 기반 검증으로 전환
- fix(vector_db): FAISS 로드 시 `documents_cache.pkl`이 비어 있으면 docstore에서 문서 캐시 복구 — 캐시 비었을 때 키워드 검색(BM25/TF-IDF)이 조용히 무력화되던 문제
- fix(vector_db): 불완전한 기존 인덱스 발견(471/476 청크가 몽촌토성4+상.pdf 단일 문서, 하.pdf·KERIS 미인덱싱) → 마크다운 기반 재구축 + KERIS 임베딩 429로 증분 보강
- chore(tracing): LangSmith 트레이싱 활성화 경로 정리 — config의 `load_dotenv`로 .env 키가 langchain-core에 자동 반영되어 별도 와이어링 불필요 확인. **`.env.example` 보호 파일로 수동 추가 필요: `UPSTAGE_EMBEDDING_DOC_MODEL=solar-embedding-1-large-passage`, `LANGSMITH_TRACING=true`**
- 설계 결정: LangGraph 미채택(선형 파이프라인에 과설계), 리랭커·RRF·Parent-Child 청킹은 평가 하네스 기반 후속 회기 과제
- 결과: 평가 스크립트 Hit@5 100% / MRR 0.875 (10질의, passage 임베딩 인덱스). 재구축 전 동일 셋은 0%(정답 문서 미인덱싱 상태의 올바른 측정)

## 2026-09-10 (테스트 인프라 개선 — 네트워크 테스트 분리)
- feat(test): `pytest tests/` 기본 실행을 오프라인 안전화 — `tests/conftest.py` 신설로 API 호출 테스트(debug·integration·legacy·processing·utils 디렉토리 + 루트 5파일)에 `network` 마커 자동 부여, pyproject `[tool.pytest.ini_options]`에 `-m "not network"` 기본 제외. 기존엔 디버그 테스트가 실제 LLM API를 호출해 전체 스위트가 수 분 이상 멈추고 API 비용이 발생. 단, tests/utils 중 순수 오프라인 회귀 테스트(test_log_masking·test_retry_contract)는 OFFLINE_EXCEPTIONS로 기본 실행에 포함
- fix(test): `tests/debug/test_image_metadata.py`가 구형 `VectorDatabase` API(`embedding_model=` 인자, `is_initialized`, `similarity_search`)를 호출하던 것을 현행 API(`VectorDatabase()`, `get_document_count`, `search` 튜플 반환)에 맞게 수정 — 기존 FAILED 해소
- chore(test): `.gitignore`의 `test_*.py` 규칙에 묻혀 커밋 누락됐던 오프라인 회귀 테스트 파일 12건 예외 등록·추적 시작(이 커밋 이전 feat/fix/refactor 커밋들에서 처리)
- 결과: 기본 실행 185 passed / 52 deselected(네트워크), 12초 내 완료. 네트워크 테스트는 `pytest tests/ -m network` 로 명시 실행
- 문서 동기화: AGENTS.md 테스트 명령, tests/README.md 실행 방법·마커 규칙

## 2026-09-10 (보안 개선 — 로그 민감정보 마스킹)
- fix(security): 로그 민감정보 마스킹 — `logging_config`에 `mask_secrets` 순수 함수와 `SecretMaskingFormatter` 신설. 환경변수에 설정된 실제 API 키 값(OPENAI/OPENROUTER/OPNEROUTER/GOOGLE/ANTHROPIC/UPSTAGE)과 알려진 토큰 패턴(`sk-`, `AIza`, `Bearer …`)을 `***`로 치환. `setup_logging`의 파일·스트림 전 핸들러에 적용(app.py 진입점 자동 커버), 핸들러 교체 판정도 마스킹 포맷터 기준으로 정합화
- test: `tests/utils/test_log_masking.py` 신규 6건(환경변수 값·패턴·일반 텍스트 무손상·포맷터 E2E), 오프라인 회귀망 157 passed, 전체 수집 237개, ruff 통과

## 2026-09-10 (신뢰성 개선 4차 — 요약 길이 자동 튜닝)
- feat(rag): 요약 길이 자동 튜닝 — `summarizer.resolve_summary_length` 순수 함수 신설(모델 출력 토큰 한도 기반, 한글 1토큰≈1문자 보수 가정, 하한 150자/기본 500자). 하드코딩된 `max_summary_length=500`을 rag_chain 초기화 시 모델별로 산출하고, `update_llm` 시에도 재계산 — 작은 출력 한도 모델에서 요약 잘림 방지
- test: `resolve_summary_length` 테스트 3건 추가, AppTest(app.py 로드) 무예외, 오프라인 회귀망 151 passed, 전체 수집 231개, ruff 통과

## 2026-09-10 (UX 개선 — 출처 하이라이트·문단 점프)
- feat(ui): 출처 하이라이트 및 문단 점프 — 신규 순수 헬퍼 `src/utils/source_highlight.py`(질문 키워드 추출(조사 제거·불용어 필터), 관련도 최고 문단 중심 ±1 미리보기, 키워드 굵게 표시)
- feat(ui): 채팅 인터페이스에 "📑 출처 보기 (n개)" expander 추가 — 문서별 `[n] 파일명 (p.X)` 제목, 관련 문단 하이라이트 미리보기 + "전체 내용 보기", 스트리밍/동기/대화 기록 재렌더링 경로 모두 적용 (`ui/components/chat_interface.py`)
- 설계 노트: Streamlit은 채팅 메시지 내부 앵커 이동을 지원하지 않으므로 문단 점프는 '관련 문단을 먼저 표시' 방식으로 구현
- test: `tests/test_source_highlight.py` 신규 9건, AppTest(app.py 로드) 무예외 확인, 오프라인 회귀망 148 passed, 전체 수집 228개, ruff 통과

## 2026-09-10 (신뢰성 개선 3차 — 잔여 예외 정비·유사 질의 캐시)
- fix(예외): 잔여 `except Exception` 6곳 정비 — `openrouter_models`(죽은 `_now` 폴백 제거, 모델 목록 조회 실패 시 캐시 사용 로그), `pdf_loader_advanced`(페이지 수 확인을 `(OSError, PdfReadError)`로 축소, OCR 확인을 `OSError`로 축소), `docx_loading`(텍스트 폴백 실패 원인 로그), `image_analyzer.get_image_type`(`OSError`로 축소 — UnidentifiedImageError 포함)
- feat(검색): 유사 질의 캐시 — `EnhancedVectorDatabase.search`에 TTL 캐시 도입(config 신규: `QUERY_CACHE_TTL_SECONDS` 기본 300초, `QUERY_CACHE_MAX_ENTRIES` 기본 64, FIFO 제거), `add_documents`·`clear_database` 시 무효화. 전처리 완료 쿼리의 정확 일치만 캐시하는 의도적 한계(ponytail)
- test: 캐시 테스트 신규(`tests/test_query_cache.py`, 히트/미스/TTL 만료/무효화/최대 항목 6건), 오프라인 회귀망 142 passed, 전체 수집 219개, ruff check·format 통과
- 참고: `.env.example`은 보호 파일로 편집 차단 — `QUERY_CACHE_TTL_SECONDS=300`, `QUERY_CACHE_MAX_ENTRIES=64` 두 줄을 수동 추가 필요

## 2026-09-10 (신뢰성 개선 2차 — 레거시 테스트 정리·예외 정비 확대)
- fix(test): pytest 전체 수집 실패 해결 — 삭제된 모듈을 import하던 레거시 테스트 4건 정리
  - `tests/debug/test_image_metadata.py`: `src.rag.vector_database` → `src.vectorstore` import 경로 수정
  - `tests/test_large_context.py`: `src.rag.parallel_processor` → `src.rag.rag_parallel_processor` import 경로 수정
  - `tests/processing/test_md_postprocessing.py`: 제품에서 삭제된 병렬 후처리(`ParallelProcessor`) 테스트 부분 제거, 단일 파일 테스트 유지
  - `tests/test_single_conversion.py`: 대상 API(`src.converters.PDFToMarkdownConverter`)가 이전 회기에서 삭제되어 대체물 없음 → 파일 삭제 (PDF 변환 로직은 `test_pdf_loading_helpers` 등이 커버)
- refactor(rag): `rag_chain.py` 죽은 방어 코드 제거 — context_chunker/summarizer/rag_parallel_processor try-import와 perf try-import를 직접 import로 정리, `_initialize_advanced_features` 단순화
- fix(agents): `base_agent.py` 재시도 계약 개선 — API 키 미설정·패키지 미설치를 `ValueError`로 승격하여 재시도 없이 즉시 실패(무의미한 백오프 대기 제거), 재시도 로그에 예외 유형 추가
- fix(processing): `preprocessing_model.py` — 응답 구조 폴백 예외를 `(AttributeError, IndexError, TypeError)`로 축소, Responses API 폴백 시 원인 로그 추가
- fix(로깅): `rag_parallel_processor.py` 청크 처리/재시도/요약 폴백 3곳에 예외 유형·스택 로깅 추가
- test: 재시도 계약 테스트 신규(`tests/utils/test_retry_contract.py`, ValueError 미재시도·네트워크 오류 재시도), 전체 수집 213개, 오프라인 회귀망 136 passed, ruff check·format 통과

## 2026-09-10 (신뢰성·테스트·출처인용 개선)
- fix(로깅): `query_engine` 예외 처리 정비 — 경계 catch에 스택 트레이스(`exc_info`)와 예외 클래스명 추가, 사용자 오류 메시지에서 내부 오류 문자열 노출 제거, 전처리 확장 실패 로그에 원인 분류 정보 포함
- refactor(rag): 죽은 방어 코드 정리 — `query_engine`의 `src.utils.perf` try-import(perf.py 실존)와 무효 `now()` 호출 제거, `rag_chain`·`query_engine`의 의미 없는 주석 정리
- feat(rag): 출처 인라인 인용 — 프롬프트에 `[출처 n]` 표기 규칙 추가(`llm_manager.create_prompt_template`), `DocumentProcessor.prepare_documents` 신설로 컨텍스트 `[문서 n]` 라벨과 출처 목록 번호 정합화, `AnswerFormatter` 참고 자료 블록을 `- [출처 n]` 형식으로 정렬
- test: 무테스트 핵심 모듈 단위테스트 신규 — `tests/test_context_chunker.py`(7), `tests/test_document_processor.py`(7), `tests/test_summarizer.py`(8), 인용 형식 회귀 2건 추가, 회귀망 131 passed
- docs: 개선 계획 `docs/improvement/IMPROVEMENT_2026-09-10_RELIABILITY.md` 신설, `docs/TODO_TASKS.md`의 AnswerFormatter·출처 인용 항목 완료 처리
- 관찰: 레거시 테스트 4개(`tests/debug/test_image_metadata.py`, `tests/processing/test_md_postprocessing.py`, `tests/test_large_context.py`, `tests/test_single_conversion.py`)가 삭제된 모듈을 import해 pytest 전체 수집이 실패함 — 다음 정리 회기 후보

## 2026-09-10 (정리 회기 2차 — 잔여 과제)
- refactor(utils): 잔여 대형 함수 분해 — `pdf_text_extraction`(_extract_text_and_images 96→헤더/수집/변환 3메서드, _clean_and_format_text 62→노이즈 제거+마크다운 변환), `semantic_chunker`(create_semantic_chunks→객체 생성 분리, _create_adaptive_chunks→키워드 병합/재분할 3메서드), `docx_loading._extract_docx_images`(91→단일 이미지 추출+확장자 추정)
- refactor(utils): `keyword_expander.py`(726줄)의 동의어·도메인·불용어 사전을 `keyword_data.py`로 추출, 본체 451줄
- refactor(loaders): `text_processor.py`(603줄)의 정규식 패턴 4종을 `text_patterns.py`로 추출, 본체 519줄 / `pdf_loading.py`의 모듈 헬퍼 9종을 `pdf_loading_helpers.py`로 추출, 본체 427줄 — 외부 import 경로 유지
- chore: 미참조 스크립트 `scripts/data_management/load_documents.py` 삭제, README·.cursor 규칙의 존재하지 않는 실행 파일 나열 정리
- fix(로깅): 삼켜진 예외 6곳에 `logger.debug` 보강(structure_parser, logging_config, openrouter_image_service, document_loader)
- fix(test): `test_simple.py`가 RAGChain에 벡터 DB를 주입하지 않아 질의가 항상 `no_documents`이던 갭 수정 — E2E status:success 확인
- test: `test_pdf_loading_helpers` 픽스처를 헬퍼 모듈 기준으로 패치 대상 갱신, 회귀 101 passed

## 2026-09-10 (프로젝트 정리 회기)
- chore(config): `.env.example`에 config.py 참조 누락 환경변수 35개 보완(청킹·검색·스트리밍·LangSmith 등), 중복 키 제거
- docs: 로컬 모델(LM Studio/GGUF) 삭제 잔여 문서 정리 — `docs/LM_STUDIO_INTEGRATION.md` 삭제, ARCHITECTURE/PRD/FEATURE_SPEC의 레거시 문구를 외부 API 전용 현행화
- refactor(utils): `pdf_converter.py`(826줄)를 3모듈로 분할 — `pdf_heading_utils`(순수 헬퍼), `pdf_text_extraction`(추출·정제 믹스인), `pdf_semantic_chunking`(청킹 믹스인), 본체 195줄, 외부 import 경로 유지
- refactor(loaders): `pdf_loading._load_pdf_file`(206줄)을 5단계 폴백 체인 전용 메서드로 분해, `docx_loading._load_docx_file`(148줄)을 추출·생성·저장 단계로 분해
- refactor(utils): `semantic_chunker`의 사전 데이터를 모듈 상수로 추출(`TOPIC_TRANSITION_KEYWORDS`, `DOMAIN_KEYWORDS`), `__init__` 105→15줄
- test: `tests/test_pdf_heading_utils.py` 신규(헤딩/문장 판정 회귀), 관련 스위트 101 passed
- test(simple): 스모크 픽스처 `domain.md` 추가로 로더→벡터DB→RAG 질의 전체 경로 실행 가능

## 2026-09-09 (대형 함수 리팩터링)
- refactor(loaders): document_loader.py를 믹스인 4종(TextCleaning·Chunking·DocxLoading·PdfLoading)으로 분할
- refactor(utils): `_is_incomplete_sentence`를 `src/utils/sentence_completion.py`로 순수 분리, 골든 테스트 61케이스로 동작 고정
- refactor(loaders): PDF 로더의 provider/model 해석·이미지 스캔·복사·메타데이터 변환·MD 저장·2단계 후처리 실행을 검증 가능한 모듈 헬퍼로 추출 (`_load_pdf_file` 311→206줄, `_process_converted_content` 227→96줄)
- refactor(processing): `preprocess_text`(108줄)를 제공자별 호출 메서드와 순수 변환 함수 7종으로 분할 (108→26줄)
- refactor(utils): `agent_pdf_converter._extract_text_blocks`(121줄)를 `build_text_blocks` 순수 함수 + 페이지 이미지 추출/설명 생성 메서드로 분리 (121→41줄)
- test: 오프라인 단위 테스트 신규 추가(헬퍼 25케이스 포함 스위트 86 passed), 스모크 6·AppTest 무예외, 전 커밋 GitHub Actions CI 녹색

## 2025-09-10
- feat(preprocessing): 기본 전처리 모델을 저비용·멀티모달 지향으로 재정렬
  - openai: gpt-4o-mini, google: gemini-1.5-flash-8b, anthropic: claude-3-5-haiku-20241022
- feat(multimodal): 멀티모달 전처리 기본 ON (ENABLE_MULTIMODAL_PREPROCESSING=true)
- feat(models): 멀티모달 지원 목록 최신화 + .env로 동적 확장(EXTRA_MULTIMODAL_*)
- feat(openrouter): 이미지 분석 기본 프로바이더를 OpenRouter로 전환, 모델 z-ai/glm-4.5v 설정
- fix(ui): app.py에 PreprocessingModelFactory 임포트 누락 보완
- feat(openai): Responses API 우선, 실패 시 Chat Completions 폴백

## 0.1.0 (초기 설계 문서화)
- PRD, 아키텍처, UI/UX, 기술 스택, 기능, API, ERD, 테이블, TODO/TASK 문서 최초 작성
- 코드베이스 역분석 기반 문서화 정합성 1차 확보
