# 변경 이력(Changelog)

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
