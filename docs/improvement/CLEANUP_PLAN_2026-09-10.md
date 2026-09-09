# 정리 작업 계획 (2026-09-10)

> **상태: 완료(2026-09-10)** — 1~5단계 전부 수행, 회귀 101 passed, 스모크 전체 경로 실행.
> **차기 회기 잔여 과제도 완료(2026-09-10 2차)** — 실제 결과는 `docs/CHANGELOG.md` 2026-09-10 항목 참조.

## 배경
프로젝트 전반 점검에서 도출된 잔여 정리 항목 중 3·4단계(코드 구조) 작업 계획.
1단계(.env.example 35키 보완)와 2단계(레거시 문서 삭제·수정)는 완료.

## 3단계: pdf_converter.py(826줄) 3모듈 분할

선례: document_loader.py의 믹스인 4종 분할 패턴을 그대로 따른다.

| 신규 파일 | 내용 | 예상 줄수 |
|---|---|---|
| `src/utils/pdf_heading_utils.py` | 무상태 순수 헬퍼: `is_real_heading`, `join_paragraph_lines`, `get_heading_level` (모듈 수준 함수로 추출) | ~150 |
| `src/utils/pdf_semantic_chunking.py` | `SemanticChunkingMixin`: create_semantic_chunks, _create_basic_chunks, convert_pdf_to_semantic_chunks | ~130 |
| `src/utils/pdf_text_extraction.py` | `PdfTextExtractionMixin`: _extract_text_and_images, _connect_cross_page_text, _can_connect_to_next, _extract_and_process_images, _clean_and_format_text(_with_images) | ~380 |

- `pdf_converter.py` 본체: `ImprovedPDFConverter(믹스인 상속)` + `convert_pdf_to_markdown` 진입점 유지. 외부 import 경로(`from ..utils.pdf_converter import ImprovedPDFConverter`)는 그대로 유지(사용처: pdf_loading.py, agent_pdf_converter.py, tests).

## 4단계: 대형 함수 분해 (30줄 초과 134개 중 의미 있는 로직 5개)

| 대상 | 줄수 | 분해 방향 |
|---|---|---|
| `pdf_loading._load_pdf_file` | 206 | 변환 경로 선택 / 변환 실행 / 마크다운 후처리·저장 블록 분리 |
| `docx_loading._load_docx_file` | 148 | 문서 파싱 / 이미지 추출 / 청킹 블록 분리 |
| `convert_pdf_to_markdown` | 124 | 페이지 반복 / 이미지 처리 / 텍스트 정제·저장 블록 분리 |
| `_extract_text_and_images` | 96 | 텍스트 추출 / 이미지 메타 수집 분리 |
| `semantic_chunker.__init__` | 105 | 파라미터 검증·정규화 헬퍼 분리 |

제외(의도적 결정): `keyword_expander`의 130/126줄 함수는 사전 데이터 정의라 쪼개는 의미 없음.

## 검증
- 각 단계 후: `ruff check/format` + 기존 회귀망(test_context_improvement, test_pdf_loading_helpers 등) + 골든 테스트
- 신규 순수 헬퍼(`pdf_heading_utils`)는 최소 단위 테스트 추가
- 완료 후 CHANGELOG 동기화

## 진행 순서
1. pdf_heading_utils 추출 + 테스트 → 검증
2. SemanticChunkingMixin 이동 → 검증
3. PdfTextExtractionMixin 이동 + convert_pdf_to_markdown 분해 → 검증
4. pdf_loading._load_docx_file 분해 → 검증
5. semantic_chunker.__init__ 분해 → 검증

## 차기 회기 잔여 과제 (이번 범위 제외)
> ✅ 아래 5건 모두 2026-09-10 2차 회기에서 완료 처리. 잔여 관찰 대상만 남음.

- ~~이동된 로직 중 30줄 초과 잔존 함수 추가 분해: `_extract_text_and_images`(96), `_clean_and_format_text`(62), `create_semantic_chunks`(62), `_create_adaptive_chunks`(62), `docx_loading._extract_docx_images`(91) 등~~ 완료
- ~~`keyword_expander.py`(726줄), `text_processor.py`(603줄) 600줄 기준 초과 파일 분할~~ 완료(데이터·패턴 모듈 추출, 전 파일 600줄 미만)
- ~~`scripts/data_management/load_documents.py` 미참조 확인 후 삭제~~ 완료(0 참조 확인 후 삭제, README/.cursor 동기화)
- ~~삼켜진 예외 6곳에 최소 `logger.debug` 추가~~ 완료
- ~~`tests/test_simple.py`가 임시 VectorDatabase와 RAGChain 내부 DB를 연결하지 않아 질의가 항상 `no_documents`인 기존 갭~~ 완료(vector_db 주입, E2E success)

## 잔여 관찰 대상 (다음 정리 회기 후보)
- `vector_db.py`(550줄), `rag_parallel_processor.py`(522줄) — 600줄 미만이지만 대형, 분할 여부 관찰
- 믹스인 이동분 중 30줄 초과 잔존 함수(`_connect_cross_page_text` 53, `_extract_and_process_images` 54, `_create_basic_chunks` 56 등) — 로직 밀도 낮아 우선순위 하향

