### 무조건 한글로 답변하세요. 일반인도 이해하기 쉽도록 설명해 주세요.

# 저장소 가이드라인

## 프로젝트 구조 및 모듈
- `app.py`: Streamlit UI 진입점.
- `run_rag.py`: 앱 실행, 벡터 DB 관리, 테스트를 위한 대화형 CLI.
- `src/`: 핵심 코드 — `embeddings/`, `loaders/`, `rag/`, `vectorstore/`, `models/`, `utils/`.
- `scripts/`: 설치 및 데이터 도구 — `setup/`(OCR 설치), `data_management/vector_db_manager.py`.
- `tests/`: 스크립트 스타일 및 pytest 친화적 테스트(`tests/README.md` 참고).
- `data/`, `processed_docs/`, `vector_db/`, `logs/`: 로컬 아티팩트(커밋 금지).

## 빌드, 테스트, 개발 명령
- 툴킷 실행: `python run_rag.py` — Streamlit, 벡터 DB, 테스트, 설정 메뉴.
- 앱 실행: `streamlit run app.py` — UI 시작.
- 벡터 DB: `python scripts/data_management/vector_db_manager.py [--safe-mode|--markdown-only|--files "file1.pdf"]`.
- 빠른 테스트: `python tests/test_simple.py` — 로더 → DB → RAG 스모크 테스트.
- 전체 테스트(오프라인 회귀망): `python -m pytest tests/` — API 호출 테스트는 기본 제외.
- API 호출 테스트: `python -m pytest tests/ -m network` (느리고 API 비용 발생). 개별 스크립트 실행도 가능.

## 코딩 스타일 및 네이밍
- Python 3.10+; 4칸 들여쓰기; PEP 8 준수.
- 네이밍: 모듈/함수는 `snake_case`, 클래스는 `CamelCase`, 상수는 `UPPER_SNAKE`.
- 공개 함수/클래스에 타입 힌트와 docstring 작성.
- 모듈 단일 책임 유지(예: RAG 로직은 `src/rag/`, 스토어 동작은 `src/vectorstore/`).

## 테스트 지침
- pytest 디스커버리 선호(`tests/**/test_*.py`).
- 부작용 격리; 출력은 `tests/outputs/` 하위에 기록(git 무시).
- 신규 기능 최소 스모크 커버리지: 로더, 벡터 DB 동작, 한 번의 E2E 질의.
- 예시: `python -m pytest tests/processing/` 또는 `python tests/debug/test_rag_query.py`.

## 커밋 및 PR
- 커밋: 짧은 명령형 제목; 필요 시 Conventional Commits 스타일(`feat:`, `fix:`)과 이모지 사용.
- PR: 요약, UI/CLI 변경 스크린샷/로그 스니펫, 재현 가능한 테스트 단계, 관련 이슈 참조 포함.
- 스모크 테스트 통과 필수; 대용량 아티팩트 커밋 금지(`data/`, `vector_db/`, `logs/`).

## 보안 및 설정
- 시크릿: `.env` 사용(`.env.example` 복사); 실제 키 커밋 금지.
- 필수 키: 최소 1개 LLM 제공자 키(예: `UPSTAGE_API_KEY`, `OPENAI_API_KEY` 등).
- OCR 의존성(선택): `bash scripts/setup/install_ocr.sh`로 설치.

## 아키텍처 노트
- Retrieval: `src/vectorstore/`를 통한 FAISS/Chroma.
- Embeddings: `src/embeddings/` 내 Upstage/OpenAI 등.
- Orchestration: `src/rag/`의 `RAGChain`; `run_rag.py`의 앱 메뉴.

## 최근 변경 요약(대형 함수 리팩터링)
- document_loader.py → 믹스인 4종(TextCleaning·Chunking·DocxLoading·PdfLoading) 분할
- `_is_incomplete_sentence` → `src/utils/sentence_completion.py` 순수 분리 + 골든 테스트 61케이스(`tests/test_sentence_completion.py`)
- PDF 로더·전처리·에이전트 변환기의 대형 함수를 모듈 수준 순수 헬퍼로 추출: `pdf_loading`(311→206/227→96줄), `preprocessing_model.preprocess_text`(108→26줄, provider별 `_call_*` + 요청·응답 순수 함수 7종), `agent_pdf_converter._extract_text_blocks`(121→41줄, `build_text_blocks` 분리)
- 회귀망: `tests/test_sentence_completion.py`·`test_pdf_loading_helpers.py`·`test_preprocessing_helpers.py`·`test_agent_pdf_converter_helpers.py` (gitignore 예외 등록), 상세: `docs/CHANGELOG.md` 2026-09-09

## 최근 변경 요약(내장 모델 → 외부 API 전면 전환)
- 모든 내장(로컬) 모델 호출 경로 제거: LM Studio·GGUF(llama.cpp)·Gemma/A.X 멀티모달·로컬 임베딩·로컬 이미지 서버 코드 삭제
- LLM/전처리/후처리/이미지분석/임베딩 전부 외부 API만 사용 (openrouter 기본, openai/google/anthropic/upstage)
- `LocalLLMAgent` → `BaseAgent`(외부 API 전용)으로 교체, 임베딩 키 미설정 시 명확한 한국어 오류
- config.py: local_*/lm_studio_*/gemma/ax 설정 삭제, openrouter 중복 정의 제거, OPENROUTER_API_KEY 양쪽 철자 인식
- 상세: `docs/improvement/IMPROVEMENT_PLAN.md` Phase 0

## 최근 변경 요약(전처리/멀티모달/모델)
- 기본 전처리 모델(저비용·멀티모달 지향) 업데이트
  - OpenAI: `gpt-4o-mini`
  - Google: `gemini-1.5-flash-8b`
  - Anthropic: `claude-3-5-haiku-20241022`
- 멀티모달 전처리 기본값 ON: `ENABLE_MULTIMODAL_PREPROCESSING=true`(config 기본)
- 멀티모달 지원 목록 최신화 + 환경변수로 동적 확장
  - `EXTRA_MULTIMODAL_OPENAI_MODELS`, `EXTRA_MULTIMODAL_GOOGLE_MODELS`, `EXTRA_MULTIMODAL_ANTHROPIC_MODELS`
  - OpenAI 목록에 `gpt-5-mini`, `gpt-5-nano` 지원(공식 가격 페이지 미러 확인 근거)
- OpenRouter 기반 지능형 이미지 추출 기본 경로 도입
  - 프로바이더: `IMAGE_ANALYSIS_PROVIDER=openrouter`
  - 모델: `OPENROUTER_MM_MODEL=z-ai/glm-4.5v`
  - 키: `OPNEROUTER_API_KEY`(주의: 프로젝트 사양상 철자 고정)
- OpenAI 호출 안정성 개선: Responses API 우선 → 실패 시 Chat Completions 폴백
- UI 임포트 수정: `app.py`에 `from src.processing import PreprocessingModelFactory`

## 최근 변경 요약(프로젝트 정리 회기 2026-09-10)
- `.env.example`: config.py 참조 누락 환경변수 35개 보완(청킹·검색·스트리밍·LangSmith 등)
- 문서: `docs/LM_STUDIO_INTEGRATION.md` 삭제, ARCHITECTURE/PRD/FEATURE_SPEC 외부 API 전용 현행화
- pdf_converter 3모듈 분할: `pdf_heading_utils`(순수 헬퍼) + `pdf_text_extraction`(믹스인) + `pdf_semantic_chunking`(믹스인), 본체 826→195줄, 외부 import 경로 유지
- 대형 함수 분해: `pdf_loading._load_pdf_file`(206→폴백 체인 5메서드), `docx_loading._load_docx_file`(148→단계별), `semantic_chunker` 사전 데이터 모듈 상수화
- 회귀망: `tests/test_pdf_heading_utils.py` 신규(gitignore 예외 등록), 스모크 픽스처 `domain.md` 추가
- 2차 정리(같은 날): 잔여 대형 함수 분해 완료, `keyword_data.py`·`text_patterns.py`·`pdf_loading_helpers.py` 추출로 전 소스 600줄 미만, 미참조 `load_documents.py` 삭제, 삼켜진 예외 6곳 로깅 보강, `test_simple.py` 벡터DB 주입으로 E2E success 확보
- 잔여 관찰 대상: `docs/improvement/CLEANUP_PLAN_2026-09-10.md` 하단 참조

우리는 함께 프로덕션 수준의 코드를 개발하고 있습니다. 여러분의 역할은 유지보수 가능하고 효율적인 솔루션을 만들면서 잠재적인 문제를 미리 발견하는 것입니다.

제가 복잡하거나 막혀있는 것 같으면 방향을 잡아드릴 것이며, 제 가이드는 여러분이 목표에 집중할 수 있도록 도와줍니다.

## 🚨 자동화된 검사는 필수입니다

**모든 후크(hook) 문제는 차단 대상 - 모든 것이 ✅ 녹색이어야 합니다!**

오류 없음. 포맷팅 문제 없음. 린팅 문제 없음. 절대 용인하지 않습니다.
이는 제안이 아닙니다. 계속 진행하기 전에 모든 문제를 해결하세요.

### 프로젝트 분석시

- 한번에 많은 양의 코드를 분석하려고 하지 말 것
- 전체프로젝트의 소스 코드 전체를 분석하려고 하지 말 것
- 거시적으로 디렉토리 구조만 1차 파악
- 질문에 해당하는 파일만 축소해서 분석할 것
- 질문에 해당하는 로직만 찾아서 분석할 것
- 작업 계획을 세워서 단계별로 분석하기 쉬운 루트를 찾아서 분석하기

### 신규 코드 작성 및 리팩토링시

💎 DRY 원칙과 SOLID 원칙, 가독성 우선
RULE: 중복 코드 제거와 가독성을 동시에 추구하라
1.1 함수/메서드 설계

단일 책임 원칙: 하나의 함수는 하나의 기능만 수행
의미있는 네이밍: getUserData() ✅ vs getData() ❌
적절한 길이: 15줄 이내 권장, 최대 30줄 초과 금지

1.2 성능 고려사항
반복문 최적화: 불필요한 중첩 반복 제거
메모리 효율성: 대용량 데이터 처리 시 스트리밍 방식 고려
네트워크 호출: 배치 처리 및 캐싱 전략 적용

1.3 코드 경량화
하나의 소스코드 파일에 소스코드 라인이 600줄이 넘어 갈 경우 리팩토링 진행 고려

### 구현전 문서 작성

1. 코드 작성 및 구현 전에 항상 계획을 먼저 세우고 해당 계획을 잊어버리지 않도록 문서를 먼저 생성하도록 한다.
2. 필요한 설계 분석 문서들을 모두 작성할 것. 
3. PRD(요구사항정의서), 아키텍쳐 명세서, UI/UX 명세서, 기술 스택 명세서, 기능명세서, API 명세서, ERD 문서, 테이블 명세서, TODOLIST 문서 및 TASK 문서, 구현완료 후 추가 기능개선, 삭제, 수정 및 버그와 오류에 대한 전반적인 버전이력 문서
4. TODOLIST 또는 TASK 문서의 경우, 프로젝트 규모와 복잡성에 따라 대분류, 중분류, 소분류로 작업단위를 하나의 기능 단위로 최대한 세밀하고 상세하게 나눠서 작성할 것.  

### 구현과 문서의 일치

 - 사용자의 질문 요청과 구현 과정 속에서 업데이트되거나 확장된, 또는 삭제된 모든 설계 및 분석 문서들과의 최신정보를 일치시켜야 됨.

### 조사 → 계획 → 구현

**절대로 코딩에 바로 뛰어들지 마세요!** 다음 순서를 항상 따르세요:

1. **조사**: 코드베이스 탐색, 기존 패턴 이해
2. **계획**: 상세한 구현 계획 수립 및 제게 확인 요청
3. **구현**: 검증 지점과 함께 계획 실행

어떤 기능을 구현해야 할 때는 먼저 말씀드립니다: "구현 전에 코드베이스를 조사하고 계획을 세우겠습니다."

복잡한 아키텍처 결정이나 도전적인 문제의 경우, 최대한의 추론 능력을 발휘하기 위해 **"초고도 사고(ultrathink)"**를 사용합니다. "이 아키텍처에 대해 해결책을 제안하기 전에 깊이 생각해보겠습니다."

### 코드 구현 시

- 구현전 작성한 문서를 참조하고 특히 TODOLIST 문서를 통해 구현 진행 사항을 파악하며 진행.
- 무엇부터 구현해야 할지 계획을 세우기
- 계획은 내가 구현 완성도 기준으로 최적의 단위까지 세밀하게 나누어서 목록을 만들 것
- 목록을 작성 후 우선순위를 정하기, 우선순위는 개발 난이도, 선행되어야할 필수 세팅 등을 고려할 것
- 복잡하고 난이도 높은 로직은 더 작게 일 단위를 쪼개서 이해하기 쉬운 단계까지 나누어 진행할 것
- 한꺼번에 구현하지 말고 하나의 기능 구현이 완료 될 때 마다 테스트 후 다음 구현 단계로 넘어갈 것
- 구현이 완료되면 꼭 프로그램이 정상 동작하는지, 구현한 기능이 정상적으로 동작 하는지 테스트 할 것(playwright mcp 활용)
- 클린 코드 및 리팩토링 원칙을 잘 지쳐서 구현 되었는지 확인하고. 리팩토링이 필요하다 판단되면 리팩토링 후 동작에 문제가 없는지 다시 테스트 진행.
- 모든 구현 정보와 문서를 일치화 시킬 것.
- 정보가 부족한 도구, 라이브러리, 에이전트, llm 모델 코드 활용 - context7 mcp를 적극 활용하고 그래도 안될 경우 인터넷 검색 : 허깅페이스, 깃허브, 레퍼런스 문서 정보, 스택오버플로우, 레딧 등 기술 커뮤니티 정보를 적극활용하여 문제 해결에 이용할 것

### 현실 검증 지점

다음 순간에 **멈추고 검증**:

- 완전한 기능 구현 후
- 새로운 주요 구성 요소 시작 전
- 뭔가 잘못된 것 같을 때
- "완료" 선언 전
- **후크가 오류와 함께 실패할 때** ❌

실행: `make fmt && make test && make lint`

> 이유: 실제로 작동하는 것을 놓칠 수 있습니다. 이러한 검증 지점은 연쇄 실패를 방지합니다.
> 

### 🚨 중요: 후크 실패는 차단됩니다

**후크가 어떤 문제라도 보고하면(종료 코드 2) 반드시 다음을 수행해야 합니다:**

1. **즉시 중단** - 다른 작업 진행 금지
2. **모든 문제 해결** - 모든 ❌ 문제를 해결할 때까지 처리
3. **수정 확인** - 실패한 명령을 다시 실행하여 해결 확인
4. **원래 작업 계속** - 중단 전 하던 작업으로 돌아가기
5. **절대 무시 금지** - 경고 없음, 오직 요구사항만 존재

여기에는 다음이 포함됩니다:

- 포맷팅 문제 (gofmt, black, prettier 등)
- 린팅 위반 (golangci-lint, eslint 등)
- 금지된 패턴 (time.Sleep, panic(), interface{})
- 기타 모든 검사

코드는 100% 깨끗해야 합니다. 예외 없음.

**복구 프로토콜:**

- 후크 실패로 중단된 경우, 원래 작업에 대한 인식 유지
- 모든 문제 해결 및 수정 확인 후, 중단된 지점부터 계속 진행
- TODO 목록을 사용하여 수정 및 원래 작업 추적

## 작업 메모리 관리

### 컨텍스트가 길어질 때:

- [CLAUDE.md](http://claude.md/) 파일 다시 읽기(전역 문서, 프로젝트 문서 모두 포함)
- 설계 문서 다시 읽기
- [PROGRESS.md](http://progress.md/) 파일에 진행 상황 요약
- 주요 변경 전 현재 상태 문서화

### [TODO.md](http://todo.md/) 유지 관리:

```
## 현재 작업
- [ ] 지금 하고 있는 것

## 완료
- [x] 실제로 완료되고 테스트된 것

## 다음 단계
- [ ] 다음에 할 일

```

- var, any 같은 추상적인 변수, 클래스 타입 사용 금지 - 구체적인 타입 사용!
- **time.Sleep()** 또는 바쁜 대기 금지 - 동기화에는 채널 사용!
- 구형과 신형 코드를 함께 보관 금지
- 마이그레이션 함수 또는 호환성 계층 금지
- 버전이 붙은 함수 이름 금지 (processV2, handleNew)
- 사용자 정의 오류 구조체 계층 금지
- 최종 코드에 TODO 금지

### 필수 표준:

- 대체 시 이전 코드 삭제
- 의미 있는 이름: `userID`와 같이 사용, `id` 금지
- 중첩 줄이기 위한 조기 반환
- 생성자의 구체적인 타입: `func NewServer() *Server`
- 간단한 오류: `return fmt.Errorf("context: %w", err)`
- 복잡한 로직은 테이블 기반 테스트
- 동기화에 채널 사용: sleep 대신 채널로 준비 신호
- 타임아웃에는 select 사용: sleep 루프 대신 타임아웃 채널과 select 사용

### 오류 및 버그를 분석해야 할 때

1. 에러 로그 및 출력된 로그를 분석 
2. 현재의 로그 만으로는 오류 및 버그를 예측하기 어려울 경우 - 예상되는 로직 및 함수 호출부 등 상세한 로그를 추가 하여 로직을 추적하세요. 
3. 로그가 지나치게 많아져도 상관 없음. 분석에 필요하다면 과감하게 추가할 것
4. 테스트 코드 작성 - 복잡한 로직에 대해 검증할 수 있는 테스트 코드를 작성할 것. 
5. 어떤 버그에 대해서 어떤 구체적인 증상에 대한 정보를 받았다면 - playwright mcp 를 활용해서 실제 그러한지 재연하고 로그와 같이 분석 진행하기 
6. 로그는 운영서버에 배포시에는 노출되지 않도록 개발모드에서만 출력하도록 할 것. 
7. 버그 추적용 로그 상세 로그 생성의 경우는 주석처리하여 추후에도 사용할 수 있도록 할 것.    

### 함께 문제 해결하기

막히거나 혼란스러울 때:

1. **멈추기** - 복잡한 해결책으로 빠져들지 마세요
2. **위임** - 병렬 조사를 위한 에이전트 고려
3. **초고도 사고 또는 심층 사고모드 활성화** - 복잡한 문제의 경우 "이 도전을 깊이 생각해야 합니다"라고 말해 더 깊은 추론 유도
4. **뒤로 물러서기** - 요구사항 다시 읽기
5. **단순화** - 대개 간단한 해결책이 정확합니다
6. **질문하기** - "두 가지 접근법을 보았습니다: [A] 대 [B]. 어느 것을 선호하시나요?"
7. 상세로그 출력 및 수집 - 로그 값을 찍으며 버그 및 오류의 원인을 찾을 수 있습니다. 
8. playwright mcp 를 활용하여 테스트를 진행

### 중요: 초고도 사고 모드 활성화

 - 항상 심층추론, 심층 사고모드, Deep-think, Ultra-think를 활성화하여 작업을 진행할 것.
