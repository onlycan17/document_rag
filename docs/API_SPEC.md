# API 명세서(내부 파이썬/CLI 중심)

본 프로젝트는 외부 REST API 서버가 아니라, 내부 파이썬/CLI 진입점 중심입니다. 아래는 개발자용 명세입니다.

## 1. Python 클래스/메서드

### `src/rag/rag_chain.py`
- `class RAGChain(provider: Optional[str], model: Optional[str], vector_db: Optional[VectorDatabase])`
  - 역할: 검색→컨텍스트 최적화→LLM 응답 생성 전체 오케스트레이션(용어(설명: 단계 조율)).
  - 주요 메서드:
    - `query(question: str) -> dict`
      - 입력: 사용자 질문 문자열
      - 출력: `{answer: str, sources: list, status: str, search_info: dict}`
      - 동작: 전처리→검색→대용량 분기→프롬프트→LLM 호출.
    - `get_last_context_tokens() -> int`: 마지막 컨텍스트 토큰 수

### `src/vectorstore/vector_db.py`
- `class EnhancedVectorDatabase`
  - `add_documents(docs: List[Document])` : 문서 인덱싱/캐시/키워드 인덱스 구축
  - `search(query: str, k: int=None) -> List[Tuple[Document, float]]` : 하이브리드/MMR 포함 검색
  - `clear_database()` : 인덱스/캐시 삭제
  - `get_document_count() -> int`

### `src/embeddings/embedding_model.py`
- `class EmbeddingModel`
  - `embed_documents(texts: List[str]) -> List[List[float]]`
  - `embed_query(text: str) -> List[float]`

## 2. CLI/스크립트

### `run_rag.py`
- `1) Streamlit 앱 실행` : `streamlit run app.py`
- `2) 벡터 DB 관리` : `python scripts/data_management/vector_db_manager.py [--safe-mode|--markdown-only|--files ...]`
- `3) 테스트 실행` : `python tests/test_*.py` 선택 실행
- `4) 환경 설정` : `pip install -r requirements.txt`, `install_ocr.sh`, `.env` 템플릿 생성
- `5) 프로젝트 정보` : 구조/설정 상태 요약

### `scripts/data_management/vector_db_manager.py`
- 옵션: `--safe-mode`, `--markdown-only`, `--files`, `--types`, `--help`
- 역할: 문서 로드→전처리→인덱싱 일괄 수행

## 3. 설정(환경 변수)
- 필수(택1): `UPSTAGE_API_KEY` 또는 `OPENAI_API_KEY` 등 LLM 키, 혹은 로컬 LLM 설정
- 로컬 LLM(HTTP) 기본값:
  - `LOCAL_LLM_BASE_URL=http://210.126.109.57:1620`
  - `LOCAL_LLM_API_KEY`(일부 서버는 불필요, 기본 `not-needed`)
- 선택: `ANONYMIZED_TELEMETRY=False`, `CHROMA_TELEMETRY=False`

## 3.1 로컬 HTTP 엔드포인트(최신)
- `GET /health` : 서비스 상태
- `GET /v1/queue/stats` : 동시성/대기열 상태
- `GET /v1/models` : 사용 가능 모델 목록
- `POST /v1/chat/completions` : 채팅(스트리밍 지원)
- `POST /v1/embeddings` : 임베딩 생성
- `POST /v1/images/generations`, `POST /v1/images/edits` : 이미지 생성/편집(옵션)

### 이미지 처리 연동
- 모듈: `src/utils/local_image_service.py`(로컬), `src/utils/openrouter_image_service.py`(외부)
- OCR/관련성 판정: `POST {BASE}/v1/chat/completions` 멀티모달 메시지로 이미지 전달(OpenAI 호환)
  - 프롬프트: 관련성 0~1, 설명, 텍스트 추출 형식으로 응답 지시
  - 응답 파싱: `RELEVANCE:`/`DESCRIPTION:`/`TEXT:` 라벨 기반 단순 파싱
- PDF 파이프라인: `DocumentLoader._load_pdf_file()`에서 서버 우선 사용, 실패 시 내장 추출기로 폴백

#### OpenRouter 설정
- 키: `.env`의 `OPNEROUTER_API_KEY`
- 기본 모델: `z-ai/glm-4.5v`
- 엔드포인트: `https://openrouter.ai/api/v1/chat/completions`

#### 멀티모달 모델 확장(.env)
- `EXTRA_MULTIMODAL_OPENAI_MODELS`: OpenAI 멀티모달 모델 추가(예: `gpt-5-mini,gpt-5-nano`)
- `EXTRA_MULTIMODAL_GOOGLE_MODELS`: Google 멀티모달 모델 추가(예: `gemini-2.5-pro`)
- `EXTRA_MULTIMODAL_ANTHROPIC_MODELS`: Anthropic 멀티모달 모델 추가

#### 로컬 멀티모달 선호
- 멀티 서버 환경에서 1620 포트를 우선 사용(현재 멀티모달 모델은 1620에만 상주)

#### 큐 상태 기반 스로틀링
- 로컬 서버: `GET /v1/queue/stats` 응답을 사용해 혼잡 시 소폭 대기(스로틀)
- 기준(예시): `pending > 2*concurrency` 또는 `running >= concurrency`이면 0.5초 대기

#### 이미지 향상(옵션)
- 로컬: `POST /v1/images/edits` 활용, 실패 시 원본 사용
- 설정: `ENABLE_IMAGE_ENHANCEMENT=true`와 `IMAGE_ENHANCEMENT_PROMPT`로 제어

## 4. 사용 예시

```python
from src.rag import RAGChain
rag = RAGChain(provider="openai", model="gpt-4o-mini")
result = rag.query("지침 문서에서 보안 관련 핵심 원칙은?")
print(result["answer"])  # 한국어 요약 + 출처
```

```bash
python run_rag.py           # 메뉴 실행
python tests/test_simple.py # 스모크 테스트
```
#### 로컬 모델 선택(멀티 서버)
- 환경: 여러 로컬 서버가 각 포트에서 동작(예: 1620/1621/1622)
- 설정: `LOCAL_LLM_BASE_URLS=http://210.126.109.57:1620,http://210.126.109.57:1621,http://210.126.109.57:1622`
- 모델 목록 조회: 각 서버의 `/v1/models`를 순회해 통합
- 모델 식별자: `model` 필드에 `"<id>|<base>"` 형식 저장
  - 예: `exaone-mini|http://210.126.109.57:1621`
- 선택 시 라우팅: `model`에 포함된 `base`를 파싱해 해당 서버로 호출
