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
- `class VectorDatabase`
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
- 필수(택1): `UPSTAGE_API_KEY` 또는 `OPENAI_API_KEY` 등 LLM API 키 (외부 API만 지원)
- 선택: `ANONYMIZED_TELEMETRY=False`, `CHROMA_TELEMETRY=False`

### 3.1 이미지 분석 연동 (외부 API)
- 모듈: `src/utils/openrouter_image_service.py`, `src/utils/image_analyzer.py`
- OCR/관련성 판정: OpenRouter `POST /v1/chat/completions` 멀티모달 메시지로 이미지 전달
  - 프롬프트: 관련성 0~1, 설명, 텍스트 추출 형식으로 응답 지시
  - 응답 파싱: `RELEVANCE:`/`DESCRIPTION:`/`TEXT:` 라벨 기반 단순 파싱
- PDF 파이프라인: `DocumentLoader._load_pdf_file()`에서 이미지 분석을 외부 API로 수행

#### OpenRouter 설정
- 키: `.env`의 `OPENROUTER_API_KEY` (하위 호환 `OPNEROUTER_API_KEY`도 인식)
- 기본 모델: `z-ai/glm-4.5v`
- 엔드포인트: `https://openrouter.ai/api/v1/chat/completions`

#### 멀티모달 모델 확장(.env)
- `EXTRA_MULTIMODAL_OPENAI_MODELS`: OpenAI 멀티모달 모델 추가(예: `gpt-5-mini,gpt-5-nano`)
- `EXTRA_MULTIMODAL_GOOGLE_MODELS`: Google 멀티모달 모델 추가(예: `gemini-2.5-pro`)
- `EXTRA_MULTIMODAL_ANTHROPIC_MODELS`: Anthropic 멀티모달 모델 추가

#### 이미지 향상(옵션)
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
