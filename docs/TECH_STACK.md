# 기술 스택 명세서

## 1. 언어/런타임
- Python 3.10+

## 2. 주요 라이브러리
- Streamlit: UI
- LangChain: LLM/벡터 스토어/리트리버 연동
- FAISS/Chroma: 벡터 DB
- sentence-transformers/HuggingFaceEmbeddings: 임베딩(오픈소스)
- Upstage/OpenAI/Google/Anthropic: 상용 LLM/임베딩 백엔드
- Local HTTP LLM: OpenAI 호환 서버(`{BASE}/v1`) — 채팅/임베딩 엔드포인트 제공
- OpenRouter: 외부 멀티모달 분석(Preprocessing) — 기본 `z-ai/glm-4.5v`
- scikit-learn: TF‑IDF/코사인 유사도

## 3. 구조/모듈
- `src/embeddings`: EmbeddingModel(배치, 재시도, 로컬 HTTP 임베딩 지원)
- `src/loaders`: DocumentLoader/PDF 확장
- `src/vectorstore`: EnhancedVectorDatabase(하이브리드/MMR/임계값)
- `src/rag`: RAGChain(+Chunker/요약/병렬)
- `src/utils/openrouter_image_service.py`: 이미지 분석/OCR(OpenRouter 전용, 폴백 없음)
- `run_rag.py`: CLI 허브, 앱/설정/테스트
- `app.py`: Streamlit 앱(스트리밍)

## 4. 설정/비밀정보
- `.env`에 API 키/로컬 서버 주소, `config.py`에 기본값.
 - 멀티 서버: `LOCAL_LLM_BASE_URLS`(콤마 구분)로 여러 엔드포인트 선언
 - OpenRouter: `OPENROUTER_API_KEY`, `OPENROUTER_MM_MODEL`(기본: `z-ai/glm-4.5v`)
 - 멀티모달 모델 확장: `EXTRA_MULTIMODAL_*` 3종(OpenAI/Google/Anthropic)
- 텔레메트리 비활성화 환경 변수 설정.

## 5. 테스트/품질
- pytest 기반 스모크 및 디버그 테스트.
- 권장 도구(추가): `ruff`(린트/포맷), `black`(포맷), `mypy`(타입 체크).
