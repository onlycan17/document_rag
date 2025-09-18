# 아키텍처 명세서

본 문서는 코드베이스(src/*, run_rag.py, app.py)를 기준으로 실제 동작 구조를 설명합니다. 어려운 용어는 괄호로 풀이합니다.

## 1. 전체 개요
- 유형: RAG 파이프라인(용어(설명: 문서→임베딩→검색→LLM 생성)).
- UI: Streamlit(app.py), CLI: run_rag.py.
- 코어: `src/rag/RAGChain`이 검색→컨텍스트 최적화→LLM 호출을 오케스트레이션(용어(설명: 여러 단계를 순서대로 조율))합니다.
- 전처리: PDF 이미지 추출 후 OpenRouter(멀티모달 chat/completions)로 OCR/관련성 판정 → 관련 이미지만 저장. 실패 시 로컬 HTTP 서버 → 내장 로컬 모델 순으로 폴백.

## 2. 컴포넌트
- `src/loaders/DocumentLoader`: PDF/OCR/Markdown/TXT 로딩 및 전처리/청크 분할.
   - 지능형 이미지 처리(우선순위): `openrouter` → `local` → `내장`
     - OpenRouter: `src/utils/openrouter_image_service.py`(모델: `z-ai/glm-4.5v`)
     - Local: `src/utils/local_image_service.py`
- `src/embeddings/EmbeddingModel`: Upstage/OpenAI/HuggingFace 임베딩 선택, 배치 처리, 재시도.
- `src/vectorstore/VectorDatabase`(실제 구현은 EnhancedVectorDatabase): FAISS/Chroma 선택, 하이브리드 검색(BM25+TF‑IDF)·MMR·임계값 동적 조정.
- `src/rag/RAGChain`: 
  - LLM 초기화(provider: openai/google/anthropic/local)
  - local: OpenAI 호환 HTTP 엔드포인트 사용(`{BASE}/v1`), 멀티 서버 지원
    - `LOCAL_LLM_BASE_URLS`(콤마 구분) 순회로 모델 목록 통합
    - 선택된 모델의 `model` 값에 `"<id>|<base>"`를 포함해 서버 라우팅
    - 헬스: `GET {BASE}/health`, 큐: `GET {BASE}/v1/queue/stats`, 모델: `GET {BASE}/v1/models`
    - 채팅: `POST {BASE}/v1/chat/completions`(스트리밍 지원), 임베딩: `POST {BASE}/v1/embeddings`
    - 멀티모달(이미지 분석)은 1620 포트 서버의 모델만 사용(현재 구성)
  - 대용량 컨텍스트 처리(ContextChunker, Summarizer, 병렬 처리)
  - 프롬프트 템플릿(Qwen ChatML/일반 템플릿)
  - 스트리밍/비스트리밍 체인
- `config.py`: 전역 설정(모델/토큰/DB 경로/검색 옵션 등).
- `run_rag.py`: 실행 허브(앱 실행/벡터DB 관리/테스트/설정).
- `app.py`: Streamlit UI(사이드바 옵션/스트리밍 토글 등).

## 3. 데이터 흐름(간단 다이어그램)

```
 [data/documents/*] → Loader → Chunks → Embeddings → Vector DB(FAISS/Chroma)
                                                ↑
  User Query → RAGChain.preprocess → VectorDB.search → 후보 문서 → 컨텍스트 최적화/요약 → Prompt → LLM → 답변 스트리밍
```

비유: “도서관에서(벡터 DB) 관련 책(문서)을 고른 뒤(검색), 필요한 페이지만 추려서(컨텍스트 최적화) 선생님(LLM)에게 보여주고 답을 듣는 과정”과 같습니다.

## 4. LLM/컨텍스트 전략
- 모델별 컨텍스트/토큰 한도 계산(안전 마진) 후 최대 컨텍스트를 문자 수로 환산.
- 대량 문서: 임계값 초과 시 컨텍스트 분할→부분 요약→병렬 처리로 결합.
- 로컬 LLM(GGUF)일 때 샘플링 파라미터(Qwen/Midm 특화) 및 n_ctx 보호.

### LM Studio (로컬 모델 호스팅) 통합

- 본 프로젝트는 로컬 LM Studio 인스턴스에서 실행 중인 모델을 자동으로 탐색하여 UI에 노출합니다. 탐색은 HTTP API(`/v1/models`) 호출을 우선 시도하고 실패 시 로컬 모델 디렉토리(`~/Library/Application Support/lm-studio/models`)를 스캔하여 모델 파일을 감지합니다.
- 발견된 모델들은 `ModelRegistry.get_all_models()`를 통해 'local' 제공자 항목으로 병합되어 사용자가 사이드바에서 선택할 수 있습니다. 구체 구현 파일: `src/models/lm_studio.py`, `src/models/model_registry.py`, `src/rag/llm_manager.py`.

## 5. 검색 전략
- 기본: 벡터 유사도 검색(similarity, 거리 낮을수록 유사).
- 옵션: 
  - 하이브리드(BM25/TF‑IDF) 결과와 벡터 결과 가중 통합(가중 평균).
  - MMR(용어(설명: 상호 유사하지 않게 다양성을 보장하도록 선택)).
  - 동적 임계값(검색 품질 낮을 때 완화).

## 6. 설정/보안
- `.env`에 API 키(Upstage/OpenAI/Google/Anthropic) 또는 로컬 모델 경로.
- `config.settings`로 모든 파라미터 제어(검색 k, 임계값, MMR 다양성, 스트리밍 여부 등).
- 텔레메트리 비활성화: `ANONYMIZED_TELEMETRY=False`, `CHROMA_TELEMETRY=False`.

## 7. 로깅/오류 복구
- 재시도: 임베딩 API(429/네트워크 오류 지수 백오프), 검색 실패 시 폴백.
- 상태 노출: 스트리밍 콜백, 검색 문서 수/길이 로그.

## 8. 성능/규모
- 배치 임베딩, 병렬 처리, 컨텍스트 길이 제어로 안정성 확보.
- 단일 노드 기준(필요 시 FAISS 인덱스 스냅샷/복구).

## 9. 테스트/운영
- 테스트: `tests/test_simple.py` 스모크, 디버그용 스크립트.
- 운영: `run_rag.py` 메뉴 기반 실행/설정/DB 관리.
