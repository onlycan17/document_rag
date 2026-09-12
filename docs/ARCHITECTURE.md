# 아키텍처 명세서

본 문서는 코드베이스(src/*, run_rag.py, app.py)를 기준으로 실제 동작 구조를 설명합니다. 어려운 용어는 괄호로 풀이합니다.

## 1. 전체 개요
- 유형: RAG 파이프라인(용어(설명: 문서→임베딩→검색→LLM 생성)).
- UI: Streamlit(app.py), CLI: run_rag.py.
- 코어: `src/rag/RAGChain`이 검색→컨텍스트 최적화→LLM 호출을 오케스트레이션(용어(설명: 여러 단계를 순서대로 조율))합니다.
- 전처리: PDF 이미지 추출 후 OpenRouter(멀티모달 chat/completions)로 OCR/관련성 판정 → 관련 이미지만 저장. 실패 시 폴백 없이 즉시 중단(엄격 모드).

## 2. 컴포넌트
- `src/loaders/DocumentLoader`: PDF/OCR/Markdown/TXT 로딩 및 전처리/청크 분할.
   - 지능형 이미지 처리: OpenRouter만 사용(폴백 없음, 엄격 모드)
     - OpenRouter: `src/utils/openrouter_image_service.py`(모델: `z-ai/glm-4.5v`)
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
- `src/utils/answer_formatter`: RAG 응답을 `핵심 요약/상세 설명/참고 자료` 구조로 정리하고 글머리표/표/문단 간격을 보정하는 계층. 스트리밍·동기 응답에서 공통 사용.
- `src/utils/answer_render`: 포맷된 응답을 화면용으로 정규화(붙은 헤딩·불릿 분리)하고 `[출처 N]` 뱃지·강조를 안전하게 HTML로 변환하며 섹션으로 분해하는 순수 헬퍼.
- `config.py`: 전역 설정(모델/토큰/DB 경로/검색 옵션 등).
- `run_rag.py`: 실행 허브(앱 실행/벡터DB 관리/테스트/설정).
- `app.py`: Streamlit UI(사이드바 옵션/스트리밍 토글 등). 커스텀 CSS는 `ui/styles.py`의 `inject_custom_css()`로 주입.

## 3. 데이터 흐름(간단 다이어그램)

```
 [data/documents/*] → Loader → Chunks → Embeddings → Vector DB(FAISS/Chroma)
                                                ↑
  User Query → RAGChain.preprocess → VectorDB.search → 후보 문서 → 컨텍스트 최적화/요약 → Prompt → LLM → 답변 스트리밍 → AnswerFormatter 구조화 출력
```

비유: “도서관에서(벡터 DB) 관련 책(문서)을 고른 뒤(검색), 필요한 페이지만 추려서(컨텍스트 최적화) 선생님(LLM)에게 보여주고, 필기 선생님(AnswerFormatter)이 보기 좋게 정리해주는 과정”과 같습니다.

## 4. LLM/컨텍스트 전략
- 모델별 컨텍스트/토큰 한도 계산(안전 마진) 후 최대 컨텍스트를 문자 수로 환산.
- 대량 문서: 임계값 초과 시 컨텍스트 분할→부분 요약→병렬 처리로 결합.

## 5. 검색 전략
- 기본: 벡터 유사도 검색(similarity, 거리 낮을수록 유사).
- 옵션: 
  - 하이브리드(BM25/TF‑IDF) 결과와 벡터 결과 가중 통합(가중 평균).
  - MMR(용어(설명: 상호 유사하지 않게 다양성을 보장하도록 선택)).
  - 동적 임계값(검색 품질 낮을 때 완화).

## 6. 설정/보안
- `.env`에 API 키(Upstage/OpenAI/Google/Anthropic/OpenRouter) 설정. 모든 LLM 호출은 외부 API 전용(로컬 모델 미지원).
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

## 10. 아키텍처 개선 방향

### 현재 아키텍처 한계점
- **모놀리식 구조**: 단일 애플리케이션에서 모든 기능 처리로 인한 확장성 제한
- **동기적 처리**: 문서 업로드 시 UI 블로킹 발생
- **메모리 사용량**: 문서 캐시와 TF-IDF 매트릭스의 메모리 상주 문제
- **강한 결합**: 컴포넌트 간 의존성으로 인한 유지보수 어려움

### 개선 계획 (3단계 접근)

#### 1단계: 즉시 개선 (High Impact, Low Effort)
- **문서 처리 모듈 분리**: `EnhancedDocumentLoader` 클래스를 작은 책임 단위로 분리 (SRP 적용)
- **비동기 처리 도입**: 문서 업로드 시 백그라운드 처리로 UI 응답성 개선
- **캐시 계층 추가**: Redis를 이용한 검색 결과 캐싱으로 성능 향상

#### 2단계: 중기 개선 (Medium Impact, Medium Effort)
- **마이크로서비스 아키텍처 전환**:
  - 문서 처리 서비스 (Document Processing Service)
  - 벡터 데이터베이스 서비스 (Vector DB Service)
  - 검색 서비스 (Search Service)
- **API Gateway 도입**: 통합 엔드포인트 제공 및 로드 밸런싱
- **설정 관리 중앙화**: 중앙 집중식 설정 관리 시스템 구축
- **모니터링 시스템 강화**: 성능 메트릭스와 로깅 개선

#### 3단계: 장기 개선 (High Impact, High Effort)
- **분산 벡터 데이터베이스**: 여러 벡터 DB 샤드에 문서 분산 저장
- **머신러닝 파이프라인**: 문서 품질 자동 평가 및 최적화 시스템
- **실시간 스트리밍 아키텍처**: WebSocket을 이용한 실시간 문서 처리 상태 업데이트
- **이벤트 기반 아키텍처**: 문서 처리 상태를 이벤트로 발행하는 시스템

### 개선 후 아키텍처 다이어그램 (목표)

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Streamlit UI │ ── │   API Gateway    │ ── │ Document        │
└─────────────┘    └──────────────────┘    │ Processor       │
                         │                 └─────────────────┘
                         │                 ┌─────────────────┐
                         ├─────────────────│ Vector DB        │
                         │                 │ Service         │
                         │                 └─────────────────┘
                         │                 ┌─────────────────┐
                         └─────────────────│ Search Service  │
                                           └─────────────────┘
```

### 예상 성능 향상
- **문서 처리 속도**: 비동기 처리로 60% 이상 개선
- **검색 응답 시간**: 캐싱 계층으로 70% 이상 개선
- **확장성**: 마이크로서비스로 수평 확장 가능
- **가용성**: 서비스 분리로 장애 격리 및 복구 시간 단축
