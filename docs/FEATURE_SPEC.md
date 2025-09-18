# 기능 명세서

## 1. 문서 처리
- 입력 형식: PDF(선택 OCR), Markdown, TXT
- 전처리: 텍스트 정리/불용어 제거 옵션, 청크 분할
- 출력: LangChain `Document`(content+metadata)
 - 이미지 처리(지능형):
   - 우선 순위: OpenRouter(기본) → 로컬 HTTP 서버 → 내장 로컬 모델
   - OpenRouter 기본 모델: `z-ai/glm-4.5v`
   - 공통: `POST {BASE}/v1/chat/completions` 멀티모달로 OCR/관련성/설명 동시 요청
 - 임계값(`LOCAL_IMAGE_RELEVANCE_THRESHOLD`) 이상만 저장
 - 실패 시 순차 폴백: OpenRouter → 로컬 서버 → 내장 로컬 모델
 - 큐 혼잡 대응: 로컬 서버 `/v1/queue/stats`로 혼잡 시 스로틀(소폭 대기)
 - 선택적 이미지 향상: `ENABLE_IMAGE_ENHANCEMENT=true` 시 `/v1/images/edits`로 품질 개선 후 분석

-## 2. 임베딩
- 제공자: Upstage/OpenAI/Local HTTP/HuggingFace
- 특징: 배치 처리, 429/네트워크 오류 재시도(지수 백오프, 용어(설명: 실패 때마다 대기시간을 2배로 늘려가며 재시도))
- 차원: 선택 모델에 따름(EmbeddingModel.get_embedding_dimension)

## 2. 임베딩
- 제공자: Upstage/OpenAI/Local HTTP/HuggingFace
- 특징: 배치 처리, 429/네트워크 오류 재시도(지수 백오프)
- 차원: 선택 모델에 따름(EmbeddingModel.get_embedding_dimension)

## 3. 벡터 DB
- 선택: FAISS(기본)/Chroma
- 하이브리드 검색: 벡터 + BM25/TF‑IDF 결과 가중 통합
- MMR: 다양성 확보로 중복 청크 억제
- 임계값: 점수 기반 동적 완화(결과 부족 시 완화)

## 4. RAG 체인
- 프롬프트: Qwen(GGUF) ChatML 전용/일반 템플릿 2체계
- 대용량 컨텍스트: 분할→요약→병렬 처리→병합
- 모델 초기화: provider에 따른 설정/토큰/컨텍스트 제어
- local 프로바이더: OpenAI 호환 HTTP 엔드포인트(`{BASE}/v1/chat/completions`) 사용
 - 스트리밍: 콜백을 통해 토큰 단위 출력
 - 멀티 서버: `LOCAL_LLM_BASE_URLS`로 다수 서버 모델 목록 통합, 선택된 모델의 `base`에 라우팅

## 7. 멀티모달 전처리 및 모델 목록 확장
- 멀티모달 전처리 기본 ON(`ENABLE_MULTIMODAL_PREPROCESSING=true`)
- 기본 전처리 모델(저비용 지향):
  - OpenAI: `gpt-4o-mini`, Google: `gemini-1.5-flash-8b`, Anthropic: `claude-3-5-haiku-20241022`
- 지원 목록 동적 확장(.env):
  - `EXTRA_MULTIMODAL_OPENAI_MODELS`, `EXTRA_MULTIMODAL_GOOGLE_MODELS`, `EXTRA_MULTIMODAL_ANTHROPIC_MODELS`
  - 예: `EXTRA_MULTIMODAL_OPENAI_MODELS=gpt-5-mini,gpt-5-nano`

## 5. 실행/운영
- run_rag.py 메뉴: 앱 실행/벡터DB 관리/테스트/설정/정보
- 설정: `.env` 자동 생성 템플릿 제공
- 로그/진단: 진행상황 로그, 검색 수/길이 표시

## 6. 에러 처리
- 키 누락/연결 불가/Rate Limit → 사용자 안내 및 재시도/폴백
- 인덱스 유실/호환성 → 경고 후 재생성 경로 안내
