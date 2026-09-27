# 기능 명세서

## 1. 문서 처리
- 입력 형식: PDF(선택 OCR), Markdown, TXT
- 전처리: 텍스트 정리/불용어 제거 옵션, 청크 분할
- 출력: LangChain `Document`(content+metadata)
 - 이미지 처리(지능형):
   - 경로: OpenRouter만 사용(폴백 없음, 엄격 모드)
   - OpenRouter 기본 모델: `qwen/qwen3.8-flash`
   - 공통: `POST {BASE}/v1/chat/completions` 멀티모달로 OCR/관련성/설명 동시 요청
 - 임계값(`LOCAL_IMAGE_RELEVANCE_THRESHOLD`) 이상만 저장
 - 실패 시 폴백 없음: 오류를 표면화하여 즉시 중단(운영 정책)
 - 큐 혼잡 대응: OpenRouter 측 Rate Limit 재시도(지수 백오프) 적용
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
- 프롬프트: 제공자 공통 단일 템플릿(`핵심 요약/상세 설명/참고 자료` 구조) — `src/rag/llm_manager.py`의 `create_prompt_template()`
- 대용량 컨텍스트: 분할→요약→병렬 처리→병합
- 모델 초기화: provider(openrouter 기본, openai/google/anthropic)에 따른 설정/토큰/컨텍스트 제어
- 스트리밍: 콜백을 통해 토큰 단위 출력
- 응답 포맷터: 모델 출력이 지침을 어겨도 `핵심 요약/상세 설명/참고 자료` 구조로 재배치하고, 문단 간 공백·불릿·표를 보정.

## 7. 문서 LLM 교정 및 모델 목록
- 문서 LLM 교정: PDF 변환 결과(`converted_docs/*.md`)를 MD 후처리 한 번으로 교정 → `processed_docs/*_processed.md`(검토 후 색인)
  - `src/utils/chunked_cleanup.py`: 문장이 끝난 문단에서만 약 3,000자로 분할, 앞뒤 청크 원문을 참고 문맥으로 전달, 3개 병렬
  - 검증: 공백 제외 글자 수 85~115%, `[페이지 N]` 표시·표 행 수 동일 — 어긋난 청크는 원문 유지
  - 페이지 경계 문장은 LLM 전에 규칙으로 연결(`sentence_completion.join_page_boundaries`)
- 기본 모델(저비용 지향):
  - OpenAI: `gpt-6-luna`, Google: `gemini-3.5-flash-lite`, Anthropic: `claude-haiku-4-5-20251001`, OpenRouter: `qwen/qwen3.8-flash`
- 지원 목록 동적 확장(.env):
  - `EXTRA_MULTIMODAL_OPENAI_MODELS`, `EXTRA_MULTIMODAL_GOOGLE_MODELS`, `EXTRA_MULTIMODAL_ANTHROPIC_MODELS`
  - 예: `EXTRA_MULTIMODAL_OPENAI_MODELS=gpt-6-sol,gpt-5.4-mini`

## 5. 실행/운영
- run_rag.py 메뉴: 앱 실행/벡터DB 관리/테스트/설정/정보
- 설정: `.env` 자동 생성 템플릿 제공
- 로그/진단: 진행상황 로그, 검색 수/길이 표시

## 6. 에러 처리
- 키 누락/연결 불가/Rate Limit → 사용자 안내 및 재시도/폴백
- 인덱스 유실/호환성 → 경고 후 재생성 경로 안내
