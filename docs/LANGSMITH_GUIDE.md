# LangSmith 트레이싱 사용 가이드

RAG 검색→생성 전 과정의 트레이스를 **LangSmith(SaaS)** 에 수집·분석한다.
LLM/전처리 호출이 외부 API 전용이므로 외부망(개발·배포 환경)에서 사용한다.

> 내부망 등 SaaS 미접속 환경에서는 `LANGSMITH_TRACING=false`로 두면 단순 no-op로
> 동작하며 앱 기능에는 영향이 없다. (셀프호스팅이 필요한 경우 LangSmith의
> 셀프호스팅 배포 또는 다른 오픈소스 관측 도구를 별도로 도입할 것)

## 아키텍처

```
[RAG 질의 흐름 (run 계층)]
rag-query (chain, trace 루트 — query_engine)
 ├─ vector-search (retriever — vector_db.search, 출처·점수 기록)
 ├─ llm-generation (llm — BaseAgent._call_llm, 모델명·토큰·메타 기록)
 └─ ChatOpenAI (llm — LangChain 자동 관측, LANGSMITH_TRACING env로 활성)

         │  src/utils/tracing.py 헬퍼 (observe_if_enabled / record_*)
         ▼
[LangSmith SDK] ──▶ [LangSmith SaaS (LANGSMITH_ENDPOINT)]
```

대화별 묶음은 Streamlit 세션 ID(`st.session_state.langsmith_session_id`)를
run의 `session_id` 메타데이터로 전파한다(`propagate_trace_attributes`).
`ensure_langsmith_env()`가 `config.settings`의 LangSmith 설정을 실제 환경변수로
반영해 랭체인 자동 관측을 활성화한다.

## 활성화 방법 (필수)

LangSmith는 **환경변수로만** 관측이 활성화된다. `.env`에 설정하세요.

```bash
# .env
LANGSMITH_TRACING=true        # 관측 활성화
LANGSMITH_API_KEY=lsv2-...    # LangSmith 계정에서 발급한 API 키
LANGSMITH_PROJECT=ragTest     # 프로젝트명 (미지정 시 default)
# LANGSMITH_ENDPOINT=...      # 셀프호스팅 시에만 (SaaS면 생략)
```

앱 시작 시 `app.py`에서 `ensure_langsmith_env()`가 `config.settings`
(`LANGSMITH_*` 변수)를 실제 환경변수에 반영하므로, `.env`만 설정하면
치환 없이 동작한다.

## 관측 진입점

계측은 `src/utils/tracing.py` 한 곳에 모여 있다 (관측 규약: 함수 인자 자동
수집 금지, `record_*`로 필요한 값만 명시 기록 — 시크릿·프롬프트 노출 방지):

- `ensure_langsmith_env()`: config의 LangSmith 설정을 환경변수로 반영 (앱 시작 시 1회)
- `observe_if_enabled(name, as_type, capture_input, capture_output)`: 조건부
  `@traceable` 데코레이터. `as_type` → run_type 매핑: span/chain→chain,
  generation→llm, retriever→retriever, tool→tool
- `record_generation(model, input, usage_details, metadata)`: llm run에 모델명·
  토큰 사용량·메타 기록
- `record_input` / `record_output` / `record_retriever_output`: run에 입력·출력·
  검색 출처 요약 기록 (내부적으로 `get_current_run_tree()`로 현재 run 갱신)
- `propagate_trace_attributes(session_id, user_id)`: `tracing_context`로 run에
  세션·사용자 메타데이터 전파

### 랭체인 자동 관측

`prompt | llm` 체인(`llm_manager.create_llm` → `query_engine`)은
`LANGSMITH_TRACING=true`로 **자동 관측**된다. 별도 CallbackHandler를 붙이지
않는다 (Langfuse 시절의 `langfuse_callbacks` 제거됨). 모델명·토큰 사용량·지연
시간이 자동으로 기록된다.

## 계측 위치

계측 위치 (검색→생성 5단계 분리):

1. `query_engine.preprocess_query`: `query-preprocessing` span
2. `vector_db.search`: `vector-retrieval` retriever (top_k·소요시간·캐시 히트 메타, 출처·점수 출력)
3. `document_processor.prepare_documents`: `reranking` span (중복제거·재랭킹, in/out 문서수·소요시간 메타)
4. `document_processor.format_documents`: `context-compression` span (컨텍스트 압축, 문서수·컨텍스트 길이 메타)
5. `query_engine._invoke_chain`/`_stream_chain_stream`: `answer-generation` span (표준·스트리밍 답변 생성)

각 단계는 `rag-query` 체인 루트의 형제(sibling) run으로 중첩된다. 그 외:
- `base_agent._call_llm`: 전처리 에이전트 llm run (모델명·토큰 기록)
- `chat_interface.py`: 세션 ID 생성·전달

> 주의: `document_processor.format_documents`는 내부에서 `prepare_documents`를 더 이상
> 재호출하지 않는다(의존 호출 시 reranking이 context-compression 안에 중첩되므로).
> 컨텍스트를 만들기 전에 반드시 `prepare_documents`를 먼저 호출할 것.

## 웹 UI

- 주소: https://smith.langchain.com (계정 대시보드)
- `LANGSMITH_PROJECT`로 지정한 프로젝트에 run이 누적된다
- 화면별 용도:
  - **Projects→Traces**: 질의별 LLM 호출 전체 기록 (체인 계층, 지연, 토큰·비용)
  - **Monitor**: 지연/비용/호출 수 추이 대시보드
  - **Datasets / Evaluation**: 평가 데이터셋과 LLM-as-judge 채점
  - **Insights**: 모델·프롬프트별 성능 비교

## 문제 해결

| 증상 | 원인/조치 |
|---|---|
| 트레이스가 안 보임 | `.env`의 `LANGSMITH_TRACING=true`·`LANGSMITH_API_KEY` 확인, 앱 프로세스 재시작 |
| 비용 열이 비어 있음 | 공급자 가격 미등록 모델 — LangSmith 콘솔에서 가격 설정 또는 metadata로 직접 기록 |
| 내부망에서 미동작 | `LANGSMITH_TRACING=false`로 꺼서 no-op 폴백 (셀프호스팅 배포는 별도 구성) |

## 검색 평가 결과 업로드

검색 평가(`scripts/eval/retrieval_eval.py`)는 랭체인/LangSmith 자동 관측과 별도로
골든 셋 기반 `hit@k`·`mrr`를 계산한다. 결과를 LangSmith Dataset으로 업로드하려면
별도 스크립트 확장이 필요하다. (현재는 텍스트 출력 중심)

## 후속 확장 (필요 시)

- 병렬 처리 경로(`rag_parallel_processor`) 활성화 시 이미 컨텍스트 전파 수정이
  반영되어 있음 — `run_in_executor`는 contextvars를 복사하지 않아
  `copy_context().run`으로 감쌈 (LangSmith 관측도 동일하게 보호됨)
- 검색·재정렬·생성 단계를 이름 있는 run으로 세분화하고 싶으면
  `observe_if_enabled(name="vector_retrieval"|"reranking"|"answer_generation")`를
  해당 메서드에 추가하고 `record_*`로 `top_k`·문서명·`chunk_id`·점수·소요시간
  메타데이터를 기록하면 된다.
