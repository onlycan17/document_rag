# Langfuse 트레이싱 사용 가이드 (셀프호스팅)

LangSmith(SaaS)는 내부망에서 사용할 수 없어, 자체 호스팅 가능한 오픈소스
LLM 관측 도구 **Langfuse v3**를 채택했다. 검색→생성 전 과정의 트레이스를
로컬(내부망)에서 수집·분석할 수 있다.

## 아키텍처

```
[RAG 질의 흐름 (트레이스 계층)]
rag-query (span, trace 루트 — query_engine)
 ├─ vector-search (retriever — vector_db.search, 출처·점수 기록)
 └─ ChatOpenAI (generation — LangChain CallbackHandler, 모델명·토큰 자동 기록)

[문서 전처리 흐름]
llm-generation (generation — BaseAgent._call_llm, 모델명·토큰 기록)

        │  src/utils/tracing.py 헬퍼
        ▼
[Langfuse SDK] ──OTLP──▶ [langfuse-web :3000] ──▶ [ClickHouse/Postgres/MinIO/Redis]
        (docker compose: infra/langfuse/)
```

대화별 묶음은 Streamlit 세션 ID(`st.session_state.langfuse_session_id`)를
trace의 `sessionId`로 전파한다(`propagate_trace_attributes`).

## 기동/중지/상태

```bash
# 기동 (백그라운드)
docker compose -f infra/langfuse/docker-compose.yml up -d

# 상태 확인 (web·worker·clickhouse·postgres·minio·redis 모두 healthy가 정상)
docker compose -f infra/langfuse/docker-compose.yml ps

# 중지 (데이터 유지) / 완전 초기화 (데이터 삭제)
docker compose -f infra/langfuse/docker-compose.yml down
docker compose -f infra/langfuse/docker-compose.yml down -v
```

## 웹 UI

- 주소: http://localhost:3000
- 관리자 계정(부트스트랩으로 자동 생성): `admin@ragtest.local`
  비밀번호는 `infra/langfuse/docker-compose.yml`의 `LANGFUSE_INIT_USER_PASSWORD` 참조
- 프로젝트: `ragTest` (조직 `ragtest`)
- 화면별 용도:
  - **Traces**: 질의별 LLM 호출 전체 기록 (입력 프롬프트, 응답, 지연, 토큰·비용)
  - **Dashboards**: 지연/비용/호출 수 추이
  - **Prompt Management**: 프롬프트 버전 관리 (실험 시 활용)
  - **Datasets / Evaluations**: 평가 데이터셋과 LLM-as-judge 채점

## 앱 연결 방식

- **SDK 선택 설치**: `pip install langfuse==4.15.2` (requirements.txt에서 의존성 해석이
  무거워 제외 — 미설치 시 트레이싱만 꺼지고 앱은 정상 동작)
- `.env`의 키로 활성화: `LANGFUSE_ENABLED=true` + `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`
  + `LANGFUSE_HOST=http://localhost:3000`
- 계측 진입점은 `src/utils/tracing.py` 한 곳 (관측 규약: langfuse/skills instrumentation 참조
  — 함수 인자 자동 수집 금지, `record_*`로 필요한 값만 명시 기록):
  - `observe_if_enabled(name, as_type, capture_input, capture_output)`: 조건부 @observe 데코레이터
  - `record_generation(model, input, usage_details, metadata)`: generation에 모델명·토큰 기록
  - `record_input` / `record_retriever_output`: span/retriever에 입력·출처 요약 기록
  - `propagate_trace_attributes(session_id, user_id)`: 트레이스에 세션·사용자 전파
  - `langfuse_callbacks()`: LangChain 체인용 CallbackHandler 목록 (`llm_manager.create_llm`에서 부착)
- 계측 위치: `query_engine.query/stream_query`(rag-query 루트), `vector_db.search`(vector-search
  retriever), `llm_manager.create_llm`(LangChain generation), `base_agent._call_llm`(전처리 에이전트
  generation), `chat_interface.py`(세션 ID 생성·전달)
- 주의: langfuse 4.x 클라이언트에는 `update_current_observation`/`update_current_trace` 메서드가
  없다 — generation은 `update_current_generation`, span/retriever는 `update_current_span`으로 갱신
  (tracing.py의 `_patch_current_observation`이 처리)
- Langfuse 미기동/키 미설정/SDK 미설치여도 앱은 정상 동작(no-op 폴백)
- LangSmith(`LANGSMITH_TRACING`)와 독립적 — 외부망 개발 시에만 켜면 됨

## 내부망 배포 시 체크리스트

1. `infra/langfuse/docker-compose.yml`의 시크릿 5종 재생성
   (`SALT`, `NEXTAUTH_SECRET`, `POSTGRES_PASSWORD`, `CLICKHOUSE_PASSWORD`, `MINIO_ROOT_PASSWORD` — `openssl rand -hex 16`)
2. `NEXTAUTH_URL`을 실제 내부 접속 주소로 변경 (예: `http://rag-server:3000`)
3. 관리자 비밀번호(`LANGFUSE_INIT_USER_PASSWORD`) 변경
4. 볼륨 백업: `langfuse_postgres_data`, `langfuse_clickhouse_data` (docker volume)

## 문제 해결

| 증상 | 원인/조치 |
|---|---|
| web·worker 재시작 반복 | `docker logs langfuse-langfuse-web-1` 확인. 대부분 시크릿·스키마 문제 |
| 트레이스가 안 보임 | `.env`의 `LANGFUSE_ENABLED=true`와 키 확인, 앱 프로세스 재시작 |
| 키 분실 | UI에서 재발급하거나 compose의 `LANGFUSE_INIT_PROJECT_*` 변경 후 `down -v` 재기동 |

## 후속 확장 (필요 시)

- 평가 하네스(`scripts/eval/retrieval_eval.py`) 결과를 Langfuse Dataset으로 업로드
- 병렬 처리 경로(`rag_parallel_processor`) 활성화 시 이미 컨텍스트 전파 수정이
  반영되어 있음 — `run_in_executor`는 contextvars를 복사하지 않아 `copy_context().run`으로 감쌈
