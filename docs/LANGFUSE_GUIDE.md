# Langfuse 트레이싱 사용 가이드 (셀프호스팅)

LangSmith(SaaS)는 내부망에서 사용할 수 없어, 자체 호스팅 가능한 오픈소스
LLM 관측 도구 **Langfuse v3**를 채택했다. 검색→생성 전 과정의 트레이스를
로컬(내부망)에서 수집·분석할 수 있다.

## 아키텍처

```
[앱: BaseAgent._call_llm]
        │  @observe (src/utils/tracing.py)
        ▼
[Langfuse SDK] ──OTLP──▶ [langfuse-web :3000] ──▶ [ClickHouse/Postgres/MinIO/Redis]
        (docker compose: infra/langfuse/)
```

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
- 코드 연결은 한 곳: `src/agents/base_agent.py`의 `_call_llm`에
  `@observe_if_enabled(name="llm_generation")` 적용 (`src/utils/tracing.py`)
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

- 검색(retrieval) 단계 트레이싱: `query_engine`에 `@observe_if_enabled` 추가
- LangChain 컴포넌트(임베딩 등): `langfuse.langchain.CallbackHandler`를 콜백으로 전달
- 평가 하네스(`scripts/eval/retrieval_eval.py`) 결과를 Langfuse Dataset으로 업로드
