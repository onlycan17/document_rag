# 보안/비밀정보 가이드

## 1. 비밀정보 관리
- `.env`에 API 키 저장, VCS 커밋 금지.
- 키 회전 주기/권한 최소화(용어(설명: 필요한 권한만 부여)).

## 2. 데이터 취급
- 입력 문서에 개인/기밀 포함 가능 → 운영 모드에서 로그 마스킹.
- 외부 LLM 사용 시 전송 데이터 범위 최소화(컨텍스트 최적화로 과도한 전송 방지).

## 3. 외부 LLM API
- 모든 LLM/임베딩 호출은 외부 API(OpenRouter·OpenAI·Google·Anthropic·Upstage)만 사용.
- 키는 `.env`에만 저장, VCS 커밋 금지, 로그에 값 노출 금지.
- 사설 네트워크에서 운영 시 아웃바운드 허용 도메인/엔드포인트를 방화벽으로 제한 권장.

## 4. 권장 설정
- `ANONYMIZED_TELEMETRY=False`, `CHROMA_TELEMETRY=False`
- 최소 권한 API 키 사용, 프로젝트 단위로 분리.
