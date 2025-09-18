# 운영 가이드

## 1. 실행 방법
- 권장: `python run_rag.py` → 메뉴에서 선택
- 수동: 
  - 인덱싱: `python scripts/data_management/vector_db_manager.py [옵션]`
  - 앱 실행: `streamlit run app.py`

## 2. 환경 변수/.env
- 필수(택1): `UPSTAGE_API_KEY` 또는 `OPENAI_API_KEY` 등
- 로컬 HTTP LLM: `LOCAL_LLM_BASE_URL`(기본: `http://210.126.109.57:1620`), `LOCAL_LLM_API_KEY`(옵션)
- 멀티 서버: `LOCAL_LLM_BASE_URLS=http://210.126.109.57:1620,http://210.126.109.57:1621,http://210.126.109.57:1622`
 - 미설정 시 기본 호스트 기반으로 1620/1621/1622를 자동 조회해 모델 목록 구성
- 텔레메트리 중지: `ANONYMIZED_TELEMETRY=False`, `CHROMA_TELEMETRY=False`
 - OpenRouter 사용: `IMAGE_ANALYSIS_PROVIDER=openrouter`, `OPNEROUTER_API_KEY`, `OPENROUTER_MM_MODEL=z-ai/glm-4.5v`
 - 멀티모달 모델 확장: `EXTRA_MULTIMODAL_*` 3종(OpenAI/Google/Anthropic)

## 3. 벡터 DB 관리
- 전체 처리: `vector_db_manager.py`
- 안전 모드(OCR 비활성화): `--safe-mode`
- 마크다운 전용: `--markdown-only`
- 특정 파일만: `--files "a.pdf" "b.md"`

## 4. 점검/복구
- 인덱스 유실/호환 안됨 → `vector_db/` 내부 파일 삭제 후 재생성
- 문서 처리 오류 많음 → 안전 모드/파일 단위 처리로 원인 좁히기
- 로컬 LLM 점검:
  - 헬스체크: `GET {BASE}/health`
  - 큐 상태: `GET {BASE}/v1/queue/stats`
  - 모델 목록: `GET {BASE}/v1/models`
  - 멀티 서버: 각 BASE로 동일 점검 수행
  - 멀티모달 모델: 현재 1620 포트에만 상주 — 이미지 분석/OCR 경로는 1620이 우선 사용됩니다.
  - OpenRouter 우선 경로: 실패 시 로컬 → 내장 순으로 자동 폴백됩니다.

## 5. 로그/모니터링
- `logs/` 디렉토리 확인(운영 배포 시 민감정보 마스킹 권장)
