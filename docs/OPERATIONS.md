# 운영 가이드

## 1. 실행 방법
- 권장: `python run_rag.py` → 메뉴에서 선택
- 수동: 
  - 인덱싱: `python scripts/data_management/vector_db_manager.py [옵션]`
  - 앱 실행: `streamlit run app.py`

## 2. 환경 변수/.env
- 필수(택1): `UPSTAGE_API_KEY` 또는 `OPENAI_API_KEY` 등
- 텔레메트리 중지: `ANONYMIZED_TELEMETRY=False`, `CHROMA_TELEMETRY=False`
- OpenRouter(이미지/전처리 고정):
  - `IMAGE_ANALYSIS_PROVIDER=openrouter`
  - `OPNEROUTER_API_KEY`(주의: 정확한 철자)
  - `OPENROUTER_MM_MODEL=z-ai/glm-4.5v`
  - `DISABLE_IMAGE_FALLBACK=true`(엄격 모드)
  - `USE_LOCAL_IMAGE_SERVER=false`(로컬 폴백 차단)
- 멀티모달 모델 확장: `EXTRA_MULTIMODAL_*` 3종(OpenAI/Google/Anthropic)

## 3. 벡터 DB 관리
- 전체 처리: `vector_db_manager.py`
- 안전 모드(OCR 비활성화): `--safe-mode`
- 마크다운 전용: `--markdown-only`
- 특정 파일만: `--files "a.pdf" "b.md"`

## 4. 점검/복구
- 인덱스 유실/호환 안됨 → `vector_db/` 내부 파일 삭제 후 재생성
- 문서 처리 오류 많음 → 안전 모드/파일 단위 처리로 원인 좁히기
- 로컬 LLM 경로는 전처리/이미지 분석에서 사용하지 않습니다(폴백 제거 정책).

## 5. 로그/모니터링
- `logs/` 디렉토리 확인(운영 배포 시 민감정보 마스킹 권장)
