# LM Studio 통합 가이드

이 문서는 로컬 LM Studio(로컬 모델 호스팅)를 이 프로젝트에서 발견하고 UI에 노출하는 방법을 설명합니다.

## 목표

- LM Studio에서 실행 중인 로컬 모델을 자동으로 검색하여 사용자가 사이드바에서 선택할 수 있도록 합니다.
- 실패 시 로컬 모델 디렉토리를 스캔하여 모델 파일을 감지합니다.

## 동작 흐름

1. 애플리케이션 시작 시 또는 사이드바에서 모델 목록을 요청하면 `src/models/lm_studio.list_lm_studio_models()`가 호출됩니다.
2. 모듈은 먼저 환경변수 `LM_STUDIO_API_URL`(기본: `http://localhost:8080`)로 HTTP API(`GET /v1/models`)를 호출하여 모델 목록을 가져옵니다.
3. API 호출이 실패하거나 응답이 없을 경우, 환경변수 `LM_STUDIO_MODEL_DIR`(기본: `~/Library/Application Support/lm-studio/models`)를 스캔하여 모델 파일(`.gguf`, `.bin`, `.pt`, `.safetensors` 등)을 찾아 목록을 구성합니다.
4. 발견된 모델은 `ModelRegistry.get_all_models()`에서 로컬 모델 항목으로 병합되어 UI에 표시됩니다.

## 지원 엔드포인트

- GET /v1/models — 모델 목록 조회
- POST /v1/chat/completions — 스트리밍/비스트리밍 채팅 생성
- POST /v1/completions — 단발성 텍스트 완성
- POST /v1/embeddings — 임베딩 생성

## 예시 기본 URL

- 내부 네트워크에서 운영중인 예시: http://192.168.0.227:3620 (프로젝트 기본값)
- 환경변수로 변경 가능: `LM_STUDIO_API_URL`

## 환경변수

- `LM_STUDIO_API_URL` (선택): LM Studio 서버 기본 URL. 예: `http://localhost:8080`
- `LM_STUDIO_MODEL_DIR` (선택): LM Studio 모델 디렉토리 경로. macOS 기본: `~/Library/Application Support/lm-studio/models`

## 파일 위치 및 코드

- API/디렉토리 탐색 로직: `src/models/lm_studio.py`
- UI 통합: `src/models/model_registry.py` (get_all_models에서 통합)
- LLM 초기화 및 로컬 모델 병합: `src/rag/llm_manager.py` (`_get_local_models`에서 병합)

## 주의 사항

- LM Studio의 HTTP API 스펙은 버전에 따라 다를 수 있으므로, 필요시 `src/models/lm_studio.py`의 파싱 로직을 조정하세요.
- 로컬 파일 스캔은 단순 파일 확장자 기반이므로, 모델 메타데이터가 필요하면 추가 파싱 로직을 구현해야 합니다.

## 예시

환경변수 설정(예시):

```bash
export LM_STUDIO_API_URL="http://localhost:8080"
export LM_STUDIO_MODEL_DIR="$HOME/Library/Application Support/lm-studio/models"
```

이후 앱을 실행하면 사이드바의 LLM 제공자에서 로컬 모델들을 선택할 수 있습니다.
