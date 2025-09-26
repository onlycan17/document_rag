# 기능 명세서

## 1. 앱 초기화
- **기능**: 세션 상태 준비, 사이드바 렌더링, 메인 탭 구성.
- **관련 파일**: `app.py`, `ui/controllers/main_controller.py`.
- **리팩토링 포인트**:
  - `initialize_application`, `handle_sidebar_changes`, `render_main_tabs`를 각각 30줄 이하 유지.
  - CSS/JS 주입을 별도 함수로 분리(`inject_lightbox_assets`).

## 2. 스트리밍 답변 처리
- **기능**: LLM 응답을 실시간으로 받아 사용자에게 표시.
- **관련 파일**: `ui/components/chat_interface.py`, `ui/services/streaming_service.py`(신규).
- **리팩토링 포인트**:
  - 스트리밍 처리 전용 클래스 `StreamingResponseHandler`를 만들어 UI 로직과 분리.
  - 상태 업데이트, 이미지 추출, 성능 측정 단계를 메서드로 쪼개기.

## 3. 이미지 추출 및 표시
- **기능**: 문서 메타데이터에서 이미지 경로를 찾아 UI에 표시.
- **관련 파일**: `src/utils/image_tools.py`(신규), `ui/services/app_service.py`.
- **리팩토링 포인트**:
  - 경로 정규화 → 파일 존재 확인 → base64 인코딩 단계를 각각 함수로 분리.
  - 콘솔 출력 대신 `logger` 사용, 필요시 디버그 모드에서만 상세 로그.

## 4. 전처리 모델 호출
- **기능**: 문서에서 텍스트 추출하고 모델로 전처리.
- **관련 파일**: `src/processing/document_pipeline.py`(신규), `src/processing/preprocessing_model.py`.
- **리팩토링 포인트**:
  - 문서 로더 호출을 공통 헬퍼 `run_document_pipeline`으로 정리.
  - `LocalPreprocessingModel`과 `APIPreprocessingModel`은 전처리에만 집중.

## 5. 모델 메타데이터 관리
- **기능**: 모델별 토큰, 컨텍스트 정보를 제공.
- **관련 파일**: `src/rag/rag_chain.py`, `src/rag/llm_manager.py`.
- **리팩토링 포인트**:
  - `RAGChain`에서 직접 하드코딩하지 않고 `LLMManager.get_model_metadata`(신규) 호출.
  - 사용하지 않는 반환문 제거.

## 6. 용어 정리
- 헬퍼 함수(설명: 큰 기능을 도와주는 작은 함수)
- 인코딩(설명: 정보를 다른 형태로 바꿔 저장하거나 전송하는 과정)
