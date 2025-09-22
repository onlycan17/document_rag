# TODOLIST / TASK 문서

## 대분류 A. 문서 및 설계 정비
- [x] A-1. PRD, 아키텍처, UI/UX 명세 작성
- [x] A-2. 기술 스택, 기능, API, ERD, 테이블 문서 작성
- [x] A-3. 기존 `TODO.md`에 리팩토링 항목 연동

## 대분류 B. 서비스 레이어 리팩토링
- [x] B-1. `ui/services/app_service.py`에서 이미지 처리 코드를 헬퍼 클래스로 분리
- [x] B-2. 로그 출력을 `logger.debug`로 전환하고 디버그 플래그 추가
- [x] B-3. 스트리밍 응답 처리 전용 모듈(`ui/services/streaming_service.py`) 도입

## 대분류 C. UI 컴포넌트 정리
- [x] C-1. `chat_interface.py`에서 이미지 추출 중복 제거
- [x] C-2. 스트리밍 처리 인터페이스를 새 서비스로 교체
- [x] C-3. 성능 정보 표시 함수 분리를 통한 30줄 이하 유지

## 대분류 D. 전처리 파이프라인 개선
- [x] D-1. `document_pipeline` 공통 헬퍼 생성
- [x] D-2. `LocalPreprocessingModel`과 `APIPreprocessingModel`에서 공통 추출 흐름 사용
- [x] D-3. 미사용 import 제거 및 에러 핸들링 일관화

## 대분류 E. RAG 체인 정리
- [x] E-1. `_get_model_max_tokens`의 하드코딩 제거
- [x] E-2. 죽은 코드 및 중복 반환 제거
- [x] E-3. 모델 메타데이터 조회를 `LLMManager`로 위임

## 대분류 F. 품질 보증
- [x] F-1. 새 헬퍼 함수 단위 테스트 작성 또는 보완
- [x] F-2. `make fmt && make test && make lint` 실행
- [x] F-3. 문서 최신화 및 버전 기록 업데이트

## 용어 정리
- 대분류(설명: 큰 작업 덩어리를 의미)
- 디버그 플래그(설명: 로그를 얼마나 자세히 남길지 제어하는 스위치)
