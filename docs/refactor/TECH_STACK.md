# 기술 스택 명세서

## 1. 언어와 프레임워크
- Python 3.10+: 메인 애플리케이션 언어.
- Streamlit: 웹 UI 프레임워크.
- LangChain: RAG 파이프라인 구성.

## 2. 데이터/저장소
- FAISS 또는 Chroma: 벡터 데이터베이스.
- 로컬 파일 시스템: 업로드 문서와 이미지 캐시 보관.

## 3. 외부 서비스
- OpenAI/Gemini/Anthropic API: 전처리 및 답변 생성 모델.
- OpenRouter: 이미지 분석 기본 경로.

## 4. 기술적 제약
- 모든 함수에 타입 힌트를 유지.
- `time.sleep` 대신 비동기 처리 또는 채널을 활용.
- 테스트/포맷/린트 명령은 `make fmt && make test && make lint`로 통합.

## 5. 리팩토링에서 새로 정리할 부분
- 이미지 처리 로직을 `src/utils/image_tools.py`로 이동하여 Streamlit 의존성을 최소화.
- 전처리 파이프라인을 `src/processing/document_pipeline.py`에서 중앙 집중 관리.

## 6. 용어 정리
- 벡터 데이터베이스(설명: 문서를 숫자 목록으로 바꿔 저장하고 비슷한 문서를 빠르게 찾게 도와주는 저장소)
- 파이프라인(설명: 여러 단계를 차례로 실행해 결과를 얻는 흐름)
