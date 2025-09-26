# RAG 개발 가이드라인

## Brief overview
이 문서는 RAG(Retrieval-Augmented Generation) 시스템 개발을 위한 프로젝트별 지침을 정의합니다. 한국어 중심의 문서 처리 및 멀티모달 기능을 갖춘 프로젝트에 특화된 개발 방식을 설명합니다.

## Communication style
- 모든 의사소통은 한국어로 진행하며, 기술 용어는 괄호 안에 쉬운 설명을 덧붙여 일반인도 이해할 수 있도록 합니다
- 복잡한 개념은 비유를 사용하여 설명합니다 (예: 벡터 DB는 문서의 지문과 같다)
- 코드 리뷰 시 구체적인 예시와 함께 피드백을 제공합니다

## Development workflow
- 구현 전 반드시 문서 조사 → 계획 수립 → 구현의 3단계를 따릅니다
- RAG 시스템의 경우 문서 처리 파이프라인, 임베딩, 검색, 생성 각 단계별로 독립적으로 테스트합니다
- 멀티모달 기능 개발 시 이미지 처리와 텍스트 처리의 통합 점검을 우선시합니다
- 벡터 DB 관리 스크립트(`vector_db_manager.py`)를 활용한 주기적인 데이터 무결성 검증

## Coding best practices
- Python 3.10+ 사용, 4칸 들여쓰기, PEP 8 준수
- 모듈 단일 책임 원칙 준수: `src/rag/`, `src/vectorstore/`, `src/processing/` 등 기능별 분리
- 한국어 텍스트 처리 시 `src/utils/korean_text_model.py`의 기능을 우선 활용
- 멀티모달 처리 시 `src/processing/multimodal_preprocessing_model.py`의 파이프라인 활용
- 에러 처리: RAG 체인의 각 단계별 명확한 오류 메시지와 복구 전략 구현

## Project context
- 주요 구성 요소: 문서 로더 → 전처리 → 임베딩 → 벡터 DB → 검색 → LLM 생성
- 지원 형식: PDF, 이미지, 텍스트 문서의 멀티모달 처리
- 기본 모델: GPT-4o-mini, Gemini-1.5-flash, Claude-3-5-Haiku
- 한국어 최적화: 형태소 분석, 키워드 확장, 의미론적 청킹
- OpenRouter를 통한 지능형 이미지 분석 파이프라인

## Testing strategies
- pytest 디스커버리 패턴 사용 (`tests/**/test_*.py`)
- 문서 처리 파이프라인은 end-to-end 테스트 우선
- 벡터 DB 연동 테스트 시 `--safe-mode` 옵션 활용
- 한국어 텍스트 처리 기능은 다양한 문장 패턴으로 검증
- 멀티모달 기능은 이미지-텍스트 조합 테스트 강화

## Performance optimization
- 대용량 문서 처리 시 스트리밍 방식 적용
- 벡터 검색 성능 모니터링 (`src/utils/perf.py`)
- 임베딩 모델 호출 배치 처리 최적화
- 캐싱 전략: 자주 사용되는 문서 임베딩 캐시 활용

## Security considerations
- API 키는 `.env` 파일 관리, 실제 키 커밋 금지
- 문서 업로드 시 파일 형식 및 크기 검증
- 벡터 DB 접근 제어 및 데이터 무결성 보장
- 외부 API 호출 시 타임아웃 및 재시도 메커니즘 구현