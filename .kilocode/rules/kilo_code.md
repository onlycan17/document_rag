# kilo_code.md

## 프로젝트 개요
이 문서는 RAG(Retrieval-Augmented Generation) 시스템 개발을 위한 코딩 규칙을 정의합니다. Python 3.10+ 기반의 한국어 중심 멀티모달 프로젝트에 특화된 규칙입니다.

## 변수명명 규칙

### 기본 규칙
- **snake_case 사용**: 모든 변수는 소문자와 언더스코어를 사용합니다
  - ✅ `user_id`, `document_content`, `embedding_vector`
  - ❌ `userId`, `DocumentContent`, `EmbeddingVector`

### 의미 있는 이름
- 변수명은 해당 변수의 용도를 명확히 설명해야 합니다
  - ✅ `max_chunk_size`, `processed_documents`, `query_embedding`
  - ❌ `a`, `temp`, `data`

### 한국어 관련 변수
- 한국어 처리를 위한 변수는 `korean_` 접두사를 권장합니다
  - ✅ `korean_text`, `korean_keywords`, `korean_morphs`
  - ❌ `text_ko`, `kr_text`, `hangul_text`

### 불리언 변수
- `is_`, `has_`, `can_` 접두사를 사용하여 의미를 명확히 합니다
  - ✅ `is_processed`, `has_content`, `can_retry`
  - ❌ `processed`, `content_exists`, `retry_flag`

### 컬렉션 변수
- 복수형을 사용하여 컬렉션임을 명시합니다
  - ✅ `documents`, `embeddings`, `chunk_list`
  - ❌ `document_list`, `embedding_array`, `chunks`

## 함수명명 규칙

### 동사 + 명사 패턴
- 함수명은 동작을 설명하는 동사로 시작합니다
  - ✅ `load_documents()`, `process_text()`, `generate_embedding()`
  - ❌ `document_loader()`, `text_processor()`, `embedding_generator()`

### CRUD 작업
- 생성: `create_`, `generate_`, `build_`
- 읽기: `get_`, `fetch_`, `retrieve_`
- 수정: `update_`, `modify_`, `transform_`
- 삭제: `delete_`, `remove_`, `clear_`

### 한국어 처리 함수
- 한국어 관련 함수는 `korean_` 접두사를 사용합니다
  - ✅ `korean_tokenize()`, `korean_normalize()`, `korean_keyword_extract()`
  - ❌ `tokenize_korean()`, `normalize_kr()`, `extract_kr_keywords()`

### RAG 특화 함수
- 문서 처리: `chunk_document()`, `embed_text()`, `search_similar()`
- 멀티모달: `extract_image_text()`, `analyze_multimodal()`, `process_visual_content()`

## 클래스명명 규칙

### PascalCase 사용
- 클래스명은 단어의 첫 글자를 대문자로 합니다
  - ✅ `DocumentLoader`, `EmbeddingModel`, `RAGChain`
  - ❌ `document_loader`, `embedding_model`, `rag_chain`

### 명사 중심 이름
- 클래스는 객체를 나타내므로 명사형 이름을 사용합니다
  - ✅ `VectorStore`, `TextProcessor`, `QueryEngine`
  - ❌ `StoreVector`, `ProcessText`, `EngineQuery`

### 인터페이스와 추상 클래스
- 인터페이스: `I` 접두사 또는 `able` 접미사
  - ✅ `ILoader`, `Processable`, `Embeddable`
- 추상 클래스: `Abstract` 접두사 또는 `Base` 접미사
  - ✅ `AbstractModel`, `LoaderBase`, `ProcessorBase`

### RAG 특화 클래스
- 코어 컴포넌트: `RAGProcessor`, `VectorDatabase`, `LLMManager`
- 유틸리티: `KoreanTextModel`, `MultimodalAnalyzer`, `DocumentSplitter`

## 파일명명 규칙

### 모듈 파일
- **snake_case 사용**: 소문자와 언더스코어를 사용합니다
  - ✅ `document_loader.py`, `vector_store.py`, `rag_chain.py`
  - ❌ `DocumentLoader.py`, `VectorStore.py`, `RAGChain.py`

### 디렉토리 구조
- 기능별 모듈화를 따라 파일을 구성하며, 현재 프로젝트의 실제 구조를 반영합니다

#### 코어 모듈 (`src/`)
- `src/agents/`: AI 에이전트 컴포넌트
  - `base_agent.py`: 에이전트 기본 클래스
  - `context_connector.py`: 컨텍스트 연결 관리
  - `quality_validator.py`: 품질 검증 에이전트
  - `structure_parser.py`: 구조 분석 에이전트

- `src/embeddings/`: 텍스트 임베딩 처리
  - `embedding_model.py`: 임베딩 모델 관리

- `src/loaders/`: 문서 로딩 및 처리
  - `document_loader.py`: 기본 문서 로더
  - `pdf_loader_advanced.py`: 고급 PDF 처리
  - `text_processor.py`: 텍스트 전처리
  - `file_loader_manager.py`: 파일 로더 관리

- `src/models/`: LLM 모델 관리
  - `model_registry.py`: 모델 레지스트리
  - `lm_studio.py`: LM Studio 통합

- `src/processing/`: 멀티모달 전처리
  - `multimodal_preprocessing_model.py`: 멀티모달 처리
  - `preprocessing_factory.py`: 전처리 팩토리
  - `document_pipeline.py`: 문서 처리 파이프라인

- `src/rag/`: RAG 체인 컴포넌트
  - `rag_chain.py`: 메인 RAG 체인
  - `query_engine.py`: 질의 처리 엔진
  - `llm_manager.py`: LLM 관리자
  - `summarizer.py`: 요약 기능

- `src/utils/`: 유틸리티 함수
  - `korean_text_model.py`: 한국어 텍스트 처리
  - `image_analyzer.py`: 이미지 분석
  - `openrouter_image_service.py`: OpenRouter 이미지 서비스
  - `intelligent_image_extractor_korean.py`: 한국어 이미지 추출

- `src/vectorstore/`: 벡터 데이터베이스
  - `vector_db.py`: 벡터 DB 관리

#### UI 모듈 (`ui/`)
- `ui/components/`: UI 컴포넌트
  - `chat_interface.py`: 채팅 인터페이스
  - `file_uploader.py`: 파일 업로더
  - `sidebar.py`: 사이드바 컴포넌트

- `ui/controllers/`: 컨트롤러
  - `main_controller.py`: 메인 컨트롤러

- `ui/services/`: 서비스 레이어
  - `app_service.py`: 애플리케이션 서비스
  - `streaming_service.py`: 스트리밍 서비스

#### 테스트 및 스크립트
- `tests/`: 단위 및 통합 테스트
- `scripts/`: 설치 및 관리 스크립트
- `docs/`: 프로젝트 문서

### 테스트 파일
- 테스트 파일은 `test_` 접두사를 사용합니다
  - ✅ `test_document_loader.py`, `test_rag_chain.py`
  - ❌ `document_loader_test.py`, `rag_chain_test.py`

### 설정 파일
- 설정 관련 파일은 명확한 이름을 사용합니다
  - ✅ `config.py`, `settings.py`, `.env.example`
  - ❌ `cfg.py`, `conf.py`, `env_sample`

## 기타 코딩 규칙

### 타입 힌트
- 모든 공개 함수와 메서드에 타입 힌트를 추가합니다
```python
def process_document(content: str, chunk_size: int = 1000) -> List[str]:
    # 구현
```

### Docstring
- 공개 API에는 Google 스타일 docstring을 작성합니다
```python
def embed_text(text: str, model_name: str = "default") -> List[float]:
    """텍스트를 벡터 임베딩으로 변환합니다.
    
    Args:
        text: 임베딩할 텍스트
        model_name: 사용할 임베딩 모델 이름
        
    Returns:
        텍스트의 벡터 임베딩
    """
```

### 에러 처리
- 구체적인 예외 타입을 사용하고 의미 있는 에러 메시지를 제공합니다
```python
try:
    document = load_document(file_path)
except FileNotFoundError:
    raise DocumentLoadError(f"문서 파일을 찾을 수 없습니다: {file_path}")
```

### 상수 정의
- 상수는 `UPPER_SNAKE_CASE`를 사용하고 모듈 상단에 정의합니다
```python
MAX_CHUNK_SIZE = 1000
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".docx"]
```

### 한국어 문자열 처리
- 문자열 상수는 별도 모듈에서 관리합니다
```python
# constants.py
KOREAN_ERROR_MESSAGES = {
    "file_not_found": "파일을 찾을 수 없습니다: {}",
    "processing_failed": "문서 처리에 실패했습니다",
    "embedding_error": "임베딩 생성 중 오류 발생"
}
```

## 코드 구조 규칙

### 함수 길이 제한
- 함수는 30줄 이내로 유지하고, 15줄 이내를 권장합니다
- 긴 함수는 의미 있는 하위 함수로 분리합니다

### 클래스 책임
- 각 클래스는 단일 책임 원칙을 따릅니다
- 너무 많은 메서드를 가진 클래스는 리팩토링을 고려합니다

### 임포트 규칙
- 표준 라이브러리, 서드파티, 로컬 모듈 순으로 그룹화합니다
- 상대 임포트보다 절대 임포트를 선호합니다

이 규칙들은 프로젝트의 일관성과 유지보수성을 높이기 위해 설계되었습니다.