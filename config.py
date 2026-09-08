import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

load_dotenv()


class Settings(BaseSettings):
    # Pydantic v2 설정: .env 사용, 알 수 없는 환경변수는 무시(에러 방지)
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # LLM 제공자 설정
    llm_provider: str = "openrouter"  # 외부 API 제공자만 지원: "openai", "google", "anthropic", "openrouter"

    # OpenAI 설정
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    # 권장 기본값: 경량 멀티모달 고성능-저비용 모델
    openai_model: str = "gpt-5-mini"

    # Google Gemini 설정
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    google_model: str = "gemini-2.5-flash"  # 또는 "gemini-2.5-pro", "gemini-2.0-ultra"

    # Anthropic Claude 설정
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    # 권장 기본값: 최신 세대 Claude 4 Sonnet
    anthropic_model: str = "claude-4-sonnet"

    # 멀티모달 추가 모델(.env에서 콤마로 확장)
    # 예) EXTRA_MULTIMODAL_OPENAI_MODELS=gpt-5-mini,gpt-5-nano
    #    EXTRA_MULTIMODAL_GOOGLE_MODELS=gemini-2.5-pro
    #    EXTRA_MULTIMODAL_ANTHROPIC_MODELS=claude-opus-4-1-20250805
    extra_multimodal_openai_models: Optional[str] = os.getenv("EXTRA_MULTIMODAL_OPENAI_MODELS")
    extra_multimodal_google_models: Optional[str] = os.getenv("EXTRA_MULTIMODAL_GOOGLE_MODELS")
    extra_multimodal_anthropic_models: Optional[str] = os.getenv("EXTRA_MULTIMODAL_ANTHROPIC_MODELS")
    # OpenRouter 멀티모달 추가 모델(.env에서 콤마로 확장)
    extra_multimodal_openrouter_models: Optional[str] = os.getenv("EXTRA_MULTIMODAL_OPENROUTER_MODELS")

    # OpenRouter 공통 설정(키/엔드포인트/모델)
    # 하위 호환: 오타 표기(OPNEROUTER_API_KEY)와 올바른 표기(OPENROUTER_API_KEY) 모두 인식
    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPNEROUTER_API_KEY")
    openrouter_api_base: str = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api")
    # 텍스트 기본 모델(선택)과 멀티모달 기본 모델
    # 텍스트 기본 모델은 무료 모델인 glm-4.5-air로 고정(요청사항)
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "z-ai/glm-4.5-air")
    openrouter_mm_model: str = os.getenv("OPENROUTER_MM_MODEL", "z-ai/glm-4.5v")

    # MD 후처리 설정
    enable_md_postprocessing: bool = os.getenv("ENABLE_MD_POSTPROCESSING", "true").lower() == "true"
    md_postprocess_target_quality: int = int(os.getenv("MD_POSTPROCESS_TARGET_QUALITY", "90"))

    # 임베딩 모델 설정
    embedding_provider: str = "upstage"  # "openai" 또는 "upstage"
    embedding_model_name: str = "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"

    # 이미지 관련도 필터 임계값
    local_image_relevance_threshold: float = float(os.getenv("LOCAL_IMAGE_RELEVANCE_THRESHOLD", "0.6"))

    # 이미지 분석 프로바이더 (pdf 전처리용): 외부 API만 지원 ("openai" | "google" | "anthropic" | "openrouter")
    image_analysis_provider: str = os.getenv("IMAGE_ANALYSIS_PROVIDER", "openrouter")

    # MD 후처리 전용 오버라이드(선택): 이 값이 설정되면 후처리만 별도 provider/model 사용
    md_postprocess_provider: Optional[str] = os.getenv("MD_POSTPROCESS_PROVIDER")
    md_postprocess_model: Optional[str] = os.getenv("MD_POSTPROCESS_MODEL")

    # 이미지 향상(선택) 설정
    enable_image_enhancement: bool = os.getenv("ENABLE_IMAGE_ENHANCEMENT", "false").lower() == "true"
    image_enhancement_prompt: str = os.getenv(
        "IMAGE_ENHANCEMENT_PROMPT",
        "Enhance readability for OCR: denoise, increase contrast, sharpen, preserve original content.",
    )

    # 업스테이지 임베딩 설정
    upstage_api_key: Optional[str] = os.getenv("UPSTAGE_API_KEY")
    upstage_embedding_model: str = "solar-embedding-1-large-query"

    # 한국어 최적화 임베딩 모델 설정 (우선 사용)
    korean_embedding_model: Optional[str] = os.getenv("KOREAN_EMBEDDING_MODEL", "jhgan/ko-sbert-multitask")

    # 벡터 데이터베이스 설정
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "./vector_db")
    vector_db_type: str = "faiss"  # "faiss" 또는 "chromadb"

    # 청크 설정 - 성능 최적화
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "150"))  # 검색 정밀도 향상을 위한 청크 크기 조정
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))  # 오버랩 조정
    # 의미 기반 청킹 설정
    use_semantic_chunking: bool = os.getenv("USE_SEMANTIC_CHUNKING", "false").lower() == "true"
    semantic_chunk_sentences: int = 3  # 의미 기반 청킹 시 문장 단위

    # RAG 설정 - 검색 성능 최적화
    k_documents: int = int(os.getenv("K_DOCUMENTS", "12"))  # 더 많은 문서를 검색하여 관련 정보 포함 가능성 높임
    search_threshold_faiss: float = 1.5  # FAISS 임계값 (거리 기반, 25% 유사도에 해당, 더 관대한 검색)
    search_threshold_chromadb: float = 0.30  # ChromaDB 임계값 (유사도 30%로 조정)
    use_mmr_search: bool = os.getenv("USE_MMR_SEARCH", "true").lower() == "true"  # MMR 검색 활성화 (다양성 향상)
    mmr_diversity_score: float = 0.3  # MMR 다양성 점수

    # 하이브리드 검색 설정
    enable_hybrid_search: bool = os.getenv("ENABLE_HYBRID_SEARCH", "false").lower() == "true"
    keyword_search_weight: float = float(os.getenv("KEYWORD_SEARCH_WEIGHT", "0.3"))  # 키워드 검색 가중치
    vector_search_weight: float = float(os.getenv("VECTOR_SEARCH_WEIGHT", "0.7"))  # 벡터 검색 가중치

    # 쿼리 최적화 설정
    enable_query_expansion: bool = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
    enable_query_preprocessing: bool = os.getenv("ENABLE_QUERY_PREPROCESSING", "true").lower() == "true"

    # LangSmith 설정
    langsmith_tracing: str = os.getenv("LANGSMITH_TRACING", "false")
    langsmith_endpoint: Optional[str] = os.getenv("LANGSMITH_ENDPOINT")
    langsmith_api_key: Optional[str] = os.getenv("LANGSMITH_API_KEY")
    langsmith_project: Optional[str] = os.getenv("LANGSMITH_PROJECT")

    # 일부 모델은 temperature를 변경할 수 없으므로 기본값을 1.0으로 설정(모델의 기본값과 일치)
    temperature: float = 1.0
    max_tokens: int = 4096  # 기본값 - 모델별로 자동 조정됨

    # 스트리밍 설정
    enable_streaming: bool = os.getenv("ENABLE_STREAMING", "true").lower() == "true"

    # 대량 문서 처리 설정
    enable_large_context_processing: bool = os.getenv("ENABLE_LARGE_CONTEXT_PROCESSING", "true").lower() == "true"
    large_context_threshold: int = int(os.getenv("LARGE_CONTEXT_THRESHOLD", "50000"))  # 50KB
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "30000"))  # 30KB
    parallel_workers: int = int(os.getenv("PARALLEL_WORKERS", "3"))
    chunk_timeout: float = float(os.getenv("CHUNK_TIMEOUT", "120.0"))  # 120초로 증가 (Rate Limit 대응)
    enable_result_merging: bool = os.getenv("ENABLE_RESULT_MERGING", "true").lower() == "true"

    # API Rate Limit 대응 설정
    api_max_retries: int = int(os.getenv("API_MAX_RETRIES", "5"))  # 재시도 횟수 증가
    api_base_delay: float = float(os.getenv("API_BASE_DELAY", "3.0"))  # 기본 대기 시간 증가
    api_max_delay: float = float(os.getenv("API_MAX_DELAY", "300.0"))  # 최대 대기 시간 증가 (5분)

    # 문서 전처리 모델 설정
    preprocessing_model: str = os.getenv("PREPROCESSING_MODEL", "openrouter")  # 전처리 기본 제공자를 외부 API로 전환
    preprocessing_max_tokens: int = int(os.getenv("PREPROCESSING_MAX_TOKENS", "2000"))  # 전처리 최대 토큰 수
    # 전처리 단계에서 사용하는 온도 (환경변수로 오버라이드 가능)
    preprocessing_temperature: float = float(os.getenv("PREPROCESSING_TEMPERATURE", "1.0"))  # 전처리 온도

    # 멀티모달 전처리 모델 설정 (이미지 처리용)
    multimodal_preprocessing_model: str = os.getenv(
        "MULTIMODAL_PREPROCESSING_MODEL", "z-ai/glm-4.5v"
    )  # OpenRouter 전용 이미지 분석 모델 (문서 명세 고수)
    multimodal_preprocessing_provider: str = os.getenv(
        "MULTIMODAL_PREPROCESSING_PROVIDER", "openrouter"
    )  # 문서 명세: 반드시 openrouter로 고정
    enable_multimodal_preprocessing: bool = (
        os.getenv("ENABLE_MULTIMODAL_PREPROCESSING", "true").lower() == "true"
    )  # 멀티모달 전처리 활성화(기본 on)

    # UI 설정
    app_title: str = "쉽게 설명하는 RAG 챗봇"
    app_description: str = (
        "복잡한 문서도 쉽게! 궁금한 내용을 질문하세요. 일반인도 이해할 수 있도록 친절하게 설명해드립니다."
    )


settings = Settings()
