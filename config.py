import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

load_dotenv()

class Settings(BaseSettings):
    # Pydantic v2 설정: .env 사용, 알 수 없는 환경변수는 무시(에러 방지)
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # LLM 제공자 설정
    llm_provider: str = "local"  # "openai", "google", "anthropic", "local"
    
    # OpenAI 설정
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = "gpt-5-mini"
    
    # Google Gemini 설정
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    google_model: str = "gemini-2.5-flash"  # 또는 "gemini-2.5-pro", "gemini-pro"
    
    # Anthropic Claude 설정
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model: str = "claude-3-haiku-20240307"
    
    # 로컬 LLM 설정 (OpenAI 호환 API)
    local_llm_base_url: str = os.getenv("LOCAL_LLM_BASE_URL", "http://210.126.109.57:1620")
    # 여러 로컬 서버 지원(콤마 구분). 예: "http://210.126.109.57:1620,http://210.126.109.57:1621,http://210.126.109.57:1622"
    local_llm_base_urls: str | None = os.getenv("LOCAL_LLM_BASE_URLS")
    local_llm_model: str = os.getenv("LOCAL_LLM_MODEL", "midm-2.0-base-instruct")
    local_llm_api_key: str = os.getenv("LOCAL_LLM_API_KEY", "not-needed")  # 일부 로컬 서버는 API 키 필요
    local_llm_max_tokens: int = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "512"))  # 로컬 모델 최대 토큰 (더 보수적으로)
    local_llm_context_window: int = int(os.getenv("LOCAL_LLM_CONTEXT_WINDOW", "4096"))  # 로컬 모델 컨텍스트 윈도우
    local_llm_timeout: int = int(os.getenv("LOCAL_LLM_TIMEOUT", "120"))  # 로컬 모델 응답 타임아웃 (초)
    local_llm_max_retries: int = int(os.getenv("LOCAL_LLM_MAX_RETRIES", "3"))  # 로컬 모델 재시도 횟수
    # 로컬 GGUF 경로(서버 없이 오프라인 추론)
    local_llm_gguf_path: Optional[str] = os.getenv("LOCAL_LLM_GGUF_PATH")
    # 업로드/전처리 과정에서 HTTP 폴백 비활성화 여부
    disable_http_fallback: bool = os.getenv("DISABLE_HTTP_FALLBACK", "false").lower() == "true"
    # 로컬 LLM 성능/안정성 파라미터
    local_llm_threads: int = int(os.getenv("LOCAL_LLM_THREADS", "8"))
    local_llm_n_gpu_layers: int = int(os.getenv("LOCAL_LLM_N_GPU_LAYERS", "0"))
    # 메모리 보호용 최대 컨텍스트 상한 (llama.cpp)
    local_llm_max_context_cap: int = int(os.getenv("LOCAL_LLM_MAX_CONTEXT_CAP", "65536"))
    
    # Gemma Multimodal 모델 설정
    gemma_multimodal_dir: Optional[str] = os.getenv("GEMMA_MULTIMODAL_DIR")
    
    # A.X Multimodal 모델 설정
    ax_multimodal_dir: Optional[str] = os.getenv("AX_MULTIMODAL_DIR")
    
    # MD 후처리 설정
    enable_md_postprocessing: bool = os.getenv("ENABLE_MD_POSTPROCESSING", "true").lower() == "true"
    md_postprocess_target_quality: int = int(os.getenv("MD_POSTPROCESS_TARGET_QUALITY", "90"))
    
    # 임베딩 모델 설정
    embedding_provider: str = "upstage"  # "openai", "local", "upstage"
    embedding_model_name: str = "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"

    # 로컬 서버 기반 이미지 처리(Flux 등) 사용 여부
    use_local_image_server: bool = os.getenv("USE_LOCAL_IMAGE_SERVER", "true").lower() == "true"
    local_image_relevance_threshold: float = float(os.getenv("LOCAL_IMAGE_RELEVANCE_THRESHOLD", "0.6"))

    # 이미지 분석 프로바이더 (pdf 전처리용): "local" | "openrouter"
    image_analysis_provider: str = os.getenv("IMAGE_ANALYSIS_PROVIDER", "local")
    # 로컬 멀티모달 선호 포트(1620 고정 요청)
    local_mm_prefer_port: str = os.getenv("LOCAL_MM_PREFER_PORT", "1620")

    # OpenRouter (외부 API) 설정 — 사용자 .env에 OPNEROUTER_API_KEY 키가 존재
    # 주의: 오타 포함 정확한 이름을 사용 (OPNEROUTER_API_KEY)
    openrouter_api_key: Optional[str] = os.getenv("OPNEROUTER_API_KEY")
    openrouter_api_base: str = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api")
    openrouter_mm_model: str = os.getenv("OPENROUTER_MM_MODEL", "qwen/qwen2.5-vl-32b-instruct")

    # 이미지 향상(선택) 설정
    enable_image_enhancement: bool = os.getenv("ENABLE_IMAGE_ENHANCEMENT", "false").lower() == "true"
    image_enhancement_prompt: str = os.getenv(
        "IMAGE_ENHANCEMENT_PROMPT",
        "Enhance readability for OCR: denoise, increase contrast, sharpen, preserve original content."
    )

    # 업스테이지 임베딩 설정
    upstage_api_key: Optional[str] = os.getenv("UPSTAGE_API_KEY")
    upstage_embedding_model: str = "solar-embedding-1-large-query"

    # 한국어 최적화 임베딩 모델 설정 (우선 사용)
    korean_embedding_model: Optional[str] = os.getenv("KOREAN_EMBEDDING_MODEL", "jhgan/ko-sbert-multitask")
    use_korean_optimized: bool = os.getenv("USE_KOREAN_OPTIMIZED", "true").lower() == "true"

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
    vector_search_weight: float = float(os.getenv("VECTOR_SEARCH_WEIGHT", "0.7"))   # 벡터 검색 가중치

    # 쿼리 최적화 설정
    enable_query_expansion: bool = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
    enable_query_preprocessing: bool = os.getenv("ENABLE_QUERY_PREPROCESSING", "true").lower() == "true"

    # LangSmith 설정
    langsmith_tracing: str = os.getenv("LANGSMITH_TRACING", "false")
    langsmith_endpoint: Optional[str] = os.getenv("LANGSMITH_ENDPOINT")
    langsmith_api_key: Optional[str] = os.getenv("LANGSMITH_API_KEY")
    langsmith_project: Optional[str] = os.getenv("LANGSMITH_PROJECT")

    temperature: float = 0.3
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
    
    # UI 설정
    app_title: str = "쉽게 설명하는 RAG 챗봇"
    app_description: str = "복잡한 문서도 쉽게! 궁금한 내용을 질문하세요. 일반인도 이해할 수 있도록 친절하게 설명해드립니다."

settings = Settings()
