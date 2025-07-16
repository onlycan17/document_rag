import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Optional

load_dotenv()

class Settings(BaseSettings):
    # LLM 제공자 설정
    llm_provider: str = "openai"  # "openai", "google", "anthropic", "local"
    
    # OpenAI 설정
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = "gpt-3.5-turbo"
    
    # Google Gemini 설정
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    google_model: str = "gemini-1.5-flash"  # 또는 "gemini-1.5-pro", "gemini-pro"
    
    # Anthropic Claude 설정
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model: str = "claude-3-haiku-20240307"
    
    # 로컬 LLM 설정 (OpenAI 호환 API)
    local_llm_base_url: str = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234")
    local_llm_model: str = os.getenv("LOCAL_LLM_MODEL", "local-model")
    local_llm_api_key: str = os.getenv("LOCAL_LLM_API_KEY", "not-needed")  # 일부 로컬 서버는 API 키 필요
    local_llm_max_tokens: int = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "512"))  # 로컬 모델 최대 토큰 (더 보수적으로)
    local_llm_context_window: int = int(os.getenv("LOCAL_LLM_CONTEXT_WINDOW", "4096"))  # 로컬 모델 컨텍스트 윈도우
    
    # 임베딩 모델 설정
    embedding_provider: str = "upstage"  # "openai", "local", "upstage"
    embedding_model_name: str = "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"

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
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))  # 더 큰 청크로 맥락 유지
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))  # 오버랩 증가로 연속성 향상
    # 의미 기반 청킹 설정
    use_semantic_chunking: bool = os.getenv("USE_SEMANTIC_CHUNKING", "false").lower() == "true"
    semantic_chunk_sentences: int = 3  # 의미 기반 청킹 시 문장 단위

    # RAG 설정 - 검색 성능 최적화
    k_documents: int = int(os.getenv("K_DOCUMENTS", "8"))  # 모델 토큰 제한 고려하여 8개로 조정
    search_threshold_faiss: float = 1.24  # FAISS 임계값 (거리 기반, 38% 유사도에 해당)
    search_threshold_chromadb: float = 0.38  # ChromaDB 임계값 (유사도 38%)
    use_mmr_search: bool = os.getenv("USE_MMR_SEARCH", "true").lower() == "true"  # 다양성 확보
    mmr_diversity_score: float = 0.3  # MMR 다양성 점수

    # 하이브리드 검색 설정
    enable_hybrid_search: bool = os.getenv("ENABLE_HYBRID_SEARCH", "false").lower() == "true"
    keyword_search_weight: float = float(os.getenv("KEYWORD_SEARCH_WEIGHT", "0.3"))  # 키워드 검색 가중치
    vector_search_weight: float = float(os.getenv("VECTOR_SEARCH_WEIGHT", "0.7"))   # 벡터 검색 가중치

    # 쿼리 최적화 설정
    enable_query_expansion: bool = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
    enable_query_preprocessing: bool = os.getenv("ENABLE_QUERY_PREPROCESSING", "true").lower() == "true"

    temperature: float = 0.3
    max_tokens: int = 4096  # 기본값 - 모델별로 자동 조정됨
    
    # 스트리밍 설정
    enable_streaming: bool = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
    
    # UI 설정
    app_title: str = "쉽게 설명하는 RAG 챗봇"
    app_description: str = "복잡한 문서도 쉽게! 궁금한 내용을 질문하세요. 일반인도 이해할 수 있도록 친절하게 설명해드립니다."

    class Config:
        env_file = ".env"

settings = Settings()