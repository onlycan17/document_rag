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
    
    # 임베딩 모델 설정
    embedding_provider: str = "local"  # "openai" 또는 "local"
    embedding_model_name: str = "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
    
    # 벡터 데이터베이스 설정
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "./vector_db")
    vector_db_type: str = "faiss"  # "faiss" 또는 "chromadb"
    
    # 청크 설정
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # RAG 설정
    k_documents: int = 10  # 검색할 문서 개수를 늘려서 더 많은 관련 문서 검색
    temperature: float = 0.3
    max_tokens: int = 2000
    
    # UI 설정
    app_title: str = "쉽게 설명하는 RAG 챗봇"
    app_description: str = "복잡한 문서도 쉽게! 궁금한 내용을 질문하세요. 일반인도 이해할 수 있도록 친절하게 설명해드립니다."

    class Config:
        env_file = ".env"

settings = Settings()