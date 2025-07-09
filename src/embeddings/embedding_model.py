from typing import List, Union
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import settings
import numpy as np

class EmbeddingModel:
    def __init__(self):
        if settings.embedding_provider == "openai" and settings.openai_api_key:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=settings.openai_api_key,
                model="text-embedding-ada-002"
            )
            self.model_type = "openai"
        else:
            # 다국어 지원 모델 사용
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.model_type = "huggingface"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 문서를 임베딩"""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리를 임베딩"""
        return self.embeddings.embed_query(text)
    
    def get_embedding_dimension(self) -> int:
        """임베딩 차원 반환"""
        if self.model_type == "openai":
            return 1536  # OpenAI text-embedding-ada-002
        else:
            # 테스트 임베딩으로 차원 확인
            test_embedding = self.embed_query("test")
            return len(test_embedding)