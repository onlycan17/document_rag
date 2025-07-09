from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config import settings
from src.embeddings import EmbeddingModel
import os
import pickle
import logging

logger = logging.getLogger(__name__)

# ChromaDB 텔레메트리 비활성화
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

class VectorDatabase:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """벡터 스토어 초기화"""
        if settings.vector_db_type == "chromadb":
            self.vector_store = Chroma(
                persist_directory=settings.vector_db_path,
                embedding_function=self.embedding_model.embeddings
            )
        elif settings.vector_db_type == "faiss":
            # FAISS 인덱스가 이미 존재하는지 확인
            faiss_index_path = os.path.join(settings.vector_db_path, "faiss_index.pkl")
            if os.path.exists(faiss_index_path):
                self.load_faiss_index()
            else:
                # 새로운 FAISS 인덱스 생성은 문서 추가 시 수행
                self.vector_store = None
        else:
            raise ValueError(f"지원하지 않는 벡터 DB 타입: {settings.vector_db_type}")
    
    def add_documents(self, documents: List[Document]):
        """문서를 벡터 DB에 추가"""
        if settings.vector_db_type == "chromadb":
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
        elif settings.vector_db_type == "faiss":
            if self.vector_store is None:
                # 첫 문서 추가 시 FAISS 인덱스 생성
                self.vector_store = FAISS.from_documents(
                    documents, 
                    self.embedding_model.embeddings
                )
            else:
                self.vector_store.add_documents(documents)
            self.save_faiss_index()
        
        print(f"{len(documents)}개의 문서가 벡터 DB에 추가되었습니다.")
    
    def search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """쿼리와 유사한 문서 검색"""
        if self.vector_store is None:
            return []
        
        if k is None:
            k = settings.k_documents
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            # 점수가 너무 낮은 결과는 제외 (관련성이 낮음)
            # FAISS의 경우 거리 기반이므로 낮을수록 좋음
            filtered_results = []
            for doc, score in results:
                if settings.vector_db_type == "faiss":
                    # FAISS는 거리 기반 (낮을수록 좋음)
                    # 임계값을 더 관대하게 설정하여 더 많은 문서 포함
                    if score < 2.0:  # 임계값을 1.5에서 2.0으로 상향
                        filtered_results.append((doc, score))
                else:
                    # ChromaDB는 유사도 기반 (높을수록 좋음)
                    if score > 0.2:  # 임계값을 0.3에서 0.2로 하향
                        filtered_results.append((doc, score))
            
            return filtered_results
        except Exception as e:
            logger.error(f"검색 중 오류 발생: {str(e)}")
            return []
    
    def save_faiss_index(self):
        """FAISS 인덱스 저장"""
        if settings.vector_db_type == "faiss" and self.vector_store:
            os.makedirs(settings.vector_db_path, exist_ok=True)
            faiss_index_path = os.path.join(settings.vector_db_path, "faiss_index.pkl")
            with open(faiss_index_path, "wb") as f:
                pickle.dump(self.vector_store, f)
    
    def load_faiss_index(self):
        """FAISS 인덱스 로드"""
        if settings.vector_db_type == "faiss":
            faiss_index_path = os.path.join(settings.vector_db_path, "faiss_index.pkl")
            with open(faiss_index_path, "rb") as f:
                self.vector_store = pickle.load(f)
    
    def clear_database(self):
        """벡터 DB 초기화"""
        if settings.vector_db_type == "chromadb":
            # ChromaDB 데이터 디렉토리 삭제
            import shutil
            if os.path.exists(settings.vector_db_path):
                shutil.rmtree(settings.vector_db_path)
                os.makedirs(settings.vector_db_path, exist_ok=True)
            self.vector_store = None
            self._initialize_vector_store()
        elif settings.vector_db_type == "faiss":
            # FAISS 인덱스 파일 삭제
            faiss_index_path = os.path.join(settings.vector_db_path, "faiss_index.pkl")
            if os.path.exists(faiss_index_path):
                os.remove(faiss_index_path)
            self.vector_store = None
        
        print("벡터 데이터베이스가 초기화되었습니다.")
    
    def get_document_count(self) -> int:
        """저장된 문서 수 반환"""
        if self.vector_store is None:
            return 0
        
        if settings.vector_db_type == "chromadb":
            # ChromaDB의 경우
            return self.vector_store._collection.count()
        elif settings.vector_db_type == "faiss":
            # FAISS의 경우
            return self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else 0