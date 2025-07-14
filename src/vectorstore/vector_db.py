from typing import List, Dict, Any, Tuple, Optional
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from config import settings
from src.embeddings import EmbeddingModel
from src.utils import TextProcessor
import os
import pickle
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ChromaDB 텔레메트리 비활성화
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

class EnhancedVectorDatabase:
    """
    향상된 벡터 데이터베이스 클래스
    - 하이브리드 검색 (벡터 + 키워드) 지원
    - MMR (Maximal Marginal Relevance) 검색
    - 개선된 임계값 처리
    - 검색 성능 최적화
    """
    
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = None
        self.documents_cache = []  # 키워드 검색을 위한 문서 캐시
        self.bm25_retriever = None  # BM25 키워드 검색기
        self.tfidf_vectorizer = None  # TF-IDF 벡터라이저
        self.tfidf_matrix = None  # TF-IDF 매트릭스
        
        self._initialize_vector_store()
        logger.info(f"벡터 데이터베이스 초기화 완료 (타입: {settings.vector_db_type})")
    
    def _initialize_vector_store(self):
        """벡터 스토어 초기화"""
        if settings.vector_db_type == "chromadb":
            self.vector_store = Chroma(
                persist_directory=settings.vector_db_path,
                embedding_function=self.embedding_model.embeddings
            )
        elif settings.vector_db_type == "faiss":
            # FAISS 인덱스가 이미 존재하는지 확인
            faiss_index_path = os.path.join(settings.vector_db_path, "index.faiss")
            old_faiss_index_path = os.path.join(settings.vector_db_path, "faiss_index.pkl")
            
            if os.path.exists(faiss_index_path) or os.path.exists(old_faiss_index_path):
                self.load_faiss_index()
            else:
                # 새로운 FAISS 인덱스 생성은 문서 추가 시 수행
                self.vector_store = None
        else:
            raise ValueError(f"지원하지 않는 벡터 DB 타입: {settings.vector_db_type}")
    
    def add_documents(self, documents: List[Document]):
        """
        문서를 벡터 DB에 추가
        - 벡터 검색용 인덱스 구축
        - 키워드 검색용 인덱스 구축
        """
        if not documents:
            logger.warning("추가할 문서가 없습니다.")
            return
        
        # 벡터 스토어에 추가
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
        
        # 키워드 검색용 문서 캐시 업데이트
        self.documents_cache.extend(documents)
        self._update_keyword_search_index()
        
        logger.info(f"{len(documents)}개의 문서가 벡터 DB에 추가되었습니다.")
        logger.info(f"총 문서 수: {len(self.documents_cache)}개")
    
    def _update_keyword_search_index(self):
        """키워드 검색 인덱스 업데이트"""
        try:
            if not self.documents_cache:
                return
            
            # BM25 검색기 업데이트
            if settings.enable_hybrid_search:
                texts = [doc.page_content for doc in self.documents_cache]
                self.bm25_retriever = BM25Retriever.from_texts(
                    texts, 
                    metadatas=[doc.metadata for doc in self.documents_cache]
                )
                logger.info("BM25 키워드 검색 인덱스 업데이트 완료")
            
            # TF-IDF 인덱스 업데이트 (추가 키워드 검색용)
            texts = [self._preprocess_text_for_keyword_search(doc.page_content) 
                    for doc in self.documents_cache]
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words=None,  # 한국어 불용어는 별도 처리
                min_df=1
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.info("TF-IDF 키워드 검색 인덱스 업데이트 완료")
            
        except Exception as e:
            logger.error(f"키워드 검색 인덱스 업데이트 실패: {str(e)}")
    
    def _preprocess_text_for_keyword_search(self, text: str) -> str:
        """키워드 검색을 위한 텍스트 전처리"""
        # TextProcessor를 사용하여 텍스트 정제 및 불용어 제거
        return TextProcessor.clean_text(text, remove_stopwords=True)
    
    def search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        향상된 검색 메서드
        - 하이브리드 검색 지원
        - MMR 검색 지원
        - 동적 임계값 조정
        """
        if self.vector_store is None:
            return []
        
        if k is None:
            k = settings.k_documents
        
        try:
            # 하이브리드 검색 사용 여부 확인
            if settings.enable_hybrid_search and self.bm25_retriever:
                return self._hybrid_search(query, k)
            else:
                return self._vector_search(query, k)
                
        except Exception as e:
            logger.error(f"검색 중 오류 발생: {str(e)}")
            return []
    
    def _vector_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """순수 벡터 검색"""
        try:
            # MMR 검색 사용 여부 확인
            if settings.use_mmr_search:
                return self._mmr_search(query, k)
            else:
                return self._similarity_search(query, k)
        except Exception as e:
            logger.error(f"벡터 검색 실패: {str(e)}")
            return []
    
    def _similarity_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """기본 유사도 검색"""
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return self._filter_by_threshold(results)
    
    def _mmr_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """MMR (Maximal Marginal Relevance) 검색"""
        try:
            # MMR 검색 수행
            docs = self.vector_store.max_marginal_relevance_search(
                query, 
                k=k,
                fetch_k=k * 2,  # 더 많은 후보에서 선택
                lambda_mult=1 - settings.mmr_diversity_score  # 다양성 조절
            )
            
            # 점수는 별도로 계산해야 함 (MMR은 점수를 반환하지 않음)
            scored_results = []
            for doc in docs:
                # 임베딩을 통한 유사도 계산
                doc_embedding = self.embedding_model.embed_query(doc.page_content)
                query_embedding = self.embedding_model.embed_query(query)
                
                # 코사인 유사도 계산
                similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                # FAISS 거리로 변환 (낮을수록 좋음)
                distance = 1 - similarity
                
                scored_results.append((doc, distance))
            
            return self._filter_by_threshold(scored_results)
            
        except Exception as e:
            logger.warning(f"MMR 검색 실패, 기본 검색 사용: {str(e)}")
            return self._similarity_search(query, k)
    
    def _hybrid_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """하이브리드 검색 (벡터 + 키워드)"""
        try:
            # 1. 벡터 검색 결과
            vector_results = self._similarity_search(query, k)
            
            # 2. 키워드 검색 결과
            keyword_results = self._keyword_search(query, k)
            
            # 3. 결과 통합 및 점수 정규화
            combined_results = self._combine_search_results(
                vector_results, 
                keyword_results, 
                settings.vector_search_weight,
                settings.keyword_search_weight
            )
            
            # 4. 상위 k개 결과 반환
            combined_results.sort(key=lambda x: x[1])  # 점수 오름차순 정렬 (낮을수록 좋음)
            return combined_results[:k]
            
        except Exception as e:
            logger.error(f"하이브리드 검색 실패, 벡터 검색만 사용: {str(e)}")
            return self._vector_search(query, k)
    
    def _keyword_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """키워드 검색"""
        keyword_results = []
        
        try:
            # BM25 검색
            if self.bm25_retriever:
                bm25_docs = self.bm25_retriever.get_relevant_documents(query)
                for doc in bm25_docs[:k]:
                    # BM25 점수를 거리로 변환 (간단한 추정)
                    score = 0.5  # BM25는 정확한 점수를 제공하지 않으므로 기본값 사용
                    keyword_results.append((doc, score))
            
            # TF-IDF 검색 (추가 검증)
            if self.tfidf_vectorizer and self.tfidf_matrix is not None:
                tfidf_results = self._tfidf_search(query, k)
                keyword_results.extend(tfidf_results)
            
        except Exception as e:
            logger.error(f"키워드 검색 실패: {str(e)}")
        
        return keyword_results
    
    def _tfidf_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """TF-IDF 기반 키워드 검색"""
        try:
            # 쿼리 벡터화
            query_processed = self._preprocess_text_for_keyword_search(query)
            query_vector = self.tfidf_vectorizer.transform([query_processed])
            
            # 코사인 유사도 계산
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # 상위 k개 인덱스 추출
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents_cache) and similarities[idx] > 0.1:  # 최소 임계값
                    doc = self.documents_cache[idx]
                    # 유사도를 거리로 변환
                    distance = 1 - similarities[idx]
                    results.append((doc, distance))
            
            return results
            
        except Exception as e:
            logger.error(f"TF-IDF 검색 실패: {str(e)}")
            return []
    
    def _combine_search_results(self, vector_results: List[Tuple[Document, float]], 
                              keyword_results: List[Tuple[Document, float]],
                              vector_weight: float, keyword_weight: float) -> List[Tuple[Document, float]]:
        """검색 결과 통합"""
        combined_dict = {}
        
        # 문서 ID 생성 함수
        def get_doc_id(doc):
            return doc.metadata.get('chunk_id', doc.page_content[:100])
        
        # 벡터 검색 결과 추가
        for doc, score in vector_results:
            doc_id = get_doc_id(doc)
            combined_dict[doc_id] = {
                'doc': doc,
                'vector_score': score,
                'keyword_score': None
            }
        
        # 키워드 검색 결과 추가
        for doc, score in keyword_results:
            doc_id = get_doc_id(doc)
            if doc_id in combined_dict:
                combined_dict[doc_id]['keyword_score'] = score
            else:
                combined_dict[doc_id] = {
                    'doc': doc,
                    'vector_score': None,
                    'keyword_score': score
                }
        
        # 통합 점수 계산
        results = []
        for doc_id, data in combined_dict.items():
            vector_score = data['vector_score'] if data['vector_score'] is not None else 1.0
            keyword_score = data['keyword_score'] if data['keyword_score'] is not None else 1.0
            
            # 가중 평균으로 통합 점수 계산
            combined_score = (vector_score * vector_weight + keyword_score * keyword_weight)
            results.append((data['doc'], combined_score))
        
        return results
    
    def _filter_by_threshold(self, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """임계값에 따른 결과 필터링"""
        if not results:
            return []
        
        # 임계값 설정
        if settings.vector_db_type == "faiss":
            threshold = settings.search_threshold_faiss
        else:
            threshold = settings.search_threshold_chromadb
        
        # 동적 임계값 조정
        scores = [score for _, score in results]
        if scores:
            min_score = min(scores)
            avg_score = sum(scores) / len(scores)
            
            # 모든 점수가 임계값보다 높으면 임계값을 완화
            if min_score > threshold:
                adjusted_threshold = min(threshold * 1.5, avg_score)
                logger.info(f"임계값 동적 조정: {threshold} -> {adjusted_threshold}")
                threshold = adjusted_threshold
        
        # 필터링 적용
        filtered_results = []
        for doc, score in results:
            if settings.vector_db_type == "faiss":
                # FAISS는 거리 기반 (낮을수록 좋음)
                if score < threshold:
                    filtered_results.append((doc, score))
            else:
                # ChromaDB는 유사도 기반 (높을수록 좋음)
                if score > threshold:
                    filtered_results.append((doc, score))
        
        # 필터링 결과가 너무 적으면 원본 결과의 일부라도 반환
        if len(filtered_results) < 2 and results:
            logger.warning(f"필터링 결과가 부족함 ({len(filtered_results)}개), 상위 결과 포함")
            filtered_results = results[:max(3, len(results) // 2)]
        
        return filtered_results
    
    def get_document_count(self) -> int:
        """저장된 문서 수 반환"""
        return len(self.documents_cache)
    
    def clear_database(self):
        """데이터베이스 초기화"""
        # 벡터 스토어 초기화
        if settings.vector_db_type == "chromadb" and self.vector_store:
            try:
                self.vector_store.delete_collection()
            except:
                pass
        elif settings.vector_db_type == "faiss":
            # FAISS 파일 삭제
            faiss_index_path = os.path.join(settings.vector_db_path, "faiss_index.pkl")
            if os.path.exists(faiss_index_path):
                os.remove(faiss_index_path)
        
        # 캐시 및 인덱스 초기화
        self.documents_cache = []
        self.bm25_retriever = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.vector_store = None
        
        # 벡터 스토어 재초기화
        self._initialize_vector_store()
        
        logger.info("벡터 데이터베이스가 초기화되었습니다.")
    
    def save_faiss_index(self):
        """FAISS 인덱스 저장 (pickle 오류 방지)"""
        if settings.vector_db_type == "faiss" and self.vector_store:
            try:
                os.makedirs(settings.vector_db_path, exist_ok=True)
                
                # FAISS 벡터 스토어는 save_local 메서드 사용
                self.vector_store.save_local(settings.vector_db_path)
                
                # 문서 캐시는 별도로 pickle로 저장
                cache_path = os.path.join(settings.vector_db_path, "documents_cache.pkl")
                with open(cache_path, "wb") as f:
                    pickle.dump(self.documents_cache, f)
                
                logger.info(f"FAISS 인덱스 저장 완료: {len(self.documents_cache)}개 문서 캐시 포함")
                
            except Exception as e:
                logger.error(f"FAISS 인덱스 저장 실패: {str(e)}")
                # 저장 실패 시에도 계속 진행 (메모리에는 유지됨)
                logger.warning("인덱스는 메모리에만 유지됩니다. 다음 실행 시 재구축이 필요합니다.")
    
    def load_faiss_index(self):
        """FAISS 인덱스 로드 (pickle 오류 방지)"""
        if settings.vector_db_type == "faiss":
            try:
                # FAISS 벡터 스토어 로드
                if os.path.exists(os.path.join(settings.vector_db_path, "index.faiss")):
                    self.vector_store = FAISS.load_local(
                        settings.vector_db_path, 
                        self.embedding_model.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                    # 문서 캐시 로드
                    cache_path = os.path.join(settings.vector_db_path, "documents_cache.pkl")
                    if os.path.exists(cache_path):
                        with open(cache_path, "rb") as f:
                            self.documents_cache = pickle.load(f)
                    else:
                        self.documents_cache = []
                    
                    # 키워드 검색 인덱스 재구축
                    if self.documents_cache:
                        self._update_keyword_search_index()
                    
                    logger.info(f"FAISS 인덱스 로드 완료 (문서 수: {len(self.documents_cache)})")
                    
                else:
                    # 기존 pickle 형식 파일이 있는지 확인 (하위 호환성)
                    old_faiss_index_path = os.path.join(settings.vector_db_path, "faiss_index.pkl")
                    if os.path.exists(old_faiss_index_path):
                        logger.info("기존 pickle 형식 인덱스 발견, 새 형식으로 마이그레이션...")
                        with open(old_faiss_index_path, "rb") as f:
                            data = pickle.load(f)
                        
                        if isinstance(data, dict):
                            self.vector_store = data['vector_store']
                            self.documents_cache = data.get('documents_cache', [])
                        else:
                            self.vector_store = data
                            self.documents_cache = []
                        
                        # 새 형식으로 저장
                        self.save_faiss_index()
                        
                        # 기존 파일 제거
                        os.remove(old_faiss_index_path)
                        logger.info("마이그레이션 완료")
                    else:
                        self.vector_store = None
                        self.documents_cache = []
                        
            except Exception as e:
                logger.error(f"FAISS 인덱스 로드 실패: {str(e)}")
                self.vector_store = None
                self.documents_cache = []
    
    def get_search_stats(self) -> Dict[str, Any]:
        """검색 관련 통계 정보 반환"""
        return {
            "total_documents": len(self.documents_cache),
            "vector_db_type": settings.vector_db_type,
            "hybrid_search_enabled": settings.enable_hybrid_search,
            "mmr_search_enabled": settings.use_mmr_search,
            "search_threshold": {
                "faiss": settings.search_threshold_faiss,
                "chromadb": settings.search_threshold_chromadb
            },
            "embedding_model": self.embedding_model.get_model_info(),
            "keyword_search_available": self.bm25_retriever is not None
        }
    
    def extract_related_terms(self, query: str, top_k: int = 20) -> List[str]:
        """
        문서에서 쿼리와 관련된 용어를 동적으로 추출
        TF-IDF와 코사인 유사도를 사용하여 관련 용어 찾기
        """
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            return []
        
        try:
            # 쿼리 벡터화
            query_processed = self._preprocess_text_for_keyword_search(query)
            query_vector = self.tfidf_vectorizer.transform([query_processed])
            
            # 문서와의 유사도 계산
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # 상위 문서 선택
            top_doc_indices = np.argsort(similarities)[::-1][:5]  # 상위 5개 문서
            
            # 관련 용어 추출
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            related_terms = set()
            
            for idx in top_doc_indices:
                if idx < len(self.documents_cache) and similarities[idx] > 0.1:
                    # 해당 문서의 TF-IDF 벡터
                    doc_vector = self.tfidf_matrix[idx]
                    
                    # 중요한 용어 추출 (TF-IDF 점수가 높은 용어)
                    doc_scores = doc_vector.toarray().flatten()
                    top_term_indices = np.argsort(doc_scores)[::-1][:10]  # 문서당 상위 10개 용어
                    
                    for term_idx in top_term_indices:
                        if doc_scores[term_idx] > 0.1:  # 임계값 이상인 용어만
                            term = feature_names[term_idx]
                            if len(term) > 1 and term not in query_processed:  # 쿼리에 없는 용어만
                                related_terms.add(term)
            
            # 원본 쿼리의 단어도 포함
            for word in query.split():
                if len(word) > 1:
                    related_terms.add(word)
            
            return list(related_terms)[:top_k]
            
        except Exception as e:
            logger.error(f"관련 용어 추출 실패: {str(e)}")
            return []

# 기존 VectorDatabase와의 호환성 유지
class VectorDatabase(EnhancedVectorDatabase):
    """기존 VectorDatabase와의 호환성을 위한 클래스"""
    pass