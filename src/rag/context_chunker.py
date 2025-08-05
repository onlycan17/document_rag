"""
컨텍스트 분할 시스템

대량 문서 처리를 위해 컨텍스트를 의미있는 단위로 분할하는 클래스.
LLM의 컨텍스트 크기 제한을 극복하여 더 많은 문서를 처리할 수 있도록 합니다.
"""

from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document
import logging
import numpy as np
from collections import defaultdict
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class ContextChunk:
    """컨텍스트 청크를 나타내는 클래스"""
    
    def __init__(self, 
                 documents: List[Tuple[Document, float]], 
                 chunk_id: str,
                 total_length: int,
                 topic_keywords: List[str] = None,
                 relevance_score: float = 0.0):
        self.documents = documents
        self.chunk_id = chunk_id
        self.total_length = total_length
        self.topic_keywords = topic_keywords or []
        self.relevance_score = relevance_score
        self.processed_result: Optional[Dict[str, Any]] = None
    
    def get_content(self) -> str:
        """청크의 전체 내용을 문자열로 반환"""
        content_parts = []
        for i, (doc, score) in enumerate(self.documents):
            content_parts.append(f"[문서 {i+1}]\n{doc.page_content}\n")
        return "\n".join(content_parts)
    
    def get_metadata(self) -> Dict[str, Any]:
        """청크의 메타데이터 반환"""
        return {
            "chunk_id": self.chunk_id,
            "document_count": len(self.documents),
            "total_length": self.total_length,
            "topic_keywords": self.topic_keywords,
            "relevance_score": self.relevance_score,
            "file_sources": list(set(doc.metadata.get('file_name', 'Unknown') 
                                   for doc, _ in self.documents))
        }

class ContextChunker:
    """컨텍스트 분할 관리 클래스"""
    
    def __init__(self, max_chunk_size: int = 30000):
        """
        Args:
            max_chunk_size: 각 청크의 최대 크기 (문자 수)
        """
        self.max_chunk_size = max_chunk_size
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def split_documents(self, 
                       documents: List[Tuple[Document, float]], 
                       query: str = "",
                       strategy: str = "relevance") -> List[ContextChunk]:
        """
        문서들을 여러 청크로 분할
        
        Args:
            documents: (Document, score) 튜플의 리스트
            query: 사용자 쿼리
            strategy: 분할 전략 ("relevance", "topic", "size", "hybrid")
            
        Returns:
            ContextChunk 객체들의 리스트
        """
        if not documents:
            return []
        
        total_length = sum(len(doc.page_content) for doc, _ in documents)
        
        # 분할이 필요하지 않은 경우
        if total_length <= self.max_chunk_size:
            chunk = ContextChunk(
                documents=documents,
                chunk_id="single_chunk",
                total_length=total_length,
                relevance_score=np.mean([score for _, score in documents])
            )
            return [chunk]
        
        logger.info(f"컨텍스트 분할 시작: {len(documents)}개 문서, {total_length:,}자 → 최대 {self.max_chunk_size:,}자 청크로 분할")
        
        # 전략에 따른 분할
        if strategy == "relevance":
            chunks = self._split_by_relevance(documents, query)
        elif strategy == "topic":
            chunks = self._split_by_topic(documents, query)
        elif strategy == "size":
            chunks = self._split_by_size(documents)
        elif strategy == "hybrid":
            chunks = self._split_hybrid(documents, query)
        else:
            raise ValueError(f"지원하지 않는 분할 전략: {strategy}")
        
        logger.info(f"컨텍스트 분할 완료: {len(chunks)}개 청크 생성")
        return chunks
    
    def _split_by_relevance(self, 
                           documents: List[Tuple[Document, float]], 
                           query: str) -> List[ContextChunk]:
        """관련도 기반 분할"""
        # 관련도 순으로 정렬
        sorted_docs = sorted(documents, key=lambda x: x[1], reverse=True)
        
        chunks = []
        current_chunk_docs = []
        current_length = 0
        chunk_id = 0
        
        for doc, score in sorted_docs:
            doc_length = len(doc.page_content)
            
            # 현재 청크에 추가할 수 있는지 확인
            if current_length + doc_length <= self.max_chunk_size:
                current_chunk_docs.append((doc, score))
                current_length += doc_length
            else:
                # 현재 청크 완성
                if current_chunk_docs:
                    chunk = ContextChunk(
                        documents=current_chunk_docs,
                        chunk_id=f"relevance_chunk_{chunk_id}",
                        total_length=current_length,
                        relevance_score=np.mean([s for _, s in current_chunk_docs])
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                # 새 청크 시작
                current_chunk_docs = [(doc, score)]
                current_length = doc_length
        
        # 마지막 청크 처리
        if current_chunk_docs:
            chunk = ContextChunk(
                documents=current_chunk_docs,
                chunk_id=f"relevance_chunk_{chunk_id}",
                total_length=current_length,
                relevance_score=np.mean([s for _, s in current_chunk_docs])
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_topic(self, 
                       documents: List[Tuple[Document, float]], 
                       query: str) -> List[ContextChunk]:
        """주제별 분할 (클러스터링 기반)"""
        if len(documents) < 3:
            return self._split_by_relevance(documents, query)
        
        # 문서 내용 추출
        texts = [doc.page_content for doc, _ in documents]
        
        try:
            # TF-IDF 벡터화
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # 최적 클러스터 수 결정 (문서 수의 1/3, 최소 2개, 최대 5개)
            n_clusters = max(2, min(5, len(documents) // 3))
            
            # K-means 클러스터링
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # 클러스터별 문서 그룹핑
            clusters = defaultdict(list)
            for i, (doc, score) in enumerate(documents):
                cluster_id = cluster_labels[i]
                clusters[cluster_id].append((doc, score, i))
            
            # 각 클러스터의 주요 키워드 추출
            feature_names = self.vectorizer.get_feature_names_out()
            
            chunks = []
            for cluster_id, cluster_docs in clusters.items():
                # 클러스터 크기가 너무 큰 경우 재귀적으로 분할
                cluster_length = sum(len(doc.page_content) for doc, _, _ in cluster_docs)
                
                if cluster_length > self.max_chunk_size:
                    # 크기 기반으로 재분할
                    sub_docs = [(doc, score) for doc, score, _ in cluster_docs]
                    sub_chunks = self._split_by_size(sub_docs)
                    chunks.extend(sub_chunks)
                else:
                    # 주제 키워드 추출
                    cluster_indices = [idx for _, _, idx in cluster_docs]
                    cluster_center = kmeans.cluster_centers_[cluster_id]
                    top_features = np.argsort(cluster_center)[-5:][::-1]
                    topic_keywords = [feature_names[i] for i in top_features]
                    
                    chunk = ContextChunk(
                        documents=[(doc, score) for doc, score, _ in cluster_docs],
                        chunk_id=f"topic_chunk_{cluster_id}",
                        total_length=cluster_length,
                        topic_keywords=topic_keywords,
                        relevance_score=np.mean([score for _, score, _ in cluster_docs])
                    )
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.warning(f"주제별 분할 실패, 관련도 기반으로 대체: {e}")
            return self._split_by_relevance(documents, query)
    
    def _split_by_size(self, documents: List[Tuple[Document, float]]) -> List[ContextChunk]:
        """크기 기반 균등 분할"""
        chunks = []
        current_chunk_docs = []
        current_length = 0
        chunk_id = 0
        
        for doc, score in documents:
            doc_length = len(doc.page_content)
            
            # 단일 문서가 최대 크기를 초과하는 경우 강제 포함
            if doc_length > self.max_chunk_size:
                if current_chunk_docs:
                    # 기존 청크 완성
                    chunk = ContextChunk(
                        documents=current_chunk_docs,
                        chunk_id=f"size_chunk_{chunk_id}",
                        total_length=current_length,
                        relevance_score=np.mean([s for _, s in current_chunk_docs])
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    current_chunk_docs = []
                    current_length = 0
                
                # 큰 문서를 별도 청크로 처리
                chunk = ContextChunk(
                    documents=[(doc, score)],
                    chunk_id=f"large_doc_chunk_{chunk_id}",
                    total_length=doc_length,
                    relevance_score=score
                )
                chunks.append(chunk)
                chunk_id += 1
                continue
            
            # 현재 청크에 추가 가능한지 확인
            if current_length + doc_length <= self.max_chunk_size:
                current_chunk_docs.append((doc, score))
                current_length += doc_length
            else:
                # 현재 청크 완성
                if current_chunk_docs:
                    chunk = ContextChunk(
                        documents=current_chunk_docs,
                        chunk_id=f"size_chunk_{chunk_id}",
                        total_length=current_length,
                        relevance_score=np.mean([s for _, s in current_chunk_docs])
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                # 새 청크 시작
                current_chunk_docs = [(doc, score)]
                current_length = doc_length
        
        # 마지막 청크 처리
        if current_chunk_docs:
            chunk = ContextChunk(
                documents=current_chunk_docs,
                chunk_id=f"size_chunk_{chunk_id}",
                total_length=current_length,
                relevance_score=np.mean([s for _, s in current_chunk_docs])
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_hybrid(self, 
                     documents: List[Tuple[Document, float]], 
                     query: str) -> List[ContextChunk]:
        """하이브리드 분할 (주제별 + 관련도 혼합)"""
        # 먼저 주제별로 분할 시도
        topic_chunks = self._split_by_topic(documents, query)
        
        # 각 주제 청크에서 크기가 큰 것은 관련도 기반으로 재분할
        final_chunks = []
        
        for chunk in topic_chunks:
            if chunk.total_length > self.max_chunk_size * 0.8:  # 80% 이상이면 재분할
                logger.info(f"청크 {chunk.chunk_id} 크기 초과로 재분할: {chunk.total_length:,}자")
                sub_chunks = self._split_by_relevance(chunk.documents, query)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def optimize_chunk_order(self, 
                           chunks: List[ContextChunk], 
                           query: str) -> List[ContextChunk]:
        """청크 순서 최적화 (관련도 기반)"""
        if not chunks:
            return chunks
        
        # 쿼리와의 관련도 기반으로 정렬
        def calculate_query_relevance(chunk: ContextChunk) -> float:
            """청크와 쿼리 간 관련도 계산"""
            chunk_text = chunk.get_content().lower()
            query_lower = query.lower()
            
            # 단순한 키워드 매칭 점수
            query_words = set(query_lower.split())
            chunk_words = set(chunk_text.split())
            
            if not query_words:
                return chunk.relevance_score
            
            # 공통 단어 비율
            common_words = query_words.intersection(chunk_words)
            keyword_score = len(common_words) / len(query_words)
            
            # 기존 관련도 점수와 결합
            combined_score = (chunk.relevance_score * 0.7) + (keyword_score * 0.3)
            return combined_score
        
        # 관련도 기반 정렬
        sorted_chunks = sorted(chunks, key=calculate_query_relevance, reverse=True)
        
        logger.info(f"청크 순서 최적화 완료: {len(sorted_chunks)}개 청크")
        return sorted_chunks
    
    def get_chunk_summary(self, chunks: List[ContextChunk]) -> Dict[str, Any]:
        """청크 분할 결과 요약"""
        if not chunks:
            return {"total_chunks": 0, "total_documents": 0, "total_length": 0}
        
        total_documents = sum(len(chunk.documents) for chunk in chunks)
        total_length = sum(chunk.total_length for chunk in chunks)
        avg_chunk_size = total_length / len(chunks) if chunks else 0
        
        chunk_sizes = [chunk.total_length for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_documents": total_documents,
            "total_length": total_length,
            "average_chunk_size": int(avg_chunk_size),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "chunk_distribution": {
                f"chunk_{i}": {
                    "size": chunk.total_length,
                    "documents": len(chunk.documents),
                    "relevance": chunk.relevance_score,
                    "topics": chunk.topic_keywords[:3] if chunk.topic_keywords else []
                }
                for i, chunk in enumerate(chunks)
            }
        }