import logging
import os
import pickle
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import settings
from src.embeddings import EmbeddingModel
from src.utils import TextProcessor
from src.utils.tracing import observe_if_enabled, record_input, record_metadata, record_retriever_output

logger = logging.getLogger(__name__)

# ChromaDB 텔레메트리 비활성화
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


def _retriever_sources(results: List[Tuple[Document, float]]) -> List[dict]:
    """검색 결과를 LangSmith retriever run용 요약(출처·점수)으로 변환한다."""
    return [
        {"source": getattr(doc, "metadata", {}).get("source", "?"), "score": round(float(score), 4)}
        for doc, score in results
    ]


def reciprocal_rank_fusion(rankings: List[List[Document]], rank_constant: int = 60) -> List[Document]:
    """여러 검색 순위를 RRF(순위 역수 합)로 융합한다. 점수 척도가 달라도 순위만 쓰므로 안전하다.

    같은 청크는 본문으로 식별한다. 반환 순서 = 융합 점수 내림차순.
    """
    fused: Dict[str, float] = {}
    docs: Dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = doc.page_content
            fused[key] = fused.get(key, 0.0) + 1.0 / (rank_constant + rank)
            docs.setdefault(key, doc)
    return [docs[key] for key in sorted(fused, key=fused.get, reverse=True)]


class VectorDatabase:
    """
    벡터 데이터베이스 클래스
    - 하이브리드 검색 (벡터 + 키워드) 지원
    - MMR (Maximal Marginal Relevance) 검색
    - 개선된 임계값 처리
    - 검색 성능 최적화
    """

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = None
        self.documents_cache = []  # 키워드 검색을 위한 문서 캐시
        self.tfidf_vectorizer = None  # 단어 TF-IDF (관련 용어 추출용)
        self.tfidf_matrix = None
        self.char_vectorizer = None  # 문자 n-gram TF-IDF (하이브리드 키워드 검색용)
        self.char_matrix = None
        self._query_cache = {}  # 유사 질의 캐시: {f"{query}|{k}": (저장 시각, 결과)}

        self._initialize_vector_store()
        logger.info(f"벡터 데이터베이스 초기화 완료 (타입: {settings.vector_db_type})")

    def _initialize_vector_store(self):
        """벡터 스토어 초기화"""
        if settings.vector_db_type == "chromadb":
            self.vector_store = Chroma(
                persist_directory=settings.vector_db_path, embedding_function=self.embedding_model.embeddings
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
                self.vector_store = FAISS.from_documents(documents, self.embedding_model.embeddings)
            else:
                self.vector_store.add_documents(documents)

        # 키워드 검색용 문서 캐시 업데이트 (저장 전에 반영해야 재시작 후에도 캐시가 인덱스와 일치)
        self.documents_cache.extend(documents)
        if settings.vector_db_type == "faiss":
            self.save_faiss_index()
        # 문서 구성이 바뀌었으므로 질의 캐시 무효화
        self._query_cache.clear()
        self._update_keyword_search_index()

        logger.info(f"{len(documents)}개의 문서가 벡터 DB에 추가되었습니다.")
        logger.info(f"총 문서 수: {len(self.documents_cache)}개")

    def _update_keyword_search_index(self):
        """키워드 검색 인덱스 업데이트"""
        try:
            if not self.documents_cache:
                return

            # 문자 n-gram 인덱스: 형태소 분석 없이 조사 붙은 어절·"Ⅰ-25호" 같은 식별자를 매칭
            if settings.enable_hybrid_search:
                self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), sublinear_tf=True)
                self.char_matrix = self.char_vectorizer.fit_transform(
                    [doc.page_content for doc in self.documents_cache]
                )
                logger.info("문자 n-gram 키워드 검색 인덱스 업데이트 완료")

            # 단어 TF-IDF 인덱스 (extract_related_terms가 어휘를 사용)
            texts = [self._preprocess_text_for_keyword_search(doc.page_content) for doc in self.documents_cache]

            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words=None,  # 한국어 불용어는 별도 처리
                min_df=1,
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.info("TF-IDF 키워드 검색 인덱스 업데이트 완료")

        except Exception as e:
            logger.error(f"키워드 검색 인덱스 업데이트 실패: {str(e)}")

    def _preprocess_text_for_keyword_search(self, text: str) -> str:
        """키워드 검색을 위한 텍스트 전처리"""
        # TextProcessor를 사용하여 텍스트 정제 및 불용어 제거
        return TextProcessor.clean_text(text, remove_stopwords=True)

    @observe_if_enabled(name="vector-retrieval", as_type="retriever", capture_input=False, capture_output=False)
    def search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        향상된 검색 메서드
        - 하이브리드 검색 지원
        - MMR 검색 지원
        - 동적 임계값 조정
        """
        start_time = time.perf_counter()
        if self.vector_store is None:
            return []

        if k is None:
            k = settings.k_documents
        record_input({"query": query, "top_k": k})

        cache_key = f"{query}|{k}"
        cached = self._query_cache.get(cache_key)
        if cached is not None and time.time() - cached[0] < settings.query_cache_ttl_seconds:
            logger.debug(f"질의 캐시 히트: '{query[:50]}...'")
            results = list(cached[1])
            record_metadata({"cache_hit": True, "duration_ms": round((time.perf_counter() - start_time) * 1000, 2)})
            record_retriever_output(_retriever_sources(results))
            return results

        try:
            # 하이브리드 검색 사용 여부 확인
            if settings.enable_hybrid_search:
                results = self._hybrid_search(query, k)
            else:
                results = self._vector_search(query, k)
        except Exception as e:
            logger.error(f"검색 중 오류 발생 ({type(e).__name__}): {e}", exc_info=True)
            return []

        self._store_query_cache(cache_key, results)
        record_metadata(
            {
                "cache_hit": False,
                "search_method": "hybrid" if settings.enable_hybrid_search else "vector",
                "duration_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }
        )
        record_retriever_output(_retriever_sources(results))
        return results

    def _store_query_cache(self, cache_key: str, results: List[Tuple[Document, float]]) -> None:
        """검색 결과를 질의 캐시에 저장 (최대 항목 수 초과 시 가장 오래된 항목 제거)"""
        if not results:
            return
        self._query_cache[cache_key] = (time.time(), list(results))
        while len(self._query_cache) > max(1, settings.query_cache_max_entries):
            del self._query_cache[next(iter(self._query_cache))]

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
        # 점수 포함 MMR은 FAISS만 지원 — 그 외 스토어는 유사도 검색 사용
        if settings.vector_db_type != "faiss":
            return self._similarity_search(query, k)
        try:
            # 질의는 한 번만 임베딩하고, 점수는 FAISS 거리 그대로 사용 (임계값과 동일 척도)
            query_embedding = self.embedding_model.embeddings.embed_query(query)
            results = self.vector_store.max_marginal_relevance_search_with_score_by_vector(
                query_embedding,
                k=k,
                fetch_k=k * 2,  # 더 많은 후보에서 선택
                lambda_mult=1 - settings.mmr_diversity_score,  # 다양성 조절
            )
            return self._filter_by_threshold(results)

        except Exception as e:
            logger.warning(f"MMR 검색 실패, 기본 검색 사용: {str(e)}")
            return self._similarity_search(query, k)

    def _hybrid_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """하이브리드 검색: 벡터 순위와 문자 n-gram 키워드 순위를 RRF로 융합

        반환 점수는 FAISS 거리 그대로 두어 임계값·재랭킹·관련도 표시와 같은 척도를 유지한다.
        키워드로만 찾은 청크를 살리는 것이 목적이므로 임계값 필터는 적용하지 않는다.
        """
        if settings.vector_db_type != "faiss" or self.char_matrix is None:
            return self._vector_search(query, k)
        query_embedding = self.embedding_model.embeddings.embed_query(query)
        # ponytail: 평면 인덱스 전수 검색으로 모든 청크 거리를 얻음 — 수십만 청크면 후보만 index.reconstruct로 계산
        all_hits = self.vector_store.similarity_search_with_score_by_vector(
            query_embedding, k=self.vector_store.index.ntotal
        )
        distances = {doc.page_content: float(score) for doc, score in all_hits}
        candidates = k * 2
        fused = reciprocal_rank_fusion(
            [[doc for doc, _ in all_hits[:candidates]], self._keyword_search(query, candidates)]
        )
        worst = max(distances.values(), default=0.0)
        return [(doc, distances.get(doc.page_content, worst)) for doc in fused[:k]]

    def _keyword_search(self, query: str, k: int) -> List[Document]:
        """문자 n-gram TF-IDF 코사인 유사도 상위 k개 청크 (유사도 0 제외)"""
        similarities = cosine_similarity(self.char_vectorizer.transform([query]), self.char_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:k]
        return [self.documents_cache[i] for i in top_indices if similarities[i] > 0]

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
            filtered_results = results[: max(3, len(results) // 2)]

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
                logger.info("ChromaDB 컬렉션 삭제 완료")
            except Exception as e:
                logger.error(f"ChromaDB 컬렉션 삭제 실패: {str(e)}")
                # 물리적 파일도 삭제 시도
                import shutil

                if os.path.exists(settings.vector_db_path):
                    try:
                        shutil.rmtree(settings.vector_db_path)
                        logger.info("ChromaDB 디렉토리 삭제 완료")
                    except Exception as dir_e:
                        logger.error(f"ChromaDB 디렉토리 삭제 실패: {str(dir_e)}")
        elif settings.vector_db_type == "faiss":
            # 모든 FAISS 관련 파일 삭제
            faiss_files = [
                "index.faiss",
                "index.pkl",
                "documents_cache.pkl",
                "faiss_index.pkl",  # 기존 호환성 유지
            ]

            for filename in faiss_files:
                file_path = os.path.join(settings.vector_db_path, filename)
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"FAISS 파일 삭제: {filename}")
                    except Exception as e:
                        logger.error(f"FAISS 파일 삭제 실패 ({filename}): {str(e)}")

        # 캐시 및 인덱스 초기화
        self.documents_cache = []
        self._query_cache.clear()
        self.bm25_retriever = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.vector_store = None

        # 벡터 스토어 재초기화
        self._initialize_vector_store()

        logger.info("벡터 데이터베이스가 완전히 초기화되었습니다.")

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
                        settings.vector_db_path, self.embedding_model.embeddings, allow_dangerous_deserialization=True
                    )

                    # 문서 캐시 로드
                    cache_path = os.path.join(settings.vector_db_path, "documents_cache.pkl")
                    if os.path.exists(cache_path):
                        with open(cache_path, "rb") as f:
                            self.documents_cache = pickle.load(f)
                    else:
                        self.documents_cache = []

                    # 캐시가 비어 있으면 FAISS docstore에서 복구 — 캐시가 비면 키워드 검색이 무력화됨
                    if not self.documents_cache and self.vector_store is not None:
                        self.documents_cache = list(self.vector_store.docstore._dict.values())
                        logger.info(f"documents_cache가 비어 있어 docstore에서 {len(self.documents_cache)}개 복구")

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
                            self.vector_store = data["vector_store"]
                            self.documents_cache = data.get("documents_cache", [])
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
                "chromadb": settings.search_threshold_chromadb,
            },
            "embedding_model": self.embedding_model.get_model_info(),
            "keyword_search_available": self.bm25_retriever is not None,
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
