"""MMR 점수 척도, 문서 캐시 저장 회귀 테스트"""

from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from src.vectorstore.vector_db import VectorDatabase


class _CountingEmbedding(DeterministicFakeEmbedding):
    query_calls: int = 0

    def embed_query(self, text: str) -> List[float]:
        self.query_calls += 1
        return super().embed_query(text)


class _StubEmbeddingModel:
    """EmbeddingModel 인터페이스 모사 (embed_query 직접 호출도 집계됨)"""

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.embed_query = embeddings.embed_query


def test_mmr_search_embeds_query_once_and_returns_faiss_distances():
    embeddings = _CountingEmbedding(size=16)
    docs = [Document(page_content=f"문서 {i}") for i in range(6)]
    db = object.__new__(VectorDatabase)
    db.embedding_model = _StubEmbeddingModel(embeddings)
    db.vector_store = FAISS.from_documents(docs, embeddings)

    expected = dict(
        (doc.page_content, score) for doc, score in db.vector_store.similarity_search_with_score("문서 0", k=6)
    )
    embeddings.query_calls = 0
    results = db._mmr_search("문서 0", k=3)

    assert embeddings.query_calls == 1
    assert results
    for doc, score in results:
        assert abs(score - expected[doc.page_content]) < 1e-4


def test_add_documents_persists_cache_including_latest_batch(tmp_path, monkeypatch):
    import pickle

    from config import settings

    monkeypatch.setattr(settings, "vector_db_path", str(tmp_path))
    db = object.__new__(VectorDatabase)
    db.embedding_model = _StubEmbeddingModel(DeterministicFakeEmbedding(size=16))
    db.vector_store, db.documents_cache, db._query_cache = None, [], {}
    db.bm25_retriever = db.tfidf_vectorizer = db.tfidf_matrix = None

    db.add_documents([Document(page_content="첫 번째 배치 문서")])
    db.add_documents([Document(page_content="두 번째 배치 문서")])

    with open(tmp_path / "documents_cache.pkl", "rb") as f:
        assert len(pickle.load(f)) == 2
