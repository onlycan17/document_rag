"""MMR 점수 척도, 문서 캐시 저장, 하이브리드(RRF) 검색 회귀 테스트"""

from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from src.vectorstore.vector_db import VectorDatabase, reciprocal_rank_fusion


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
    db.tfidf_vectorizer = db.tfidf_matrix = db.char_vectorizer = db.char_matrix = None

    db.add_documents([Document(page_content="첫 번째 배치 문서")])
    db.add_documents([Document(page_content="두 번째 배치 문서")])

    with open(tmp_path / "documents_cache.pkl", "rb") as f:
        assert len(pickle.load(f)) == 2


def test_reciprocal_rank_fusion_prefers_docs_ranked_high_in_both_lists():
    a, b, c = (Document(page_content=text) for text in "abc")
    fused = reciprocal_rank_fusion([[a, b, c], [b, c, a]])
    assert [doc.page_content for doc in fused] == ["b", "a", "c"]


def test_reciprocal_rank_fusion_includes_docs_from_single_list():
    a, b = Document(page_content="a"), Document(page_content="b")
    assert [doc.page_content for doc in reciprocal_rank_fusion([[a], [b]])] == ["a", "b"]


def test_hybrid_search_rescues_exact_identifier_and_keeps_faiss_distance(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "enable_hybrid_search", True)
    monkeypatch.setattr(settings, "vector_db_type", "faiss")
    embeddings = DeterministicFakeEmbedding(size=16)  # 의미 없는 무작위 벡터 → 벡터 순위는 우연
    docs = [Document(page_content=f"일반 서술 문단 {i}") for i in range(20)]
    target = Document(page_content="Ⅰ-25호 수혈유구 규모는 길이 316㎝이다")
    db = object.__new__(VectorDatabase)
    db.embedding_model = _StubEmbeddingModel(embeddings)
    db.vector_store = FAISS.from_documents(docs + [target], embeddings)
    db.documents_cache = docs + [target]
    db.tfidf_vectorizer = db.tfidf_matrix = db.char_vectorizer = db.char_matrix = None
    db._update_keyword_search_index()

    results = db._hybrid_search("Ⅰ-25호 수혈유구 규모", k=3)

    assert target.page_content in [doc.page_content for doc, _ in results]
    query_distances = dict(
        (doc.page_content, score)
        for doc, score in db.vector_store.similarity_search_with_score("Ⅰ-25호 수혈유구 규모", k=21)
    )
    for doc, score in results:
        assert abs(score - query_distances[doc.page_content]) < 1e-4
