"""MMR 점수 척도와 멀티모달 전처리 메타데이터 회귀 테스트"""

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


class _EchoMultimodalModel:
    """전달받은 이미지 목록을 본문으로 돌려주는 가짜 멀티모달 전처리 모델"""

    def preprocess_document_with_images(self, content: str, images: list) -> str:
        return f"{content} images={images}"

    def preprocess_text(self, content: str) -> str:
        return content


def test_multimodal_preprocessing_keeps_each_page_metadata_and_images():
    from src.loaders.document_loader import EnhancedDocumentLoader

    loader = object.__new__(EnhancedDocumentLoader)
    loader._preprocessing_model = _EchoMultimodalModel()
    loader.preprocessing_model = "fake"
    loader.enable_multimodal_preprocessing = True
    body = "몽촌토성 발굴 조사 내용입니다. " * 10
    pages = [Document(page_content=body, metadata={"page": i, "images": [f"img{i}.png"]}) for i in range(2)]

    processed = loader._preprocess_documents(pages, ".pdf")

    assert [doc.metadata["page"] for doc in processed] == [0, 1]
    assert "img0.png" in processed[0].page_content and "img1.png" not in processed[0].page_content
