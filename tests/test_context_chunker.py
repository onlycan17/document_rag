#!/usr/bin/env python3
"""ContextChunker 최소 단위테스트 - 외부 API 없이 순수 분할 로직만 검증."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from langchain.schema import Document

from src.rag.context_chunker import ContextChunker


def make_doc(content: str, score: float = 0.5, file_name: str = "테스트문서.pdf") -> tuple:
    doc = Document(page_content=content, metadata={"file_name": file_name, "source": file_name})
    return (doc, score)


def test_empty_documents_returns_empty_list():
    chunker = ContextChunker(max_chunk_size=1000)
    assert chunker.split_documents([]) == []


def test_small_context_returns_single_chunk():
    chunker = ContextChunker(max_chunk_size=1000)
    docs = [make_doc("짧은 문서", 0.9), make_doc("또 다른 문서", 0.5)]
    chunks = chunker.split_documents(docs, query="테스트")
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "single_chunk"
    assert chunks[0].total_length == sum(len(d.page_content) for d, _ in docs)


def test_unsupported_strategy_raises_value_error():
    chunker = ContextChunker(max_chunk_size=10)
    docs = [make_doc("가" * 100), make_doc("나" * 100)]
    with pytest.raises(ValueError):
        chunker.split_documents(docs, strategy="unknown")


def test_split_by_size_respects_max_chunk_size():
    chunker = ContextChunker(max_chunk_size=100)
    docs = [make_doc("가" * 60, 0.9), make_doc("나" * 60, 0.8), make_doc("다" * 60, 0.7)]
    chunks = chunker.split_documents(docs, strategy="size")
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.total_length <= 100
        assert chunk.get_metadata()["document_count"] >= 1


def test_oversized_document_gets_own_chunk():
    chunker = ContextChunker(max_chunk_size=100)
    big = make_doc("대" * 300, 0.9)
    small = make_doc("소", 0.8)
    chunks = chunker.split_documents([big, small], strategy="size")
    large_chunks = [c for c in chunks if c.chunk_id.startswith("large_doc_chunk")]
    assert len(large_chunks) == 1
    assert large_chunks[0].total_length == 300


def test_relevance_strategy_prefers_high_score_first():
    chunker = ContextChunker(max_chunk_size=100)
    docs = [make_doc("가" * 60, 0.3), make_doc("나" * 60, 0.9)]
    chunks = chunker.split_documents(docs, strategy="relevance")
    assert len(chunks) >= 2
    # 관련도 높은 문서(나)가 첫 청크에 먼저 배치되어야 한다
    assert chunks[0].documents[0][0].page_content == "나" * 60


def test_get_chunk_summary_counts_all_documents():
    chunker = ContextChunker(max_chunk_size=100)
    docs = [make_doc("가" * 60, 0.9), make_doc("나" * 60, 0.8), make_doc("다" * 60, 0.7)]
    chunks = chunker.split_documents(docs, strategy="size")
    summary = chunker.get_chunk_summary(chunks)
    assert summary["total_documents"] == 3
    assert summary["total_length"] == 180
    assert summary["min_chunk_size"] <= summary["average_chunk_size"] <= summary["max_chunk_size"]
