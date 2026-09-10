#!/usr/bin/env python3
"""DocumentProcessor(src/rag) 최소 단위테스트 - 중복제거/재랭킹/정제/출처 생성 검증."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.schema import Document

from src.rag.document_processor import DocumentProcessor


def make_doc(content: str, score: float = 0.5, **metadata) -> tuple:
    base = {"file_name": "테스트문서.pdf", "source": "테스트문서.pdf", "chunk_id": "chunk_0"}
    base.update(metadata)
    return (Document(page_content=content, metadata=base), score)


def test_remove_exact_duplicate_documents():
    processor = DocumentProcessor()
    docs = [make_doc("같은 내용입니다", 0.9), make_doc("같은 내용입니다", 0.7), make_doc("다른 내용입니다", 0.5)]
    unique = processor.remove_duplicate_documents(docs)
    assert len(unique) == 2


def test_similarity_jaccard_edges():
    processor = DocumentProcessor()
    assert processor.calculate_content_similarity("", "") == 1.0
    assert processor.calculate_content_similarity("사과 배", "") == 0.0
    assert processor.calculate_content_similarity("사과 배", "사과 배") == 1.0
    assert processor.calculate_content_similarity("사과", "배") == 0.0


def test_normalize_score_faiss_distance_to_similarity(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "vector_db_type", "faiss")
    processor = DocumentProcessor()
    assert processor.normalize_score(0.0) == 1.0  # 거리 0 = 완전 유사
    assert processor.normalize_score(2.0) == 0.0

    monkeypatch.setattr(settings, "vector_db_type", "chroma")
    assert processor.normalize_score(0.95) == 0.95
    assert processor.normalize_score(1.5) == 1.0


def test_sanitize_output_chunk_removes_tokens_and_role_lines():
    processor = DocumentProcessor()
    dirty = "<|im_start|>assistant\n답변 예시: 가짜 답변\n진짜 답변 내용입니다\nuser 안녕"
    cleaned = processor.sanitize_output_chunk(dirty)
    assert "<|im_start|>" not in cleaned
    assert "가짜 답변" not in cleaned
    assert "진짜 답변 내용입니다" in cleaned
    assert not any(line.strip().lower().startswith("user") for line in cleaned.splitlines())


def test_optimize_content_filters_meta_instruction_lines():
    processor = DocumentProcessor()
    content = "핵심 문서 내용입니다.\n답변 시 지켜야 할 규칙: 정중하게\n<|im_end|> 오염 토큰"
    optimized = processor.optimize_content(content)
    assert "지켜야 할 규칙" not in optimized
    assert "핵심 문서 내용입니다" in optimized


def test_generate_enhanced_sources_exposes_fields():
    processor = DocumentProcessor()
    docs = [make_doc("내용", 0.8, page=3)]
    sources = processor.generate_enhanced_sources(docs)
    assert len(sources) == 1
    source = sources[0]
    assert source["index"] == 1
    assert source["file_name"] == "테스트문서.pdf"
    assert source["page"] == 3
    assert 0 <= source["relevance_score"] <= 100


def test_context_labels_and_source_numbers_match():
    processor = DocumentProcessor()
    docs = [
        make_doc("후순위 문서 " + "나" * 50, 0.3, file_name="후순위.pdf"),
        make_doc("최우선 문서 " + "가" * 50, 0.9, file_name="최우선.pdf"),
    ]
    prepared = processor.prepare_documents(docs)
    context = processor.format_documents(prepared)
    sources = processor.generate_enhanced_sources(prepared)

    first_source = sources[0]["file_name"]
    assert f"[문서 1: {first_source}" in context
    for i, source in enumerate(sources, start=1):
        assert f"[문서 {i}: {source['file_name']}" in context
