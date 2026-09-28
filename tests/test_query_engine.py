"""QueryEngine 회귀 테스트 — 쿼리 확장은 요청당 한 번만 실행되어야 한다."""

from unittest.mock import MagicMock

from langchain.schema import Document

from src.rag.query_engine import QueryEngine


def _make_engine() -> QueryEngine:
    vector_db = MagicMock()
    vector_db.get_document_count.return_value = 1
    vector_db.search.return_value = [(Document(page_content="몽촌토성 본문", metadata={}), 0.5)]
    document_processor = MagicMock()
    document_processor.prepare_documents.side_effect = lambda docs: docs
    document_processor.format_documents.return_value = "컨텍스트"
    document_processor.sanitize_output_chunk.side_effect = lambda text: text
    document_processor.generate_enhanced_sources.return_value = []
    engine = QueryEngine(vector_db, MagicMock(), document_processor)
    engine.chain = MagicMock()
    engine.chain.invoke.return_value = "답변"
    engine.streaming_chain = MagicMock()
    engine.streaming_chain.stream.return_value = iter(["답", "변"])
    engine.preprocess_query = MagicMock(return_value="몽촌토성 확장")
    return engine


def test_query_expands_once():
    engine = _make_engine()
    result = engine._query_impl("몽촌토성")
    assert engine.preprocess_query.call_count == 1
    assert result["search_info"]["processed_query"] == "몽촌토성 확장"


def test_stream_query_expands_once():
    engine = _make_engine()
    chunks = list(engine._stream_query_impl("몽촌토성"))
    assert engine.preprocess_query.call_count == 1
    assert chunks[-1]["search_info"]["processed_query"] == "몽촌토성 확장"
