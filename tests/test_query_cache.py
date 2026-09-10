#!/usr/bin/env python3
"""유사 질의 캐시 동작 검증 — 히트/미스/TTL 만료/무효화/최대 항목 수."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.schema import Document

from src.vectorstore.vector_db import EnhancedVectorDatabase


def make_db() -> EnhancedVectorDatabase:
    db = object.__new__(EnhancedVectorDatabase)
    db._query_cache = {}
    db.bm25_retriever = None
    db.vector_store = object()  # None이면 search()가 조기 반환하므로 더미 객체 사용
    db.documents_cache = []
    return db


def test_cache_hit_avoids_second_search(monkeypatch):
    monkeypatch.setattr("src.vectorstore.vector_db.settings.query_cache_ttl_seconds", 300, raising=False)
    db = make_db()
    calls = []

    db._vector_search = lambda query, k: (calls.append(query), [(Document(page_content="문서"), 0.5)])[1]

    first = db.search("몽촌토성", k=3)
    second = db.search("몽촌토성", k=3)

    assert len(calls) == 1
    assert first == second


def test_different_query_misses_cache(monkeypatch):
    monkeypatch.setattr("src.vectorstore.vector_db.settings.query_cache_ttl_seconds", 300, raising=False)
    db = make_db()
    calls = []

    db._vector_search = lambda query, k: (calls.append(query), [(Document(page_content="문서"), 0.5)])[1]

    db.search("몽촌토성", k=3)
    db.search("발굴조사", k=3)

    assert len(calls) == 2


def test_expired_entry_is_refreshed(monkeypatch):
    monkeypatch.setattr("src.vectorstore.vector_db.settings.query_cache_ttl_seconds", 0, raising=False)
    db = make_db()
    calls = []

    db._vector_search = lambda query, k: (calls.append(query), [(Document(page_content="문서"), 0.5)])[1]

    db.search("몽촌토성", k=3)
    db.search("몽촌토성", k=3)

    assert len(calls) == 2


def test_add_documents_invalidates_cache(monkeypatch):
    monkeypatch.setattr("src.vectorstore.vector_db.settings.vector_db_type", "faiss", raising=False)
    monkeypatch.setattr("src.vectorstore.vector_db.settings.query_cache_ttl_seconds", 300, raising=False)
    db = make_db()
    db.vector_store = type("FakeStore", (), {"add_documents": staticmethod(lambda docs: None)})()
    db.save_faiss_index = lambda: None

    db._query_cache["몽촌토성|3"] = (time.time(), [(Document(page_content="이전 결과"), 0.5)])

    db.add_documents([Document(page_content="새 문서", metadata={})])

    assert db._query_cache == {}


def test_max_entries_evicts_oldest(monkeypatch):
    monkeypatch.setattr("src.vectorstore.vector_db.settings.query_cache_max_entries", 2, raising=False)
    db = make_db()

    db._store_query_cache("a", [(Document(page_content="1"), 0.5)])
    db._store_query_cache("b", [(Document(page_content="2"), 0.5)])
    db._store_query_cache("c", [(Document(page_content="3"), 0.5)])

    assert set(db._query_cache.keys()) == {"b", "c"}


def test_empty_results_are_not_cached(monkeypatch):
    monkeypatch.setattr("src.vectorstore.vector_db.settings.query_cache_max_entries", 2, raising=False)
    db = make_db()

    db._store_query_cache("empty", [])

    assert db._query_cache == {}
