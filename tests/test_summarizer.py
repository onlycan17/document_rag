#!/usr/bin/env python3
"""HierarchicalSummarizer 최소 단위테스트 - LLM 없는 순수 요약 로직 검증."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.schema import Document

from src.rag.summarizer import DocumentSummary, HierarchicalSummarizer, resolve_summary_length


CONTENT = (
    "백제의 몽촌토성은 서울 송파구에 위치한 토성 유적이다. "
    "2024년 3월 발굴조사에서 다양한 토기편이 출토되었다. "
    "조사 결과 고배와 파수편 등 주요 유물이 확인되었다. "
    "이 유적은 한성백제 시대의 도성으로 추정된다. "
    "향후 추가 조사 계획이 발표되었다. "
    "발굴 유물은 국립서울박물관에 보관된다."
)


def make_doc(content: str = CONTENT, score: float = 0.8) -> tuple:
    return (Document(page_content=content, metadata={"file_name": "테스트문서.pdf"}), score)


def test_extract_key_sentences_short_content_returns_all():
    summarizer = HierarchicalSummarizer()
    short = "첫 문장입니다. 둘째 문장입니다."
    assert len(summarizer._extract_key_sentences(short)) == 2


def test_extract_key_sentences_limits_count():
    summarizer = HierarchicalSummarizer()
    keys = summarizer._extract_key_sentences(CONTENT, query="몽촌토성", num_sentences=3)
    assert len(keys) == 3
    # TF-IDF 폴백이든 정상 경로든 문장은 원본에서 선택되어야 한다
    for sentence in keys:
        assert sentence in CONTENT


def test_importance_score_empty_content_is_zero():
    summarizer = HierarchicalSummarizer()
    assert summarizer._calculate_importance_score("") == 0.0
    assert summarizer._calculate_importance_score("   ") == 0.0


def test_importance_score_rewards_information_density():
    summarizer = HierarchicalSummarizer()
    plain = summarizer._calculate_importance_score("특별한 정보가 없는 평범한 문장이다.")
    informative = summarizer._calculate_importance_score("2024년 3월 15일 30% 증가했고 5개 기관이 참여했다.")
    assert informative > plain


def test_extractive_summary_respects_target_length():
    summarizer = HierarchicalSummarizer()
    keys = ["가" * 80, "나" * 80, "다" * 80]
    summary = summarizer._create_extractive_summary(keys, target_length=100)
    assert len(summary) <= 110  # 말줄임 여유


def test_document_summary_get_compressed_content():
    summary = DocumentSummary(
        original_content="원본",
        summary="요약 본문이 목표 길이보다 깁니다",
        key_sentences=["첫 문장", "둘째 문장"],
        importance_score=0.5,
        compression_ratio=0.1,
    )
    assert summary.get_compressed_content() == "요약 본문이 목표 길이보다 깁니다"
    assert summary.get_compressed_content(target_length=100) == "요약 본문이 목표 길이보다 깁니다"
    # 요약이 목표 길이보다 길면 핵심 문장부터 우선 담는다
    assert summary.get_compressed_content(target_length=8) == "첫 문장"


def test_create_document_summary_without_llm_uses_extractive():
    summarizer = HierarchicalSummarizer(llm=None)
    result = summarizer.create_document_summary(make_doc()[0], query="몽촌토성")
    assert result.summary
    assert result.key_sentences
    assert 0.0 <= result.importance_score <= 1.0
    assert 0.0 <= result.compression_ratio <= 1.0


def test_compress_context_short_input_returns_original():
    summarizer = HierarchicalSummarizer()
    docs = [make_doc()]
    result = summarizer.compress_context(docs, target_length=10_000)
    assert result == CONTENT


def test_resolve_summary_length_defaults_for_large_output_budget():
    assert resolve_summary_length(4096) == 500
    assert resolve_summary_length(16384) == 500


def test_resolve_summary_length_shrinks_for_small_output_budget():
    assert resolve_summary_length(300) == 300
    assert resolve_summary_length(100) == 150  # 최소 길이 보장


def test_resolve_summary_length_handles_invalid_budget():
    assert resolve_summary_length(0) == 500
    assert resolve_summary_length(-5) == 500
