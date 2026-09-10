"""retrieval_metrics 순수 함수 테스트 (오프라인)."""

from src.utils.retrieval_metrics import best_relevant_rank, hit_at_k, mrr


def test_best_relevant_rank_finds_first_match():
    sources = ["a.pdf", "/data/몽촌토성4+상.pdf", "b.pdf"]
    assert best_relevant_rank(sources, ["몽촌토성4+상.pdf"]) == 2


def test_best_relevant_rank_any_of_expected():
    sources = ["x.pdf", "y.pdf", "KERIS_report.pdf"]
    assert best_relevant_rank(sources, ["KERIS", "몽촌토성"]) == 3


def test_best_relevant_rank_returns_none_when_missing():
    assert best_relevant_rank(["a.pdf", "b.pdf"], ["없는문서.pdf"]) is None


def test_hit_at_k_counts_ranks_within_k():
    ranks = [1, 3, 7, None]
    assert hit_at_k(ranks, k=5) == 0.5


def test_mrr_averages_reciprocal_ranks():
    ranks = [1, 2, None]
    assert abs(mrr(ranks) - (1.0 + 0.5 + 0.0) / 3) < 1e-9


def test_metrics_handle_empty_input():
    assert hit_at_k([], k=5) == 0.0
    assert mrr([]) == 0.0
