"""검색 평가 지표 (순수 함수).

골든 셋 기반 검색 품질 측정에 쓰이는 지표 계산만 담당한다.
입출력이 단순 값이므로 네트워크·DB 없이 단위 테스트 가능하다.
"""

from typing import Optional


def best_relevant_rank(sources: list[str], expected_sources: list[str]) -> Optional[int]:
    """검색 결과 순서에서 기대 문서가 처음 등장하는 1-based 순위.

    Args:
        sources: 검색된 문서들의 식별 문자열(메타데이터 source 등) 목록 (순서 = 관련도 순)
        expected_sources: 정답으로 인정할 문서 식별 문자열 목록 (부분 일치)

    Returns:
        1-based 순위. 기대 문서가 결과에 없으면 None
    """
    for rank, source in enumerate(sources, start=1):
        if any(expected in source for expected in expected_sources):
            return rank
    return None


def hit_at_k(ranks: list[Optional[int]], k: int) -> float:
    """상위 k 안에 정답 문서가 포함된 질문 비율 (Hit@k).

    Args:
        ranks: 각 질문의 정답 순위(best_relevant_rank 결과), 미검색은 None
        k: 판정 기준 순위
    """
    if not ranks:
        return 0.0
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits / len(ranks)


def mrr(ranks: list[Optional[int]]) -> float:
    """정답 순위 역수의 평균 (Mean Reciprocal Rank). 미검색 질문은 0으로 계산."""
    if not ranks:
        return 0.0
    return sum(1.0 / rank if rank is not None else 0.0 for rank in ranks) / len(ranks)
