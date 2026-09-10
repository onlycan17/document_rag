#!/usr/bin/env python3
"""골든 셋 기반 검색 품질 평가 스크립트.

사용법:
    python scripts/eval/retrieval_eval.py                  # 기본 골든 셋, k=5
    python scripts/eval/retrieval_eval.py --k 3            # Hit@3 기준
    python scripts/eval/retrieval_eval.py --golden <path>  # 다른 골든 셋

실제 임베딩 API를 호출하므로 질의 수만큼 API 비용이 발생한다.
벡터 DB를 재구축한 뒤 다시 실행하면 임베딩 모델 변경 효과를 숫자로 비교할 수 있다.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.retrieval_metrics import best_relevant_rank, hit_at_k, mrr  # noqa: E402
from src.vectorstore import VectorDatabase  # noqa: E402

DEFAULT_GOLDEN = PROJECT_ROOT / "tests" / "eval" / "golden_retrieval.json"


def evaluate(vector_db: VectorDatabase, cases: list[dict], k: int) -> dict:
    """골든 셋 전체를 검색해 사례별 순위와 요약 지표를 반환한다."""
    detail = []
    ranks: list = []
    for case in cases:
        results = vector_db.search(case["question"], k=k)
        sources = [str(getattr(doc, "metadata", {}).get("source", "")) for doc, _ in results]
        rank = best_relevant_rank(sources, case["expected_sources"])
        ranks.append(rank)
        detail.append({"question": case["question"], "rank": rank, "sources": sources})
    return {"ranks": ranks, "detail": detail, "hit_at_k": hit_at_k(ranks, k), "mrr": mrr(ranks)}


def print_report(result: dict, k: int) -> None:
    for i, item in enumerate(result["detail"], 1):
        status = f"✅ 순위 {item['rank']}" if item["rank"] else "❌ 미검색"
        print(f"[{i:02d}] {status} — {item['question']}")
        if item["rank"] is None:
            print(f"     상위 결과: {item['sources'][:3]}")
    print("\n" + "=" * 60)
    print(f"Hit@{k}: {result['hit_at_k']:.1%}")
    print(f"MRR   : {result['mrr']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="골든 셋 기반 검색 품질 평가")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="골든 셋 JSON 경로")
    parser.add_argument("--k", type=int, default=5, help="Hit@k 판정 기준 순위 (기본 5)")
    args = parser.parse_args()

    with open(args.golden, encoding="utf-8") as f:
        golden = json.load(f)
    cases = golden["cases"]

    vector_db = VectorDatabase()
    if vector_db.get_document_count() == 0:
        print("벡터 DB가 초기화되지 않았습니다. 먼저 vector_db_manager로 구축해주세요.")
        sys.exit(1)

    print(f"문서 {vector_db.get_document_count()}개 청크에서 {len(cases)}개 질의 평가 (Hit@{args.k})\n")
    result = evaluate(vector_db, cases, args.k)
    print_report(result, args.k)


if __name__ == "__main__":
    main()
