#!/usr/bin/env python3
"""골든 셋 기반 검색 품질 평가 스크립트.

사용법:
    python scripts/eval/retrieval_eval.py                  # 기본 골든 셋, k=5
    python scripts/eval/retrieval_eval.py --k 3            # Hit@3 기준
    python scripts/eval/retrieval_eval.py --golden <path>  # 다른 골든 셋
    python scripts/eval/retrieval_eval.py --upload         # 결과를 Langfuse에 기록

실제 임베딩 API를 호출하므로 질의 수만큼 API 비용이 발생한다.
벡터 DB를 재구축한 뒤 다시 실행하면 임베딩 모델 변경 효과를 숫자로 비교할 수 있다.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.retrieval_metrics import best_relevant_rank, hit_at_k, mrr  # noqa: E402
from src.utils.tracing import (  # noqa: E402
    langfuse_enabled,
    observe_if_enabled,
    propagate_trace_attributes,
    record_input,
    record_output,
)
from src.vectorstore import VectorDatabase  # noqa: E402

DEFAULT_GOLDEN = PROJECT_ROOT / "tests" / "eval" / "golden_retrieval.json"
DATASET_NAME = "retrieval-golden"


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


@observe_if_enabled(name="retrieval-eval", capture_input=False)
def _record_eval_trace(input_data: dict, output_data: dict, session_id: str) -> Optional[str]:
    """평가 1건(사례 또는 요약)을 트레이스로 기록하고 trace id를 반환한다."""
    record_input(input_data)
    record_output(output_data)
    with propagate_trace_attributes(session_id=session_id):
        from langfuse import get_client

        return get_client().get_current_trace_id()


def upload_to_langfuse(golden: dict, result: dict, k: int) -> None:
    """평가 결과를 Langfuse에 기록한다: 골든 셋 Dataset, 사례별·요약 점수.

    Dataset 아이템은 질의 기준 중복 생성을 피한다. 점수는 실행마다 새 트레이스에
    기록되어 시계열로 누적된다.
    """
    if not langfuse_enabled():
        print("\nLangfuse 비활성화 — 업로드 생략 (LANGFUSE_ENABLED=true + 키 설정 필요)")
        return
    from langfuse import get_client

    client = get_client()
    try:
        client.get_dataset(DATASET_NAME)
    except Exception:
        client.create_dataset(name=DATASET_NAME, description=golden.get("description"))

    existing = {
        item.input.get("question") for item in client.get_dataset(DATASET_NAME).items if isinstance(item.input, dict)
    }
    for case in golden["cases"]:
        if case["question"] not in existing:
            client.create_dataset_item(
                dataset_name=DATASET_NAME,
                input={"question": case["question"]},
                expected_output=case["expected_sources"],
            )

    session_id = f"retrieval-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    for detail in result["detail"]:
        trace_id = _record_eval_trace(
            {"type": "case", "question": detail["question"], "k": k},
            {"best_rank": detail["rank"], "sources": detail["sources"][:k]},
            session_id,
        )
        if trace_id:
            client.create_score(
                name=f"hit@{k}", value=1.0 if detail["rank"] else 0.0, trace_id=trace_id, data_type="NUMERIC"
            )
            # best_rank: 1위=1.0, 미검색=0.0
            client.create_score(
                name="best_rank", value=float(detail["rank"] or 0), trace_id=trace_id, data_type="NUMERIC"
            )

    summary_id = _record_eval_trace(
        {"type": "summary", "k": k, "cases": len(result["detail"])},
        {"hit_at_k": result["hit_at_k"], "mrr": result["mrr"]},
        session_id,
    )
    if summary_id:
        client.create_score(name=f"hit@{k}", value=result["hit_at_k"], trace_id=summary_id, data_type="NUMERIC")
        client.create_score(name="mrr", value=result["mrr"], trace_id=summary_id, data_type="NUMERIC")

    if hasattr(client, "flush"):
        client.flush()
    print(f"Langfuse 업로드 완료: dataset={DATASET_NAME}, session={session_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="골든 셋 기반 검색 품질 평가")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="골든 셋 JSON 경로")
    parser.add_argument("--k", type=int, default=5, help="Hit@k 판정 기준 순위 (기본 5)")
    parser.add_argument("--upload", action="store_true", help="결과를 Langfuse Dataset·점수로 기록")
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
    if args.upload:
        upload_to_langfuse(golden, result, args.k)


if __name__ == "__main__":
    main()
