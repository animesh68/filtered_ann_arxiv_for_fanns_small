"""
Experiment 02: Exact Filtered Nearest-Neighbor Search Baseline.
Evaluates exact baseline across unfiltered, EM, Range, and EMIS workloads.
Verifies ground-truth fidelity, latency, candidate count, and throughput.
"""

import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import load_fanns_dataset
from src.filters import MetadataIndex, parse_query_filter
from src.exact_search import ExactFilteredSearcher
from src.metrics import (
    QueryEvaluationResult,
    aggregate_query_results,
    compute_recall_at_k,
    results_to_dataframe,
)
from src.timing import PrecisionTimer


def run_exact_search_benchmark(num_eval_queries: int = 500, k: int = 10):
    print("=" * 70)
    print(f"EXPERIMENT 02: EXACT FILTERED SEARCH BASELINE (K={k}, N={num_eval_queries})")
    print("=" * 70)

    dataset = load_fanns_dataset("data/arxiv-for-fanns-small")
    meta_idx = MetadataIndex(dataset.database_attributes)
    searcher = ExactFilteredSearcher(dataset.database_vectors, meta_idx, metric="l2")

    aggregate_results = []
    output_dir = Path("results/exact_search")
    output_dir.mkdir(parents=True, exist_ok=True)

    for w_name, workload in dataset.workloads.items():
        print(f"\nEvaluating workload: {w_name} ...")
        query_results = []

        for q_idx in range(min(num_eval_queries, dataset.num_queries)):
            q_vec = dataset.query_vectors[q_idx]
            q_attr = workload.query_attributes[q_idx] if workload.query_attributes else None
            predicate = parse_query_filter(workload.filter_type, q_attr)

            selectivity = meta_idx.get_selectivity(predicate)
            gt_ids = workload.ground_truth[q_idx]

            timer = PrecisionTimer()
            with timer:
                pred_ids, pred_dists, cand_count = searcher.search(q_vec, predicate=predicate, k=k)

            recall = compute_recall_at_k(pred_ids, gt_ids, k=k)

            query_results.append(
                QueryEvaluationResult(
                    query_id=q_idx,
                    filter_type=w_name,
                    selectivity=selectivity,
                    candidate_count=cand_count,
                    k=k,
                    recall_at_k=recall,
                    latency_ms=timer.elapsed_ms,
                )
            )

        agg = aggregate_query_results(
            method_name="ExactSearch",
            filter_type=w_name,
            k=k,
            query_results=query_results,
            parameters={"metric": "l2", "k": k},
        )
        aggregate_results.append(agg)

        print(f"[{w_name}] Mean Recall@{k}: {agg.mean_recall:.4f} | Mean Latency: {agg.mean_latency_ms:.3f} ms | QPS: {agg.qps:.1f} | Mean Candidates: {agg.mean_candidate_count:.1f}")

    df = results_to_dataframe(aggregate_results)
    df.to_csv(output_dir / f"exact_search_summary_k{k}.csv", index=False)
    print(f"\n[OK] Results saved to {output_dir / f'exact_search_summary_k{k}.csv'}")
    return df


if __name__ == "__main__":
    run_exact_search_benchmark(num_eval_queries=500, k=10)
