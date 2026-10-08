"""
Experiment 04: HNSW Overfetching Strategy Sweep.
Evaluates recall vs latency trade-offs across overfetch factors [1.0, 2.0, 5.0, 10.0, 20.0, 50.0].
"""

import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import load_fanns_dataset
from src.filters import MetadataIndex, parse_query_filter
from src.hnsw_index import HNSWIndex
from src.overfetch import HNSWOverfetchSearcher
from src.metrics import (
    QueryEvaluationResult,
    aggregate_query_results,
    compute_recall_at_k,
    results_to_dataframe,
)
from src.timing import PrecisionTimer


def run_hnsw_overfetch_benchmark(
    num_eval_queries: int = 500,
    k: int = 10,
    overfetch_factors=(1.0, 2.0, 5.0, 10.0, 20.0, 50.0),
    M: int = 16,
    ef_construction: int = 100,
):
    print("=" * 70)
    print(f"EXPERIMENT 04: HNSW OVERFETCHING SWEEP (K={k}, N={num_eval_queries})")
    print("=" * 70)

    dataset = load_fanns_dataset("data/arxiv-for-fanns-small")
    meta_idx = MetadataIndex(dataset.database_attributes)

    t0 = time.perf_counter()
    hnsw = HNSWIndex(dimension=dataset.dimension, metric="l2", M=M, ef_construction=ef_construction, random_seed=42)
    hnsw.build_index(dataset.database_vectors)
    build_time = time.perf_counter() - t0
    index_mem = hnsw.estimate_memory_mb()

    overfetch_searcher = HNSWOverfetchSearcher(hnsw, meta_idx)
    output_dir = Path("results/hnsw_overfetch")
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_results = []

    for factor in overfetch_factors:
        print(f"\n--- Overfetch Factor: {factor}x (k_fetch = {int(k * factor)}) ---")
        for w_name in ["EM", "Range", "EMIS"]:
            workload = dataset.get_workload(w_name)
            q_results = []

            for q_idx in range(min(num_eval_queries, dataset.num_queries)):
                q_vec = dataset.query_vectors[q_idx]
                q_attr = workload.query_attributes[q_idx]
                predicate = parse_query_filter(workload.filter_type, q_attr)
                selectivity = meta_idx.get_selectivity(predicate)
                gt_ids = workload.ground_truth[q_idx]

                timer = PrecisionTimer()
                with timer:
                    pred_ids, pred_dists, cand_count = overfetch_searcher.search(
                        q_vec, predicate=predicate, k=k, overfetch_factor=factor
                    )

                recall = compute_recall_at_k(pred_ids, gt_ids, k=k)
                q_results.append(
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
                method_name=f"HNSW_Overfetch_{factor}x",
                filter_type=w_name,
                k=k,
                query_results=q_results,
                index_build_time_s=build_time,
                index_memory_mb=index_mem,
                parameters={"overfetch_factor": factor, "k": k, "M": M, "ef_construction": ef_construction},
            )
            aggregate_results.append(agg)
            print(f"[{w_name} | {factor}x] Recall@{k}: {agg.mean_recall:.4f} | Latency: {agg.mean_latency_ms:.3f} ms | QPS: {agg.qps:.1f}")

    df = results_to_dataframe(aggregate_results)
    df.to_csv(output_dir / "hnsw_overfetch_summary.csv", index=False)
    print(f"\n[OK] Summary saved to {output_dir / 'hnsw_overfetch_summary.csv'}")
    return df


if __name__ == "__main__":
    run_hnsw_overfetch_benchmark(num_eval_queries=500, k=10)
