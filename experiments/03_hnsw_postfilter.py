"""
Experiment 03: HNSW Baseline and Naive Post-Filtering.
Measures unfiltered HNSW recall baseline and the severe recall degradation under naive post-filtering.
"""

import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import load_fanns_dataset
from src.filters import MetadataIndex, parse_query_filter
from src.hnsw_index import HNSWIndex
from src.postfilter import HNSWPostFilterSearcher
from src.metrics import (
    QueryEvaluationResult,
    aggregate_query_results,
    compute_recall_at_k,
    results_to_dataframe,
)
from src.timing import PrecisionTimer


def run_hnsw_postfilter_benchmark(
    num_eval_queries: int = 500,
    k: int = 10,
    M: int = 16,
    ef_construction: int = 100,
    ef_search: int = 50,
):
    print("=" * 70)
    print(f"EXPERIMENT 03: HNSW BASELINE & POST-FILTERING (K={k}, N={num_eval_queries}, M={M}, efC={ef_construction}, efS={ef_search})")
    print("=" * 70)

    dataset = load_fanns_dataset("data/arxiv-for-fanns-small")
    meta_idx = MetadataIndex(dataset.database_attributes)

    # 1. Build HNSW Index
    print("\nBuilding HNSW Index...")
    t0 = time.perf_counter()
    hnsw = HNSWIndex(
        dimension=dataset.dimension,
        metric="l2",
        M=M,
        ef_construction=ef_construction,
        random_seed=42,
    )
    hnsw.build_index(dataset.database_vectors)
    build_time = time.perf_counter() - t0
    index_mem = hnsw.estimate_memory_mb()
    print(f"[OK] HNSW Index built in {build_time:.3f} s. Estimated memory: {index_mem:.2f} MB")

    post_searcher = HNSWPostFilterSearcher(hnsw, meta_idx)
    output_dir = Path("results/hnsw_postfilter")
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_results = []
    per_query_records = []

    # 2. Evaluate Unfiltered baseline first
    unfiltered_workload = dataset.get_workload("unfiltered")
    print("\nEvaluating Unfiltered HNSW Baseline...")
    unfiltered_q_results = []
    for q_idx in range(min(num_eval_queries, dataset.num_queries)):
        q_vec = dataset.query_vectors[q_idx]
        gt_ids = unfiltered_workload.ground_truth[q_idx]

        timer = PrecisionTimer()
        with timer:
            pred_ids, pred_dists, cand_count = hnsw.search_unfiltered(
                q_vec, k=k, ef_search=ef_search
            )

        recall = compute_recall_at_k(pred_ids, gt_ids, k=k)
        res = QueryEvaluationResult(
            query_id=q_idx,
            filter_type="unfiltered",
            selectivity=1.0,
            candidate_count=cand_count,
            k=k,
            recall_at_k=recall,
            latency_ms=timer.elapsed_ms,
        )
        unfiltered_q_results.append(res)
        per_query_records.append(res)

    agg_unfilt = aggregate_query_results(
        method_name="HNSW_Unfiltered",
        filter_type="unfiltered",
        k=k,
        query_results=unfiltered_q_results,
        index_build_time_s=build_time,
        index_memory_mb=index_mem,
        parameters={"M": M, "ef_construction": ef_construction, "ef_search": ef_search},
    )
    aggregate_results.append(agg_unfilt)
    print(f"[Unfiltered HNSW] Mean Recall@{k}: {agg_unfilt.mean_recall:.4f} | Mean Latency: {agg_unfilt.mean_latency_ms:.3f} ms | QPS: {agg_unfilt.qps:.1f}")

    # 3. Evaluate Naive Post-Filtering on Filtered Workloads
    for w_name in ["EM", "Range", "EMIS"]:
        workload = dataset.get_workload(w_name)
        print(f"\nEvaluating Naive Post-Filtering on workload: {w_name} ...")
        w_q_results = []

        for q_idx in range(min(num_eval_queries, dataset.num_queries)):
            q_vec = dataset.query_vectors[q_idx]
            q_attr = workload.query_attributes[q_idx]
            predicate = parse_query_filter(workload.filter_type, q_attr)
            selectivity = meta_idx.get_selectivity(predicate)
            gt_ids = workload.ground_truth[q_idx]

            timer = PrecisionTimer()
            with timer:
                pred_ids, pred_dists, cand_count = post_searcher.search(
                    q_vec, predicate=predicate, k=k, ef_search=ef_search
                )

            recall = compute_recall_at_k(pred_ids, gt_ids, k=k)
            res = QueryEvaluationResult(
                query_id=q_idx,
                filter_type=w_name,
                selectivity=selectivity,
                candidate_count=cand_count,
                k=k,
                recall_at_k=recall,
                latency_ms=timer.elapsed_ms,
            )
            w_q_results.append(res)
            per_query_records.append(res)

        agg_post = aggregate_query_results(
            method_name="HNSW_PostFilter",
            filter_type=w_name,
            k=k,
            query_results=w_q_results,
            index_build_time_s=build_time,
            index_memory_mb=index_mem,
            parameters={"M": M, "ef_construction": ef_construction, "ef_search": ef_search},
        )
        aggregate_results.append(agg_post)
        print(f"[{w_name} PostFilter] Mean Recall@{k}: {agg_post.mean_recall:.4f} | Mean Latency: {agg_post.mean_latency_ms:.3f} ms | Mean Selectivity: {agg_post.mean_selectivity:.3f}")

    df = results_to_dataframe(aggregate_results)
    df.to_csv(output_dir / "hnsw_postfilter_summary.csv", index=False)
    print(f"\n[OK] Summary saved to {output_dir / 'hnsw_postfilter_summary.csv'}")
    return df


if __name__ == "__main__":
    run_hnsw_postfilter_benchmark(num_eval_queries=500, k=10)
