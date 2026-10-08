"""
Experiment 08: Comprehensive Controlled Comparative Benchmark.
Runs all evaluated FANNS strategies on identical queries, identical seeds,
stratified across natural selectivity bands and filter types.
Saves granular per-query results and comprehensive summary CSVs.
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import load_fanns_dataset
from src.filters import MetadataIndex, parse_query_filter
from src.hnsw_index import HNSWIndex
from src.exact_search import ExactFilteredSearcher
from src.postfilter import HNSWPostFilterSearcher
from src.overfetch import HNSWOverfetchSearcher
from src.prefilter import PreFilterSearcher
from src.ingraph import InGraphFilterSearcher
from src.filter_aware import ACORNFilterSearcher, SIEVEFilterSearcher, UNIFYFilterSearcher
from src.metrics import (
    QueryEvaluationResult,
    aggregate_query_results,
    compute_recall_at_k,
    results_to_dataframe,
)
from src.timing import PrecisionTimer


def run_comprehensive_controlled_benchmark(
    num_eval_queries: int = 500,
    k: int = 10,
    M: int = 16,
    ef_construction: int = 100,
    ef_search: int = 50,
):
    print("=" * 80)
    print(f"EXPERIMENT 08: COMPREHENSIVE CONTROLLED BENCHMARK (K={k}, Queries/Workload={num_eval_queries})")
    print("=" * 80)

    dataset = load_fanns_dataset("data/arxiv-for-fanns-small")
    meta_idx = MetadataIndex(dataset.database_attributes)

    # Build HNSW Index
    print("\n[1/3] Building Shared HNSW Index (M=16, efC=100)...")
    t0 = time.perf_counter()
    hnsw = HNSWIndex(dimension=dataset.dimension, metric="l2", M=M, ef_construction=ef_construction, random_seed=42)
    hnsw.build_index(dataset.database_vectors)
    build_time = time.perf_counter() - t0
    index_mem = hnsw.estimate_memory_mb()
    print(f"[OK] Index built in {build_time:.2f}s | RAM: {index_mem:.2f} MB")

    # Instantiate all search strategies
    exact_searcher = ExactFilteredSearcher(dataset.database_vectors, meta_idx, metric="l2")
    post_searcher = HNSWPostFilterSearcher(hnsw, meta_idx)
    overfetch_2x = HNSWOverfetchSearcher(hnsw, meta_idx)
    overfetch_5x = HNSWOverfetchSearcher(hnsw, meta_idx)
    overfetch_10x = HNSWOverfetchSearcher(hnsw, meta_idx)
    overfetch_20x = HNSWOverfetchSearcher(hnsw, meta_idx)
    pre_searcher = PreFilterSearcher(dataset.database_vectors, meta_idx, metric="l2")
    ingraph_searcher = InGraphFilterSearcher(hnsw, meta_idx)
    acorn_searcher = ACORNFilterSearcher(hnsw, meta_idx, n_hops=2)
    sieve_searcher = SIEVEFilterSearcher(hnsw, meta_idx)
    unify_searcher = UNIFYFilterSearcher(hnsw, meta_idx)

    # Strategy dispatcher map
    strategies = {
        "ExactSearch": lambda q, p: exact_searcher.search(q, predicate=p, k=k),
        "PreFilter": lambda q, p: pre_searcher.search(q, predicate=p, k=k),
        "HNSW_PostFilter": lambda q, p: post_searcher.search(q, predicate=p, k=k, ef_search=ef_search),
        "HNSW_Overfetch_2x": lambda q, p: overfetch_2x.search(q, predicate=p, k=k, overfetch_factor=2.0),
        "HNSW_Overfetch_5x": lambda q, p: overfetch_5x.search(q, predicate=p, k=k, overfetch_factor=5.0),
        "HNSW_Overfetch_10x": lambda q, p: overfetch_10x.search(q, predicate=p, k=k, overfetch_factor=10.0),
        "HNSW_Overfetch_20x": lambda q, p: overfetch_20x.search(q, predicate=p, k=k, overfetch_factor=20.0),
        "InGraphFilter": lambda q, p: ingraph_searcher.search(q, predicate=p, k=k, ef_search=ef_search),
        "ACORN_Approx": lambda q, p: acorn_searcher.search(q, predicate=p, k=k, ef_search=ef_search),
        "SIEVE_Approx": lambda q, p: sieve_searcher.search(q, predicate=p, k=k, ef_search=ef_search),
        "UNIFY_Approx": lambda q, p: unify_searcher.search(q, predicate=p, k=k, base_ef_search=ef_search),
    }

    out_dir = Path("results/controlled_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_query_records = []
    aggregate_summaries = []

    print("\n[2/3] Executing Controlled Evaluation across Workloads & Methods...")
    for w_name in ["EM", "Range", "EMIS"]:
        workload = dataset.get_workload(w_name)
        print(f"\n==================== Workload: {w_name} ====================")

        for strat_name, search_fn in strategies.items():
            strat_q_results = []

            for q_idx in range(min(num_eval_queries, dataset.num_queries)):
                q_vec = dataset.query_vectors[q_idx]
                q_attr = workload.query_attributes[q_idx]
                predicate = parse_query_filter(workload.filter_type, q_attr)
                selectivity = meta_idx.get_selectivity(predicate)
                gt_ids = workload.ground_truth[q_idx]

                timer = PrecisionTimer()
                with timer:
                    pred_ids, pred_dists, cand_count = search_fn(q_vec, predicate)

                recall = compute_recall_at_k(pred_ids, gt_ids, k=k)

                q_res = QueryEvaluationResult(
                    query_id=q_idx,
                    filter_type=w_name,
                    selectivity=selectivity,
                    candidate_count=cand_count,
                    k=k,
                    recall_at_k=recall,
                    latency_ms=timer.elapsed_ms,
                )
                strat_q_results.append(q_res)

                # Record per-query flat dict for granular CSV
                row_dict = {
                    "method": strat_name,
                    "filter_type": w_name,
                    "query_id": q_idx,
                    "k": k,
                    "selectivity": selectivity,
                    "candidate_count": cand_count,
                    "recall_at_k": recall,
                    "latency_ms": timer.elapsed_ms,
                }
                all_query_records.append(row_dict)

            agg = aggregate_query_results(
                method_name=strat_name,
                filter_type=w_name,
                k=k,
                query_results=strat_q_results,
                index_build_time_s=build_time if "HNSW" in strat_name or "Graph" in strat_name or "ACORN" in strat_name or "SIEVE" in strat_name or "UNIFY" in strat_name else 0.0,
                index_memory_mb=index_mem if "HNSW" in strat_name or "Graph" in strat_name or "ACORN" in strat_name or "SIEVE" in strat_name or "UNIFY" in strat_name else 0.0,
                parameters={"k": k, "M": M, "ef_construction": ef_construction, "ef_search": ef_search},
            )
            aggregate_summaries.append(agg)
            print(f"[{strat_name:<20} | {w_name}] Recall@{k}: {agg.mean_recall:.4f} | Latency: {agg.mean_latency_ms:>6.3f} ms | QPS: {agg.qps:>6.1f} | Mean Cand: {agg.mean_candidate_count:>5.1f}")

    # Save per-query raw data
    print("\n[3/3] Saving benchmark datasets...")
    df_per_query = pd.DataFrame(all_query_records)
    df_per_query.to_csv(out_dir / "all_queries_raw_results.csv", index=False)
    print(f"[OK] Raw per-query log saved ({len(df_per_query)} rows): {out_dir / 'all_queries_raw_results.csv'}")

    df_summary = results_to_dataframe(aggregate_summaries)
    df_summary.to_csv(out_dir / "benchmark_summary.csv", index=False)
    print(f"[OK] Summary table saved: {out_dir / 'benchmark_summary.csv'}")

    # Stratified selectivity analysis
    df_per_query["selectivity_bucket"] = pd.cut(
        df_per_query["selectivity"],
        bins=[-0.001, 0.02, 0.10, 0.30, 1.0],
        labels=["Ultra-Low (<2%)", "Low (2-10%)", "Medium (10-30%)", "High (>30%)"],
    )
    strat_grouped = df_per_query.groupby(["method", "filter_type", "selectivity_bucket"], observed=False).agg(
        mean_recall=("recall_at_k", "mean"),
        mean_latency_ms=("latency_ms", "mean"),
        median_latency_ms=("latency_ms", "median"),
        p95_latency_ms=("latency_ms", lambda x: np.percentile(x, 95) if len(x) > 0 else 0.0),
        query_count=("query_id", "count"),
    ).reset_index()

    strat_grouped.to_csv(out_dir / "stratified_selectivity_summary.csv", index=False)
    print(f"[OK] Stratified selectivity summary saved: {out_dir / 'stratified_selectivity_summary.csv'}")

    return df_summary, strat_grouped


if __name__ == "__main__":
    run_comprehensive_controlled_benchmark(num_eval_queries=500, k=10)
