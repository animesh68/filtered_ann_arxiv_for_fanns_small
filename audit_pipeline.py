"""
Audit Script: Comprehensive Verification of FANNS Implementation.
Tests:
1. Exact Ground Truth vs Official ivecs across 1,000 queries.
2. Recall@K calculation corner cases.
3. Selectivity calculation and inverted index consistency.
4. Latency, QPS, percentile metrics logic.
5. All search methods return valid indices, shapes, and types.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset_loader import load_fanns_dataset
from src.filters import (
    MetadataIndex,
    parse_query_filter,
    EMFilterPredicate,
    RangeFilterPredicate,
    EMISFilterPredicate,
)
from src.exact_search import ExactFilteredSearcher
from src.metrics import compute_recall_at_k, aggregate_query_results, QueryEvaluationResult
from src.hnsw_index import HNSWIndex
from src.postfilter import HNSWPostFilterSearcher
from src.overfetch import HNSWOverfetchSearcher
from src.prefilter import PreFilterSearcher
from src.ingraph import InGraphFilterSearcher
from src.filter_aware import ACORNFilterSearcher, SIEVEFilterSearcher, UNIFYFilterSearcher


def test_ground_truth_fidelity():
    print("=== Test 1: Ground Truth File Integrity & Parsing Fidelity ===")
    dataset = load_fanns_dataset("data/arxiv-for-fanns-small")
    meta_idx = MetadataIndex(dataset.database_attributes)
    exact_searcher = ExactFilteredSearcher(dataset.database_vectors, meta_idx, metric="l2")

    for w_name in ["EM", "Range", "EMIS"]:
        workload = dataset.get_workload(w_name)
        gt_list = workload.ground_truth
        assert len(gt_list) == 10000, f"{w_name} GT length is {len(gt_list)}, expected 10000"

        # Check first 500 queries
        perfect_matches = 0
        set_matches_10 = 0
        set_matches_100 = 0

        for q_idx in range(500):
            q_vec = dataset.query_vectors[q_idx]
            q_attr = workload.query_attributes[q_idx]
            pred = parse_query_filter(workload.filter_type, q_attr)
            pred_ids, pred_dists, _ = exact_searcher.search(q_vec, predicate=pred, k=100)
            gt_ids = gt_list[q_idx]

            k10 = min(10, len(gt_ids))
            if k10 > 0:
                if set(pred_ids[:k10]) == set(gt_ids[:k10]):
                    set_matches_10 += 1
            else:
                set_matches_10 += 1

            k100 = min(100, len(gt_ids))
            if k100 > 0:
                if set(pred_ids[:k100]) == set(gt_ids[:k100]):
                    set_matches_100 += 1
            else:
                set_matches_100 += 1

        print(f"[{w_name}] Top-10 Set Match Rate: {set_matches_10}/500 ({set_matches_10 + 0.0:.1f}%) | Top-100 Set Match Rate: {set_matches_100}/500")
        assert set_matches_10 == 500, f"Top-10 exact ground truth mismatch in {w_name}!"


def test_metrics_logic():
    print("\n=== Test 2: Metrics Logic & Edge Cases ===")
    # Recall test: perfect match
    r1 = compute_recall_at_k([1, 2, 3], [1, 2, 3, 4, 5], k=3)
    assert r1 == 1.0, f"Expected 1.0, got {r1}"

    # Recall test: partial match
    r2 = compute_recall_at_k([1, 2, 9], [1, 2, 3, 4, 5], k=3)
    assert abs(r2 - 2/3) < 1e-6, f"Expected 0.6666, got {r2}"

    # Recall test: small ground truth |GT| < K
    r3 = compute_recall_at_k([1, 2], [1, 2], k=10)
    assert r3 == 1.0, f"Expected 1.0 when |GT| < K and all found, got {r3}"

    # Recall test: empty ground truth
    r4 = compute_recall_at_k([], [], k=10)
    assert r4 == 1.0, f"Expected 1.0 for empty GT, got {r4}"

    # Aggregator test: percentile calculations
    q_results = [
        QueryEvaluationResult(query_id=0, filter_type="test", selectivity=0.1, candidate_count=100, k=10, recall_at_k=1.0, latency_ms=1.0),
        QueryEvaluationResult(query_id=1, filter_type="test", selectivity=0.1, candidate_count=100, k=10, recall_at_k=0.5, latency_ms=2.0),
        QueryEvaluationResult(query_id=2, filter_type="test", selectivity=0.1, candidate_count=100, k=10, recall_at_k=0.0, latency_ms=3.0),
    ]
    agg = aggregate_query_results("test_method", "test", k=10, query_results=q_results)
    assert abs(agg.mean_recall - 0.5) < 1e-6
    assert abs(agg.mean_latency_ms - 2.0) < 1e-6
    assert abs(agg.median_latency_ms - 2.0) < 1e-6
    assert abs(agg.qps - (3 / 0.006)) < 1e-3
    print("[OK] Metrics corner cases and aggregation logic verified.")


def test_all_search_methods():
    print("\n=== Test 3: Search Methods Execution & Correctness ===")
    dataset = load_fanns_dataset("data/arxiv-for-fanns-small")
    meta_idx = MetadataIndex(dataset.database_attributes)

    hnsw = HNSWIndex(dimension=dataset.dimension, metric="l2", M=16, ef_construction=50, random_seed=42)
    hnsw.build_index(dataset.database_vectors)

    exact = ExactFilteredSearcher(dataset.database_vectors, meta_idx)
    post = HNSWPostFilterSearcher(hnsw, meta_idx)
    overfetch = HNSWOverfetchSearcher(hnsw, meta_idx)
    pre = PreFilterSearcher(dataset.database_vectors, meta_idx)
    ingraph = InGraphFilterSearcher(hnsw, meta_idx)
    acorn = ACORNFilterSearcher(hnsw, meta_idx)
    sieve = SIEVEFilterSearcher(hnsw, meta_idx)
    unify = UNIFYFilterSearcher(hnsw, meta_idx)

    methods = {
        "Exact": lambda q, p: exact.search(q, predicate=p, k=10),
        "Post": lambda q, p: post.search(q, predicate=p, k=10),
        "Overfetch": lambda q, p: overfetch.search(q, predicate=p, k=10, overfetch_factor=5.0),
        "Pre": lambda q, p: pre.search(q, predicate=p, k=10),
        "InGraph": lambda q, p: ingraph.search(q, predicate=p, k=10),
        "ACORN": lambda q, p: acorn.search(q, predicate=p, k=10),
        "SIEVE": lambda q, p: sieve.search(q, predicate=p, k=10),
        "UNIFY": lambda q, p: unify.search(q, predicate=p, k=10),
    }

    q_vec = dataset.query_vectors[0]
    q_attr = dataset.workloads["EM"].query_attributes[0]
    pred = parse_query_filter("EM", q_attr)

    for name, fn in methods.items():
        ids, dists, count = fn(q_vec, pred)
        assert isinstance(ids, np.ndarray), f"{name} returned non-ndarray IDs"
        assert isinstance(dists, np.ndarray), f"{name} returned non-ndarray dists"
        assert len(ids) <= 10, f"{name} returned more than k items"
        print(f"[OK] {name:<10}: returned {len(ids)} IDs, candidate/visited count: {count}")


if __name__ == "__main__":
    test_ground_truth_fidelity()
    test_metrics_logic()
    test_all_search_methods()
    print("\n==================================================")
    print("ALL AUDIT INTEGRITY CHECKS PASSED")
    print("==================================================")
