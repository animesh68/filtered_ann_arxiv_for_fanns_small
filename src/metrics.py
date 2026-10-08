"""
Comprehensive evaluation metrics and logging framework for FANNS benchmarks.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd


@dataclass
class QueryEvaluationResult:
    """Individual query execution record."""
    query_id: int
    filter_type: str
    selectivity: float
    candidate_count: int
    k: int
    recall_at_k: float
    latency_ms: float
    extra: Optional[Dict[str, Any]] = None


@dataclass
class AggregateBenchmarkResult:
    """Aggregated benchmark statistics for a method over a workload."""
    method: str
    filter_type: str
    k: int
    num_queries: int
    index_build_time_s: float
    index_memory_mb: float
    mean_recall: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    qps: float
    mean_selectivity: float
    mean_candidate_count: float
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Flatten parameters for easy CSV tabular display
        for k, v in self.parameters.items():
            d[f"param_{k}"] = v
        return d


def compute_recall_at_k(predicted_ids: Sequence[int], ground_truth_ids: Sequence[int], k: int) -> float:
    """
    Computes Recall@K: |predicted[:k] ∩ ground_truth[:k]| / min(k, |ground_truth|).
    If ground_truth is empty, returns 1.0 if predicted is also empty, else 1.0 (trivial match).
    """
    if len(ground_truth_ids) == 0:
        return 1.0

    target_k = min(k, len(ground_truth_ids))
    if target_k == 0:
        return 1.0

    pred_set = set(predicted_ids[:k])
    gt_set = set(ground_truth_ids[:target_k])

    hits = len(pred_set.intersection(gt_set))
    return float(hits) / float(target_k)


def aggregate_query_results(
    method_name: str,
    filter_type: str,
    k: int,
    query_results: List[QueryEvaluationResult],
    index_build_time_s: float = 0.0,
    index_memory_mb: float = 0.0,
    parameters: Optional[Dict[str, Any]] = None,
) -> AggregateBenchmarkResult:
    """Aggregates a list of per-query evaluation records into statistical summary."""
    if not query_results:
        raise ValueError("Cannot aggregate empty query results list.")

    latencies = np.array([r.latency_ms for r in query_results], dtype=np.float64)
    recalls = np.array([r.recall_at_k for r in query_results], dtype=np.float64)
    selectivities = np.array([r.selectivity for r in query_results], dtype=np.float64)
    candidates = np.array([r.candidate_count for r in query_results], dtype=np.float64)

    total_time_s = np.sum(latencies) / 1000.0
    qps = len(latencies) / total_time_s if total_time_s > 0 else 0.0

    return AggregateBenchmarkResult(
        method=method_name,
        filter_type=filter_type,
        k=k,
        num_queries=len(query_results),
        index_build_time_s=index_build_time_s,
        index_memory_mb=index_memory_mb,
        mean_recall=float(np.mean(recalls)),
        mean_latency_ms=float(np.mean(latencies)),
        median_latency_ms=float(np.median(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        p99_latency_ms=float(np.percentile(latencies, 99)),
        qps=float(qps),
        mean_selectivity=float(np.mean(selectivities)),
        mean_candidate_count=float(np.mean(candidates)),
        parameters=parameters or {},
    )


def results_to_dataframe(results: List[AggregateBenchmarkResult]) -> pd.DataFrame:
    """Converts a list of AggregateBenchmarkResult into a clean DataFrame."""
    rows = [r.to_dict() for r in results]
    return pd.DataFrame(rows)
