"""
Exact Filtered and Unfiltered Nearest Neighbor Search Baseline.
Acts as the mathematical ground-truth correctness reference.
"""

from typing import List, Optional, Tuple, Union
import numpy as np

from .filters import FilterPredicate, MetadataIndex


class ExactFilteredSearcher:
    """
    Exact Nearest Neighbor Search with metadata predicate filtering.
    Computes exact Euclidean L2 distances (or Inner Product) over valid candidates.
    """
    def __init__(
        self,
        database_vectors: np.ndarray,
        metadata_index: MetadataIndex,
        metric: str = "l2"
    ):
        self.database_vectors = np.ascontiguousarray(database_vectors, dtype=np.float32)
        self.metadata_index = metadata_index
        self.metric = metric.lower()
        self.num_vectors, self.dim = self.database_vectors.shape

    def search(
        self,
        query_vector: np.ndarray,
        predicate: Optional[FilterPredicate] = None,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Executes exact search over candidate vectors satisfying predicate.
        Returns:
            (top_k_indices, top_k_distances, candidate_count)
        """
        candidate_ids = self.metadata_index.get_matching_ids(predicate)
        candidate_count = len(candidate_ids)

        if candidate_count == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), 0

        target_k = min(k, candidate_count)
        cand_vecs = self.database_vectors[candidate_ids]
        q_vec = np.ascontiguousarray(query_vector, dtype=np.float32)

        if self.metric == "l2":
            # Optimized squared Euclidean distance: sum((x - q)^2)
            diff = cand_vecs - q_vec
            dists = np.einsum('ij,ij->i', diff, diff)
        elif self.metric in ("ip", "cosine", "inner_product"):
            # Inner product / Cosine for normalized vectors: -dot(cand, q)
            dists = -np.dot(cand_vecs, q_vec)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        if candidate_count <= target_k:
            sorted_local = np.argsort(dists)
            topk_ids = candidate_ids[sorted_local]
            topk_dists = dists[sorted_local]
        else:
            # Use argpartition for fast top-k selection
            partitioned_local = np.argpartition(dists, target_k)[:target_k]
            sorted_sub = np.argsort(dists[partitioned_local])
            best_local = partitioned_local[sorted_sub]
            topk_ids = candidate_ids[best_local]
            topk_dists = dists[best_local]

        return topk_ids.astype(np.int32), topk_dists.astype(np.float32), candidate_count
