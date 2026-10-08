"""
Pre-Filtering Search Strategy.
Workflow:
  Query + Filter -> Metadata Index -> Valid Candidate IDs -> Restricted Vector Search -> Top-K.
"""

from typing import List, Optional, Tuple
import numpy as np

from .filters import FilterPredicate, MetadataIndex


class PreFilterSearcher:
    """
    Pre-Filtering Searcher.
    Uses the inverted metadata index to immediately prune all non-matching vectors,
    then executes restricted exact distance computation over the candidate set.
    Maintains 100% recall over valid candidates, with latency scaling directly with candidate set size.
    """
    def __init__(
        self,
        database_vectors: np.ndarray,
        metadata_index: MetadataIndex,
        metric: str = "l2",
    ):
        self.database_vectors = np.ascontiguousarray(database_vectors, dtype=np.float32)
        self.metadata_index = metadata_index
        self.metric = metric.lower()

    def search(
        self,
        query_vec: np.ndarray,
        predicate: Optional[FilterPredicate] = None,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Executes pre-filtered search.
        Returns (topk_ids, topk_dists, candidate_count)
        """
        candidate_ids = self.metadata_index.get_matching_ids(predicate)
        cand_count = len(candidate_ids)

        if cand_count == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), 0

        target_k = min(k, cand_count)
        cand_vecs = self.database_vectors[candidate_ids]
        q_vec = np.ascontiguousarray(query_vec, dtype=np.float32)

        if self.metric == "l2":
            diff = cand_vecs - q_vec
            dists = np.einsum('ij,ij->i', diff, diff)
        elif self.metric in ("ip", "cosine"):
            dists = -np.dot(cand_vecs, q_vec)
        else:
            diff = cand_vecs - q_vec
            dists = np.einsum('ij,ij->i', diff, diff)

        if cand_count <= target_k:
            order = np.argsort(dists)
            return candidate_ids[order], dists[order], cand_count

        part = np.argpartition(dists, target_k)[:target_k]
        sorted_sub = np.argsort(dists[part])
        best_indices = part[sorted_sub]

        return candidate_ids[best_indices], dists[best_indices], cand_count
