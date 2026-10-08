"""
HNSW Post-Filtering Strategy.
Workflow:
  Query -> Unfiltered HNSW Search (k_cand = K) -> Discard invalid candidates -> Return remaining Top-K.
"""

from typing import List, Optional, Tuple
import numpy as np

from .filters import FilterPredicate, MetadataIndex
from .hnsw_index import HNSWIndex


class HNSWPostFilterSearcher:
    """
    Naive HNSW Post-Filtering.
    Searches standard HNSW index for top-k candidates, then applies metadata filter.
    Demonstrates classical recall degradation under low selectivity.
    """
    def __init__(self, hnsw_index: HNSWIndex, metadata_index: MetadataIndex):
        self.hnsw_index = hnsw_index
        self.metadata_index = metadata_index

    def search(
        self,
        query_vec: np.ndarray,
        predicate: Optional[FilterPredicate] = None,
        k: int = 10,
        ef_search: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Executes post-filtered search.
        Returns (topk_ids, topk_dists, candidate_count_evaluated)
        """
        # Step 1: Unfiltered HNSW search for k candidates
        cand_ids, cand_dists, nodes_visited = self.hnsw_index.search_unfiltered(
            query_vec=query_vec,
            k=k,
            ef_search=ef_search,
        )

        if predicate is None or len(cand_ids) == 0:
            return cand_ids[:k], cand_dists[:k], len(cand_ids)

        # Step 2: Filter candidates against predicate
        valid_mask = self.metadata_index.get_boolean_mask(predicate)
        matched_indices = [i for i, cid in enumerate(cand_ids) if valid_mask[cid]]

        filtered_ids = cand_ids[matched_indices]
        filtered_dists = cand_dists[matched_indices]

        return filtered_ids, filtered_dists, len(cand_ids)
