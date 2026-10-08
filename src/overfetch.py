"""
HNSW Overfetching Strategy.
Workflow:
  Query -> Unfiltered HNSW Search (k_cand = K * overfetch_factor) -> Discard invalid candidates -> Return Top-K.
"""

from typing import List, Optional, Tuple
import numpy as np

from .filters import FilterPredicate, MetadataIndex
from .hnsw_index import HNSWIndex


class HNSWOverfetchSearcher:
    """
    HNSW with Overfetching.
    Retrieves an expanded candidate set (e.g. 2x, 5x, 10x, 20x, 50x, or dynamic based on selectivity),
    then applies metadata filtering.
    """
    def __init__(self, hnsw_index: HNSWIndex, metadata_index: MetadataIndex):
        self.hnsw_index = hnsw_index
        self.metadata_index = metadata_index

    def search(
        self,
        query_vec: np.ndarray,
        predicate: Optional[FilterPredicate] = None,
        k: int = 10,
        overfetch_factor: float = 5.0,
        ef_search: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Executes overfetched HNSW search.
        k_fetch = int(k * overfetch_factor)
        """
        k_fetch = max(k, int(k * overfetch_factor))
        effective_ef = max(ef_search or 50, k_fetch)

        # Unfiltered HNSW search for k_fetch candidates
        cand_ids, cand_dists, nodes_visited = self.hnsw_index.search_unfiltered(
            query_vec=query_vec,
            k=k_fetch,
            ef_search=effective_ef,
        )

        if predicate is None or len(cand_ids) == 0:
            return cand_ids[:k], cand_dists[:k], len(cand_ids)

        # Filter candidates
        valid_mask = self.metadata_index.get_boolean_mask(predicate)
        matched_indices = [i for i, cid in enumerate(cand_ids) if valid_mask[cid]]

        filtered_ids = cand_ids[matched_indices][:k]
        filtered_dists = cand_dists[matched_indices][:k]

        return filtered_ids, filtered_dists, len(cand_ids)
