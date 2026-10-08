"""
In-Graph Filtered Search Strategy.
Workflow:
  Query + Filter -> Single HNSW Graph Traversal -> Valid Mask check at each visited node -> Filter-aware top-k.
"""

from typing import List, Optional, Tuple
import numpy as np

from .filters import FilterPredicate, MetadataIndex
from .hnsw_index import HNSWIndex


class InGraphFilterSearcher:
    """
    In-Graph Filtering Searcher.
    Traverses the HNSW graph topology while utilizing a pre-computed or on-the-fly filter mask.
    Non-matching nodes serve as routing/stepping stones to maintain graph connectivity,
    while only matching nodes are admitted into the final result candidate pool.
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
        Executes in-graph filtered search.
        Returns (topk_ids, topk_dists, nodes_visited)
        """
        if predicate is None:
            return self.hnsw_index.search_unfiltered(query_vec, k=k, ef_search=ef_search)

        valid_mask = self.metadata_index.get_boolean_mask(predicate)
        return self.hnsw_index.search_ingraph(
            query_vec=query_vec,
            valid_mask=valid_mask,
            k=k,
            ef_search=ef_search,
        )
