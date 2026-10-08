"""
Paper-inspired Filter-Aware Search Strategies: ACORN, SIEVE, and UNIFY.

IMPORTANT RESEARCH INTEGRITY NOTICE:
These implementations are carefully designed *paper-inspired baseline approximations*
for comparative research, capturing the architectural principles of:
- ACORN (Patel et al., SIGMOD 2024): Multi-hop / relaxed neighbor routing over filtered graphs.
- SIEVE (Gao et al., VLDB 2024): Multi-entry-point routing based on metadata attributes.
- UNIFY (Zhang et al., 2024): Adaptive unified traversal combining entry routing & selectivity awareness.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .filters import FilterPredicate, MetadataIndex
from .hnsw_index import HNSWIndex


class ACORNFilterSearcher:
    """
    Paper-inspired approximation of ACORN (Patel et al., 2024).
    Key mechanism: Expands multi-hop neighborhood during graph traversal when visiting
    non-matching routing nodes, mitigating the disconnection effect in filtered graphs.
    """
    def __init__(self, hnsw_index: HNSWIndex, metadata_index: MetadataIndex, n_hops: int = 2):
        self.hnsw_index = hnsw_index
        self.metadata_index = metadata_index
        self.n_hops = n_hops

    def search(
        self,
        query_vec: np.ndarray,
        predicate: Optional[FilterPredicate] = None,
        k: int = 10,
        ef_search: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        if predicate is None:
            return self.hnsw_index.search_unfiltered(query_vec, k=k, ef_search=ef_search)

        valid_mask = self.metadata_index.get_boolean_mask(predicate)
        return self.hnsw_index.search_acorn_style(
            query_vec=query_vec,
            valid_mask=valid_mask,
            k=k,
            ef_search=ef_search,
            n_hops=self.n_hops,
        )


class SIEVEFilterSearcher:
    """
    Paper-inspired approximation of SIEVE (Gao et al., 2024).
    Key mechanism: Instead of traversing from a single arbitrary global entry point,
    selects an entry point from valid matching candidate nodes closest to query centroids,
    preventing early search termination in invalid graph regions.
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
        if predicate is None:
            return self.hnsw_index.search_unfiltered(query_vec, k=k, ef_search=ef_search)

        matching_ids = self.metadata_index.get_matching_ids(predicate)
        if len(matching_ids) == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), 0

        # SIEVE Entry Point Selection: pick a valid matching candidate as localized entry point
        # For efficiency, pick sample of candidates and select closest to query
        sample_size = min(len(matching_ids), 16)
        sample_ids = matching_ids[np.linspace(0, len(matching_ids) - 1, sample_size, dtype=int)]
        sample_vecs = self.hnsw_index.vectors[sample_ids]
        diffs = sample_vecs - query_vec
        dists = np.einsum('ij,ij->i', diffs, diffs)
        best_ep = sample_ids[np.argmin(dists)]

        # Traverse layer 0 starting directly from best_ep
        valid_mask = self.metadata_index.get_boolean_mask(predicate)
        ef = max(ef_search, k)
        import heapq

        visited = {best_ep}
        d_ep = float(np.min(dists))
        candidates = [(d_ep, best_ep)]
        w = [(-d_ep, best_ep)]

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_w_dist = -w[0][0] if len(w) >= ef else float('inf')

            if c_dist > furthest_w_dist and len(w) >= ef:
                break

            for neighbor in self.hnsw_index.layers[0].get(c_id, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    d_n = self.hnsw_index._dist(self.hnsw_index.vectors[neighbor], query_vec)
                    furthest_w_dist = -w[0][0] if len(w) >= ef else float('inf')

                    heapq.heappush(candidates, (d_n, neighbor))
                    if valid_mask[neighbor]:
                        if d_n < furthest_w_dist or len(w) < ef:
                            heapq.heappush(w, (-d_n, neighbor))
                            if len(w) > ef:
                                heapq.heappop(w)

        sorted_results = sorted([(-d, node_id) for d, node_id in w], key=lambda x: x[0])[:k]
        topk_ids = np.array([x[1] for x in sorted_results], dtype=np.int32)
        topk_dists = np.array([x[0] for x in sorted_results], dtype=np.float32)
        return topk_ids, topk_dists, len(visited)


class UNIFYFilterSearcher:
    """
    Paper-inspired approximation of UNIFY (Zhang et al., 2024).
    Key mechanism: Unified adaptive filter search that dynamically scales ef_search
    and expansion strategy according to estimated filter selectivity.
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
        base_ef_search: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        eff_base = base_ef_search if base_ef_search is not None else ef_search
        if predicate is None:
            return self.hnsw_index.search_unfiltered(query_vec, k=k, ef_search=eff_base)

        selectivity = self.metadata_index.get_selectivity(predicate)
        valid_mask = self.metadata_index.get_boolean_mask(predicate)

        # UNIFY adaptive scaling: increase ef_search when selectivity is low
        if selectivity <= 0.02:
            effective_ef = max(eff_base * 4, 150)
            use_multihop = True
        elif selectivity <= 0.10:
            effective_ef = max(eff_base * 2, 80)
            use_multihop = True
        else:
            effective_ef = eff_base
            use_multihop = False


        if use_multihop:
            return self.hnsw_index.search_acorn_style(
                query_vec=query_vec,
                valid_mask=valid_mask,
                k=k,
                ef_search=effective_ef,
                n_hops=2,
            )
        else:
            return self.hnsw_index.search_ingraph(
                query_vec=query_vec,
                valid_mask=valid_mask,
                k=k,
                ef_search=effective_ef,
            )
