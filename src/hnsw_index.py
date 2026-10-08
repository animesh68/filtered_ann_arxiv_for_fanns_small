"""
High-Performance Hierarchical Navigable Small World (HNSW) Index Implementation.
Designed for FANNS research: provides full visibility and hooks for post-filtering,
in-graph filtering, ACORN hybrid routing, SIEVE entry-points, and UNIFY graph traversal.
"""

import heapq
import math
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np


class HNSWIndex:
    """
    Hierarchical Navigable Small World (HNSW) Graph Index.
    Implements Malkov & Yashunin (2018) algorithm with research hooks for metadata filtering.
    """
    def __init__(
        self,
        dimension: int,
        metric: str = "l2",
        M: int = 16,
        ef_construction: int = 100,
        random_seed: int = 42,
    ):
        self.dimension = dimension
        self.metric = metric.lower()
        self.M = M
        self.M0 = 2 * M  # Max connections for layer 0
        self.ef_construction = ef_construction
        self.random_seed = random_seed
        self.mL = 1.0 / math.log(M) if M > 1 else 1.0

        self.rng = np.random.RandomState(random_seed)
        self.vectors: Optional[np.ndarray] = None
        self.num_nodes: int = 0
        self.enter_point: Optional[int] = None
        self.max_layer: int = -1

        # Multi-layer adjacency lists: layers[lc][node_id] = list of neighbor IDs
        self.layers: List[Dict[int, List[int]]] = []
        self.node_layers: List[int] = []

    def _dist(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute distance between two vectors."""
        if self.metric == "l2":
            diff = vec_a - vec_b
            return float(np.dot(diff, diff))
        elif self.metric in ("ip", "cosine"):
            return float(-np.dot(vec_a, vec_b))
        else:
            diff = vec_a - vec_b
            return float(np.dot(diff, diff))

    def _dist_batch(self, query_vec: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
        """Compute distances between query and batch of node IDs."""
        sub_vecs = self.vectors[node_ids]
        if self.metric == "l2":
            diff = sub_vecs - query_vec
            return np.einsum('ij,ij->i', diff, diff)
        elif self.metric in ("ip", "cosine"):
            return -np.dot(sub_vecs, query_vec)
        else:
            diff = sub_vecs - query_vec
            return np.einsum('ij,ij->i', diff, diff)

    def _get_random_layer(self) -> int:
        """Select node max level using exponential distribution."""
        return int(-math.log(self.rng.uniform(1e-9, 1.0)) * self.mL)

    def build_index(self, vectors: np.ndarray):
        """Build HNSW graph over all dataset vectors."""
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.num_nodes = self.vectors.shape[0]
        self.layers = []
        self.node_layers = []

        for i in range(self.num_nodes):
            self._insert_point(i)

    def _insert_point(self, point_id: int):
        """Insert a single point into the hierarchical graph."""
        vec = self.vectors[point_id]
        level = self._get_random_layer()
        self.node_layers.append(level)

        # Extend layers if new level exceeds current max_layer
        while len(self.layers) <= level:
            self.layers.append({})

        if self.enter_point is None:
            self.enter_point = point_id
            self.max_layer = level
            for lc in range(level + 1):
                self.layers[lc][point_id] = []
            return

        curr_ep = self.enter_point
        curr_dist = self._dist(self.vectors[curr_ep], vec)

        # 1. Greedy search from top layer down to level + 1
        for lc in range(self.max_layer, level, -1):
            changed = True
            while changed:
                changed = False
                neighbors = self.layers[lc].get(curr_ep, [])
                for neighbor in neighbors:
                    d = self._dist(self.vectors[neighbor], vec)
                    if d < curr_dist:
                        curr_dist = d
                        curr_ep = neighbor
                        changed = True

        # 2. Search and connect from min(max_layer, level) down to layer 0
        top_l = min(self.max_layer, level)
        ep_set = [curr_ep]

        for lc in range(top_l, -1, -1):
            candidates = self._search_layer_base(
                vec, ep_set, ef=self.ef_construction, lc=lc
            )
            # candidates is list of (dist, id)
            m_max = self.M0 if lc == 0 else self.M
            selected_neighbors = self._select_heuristic(candidates, m_max)

            self.layers[lc][point_id] = selected_neighbors

            # Add back-connections and shrink if needed
            for neighbor in selected_neighbors:
                n_list = self.layers[lc].setdefault(neighbor, [])
                n_list.append(point_id)
                if len(n_list) > m_max:
                    cand_n = [(self._dist(self.vectors[neighbor], self.vectors[n_id]), n_id) for n_id in n_list]
                    self.layers[lc][neighbor] = self._select_heuristic(cand_n, m_max)

            ep_set = [c[1] for c in candidates]

        if level > self.max_layer:
            self.max_layer = level
            self.enter_point = point_id

    def _select_heuristic(
        self, candidates: List[Tuple[float, int]], max_m: int
    ) -> List[int]:
        """Simple greedy neighbor selection (can be extended to diverse heuristic)."""
        sorted_cand = sorted(candidates, key=lambda x: x[0])
        return [cand_id for _, cand_id in sorted_cand[:max_m]]

    def _search_layer_base(
        self,
        query_vec: np.ndarray,
        entry_points: List[int],
        ef: int,
        lc: int,
    ) -> List[Tuple[float, int]]:
        """Beam search at layer lc."""
        visited = set(entry_points)
        # Min-heap of candidates to visit: (dist, id)
        candidates = []
        # Max-heap for W (best found so far): (-dist, id)
        w = []

        for ep in entry_points:
            d = self._dist(self.vectors[ep], query_vec)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(w, (-d, ep))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_w_dist = -w[0][0]

            if c_dist > furthest_w_dist and len(w) >= ef:
                break

            neighbors = self.layers[lc].get(c_id, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    d_n = self._dist(self.vectors[neighbor], query_vec)
                    furthest_w_dist = -w[0][0]

                    if d_n < furthest_w_dist or len(w) < ef:
                        heapq.heappush(candidates, (d_n, neighbor))
                        heapq.heappush(w, (-d_n, neighbor))
                        if len(w) > ef:
                            heapq.heappop(w)

        # Return list of (dist, id)
        return [(-d, node_id) for d, node_id in w]

    def search_unfiltered(
        self,
        query_vec: np.ndarray,
        k: int = 10,
        ef_search: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Standard unfiltered HNSW search. Returns (topk_ids, topk_dists, nodes_visited)."""
        if self.enter_point is None or self.num_nodes == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), 0

        curr_ep = self.enter_point
        curr_dist = self._dist(self.vectors[curr_ep], query_vec)
        visited_count = 1

        # Greedy routing down to layer 1
        for lc in range(self.max_layer, 0, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.layers[lc].get(curr_ep, []):
                    visited_count += 1
                    d = self._dist(self.vectors[neighbor], query_vec)
                    if d < curr_dist:
                        curr_dist = d
                        curr_ep = neighbor
                        changed = True

        # Beam search at layer 0
        ef = max(ef_search, k)
        visited = {curr_ep}
        candidates = [(curr_dist, curr_ep)]
        w = [(-curr_dist, curr_ep)]

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_w_dist = -w[0][0]

            if c_dist > furthest_w_dist and len(w) >= ef:
                break

            for neighbor in self.layers[0].get(c_id, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    visited_count += 1
                    d_n = self._dist(self.vectors[neighbor], query_vec)
                    furthest_w_dist = -w[0][0]

                    if d_n < furthest_w_dist or len(w) < ef:
                        heapq.heappush(candidates, (d_n, neighbor))
                        heapq.heappush(w, (-d_n, neighbor))
                        if len(w) > ef:
                            heapq.heappop(w)

        # Sort results
        sorted_results = sorted([(-d, node_id) for d, node_id in w], key=lambda x: x[0])[:k]
        topk_ids = np.array([x[1] for x in sorted_results], dtype=np.int32)
        topk_dists = np.array([x[0] for x in sorted_results], dtype=np.float32)
        return topk_ids, topk_dists, len(visited)

    def search_ingraph(
        self,
        query_vec: np.ndarray,
        valid_mask: np.ndarray,
        k: int = 10,
        ef_search: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        In-Graph Filtered Search:
        Traverses standard HNSW graph, but candidate result pool W only admits valid candidates.
        Traversal explores un-filtered neighbors to maintain connectivity.
        """
        if self.enter_point is None or self.num_nodes == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), 0

        curr_ep = self.enter_point
        curr_dist = self._dist(self.vectors[curr_ep], query_vec)
        visited_count = 1

        # Greedy routing down
        for lc in range(self.max_layer, 0, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.layers[lc].get(curr_ep, []):
                    visited_count += 1
                    d = self._dist(self.vectors[neighbor], query_vec)
                    if d < curr_dist:
                        curr_dist = d
                        curr_ep = neighbor
                        changed = True

        # Beam search at layer 0 with filter admission
        ef = max(ef_search, k)
        visited = {curr_ep}
        candidates = [(curr_dist, curr_ep)]
        w = []

        if valid_mask[curr_ep]:
            heapq.heappush(w, (-curr_dist, curr_ep))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_w_dist = -w[0][0] if len(w) >= ef else float('inf')

            if c_dist > furthest_w_dist and len(w) >= ef:
                break

            for neighbor in self.layers[0].get(c_id, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    visited_count += 1
                    d_n = self._dist(self.vectors[neighbor], query_vec)
                    furthest_w_dist = -w[0][0] if len(w) >= ef else float('inf')

                    # Candidate queue explores regardless of filter to navigate topology
                    heapq.heappush(candidates, (d_n, neighbor))

                    # Result pool only admits valid candidates
                    if valid_mask[neighbor]:
                        if d_n < furthest_w_dist or len(w) < ef:
                            heapq.heappush(w, (-d_n, neighbor))
                            if len(w) > ef:
                                heapq.heappop(w)

        sorted_results = sorted([(-d, node_id) for d, node_id in w], key=lambda x: x[0])[:k]
        topk_ids = np.array([x[1] for x in sorted_results], dtype=np.int32)
        topk_dists = np.array([x[0] for x in sorted_results], dtype=np.float32)
        return topk_ids, topk_dists, len(visited)

    def search_acorn_style(
        self,
        query_vec: np.ndarray,
        valid_mask: np.ndarray,
        k: int = 10,
        ef_search: int = 50,
        n_hops: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        ACORN-style Filtered Traversal (Paper-inspired approximation):
        Expands multi-hop filtered neighborhood during beam search when valid candidate density is low.
        """
        if self.enter_point is None or self.num_nodes == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), 0

        curr_ep = self.enter_point
        curr_dist = self._dist(self.vectors[curr_ep], query_vec)

        for lc in range(self.max_layer, 0, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.layers[lc].get(curr_ep, []):
                    d = self._dist(self.vectors[neighbor], query_vec)
                    if d < curr_dist:
                        curr_dist = d
                        curr_ep = neighbor
                        changed = True

        ef = max(ef_search, k)
        visited = {curr_ep}
        candidates = [(curr_dist, curr_ep)]
        w = []

        if valid_mask[curr_ep]:
            heapq.heappush(w, (-curr_dist, curr_ep))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_w_dist = -w[0][0] if len(w) >= ef else float('inf')

            if c_dist > furthest_w_dist and len(w) >= ef:
                break

            # 1-hop and 2-hop neighbor expansion
            neighbors_to_check = set(self.layers[0].get(c_id, []))
            if not valid_mask[c_id] and n_hops > 1:
                # If current node is invalid, expand 2nd hop to route through valid cluster faster
                for n1 in list(neighbors_to_check):
                    for n2 in self.layers[0].get(n1, []):
                        if n2 not in visited:
                            neighbors_to_check.add(n2)

            for neighbor in neighbors_to_check:
                if neighbor not in visited:
                    visited.add(neighbor)
                    d_n = self._dist(self.vectors[neighbor], query_vec)
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

    def estimate_memory_mb(self) -> float:
        """Estimate total RAM consumption in Megabytes."""
        bytes_total = 0
        if self.vectors is not None:
            bytes_total += self.vectors.nbytes
        for layer in self.layers:
            for k, v in layer.items():
                bytes_total += sys.getsizeof(k) + sys.getsizeof(v) + len(v) * 4
        return bytes_total / (1024 * 1024)
