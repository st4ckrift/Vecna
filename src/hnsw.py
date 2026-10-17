"""A small, educational Hierarchical Navigable Small World index.

This implementation keeps graph structure separate from ``VectorStore`` and
uses the store's vector IDs as node IDs.  It is intentionally a single in
memory graph and is not intended to be a production HNSW implementation.
"""

import heapq
import math
import random

from src.similarity import euclidean_dis


class HNSWIndex:
    """Incremental HNSW index using Euclidean distance.

    ``M`` controls the intended number of links on upper layers (layer zero
    permits ``2 * M`` links), while ``ef_construction`` and ``ef_search``
    control insertion and query exploration breadth.  ``seed`` makes random
    node levels reproducible.  Call :meth:`insert` for a newly added store
    vector, or :meth:`build` to explicitly rebuild the complete graph.
    """

    def __init__(self, store, M=8, ef_construction=100, ef_search=32, seed=None):
        if M < 1:
            raise ValueError("M must be at least 1")
        if ef_construction < 1 or ef_search < 1:
            raise ValueError("ef_construction and ef_search must be positive")
        self.store = store
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.seed = seed
        self._rng = random.Random(seed)
        self.layers = []
        self.levels = {}
        self.entry_point = None
        self.max_level = -1
        self.build()

    def _random_level(self):
        # Geometric tail: high levels become exponentially less likely.
        value = max(self._rng.random(), 1e-12)
        return int(-math.log(value) / math.log(self.M + 1))

    def _ensure_layer(self, level):
        while len(self.layers) <= level:
            self.layers.append({})

    def _max_neighbors(self, level):
        return self.M * 2 if level == 0 else self.M

    def _distance(self, left_id, right_vector):
        return euclidean_dis(self.store.get(left_id), right_vector)

    def _search_layer(self, query, entry_ids, level, ef):
        """Return up to ``ef`` closest discovered nodes at one layer."""
        if not entry_ids:
            return []
        visited = set(entry_ids)
        candidates = []
        results = []  # (-distance, id), so the farthest result is first
        for node_id in entry_ids:
            distance = self._distance(node_id, query)
            heapq.heappush(candidates, (distance, node_id))
            heapq.heappush(results, (-distance, node_id))

        while candidates:
            distance, node_id = heapq.heappop(candidates)
            farthest = -results[0][0]
            if len(results) >= ef and distance > farthest:
                break
            for neighbor_id in self.layers[level].get(node_id, ()):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                neighbor_distance = self._distance(neighbor_id, query)
                if len(results) < ef or neighbor_distance < -results[0][0]:
                    heapq.heappush(candidates, (neighbor_distance, neighbor_id))
                    heapq.heappush(results, (-neighbor_distance, neighbor_id))
                    if len(results) > ef:
                        heapq.heappop(results)

        return [node_id for _, node_id in sorted(
            ((-distance, node_id) for distance, node_id in results)
        )]

    def _greedy_search(self, query, entry_id, level):
        current = entry_id
        current_distance = self._distance(current, query)
        changed = True
        while changed:
            changed = False
            for neighbor_id in self.layers[level].get(current, ()):
                distance = self._distance(neighbor_id, query)
                if distance < current_distance:
                    current, current_distance = neighbor_id, distance
                    changed = True
        return current

    def _prune(self, node_id, level):
        neighbors = self.layers[level].get(node_id, set())
        limit = self._max_neighbors(level)
        if len(neighbors) > limit:
            vector = self.store.get(node_id)
            ordered = sorted(neighbors, key=lambda candidate: self._distance(candidate, vector))
            kept = set(ordered[:limit])
            removed = neighbors - kept
            self.layers[level][node_id] = kept
            for removed_id in removed:
                self.layers[level].get(removed_id, set()).discard(node_id)

    def insert(self, vector_id):
        """Insert one existing store vector without rebuilding other nodes."""
        if vector_id not in self.store.vectors:
            raise KeyError(f"Unknown vector ID: {vector_id}")
        if vector_id in self.levels:
            return

        level = self._random_level()
        self.levels[vector_id] = level
        self._ensure_layer(level)
        for layer in range(level + 1):
            self.layers[layer][vector_id] = set()

        if self.entry_point is None:
            self.entry_point = vector_id
            self.max_level = level
            return

        current = self.entry_point
        for layer in range(self.max_level, level, -1):
            current = self._greedy_search(self.store.get(vector_id), current, layer)

        for layer in range(min(level, self.max_level), -1, -1):
            nearby = self._search_layer(
                self.store.get(vector_id), [current], layer, self.ef_construction
            )
            selected = sorted(
                nearby,
                key=lambda candidate: self._distance(candidate, self.store.get(vector_id)),
            )[: self._max_neighbors(layer)]
            for neighbor_id in selected:
                self.layers[layer][vector_id].add(neighbor_id)
                self.layers[layer][neighbor_id].add(vector_id)
                self._prune(neighbor_id, layer)
            self._prune(vector_id, layer)
            if selected:
                current = selected[0]

        if level > self.max_level:
            self.entry_point = vector_id
            self.max_level = level

    def build(self):
        """Clear and rebuild the graph from all vectors currently in the store."""
        self.layers = []
        self.levels = {}
        self.entry_point = None
        self.max_level = -1
        self._rng = random.Random(self.seed)
        for vector_id in self.store.vectors:
            self.insert(vector_id)

    def candidates(self, query):
        """Return approximate nearest IDs discovered by hierarchical traversal."""
        if self.entry_point is None:
            return []
        current = self.entry_point
        for layer in range(self.max_level, 0, -1):
            current = self._greedy_search(query, current, layer)
        return self._search_layer(query, [current], 0, self.ef_search)
