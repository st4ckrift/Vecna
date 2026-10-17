"""Approximate nearest-neighbor indexes.

The LSH index in this module uses one table of random hyperplanes.  Queries
only inspect the exact matching bucket; this is intentionally small and
educational rather than a high-recall production ANN implementation.
"""

import random


class Index:
    """Random-hyperplane locality-sensitive hash index.

    ``build()`` should be called again after vectors are added to the store.
    A query can return fewer than ``k`` results, or no results, when its exact
    hash bucket is sparse.
    """

    def __init__(self, store, num_planes=8, seed=None):
        if num_planes < 0:
            raise ValueError("num_planes must be non-negative")
        self.store = store
        self.num_planes = num_planes
        self.seed = seed
        self.hyperplanes = []
        self.buckets = {}
        self.dimension = None
        self._generate_hyperplanes()
        self.build()

    def _current_dimension(self):
        if not self.store.vectors:
            return None
        return len(next(iter(self.store.vectors.values())).values)

    def _generate_hyperplanes(self):
        self.dimension = self._current_dimension()
        self.hyperplanes = []
        if self.dimension is None:
            return

        rng = random.Random(self.seed)
        self.hyperplanes = [
            [rng.gauss(0.0, 1.0) for _ in range(self.dimension)]
            for _ in range(self.num_planes)
        ]

    def _ensure_hyperplanes(self, vector):
        dimension = len(vector.values)
        if self.dimension != dimension:
            self._generate_hyperplanes()
        if self.dimension != dimension:
            raise ValueError("Vector dimensionality does not match the index")

    def _hash_vector(self, vector):
        """Return the hyperplane sign bits as a hashable tuple."""
        self._ensure_hyperplanes(vector)
        bits = []
        for hyperplane in self.hyperplanes:
            dot_product = sum(value * weight for value, weight in zip(vector.values, hyperplane))
            bits.append(1 if dot_product >= 0 else 0)
        return tuple(bits)

    def build(self):
        """Rebuild buckets from every vector currently in the store."""
        self._generate_hyperplanes()
        self.buckets = {}
        if self.dimension is None:
            return
        for vector_id, vector in self.store.vectors.items():
            signature = self._hash_vector(vector)
            bucket = self.buckets.setdefault(signature, [])
            if vector_id not in bucket:
                bucket.append(vector_id)

    def candidates(self, query):
        """Return IDs in the query's exact bucket, without a full-scan fallback."""
        if not self.hyperplanes:
            return []
        signature = self._hash_vector(query)
        return list(self.buckets.get(signature, []))
