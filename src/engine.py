"""High-level interface for the educational vector search engine."""

from src.hnsw import HNSWIndex
from src.index import Index
from src.search import BruteForceSearch
from src.storage import VectorStore


class VectorEngine:
    """Coordinate embedding, storage, search, and an optional ANN index.

    ``index_type`` may be ``None`` (exact brute force), ``"lsh"``, or
    ``"hnsw"``.  ANN searches remain approximate and can miss matching
    documents because filtering is applied to discovered candidates.
    ``embedder`` is injectable so callers can avoid loading a model in tests.
    """

    def __init__(self, store=None, embedder=None, index_type=None, index_kwargs=None):
        if index_type not in (None, "lsh", "hnsw"):
            raise ValueError("index_type must be None, 'lsh', or 'hnsw'")
        self.store = store if store is not None else VectorStore()
        if embedder is None:
            from src.embedding import Embedder

            embedder = Embedder()
        self.embedder = embedder
        self.index_type = index_type
        self.index_kwargs = dict(index_kwargs or {})
        self.index = self._create_index()
        self.searcher = BruteForceSearch(self.store, self.index)

    def _create_index(self):
        if self.index_type == "lsh":
            return Index(self.store, **self.index_kwargs)
        if self.index_type == "hnsw":
            return HNSWIndex(self.store, **self.index_kwargs)
        return None

    @staticmethod
    def _validate_text(text, field_name):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{field_name} must be a non-empty string")

    @staticmethod
    def _validate_k(k):
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer")

    def add_document(self, text, metadata=None):
        self._validate_text(text, "text")
        vector = self.embedder.embed(text)
        vector_id = self.store.add(vector, text, metadata)
        if self.index is not None:
            if self.index_type == "hnsw":
                self.index.insert(vector_id)
            else:
                # LSH has no incremental insertion API; rebuild explicitly.
                self.index.build()
        return vector_id

    def search(self, query, k=5, filters=None):
        self._validate_text(query, "query")
        self._validate_k(k)
        query_vector = self.embedder.embed(query)
        return self.searcher.search(query_vector, k=k, filters=filters)

    def search_text(self, query, k=5, filters=None):
        """Search text and return ``(document, distance)`` pairs."""
        self._validate_text(query, "query")
        self._validate_k(k)
        return self.searcher.search_text(query, self.embedder, k=k, filters=filters)

    def save(self, filepath):
        self.store.save(filepath)

    def load(self, filepath):
        self.store.load(filepath)
        self.index = self._create_index()
        self.searcher = BruteForceSearch(self.store, self.index)
