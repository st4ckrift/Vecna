from src.hnsw import HNSWIndex
from src.search import BruteForceSearch
from src.storage import VectorStore
from src.vector import Vector


def store_with_vectors(values):
    store = VectorStore()
    ids = [store.add(Vector(value), str(value)) for value in values]
    return store, ids


def test_empty_and_single_vector_indexes_are_safe():
    empty, _ = store_with_vectors([])
    index = HNSWIndex(empty, seed=4)
    assert index.entry_point is None
    assert index.candidates(Vector([1.0, 0.0])) == []

    store, ids = store_with_vectors([[1.0, 0.0]])
    index = HNSWIndex(store, seed=4)
    assert index.entry_point == ids[0]
    assert index.candidates(Vector([1.0, 0.0])) == ids


def test_graph_has_valid_reciprocal_links_and_connectivity_limits():
    store, ids = store_with_vectors([[float(i), 0.0] for i in range(12)])
    index = HNSWIndex(store, M=2, ef_construction=8, seed=9)
    assert set(index.levels) == set(ids)
    for level, graph in enumerate(index.layers):
        limit = 4 if level == 0 else 2
        for node_id, neighbors in graph.items():
            assert node_id in store.vectors
            assert len(neighbors) <= limit
            for neighbor_id in neighbors:
                assert neighbor_id in store.vectors
                assert node_id in graph[neighbor_id]


def test_seed_makes_levels_and_graph_reproducible():
    store, _ = store_with_vectors([[float(i), float(i % 3)] for i in range(8)])
    first = HNSWIndex(store, M=3, seed=12)
    second = HNSWIndex(store, M=3, seed=12)
    assert first.levels == second.levels
    assert first.entry_point == second.entry_point
    assert first.layers == second.layers


def test_candidates_and_search_use_graph_not_full_store_scan():
    store, ids = store_with_vectors([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    index = HNSWIndex(store, M=2, ef_construction=8, ef_search=2, seed=3)
    candidates = index.candidates(Vector([0.2, 0.0]))
    assert 0 < len(candidates) <= 2
    assert set(candidates).issubset(set(ids))
    results = BruteForceSearch(store, index).search(Vector([0.2, 0.0]), k=2)
    assert results
    assert results[0][0] == ids[0]


def test_incremental_insert_and_explicit_rebuild_include_vectors():
    store, ids = store_with_vectors([[0.0, 0.0], [10.0, 10.0]])
    index = HNSWIndex(store, seed=8)
    new_id = store.add(Vector([5.0, 5.0]), "new")
    assert new_id not in index.levels
    index.insert(new_id)
    assert new_id in index.levels
    assert new_id in index.candidates(Vector([5.0, 5.0]))

    rebuilt = HNSWIndex(store, seed=8)
    assert set(rebuilt.levels) == set(ids + [new_id])


def test_search_text_preserves_documents():
    store, ids = store_with_vectors([[1.0, 0.0], [0.0, 1.0]])
    index = HNSWIndex(store, ef_search=4, seed=2)

    class Embedder:
        def embed(self, text):
            return Vector([1.0, 0.0])

    results = BruteForceSearch(store, index).search_text("query", Embedder())
    assert results[0][0] == str([1.0, 0.0])
    assert results[0][1] == 0.0
