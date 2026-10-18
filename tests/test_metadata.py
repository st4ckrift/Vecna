import json

from src.hnsw import HNSWIndex
from src.index import Index
from src.search import BruteForceSearch
from src.storage import VectorStore
from src.vector import Vector


def make_store():
    store = VectorStore()
    first = store.add(
        Vector([0.0, 0.0]),
        "python tutorial",
        {"category": "tutorial", "language": "python", "difficulty": "beginner"},
    )
    second = store.add(
        Vector([1.0, 0.0]),
        "python reference",
        {"category": "reference", "language": "python"},
    )
    third = store.add(
        Vector([2.0, 0.0]),
        "javascript tutorial",
        {"category": "tutorial", "language": "javascript"},
    )
    fourth = store.add(Vector([3.0, 0.0]), "unclassified")
    return store, first, second, third, fourth


def test_metadata_api_and_backward_compatible_add():
    store, first, _, _, fourth = make_store()
    assert store.get_metadata(first)["category"] == "tutorial"
    assert store.get_metadata(fourth) == {}
    assert store.add(Vector([4.0, 0.0]), "empty", metadata={})


def test_exact_filters_use_and_semantics_before_top_k():
    store, first, second, third, fourth = make_store()
    search = BruteForceSearch(store)
    assert [item[0] for item in search.search(Vector([3.0, 0.0]), k=5)] == [fourth, third, second, first]
    assert [item[0] for item in search.search(
        Vector([3.0, 0.0]), k=5, filters={"category": "tutorial"}
    )] == [third, first]
    assert [item[0] for item in search.search(
        Vector([3.0, 0.0]), k=1, filters={"category": "tutorial", "language": "python"}
    )] == [first]
    assert search.search(Vector([0.0, 0.0]), filters={"missing": "value"}) == []
    assert search.search(Vector([0.0, 0.0]), filters={"language": "rust"}) == []
    assert len(search.search(Vector([0.0, 0.0]), k=4, filters={})) == 4
    assert len(search.search(Vector([0.0, 0.0]), k=4, filters=None)) == 4


def test_search_text_applies_filters_and_returns_documents():
    store, first, _, _, _ = make_store()

    class Embedder:
        def embed(self, text):
            return Vector([0.0, 0.0])

    results = BruteForceSearch(store).search_text(
        "query", Embedder(), k=5, filters={"category": "tutorial", "language": "python"}
    )
    assert results == [("python tutorial", 0.0)]


def test_metadata_persists_and_legacy_files_load(tmp_path):
    store, first, _, _, fourth = make_store()
    path = tmp_path / "vectors.json"
    store.save(path)
    loaded = VectorStore()
    loaded.load(path)
    assert loaded.get_metadata(first)["difficulty"] == "beginner"
    assert loaded.get_metadata(fourth) == {}
    assert loaded.next_id == store.next_id

    legacy_data = {
        "next_id": 2,
        "vectors": {"1": [1.0, 2.0]},
        "documents": {"1": "legacy"},
    }
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy_data))
    legacy = VectorStore()
    legacy.load(legacy_path)
    assert legacy.get(1).values == [1.0, 2.0]
    assert legacy.get_document(1) == "legacy"
    assert legacy.get_metadata(1) == {}
    assert legacy.next_id == 2


def test_filters_work_with_lsh_and_hnsw_candidates():
    store, first, second, third, _ = make_store()
    query = Vector([0.0, 0.0])
    lsh_results = BruteForceSearch(store, Index(store, num_planes=4, seed=1)).search(
        query, k=5, filters={"language": "python"}
    )
    # LSH remains approximate: filtering only applies to discovered candidates.
    assert {item[0] for item in lsh_results}.issubset({first, second})

    hnsw = HNSWIndex(store, ef_search=10, seed=1)
    hnsw_results = BruteForceSearch(store, hnsw).search(
        query, k=5, filters={"category": "tutorial"}
    )
    assert set(item[0] for item in hnsw_results).issubset({first, third})
