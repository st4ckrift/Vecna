from src.index import Index
from src.search import BruteForceSearch
from src.storage import VectorStore
from src.vector import Vector


def make_store():
    store = VectorStore()
    first = store.add(Vector([1.0, 0.0]), "first")
    second = store.add(Vector([0.0, 1.0]), "second")
    return store, first, second


def test_hash_has_configured_bits_and_identical_vectors_match():
    store, first, _ = make_store()
    index = Index(store, num_planes=5, seed=7)
    assert len(index._hash_vector(store.get(first))) == 5
    assert index._hash_vector(Vector([1.0, 0.0])) == index._hash_vector(store.get(first))


def test_build_places_ids_in_their_matching_buckets():
    store, first, second = make_store()
    index = Index(store, num_planes=4, seed=3)
    index.build()
    assert first in index.buckets[index._hash_vector(store.get(first))]
    assert second in index.buckets[index._hash_vector(store.get(second))]


def test_candidates_only_returns_exact_bucket_and_empty_bucket_is_empty():
    store, first, second = make_store()
    index = Index(store, num_planes=4, seed=0)
    first_signature = index._hash_vector(store.get(first))
    assert index.candidates(store.get(first)) == index.buckets[first_signature]

    missing_query = None
    for value in range(-10, 11):
        candidate = Vector([value, value + 0.37])
        if index._hash_vector(candidate) not in index.buckets:
            missing_query = candidate
            break
    assert missing_query is not None
    missing_signature = index._hash_vector(missing_query)
    assert missing_signature not in index.buckets
    assert index.candidates(missing_query) == []
    assert second in index.candidates(store.get(second))


def test_search_works_with_and_without_index_and_search_text():
    store, first, _ = make_store()
    query = store.get(first)
    brute_results = BruteForceSearch(store).search(query, k=2)
    indexed_results = BruteForceSearch(store, Index(store, num_planes=4, seed=4)).search(query, k=2)
    assert brute_results[0][0] == first
    assert indexed_results[0][0] == first

    class Embedder:
        def embed(self, text):
            return query

    text_results = BruteForceSearch(store, Index(store, seed=4)).search_text("ignored", Embedder())
    assert text_results[0][0] == "first"
    assert text_results[0][1] == indexed_results[0][1]


def test_rebuild_includes_new_vectors_and_empty_store_is_safe():
    store = VectorStore()
    index = Index(store, seed=2)
    assert index.candidates(Vector([1.0, 2.0])) == []
    new_id = store.add(Vector([1.0, 2.0]), "new")
    index.build()
    assert new_id in index.candidates(Vector([1.0, 2.0]))


def test_store_json_persistence_remains_compatible(tmp_path):
    store, first, _ = make_store()
    path = tmp_path / "vectors.json"
    store.save(path)
    loaded = VectorStore()
    loaded.load(path)
    assert loaded.get(first).values == store.get(first).values
    assert loaded.get_document(first) == "first"
