import pytest

from src.engine import VectorEngine
from src.storage import VectorStore
from src.vector import Vector


class FakeEmbedder:
    def embed(self, text):
        values = {
            "near": [0.0, 0.0],
            "far": [10.0, 0.0],
            "middle": [5.0, 0.0],
        }
        return Vector(values[text])


def make_engine(index_type=None, index_kwargs=None):
    return VectorEngine(
        embedder=FakeEmbedder(), index_type=index_type, index_kwargs=index_kwargs
    )


def test_default_engine_adds_and_searches_documents():
    engine = make_engine()
    document_id = engine.add_document("near", {"kind": "close"})
    assert document_id == 1
    assert engine.search("near", k=1) == [(document_id, 0.0)]
    assert engine.search_text("near", k=1) == [("near", 0.0)]


def test_metadata_filtering_and_empty_store():
    engine = make_engine()
    assert engine.search("near") == []
    engine.add_document("near", {"kind": "close"})
    engine.add_document("far", {"kind": "distant"})
    assert engine.search("near", k=5, filters={"kind": "distant"}) == [(2, 10.0)]
    assert engine.search("near", k=5, filters={"kind": "missing"}) == []


def test_save_load_restores_data_and_rebuilds_index(tmp_path):
    path = tmp_path / "vectors.json"
    engine = make_engine("hnsw", {"seed": 3, "ef_search": 10})
    engine.add_document("near", {"kind": "close"})
    engine.add_document("far", {"kind": "distant"})
    engine.save(path)

    loaded = make_engine("hnsw", {"seed": 3, "ef_search": 10})
    loaded.load(path)
    assert loaded.search("near", k=1) == [(1, 0.0)]
    assert loaded.store.get_metadata(1) == {"kind": "close"}
    assert set(loaded.index.levels) == {1, 2}


def test_index_modes_incrementally_accept_documents():
    for index_type, kwargs in (("lsh", {"num_planes": 1, "seed": 2}), ("hnsw", {"seed": 2})):
        engine = make_engine(index_type, kwargs)
        document_id = engine.add_document("middle")
        assert document_id in engine.index.candidates(Vector([5.0, 0.0]))


def test_invalid_engine_configuration_and_inputs():
    with pytest.raises(ValueError, match="index_type"):
        make_engine("unknown")
    engine = make_engine()
    with pytest.raises(ValueError, match="text"):
        engine.add_document("   ")
    with pytest.raises(ValueError, match="query"):
        engine.search("\t")
    with pytest.raises(ValueError, match="positive"):
        engine.search("near", k=0)
    with pytest.raises(ValueError, match="positive"):
        engine.search("near", k=True)


def test_existing_store_can_be_injected():
    store = VectorStore()
    store.add(Vector([0.0, 0.0]), "near", {"kind": "close"})
    engine = VectorEngine(store=store, embedder=FakeEmbedder())
    assert engine.search("near") == [(1, 0.0)]
