from src.vector import Vector
from vec import ingest_directory, iter_files


class FakeEngine:
    def __init__(self):
        self.documents = []

    def add_document(self, text, metadata=None):
        self.documents.append((text, metadata))
        return len(self.documents)


def test_directory_ingestion_is_recursive_and_deterministic(tmp_path):
    (tmp_path / "b.txt").write_text("second", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.md").write_text("first", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("  ", encoding="utf-8")
    engine = FakeEngine()

    assert [path.name for path in iter_files(tmp_path)] == ["b.txt", "empty.txt", "a.md"]
    ids = ingest_directory(engine, tmp_path)
    assert ids == [1, 2]
    assert [item[0] for item in engine.documents] == ["second", "first"]
    assert engine.documents[1][1]["extension"] == ".md"
