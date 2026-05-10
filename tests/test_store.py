from search.schemas import Document
from search.store import DocumentStore


def test_empty_store():
    s = DocumentStore()
    assert len(s) == 0
    assert s.get("missing") is None


def test_upsert_new_documents_returns_count():
    s = DocumentStore()
    new = s.upsert(
        [
            Document(id="a", text="alpha"),
            Document(id="b", text="bravo"),
        ]
    )
    assert new == 2
    assert len(s) == 2


def test_upsert_overwrites_existing_returns_only_new():
    s = DocumentStore()
    s.upsert([Document(id="a", text="alpha v1")])
    new = s.upsert(
        [
            Document(id="a", text="alpha v2"),
            Document(id="b", text="bravo"),
        ]
    )
    assert new == 1
    assert s.get("a").text == "alpha v2"


def test_metadata_round_trip():
    s = DocumentStore()
    s.upsert([Document(id="x", text="t", metadata={"src": "wiki", "lang": "en"})])
    doc = s.get("x")
    assert doc.metadata == {"src": "wiki", "lang": "en"}


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "store.jsonl"
    s = DocumentStore()
    s.upsert(
        [
            Document(id="a", text="alpha"),
            Document(id="b", text="bravo", metadata={"k": "v"}),
        ]
    )
    s.save(path)

    restored = DocumentStore.load(path)
    assert len(restored) == 2
    assert restored.get("a").text == "alpha"
    assert restored.get("b").metadata == {"k": "v"}


def test_load_missing_path_returns_empty(tmp_path):
    restored = DocumentStore.load(tmp_path / "missing.jsonl")
    assert len(restored) == 0
