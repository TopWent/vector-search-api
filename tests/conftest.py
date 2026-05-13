import pytest
from fastapi.testclient import TestClient

from search.app import create_app
from search.config import Config
from search.embedder import HashEmbedder
from search.index import VectorIndex
from search.store import DocumentStore


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        model_name="stub",
        index_path=str(tmp_path / "idx.bin"),
        store_path=str(tmp_path / "store.jsonl"),
        batch_size=8,
        http_addr=":8000",
        log_level="info",
    )


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=16)


@pytest.fixture
def empty_index(embedder) -> VectorIndex:
    return VectorIndex(dim=embedder.dim)


@pytest.fixture
def store() -> DocumentStore:
    return DocumentStore()


@pytest.fixture
def client(embedder, empty_index, store, cfg) -> TestClient:
    app = create_app(embedder, empty_index, store, cfg)
    return TestClient(app)
