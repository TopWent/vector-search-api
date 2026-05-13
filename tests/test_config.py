import pytest

from search.config import load_config


def test_defaults(monkeypatch):
    for key in ("MODEL_NAME", "INDEX_PATH", "STORE_PATH", "BATCH_SIZE", "HTTP_ADDR", "LOG_LEVEL"):
        monkeypatch.delenv(key, raising=False)

    cfg = load_config()
    assert "all-MiniLM-L6-v2" in cfg.model_name
    assert cfg.batch_size == 32
    assert cfg.http_addr == ":8000"
    assert cfg.log_level == "info"


def test_overrides(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "custom/model")
    monkeypatch.setenv("BATCH_SIZE", "16")
    monkeypatch.setenv("HTTP_ADDR", "127.0.0.1:9000")
    monkeypatch.setenv("LOG_LEVEL", "debug")

    cfg = load_config()
    assert cfg.model_name == "custom/model"
    assert cfg.batch_size == 16
    assert cfg.http_addr == "127.0.0.1:9000"
    assert cfg.log_level == "debug"


def test_rejects_zero_batch_size(monkeypatch):
    monkeypatch.setenv("BATCH_SIZE", "0")
    with pytest.raises(ValueError):
        load_config()


def test_rejects_non_integer_batch_size(monkeypatch):
    monkeypatch.setenv("BATCH_SIZE", "abc")
    with pytest.raises(ValueError):
        load_config()
