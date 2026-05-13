import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    model_name: str
    index_path: str
    store_path: str
    batch_size: int
    http_addr: str
    log_level: str


def load_config() -> Config:
    return Config(
        model_name=os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        index_path=os.environ.get("INDEX_PATH", "./data/index.bin"),
        store_path=os.environ.get("STORE_PATH", "./data/docs.jsonl"),
        batch_size=_int_env("BATCH_SIZE", 32, minimum=1),
        http_addr=os.environ.get("HTTP_ADDR", ":8000"),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )


def _int_env(key: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {value}")
    return value
