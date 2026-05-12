from typing import Protocol

import numpy as np


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size

    @property
    def dim(self) -> int:
        getter = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        return int(getter())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)


class HashEmbedder:
    """Deterministic stub used in tests so the suite runs without downloading weights."""

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            rng = np.random.default_rng(seed=_stable_hash(text))
            vec = rng.standard_normal(self._dim).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-12
            out[i] = vec
        return out


def _stable_hash(text: str) -> int:
    h = 1469598103934665603
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h
