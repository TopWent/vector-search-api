from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from .rwlock import RWLock


class VectorIndex:
    """Inner-product index with id mapping. Caller must pass L2-normalised vectors."""

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []
        self._lock = RWLock()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def size(self) -> int:
        return self._index.ntotal

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if vectors.dtype != np.float32:
            raise TypeError(f"vectors must be float32, got {vectors.dtype}")
        if vectors.shape != (len(ids), self._dim):
            raise ValueError(
                f"shape mismatch: vectors {vectors.shape}, expected ({len(ids)}, {self._dim})"
            )
        with self._lock.write():
            self._index.add(vectors)
            self._ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if query.dtype != np.float32:
            raise TypeError(f"query must be float32, got {query.dtype}")
        if query.shape != (self._dim,):
            raise ValueError(f"query shape {query.shape}, expected ({self._dim},)")
        if self.size == 0:
            return []

        with self._lock.read():
            k = min(k, self.size)
            scores, idx = self._index.search(query.reshape(1, -1), k)
            return [
                (self._ids[i], float(s))
                for s, i in zip(scores[0], idx[0], strict=True)
                if i != -1
            ]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock.read():
            faiss.write_index(self._index, str(path))
            path.with_suffix(".ids").write_text("\n".join(self._ids), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, dim: int) -> VectorIndex:
        path = Path(path)
        instance = cls(dim)
        if not path.exists():
            return instance
        instance._index = faiss.read_index(str(path))
        ids_path = path.with_suffix(".ids")
        if ids_path.exists():
            instance._ids = ids_path.read_text(encoding="utf-8").splitlines()
        return instance
