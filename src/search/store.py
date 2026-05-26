from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from .schemas import Document


class DocumentStore:
    """In-memory document map, keyed by id, persisted as JSONL.

    The whole corpus lives in a dict; save() rewrites the file from scratch.
    Fine for the corpus sizes this thing targets, not for millions of docs.
    """

    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}
        self._lock = Lock()

    def __len__(self) -> int:
        return len(self._docs)

    def upsert(self, documents: list[Document]) -> int:
        with self._lock:
            new_count = 0
            for d in documents:
                if d.id not in self._docs:
                    new_count += 1
                self._docs[d.id] = d
            return new_count

    def get(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    def all_ids(self) -> list[str]:
        with self._lock:
            return list(self._docs.keys())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, path.open("w", encoding="utf-8") as f:
            for doc in self._docs.values():
                f.write(json.dumps(doc.model_dump(), ensure_ascii=False))
                f.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> DocumentStore:
        store = cls()
        path = Path(path)
        if not path.exists():
            return store
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = Document.model_validate_json(line)
                store._docs[doc.id] = doc
        return store
