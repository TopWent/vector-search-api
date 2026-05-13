from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import TracebackType

import numpy as np
from fastapi import FastAPI, HTTPException, Response

from . import metrics
from .config import Config
from .embedder import Embedder
from .index import VectorIndex
from .schemas import (
    EmbedRequest,
    EmbedResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    UpsertRequest,
    UpsertResponse,
)
from .store import DocumentStore

log = logging.getLogger("search")


def create_app(
    embedder: Embedder, index: VectorIndex, store: DocumentStore, cfg: Config
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        log.info("ready: model_dim=%d corpus=%d", embedder.dim, len(store))
        try:
            yield
        finally:
            try:
                index.save(cfg.index_path)
                store.save(cfg.store_path)
                log.info("snapshot saved: index=%s store=%s", cfg.index_path, cfg.store_path)
            except Exception:
                log.exception("snapshot failed")

    app = FastAPI(title="vector-search-api", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "dim": embedder.dim, "indexed": index.size}

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        body, content_type = metrics.expose()
        return Response(content=body, media_type=content_type)

    @app.post("/embeddings", response_model=EmbedResponse)
    async def embeddings(req: EmbedRequest) -> EmbedResponse:
        with _track("embeddings") as record_status:
            vectors = await asyncio.to_thread(_encode_with_metrics, embedder, req.texts)
            record_status("ok")
            return EmbedResponse(dim=embedder.dim, vectors=vectors.tolist())

    @app.post("/documents", response_model=UpsertResponse)
    async def documents(req: UpsertRequest) -> UpsertResponse:
        with _track("documents") as record_status:
            texts = [d.text for d in req.documents]
            ids = [d.id for d in req.documents]
            vectors = await asyncio.to_thread(_encode_with_metrics, embedder, texts)
            try:
                await asyncio.to_thread(index.add, ids, vectors)
            except (TypeError, ValueError) as e:
                record_status("error")
                raise HTTPException(status_code=400, detail=str(e)) from e

            new_count = store.upsert(req.documents)
            metrics.INDEX_SIZE.inc(len(req.documents))
            record_status("ok")
            return UpsertResponse(upserted=new_count, total=len(store))

    @app.post("/search", response_model=SearchResponse)
    async def search(req: SearchRequest) -> SearchResponse:
        with _track("search") as record_status:
            if index.size == 0:
                record_status("empty")
                return SearchResponse(hits=[])

            vectors = await asyncio.to_thread(_encode_with_metrics, embedder, [req.query])
            results = await asyncio.to_thread(index.search, vectors[0], req.k)

            hits: list[SearchHit] = []
            for doc_id, score in results:
                doc = store.get(doc_id)
                if doc is None:
                    continue
                hits.append(SearchHit(id=doc.id, score=score, text=doc.text, metadata=doc.metadata))
            record_status("ok")
            return SearchResponse(hits=hits)

    return app


class _Timer:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.status = "error"
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()

        def record_status(value: str) -> None:
            self.status = value

        return record_status

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        elapsed = time.perf_counter() - self._start
        metrics.REQUEST_LATENCY.labels(endpoint=self.endpoint).observe(elapsed)
        metrics.REQUEST_COUNTER.labels(endpoint=self.endpoint, status=self.status).inc()


def _track(endpoint: str) -> _Timer:
    return _Timer(endpoint)


def _encode_with_metrics(embedder: Embedder, texts: list[str]) -> np.ndarray:
    start = time.perf_counter()
    vectors = embedder.encode(texts)
    elapsed = time.perf_counter() - start
    metrics.ENCODE_LATENCY.labels(batch_size_bucket=_bucket(len(texts))).observe(elapsed)
    return vectors


def _bucket(n: int) -> str:
    if n == 1:
        return "1"
    if n <= 8:
        return "2-8"
    if n <= 32:
        return "9-32"
    if n <= 128:
        return "33-128"
    return "129+"
