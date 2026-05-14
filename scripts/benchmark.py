#!/usr/bin/env python
"""Micro-benchmark for encode + search latency on the real model."""

import statistics
import time

import numpy as np

from search.embedder import SentenceTransformerEmbedder
from search.index import VectorIndex

CORPUS_SIZE = 5000
QUERY_COUNT = 200
BATCH_SIZES = [1, 8, 32, 128]


def main() -> None:
    print("loading model...")
    embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    print(f"dim={embedder.dim}")

    rng = np.random.default_rng(42)
    corpus = [
        f"document number {i} with some random {rng.integers(0, 1_000_000)}"
        for i in range(CORPUS_SIZE)
    ]

    t0 = time.perf_counter()
    vectors = embedder.encode(corpus)
    encode_total = time.perf_counter() - t0
    print(
        f"corpus encode: {CORPUS_SIZE} docs in {encode_total:.2f}s "
        f"({CORPUS_SIZE / encode_total:.0f} docs/s)"
    )

    index = VectorIndex(dim=embedder.dim)
    index.add([str(i) for i in range(CORPUS_SIZE)], vectors)

    print()
    print(f"{'batch':>5} {'mean_ms':>9} {'p50_ms':>8} {'p95_ms':>8} {'p99_ms':>8} {'docs/s':>9}")
    print("-" * 55)
    for batch in BATCH_SIZES:
        texts = [f"query {i}" for i in range(batch)]
        times = []
        for _ in range(50):
            t = time.perf_counter()
            embedder.encode(texts)
            times.append((time.perf_counter() - t) * 1000)
        mean = statistics.mean(times)
        p50 = statistics.median(times)
        p95 = statistics.quantiles(times, n=20)[-1]
        p99 = sorted(times)[int(len(times) * 0.99)]
        throughput = batch * 1000 / mean
        print(f"{batch:>5} {mean:>9.2f} {p50:>8.2f} {p95:>8.2f} {p99:>8.2f} {throughput:>9.0f}")

    print()
    queries = [f"document number {rng.integers(0, CORPUS_SIZE)}" for _ in range(QUERY_COUNT)]
    query_vecs = embedder.encode(queries)

    times = []
    for q in query_vecs:
        t = time.perf_counter()
        index.search(q, k=10)
        times.append((time.perf_counter() - t) * 1000)
    print(f"search@10 over {CORPUS_SIZE} vectors, {QUERY_COUNT} queries:")
    print(
        f"  mean={statistics.mean(times):.2f} ms "
        f"p50={statistics.median(times):.2f} ms "
        f"p95={statistics.quantiles(times, n=20)[-1]:.2f} ms"
    )


if __name__ == "__main__":
    main()
