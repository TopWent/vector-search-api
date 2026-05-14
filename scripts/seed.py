#!/usr/bin/env python
"""Upload a small sample corpus to a running API instance."""

import json
import os
import sys
import urllib.error
import urllib.request

SAMPLE = [
    {"id": "ml-1", "text": "Machine learning models learn patterns from data."},
    {"id": "ml-2", "text": "Deep learning uses neural networks with many layers."},
    {"id": "ml-3", "text": "Reinforcement learning trains agents through rewards."},
    {"id": "db-1", "text": "Relational databases store data in normalized tables."},
    {"id": "db-2", "text": "Vector databases index embeddings for similarity search."},
    {"id": "db-3", "text": "Sharding distributes data across multiple servers."},
    {"id": "ops-1", "text": "Kubernetes orchestrates containers across nodes."},
    {"id": "ops-2", "text": "Prometheus scrapes metrics over HTTP at a fixed interval."},
    {"id": "ops-3", "text": "Observability covers logs, metrics, and traces."},
    {"id": "food-1", "text": "Italian pasta is made from durum wheat semolina."},
    {"id": "food-2", "text": "Thai green curry uses coconut milk and basil."},
    {"id": "food-3", "text": "Sushi rice is seasoned with rice vinegar and sugar."},
]


def main() -> None:
    base = os.environ.get("API_URL", "http://localhost:8000")
    body = json.dumps({"documents": SAMPLE}).encode("utf-8")
    req = urllib.request.Request(
        url=f"{base}/documents",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
