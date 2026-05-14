#!/usr/bin/env python
"""Evaluate a sentence-transformer on a BEIR-format dataset.

Expected layout:
    <dir>/corpus.jsonl   lines of {"_id": str, "title": str?, "text": str}
    <dir>/queries.jsonl  lines of {"_id": str, "text": str}
    <dir>/qrels.tsv      TSV "query-id\tcorpus-id\tscore" with header

Run:
    python scripts/evaluate.py --dataset data/eval_demo --model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import json
import sys
from pathlib import Path

from search.embedder import SentenceTransformerEmbedder
from search.eval import evaluate


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as f:
        header = next(f, "").strip().split("\t")
        if "query-id" not in header:
            f.seek(0)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, cid, score = parts[0], parts[1], parts[-1]
            qrels.setdefault(qid, {})[cid] = int(score)
    return qrels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="Directory containing corpus.jsonl, queries.jsonl, qrels.tsv",
    )
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.dataset)
    corpus_rows = load_jsonl(root / "corpus.jsonl")
    query_rows = load_jsonl(root / "queries.jsonl")
    qrels = load_qrels(root / "qrels.tsv")

    corpus = {r["_id"]: (r.get("title", "") + " " + r["text"]).strip() for r in corpus_rows}
    queries = {r["_id"]: r["text"] for r in query_rows}

    print(f"dataset: corpus={len(corpus)} queries={len(queries)} qrels={len(qrels)}")
    print(f"loading model: {args.model}")
    embedder = SentenceTransformerEmbedder(args.model, batch_size=args.batch_size)
    print(f"model ready: dim={embedder.dim}")

    print("running evaluation...")
    result = evaluate(embedder, corpus, queries, qrels, k=args.k)

    print()
    print(f"queries_evaluated = {result.n_queries}")
    print(f"corpus_size       = {result.n_corpus}")
    print(f"MRR@{args.k:<3}        = {result.mrr:.4f}")
    print(f"nDCG@{args.k:<3}       = {result.ndcg:.4f}")
    print(f"Recall@{args.k:<3}     = {result.recall:.4f}")

    if args.output:
        out = {
            "dataset": args.dataset,
            "model": args.model,
            "k": args.k,
            "n_queries": result.n_queries,
            "n_corpus": result.n_corpus,
            "mrr": result.mrr,
            "ndcg": result.ndcg,
            "recall": result.recall,
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nsaved {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
