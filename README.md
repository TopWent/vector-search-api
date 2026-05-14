# Vector Search API

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Embedding search service plus the offline tooling around it. Fine-tune a sentence-transformer, evaluate it with IR metrics on BEIR-format data, export to ONNX with INT8 quantization, then serve queries over HTTP backed by FAISS.

## What it does

The HTTP service (FastAPI) embeds text, indexes it in FAISS, and answers top-k queries. Snapshots persist to disk on shutdown. Prometheus metrics and JSON logs are wired in.

The rest is scripts:

- `scripts/evaluate.py` runs BEIR-format datasets and reports MRR, nDCG, Recall at k.
- `scripts/finetune.py` trains with MultipleNegativesRankingLoss on HuggingFace datasets.
- `scripts/export_onnx.py` exports ONNX FP32, quantises to INT8 (dynamic, AVX2), and times the three backends against each other.
- `scripts/benchmark.py` measures encode throughput and search latency on the real model.

Tests swap in a stub embedder (`HashEmbedder`) so the suite runs in under a second with no model download.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/documents` | Upsert text documents; the service embeds and indexes them. |
| `POST` | `/search` | Top-k nearest neighbours for a query string. |
| `POST` | `/embeddings` | Raw embedding vectors for a list of texts (no indexing). |
| `GET` | `/health` | Readiness probe. |
| `GET` | `/metrics` | Prometheus text exposition. |

## Quick start

```bash
docker compose up --build
python scripts/seed.py
curl -s localhost:8000/search -d '{"query":"machine learning","k":3}' -H 'Content-Type: application/json' | jq
```

## ML pipeline

### Evaluate

```bash
pip install -e ".[ml]"
python scripts/evaluate.py --dataset data/eval_demo --model sentence-transformers/all-MiniLM-L6-v2 --k 10
```

The bundled `data/eval_demo` set is a 20-doc, 5-query sanity check, not a benchmark. The queries lift phrases straight from the relevant docs, so MRR and Recall come out at 1.0 by construction. It only proves the eval plumbing runs end to end.

For numbers that mean anything, point the script at BEIR (SciFact, NFCorpus, FiQA) or MTEB. The harness needs three files: `corpus.jsonl`, `queries.jsonl`, `qrels.tsv`.

### Fine-tune

```bash
# Smoke test on CPU (~2 minutes)
python scripts/finetune.py --steps 50 --batch-size 8

# Full fine-tune (GPU recommended)
python scripts/finetune.py --base-model sentence-transformers/all-MiniLM-L6-v2 \
    --dataset sentence-transformers/all-nli --config pair --split 'train[:50000]' \
    --steps 5000 --batch-size 64 --output ./output/finetuned
```

Then re-run `evaluate.py --model ./output/finetuned` and compare.

### Export to ONNX and quantise

```bash
python scripts/export_onnx.py --model sentence-transformers/all-MiniLM-L6-v2 --quantize
```

Exports FP32 ONNX, quantises to INT8 (AVX2 dynamic), times 30 iterations each for PyTorch / ONNX FP32 / ONNX INT8, and prints cosine similarity between PyTorch and ONNX FP32 so you can check the export did not drift.

Rough CPU numbers I saw on `all-MiniLM-L6-v2`, batch 8, on a laptop:

| Backend | mean (ms) | p95 (ms) | docs/s |
|---|---|---|---|
| pytorch | 35 | 48 | 230 |
| onnx-fp32 | 22 | 30 | 360 |
| onnx-int8 | 11 | 16 | 720 |

Your mileage varies by CPU. Run it yourself. Cosine similarity vs PyTorch stays above 0.9999.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model id (or local path) |
| `INDEX_PATH` | `./data/index.bin` | FAISS index snapshot path |
| `STORE_PATH` | `./data/docs.jsonl` | JSONL document store path |
| `BATCH_SIZE` | `32` | Encode batch size |
| `HTTP_ADDR` | `:8000` | Listen address |
| `LOG_LEVEL` | `info` | `debug` / `info` / `warn` / `error` |

## Layout

`src/search` is the service: `app.py` holds the FastAPI routes, `embedder.py` wraps SentenceTransformer (with a hash-based stub for tests), `index.py` wraps a FAISS `IndexFlatIP`, and `store.py` keeps the documents in a JSONL-backed dict. `config.py` reads env vars, `metrics.py` and `logging_setup.py` handle observability. The offline pieces (fine-tune, evaluate, ONNX export) live in `scripts/` and the IR metrics they use are under `src/search/eval`.

Embeddings are L2-normalised so inner product equals cosine. `IndexFlatIP` is exact and fine up to a point; for big corpora swap in `IndexHNSWFlat(dim, M=32)` and eat the recall hit.

## Development

```bash
make install     # create venv and install dev deps
make test        # pytest, no network
make bench       # micro-benchmark encode + search latency
make docker      # build the image
```

## Limitations

The index is flat and lives in memory, so the corpus has to fit in RAM and large collections will be slow. There is no auth, rate limiting, or HNSW yet. Snapshots are written on shutdown only, so a hard crash loses anything indexed since the last clean stop. Single process; scaling out means rebuilding the index per replica or moving to a shared vector store. Things I would add next if this were more than a portfolio piece: HNSW for big corpora, a cross-encoder re-rank stage, and persisting on a timer instead of just at exit.

## License

MIT. See [LICENSE](LICENSE).

## Author

[@TopWent](https://github.com/TopWent).
