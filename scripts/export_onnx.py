#!/usr/bin/env python
"""Export a sentence-transformer to ONNX, optionally quantise to INT8, and benchmark.

Run:
    python scripts/export_onnx.py --model sentence-transformers/all-MiniLM-L6-v2 --quantize

Outputs go to ./output/onnx-fp32 and ./output/onnx-int8.

Requires: pip install -e ".[ml]"
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

SAMPLE_TEXT = "Sentence embedding inference benchmark text. " * 4


def benchmark(encode_fn, batch_size: int, runs: int = 30) -> dict[str, float]:
    texts = [SAMPLE_TEXT for _ in range(batch_size)]
    # warmup
    for _ in range(3):
        encode_fn(texts)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        encode_fn(texts)
        times.append((time.perf_counter() - t) * 1000)
    p95_idx = max(int(len(times) * 0.95) - 1, 0)
    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "p95_ms": sorted(times)[p95_idx],
        "throughput": batch_size * 1000 / statistics.mean(times),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument(
        "--quantize", action="store_true", help="Also export INT8 dynamic-quantised variant"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()

    try:
        import numpy as np
        from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"missing dependency: {e}", file=sys.stderr)
        print('install ML extras: pip install -e ".[ml]"', file=sys.stderr)
        return 2

    output_root = Path(args.output_dir)
    fp32_dir = output_root / "onnx-fp32"
    int8_dir = output_root / "onnx-int8"

    print(f"exporting {args.model} to ONNX FP32...")
    onnx_model = ORTModelForFeatureExtraction.from_pretrained(args.model, export=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    onnx_model.save_pretrained(fp32_dir)
    tokenizer.save_pretrained(fp32_dir)
    print(f"  saved to {fp32_dir}")

    if args.quantize:
        print("quantising to INT8 (AVX2 dynamic)...")
        quantizer = ORTQuantizer.from_pretrained(fp32_dir)
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=str(int8_dir), quantization_config=qconfig)
        tokenizer.save_pretrained(int8_dir)
        print(f"  saved to {int8_dir}")

    if args.skip_bench:
        return 0

    print(f"\nbenchmarking encode (batch={args.batch_size}, 30 runs each)...")

    pt_model = SentenceTransformer(args.model)
    onnx_fp32 = ORTModelForFeatureExtraction.from_pretrained(fp32_dir)
    onnx_int8 = (
        ORTModelForFeatureExtraction.from_pretrained(int8_dir, file_name="model_quantized.onnx")
        if args.quantize
        else None
    )

    def encode_pt(texts):
        return pt_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def encode_onnx(model_):
        def fn(texts):
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            out = model_(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            normed = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-9)
            return normed.detach().numpy()

        return fn

    backends = [("pytorch", encode_pt), ("onnx-fp32", encode_onnx(onnx_fp32))]
    if onnx_int8 is not None:
        backends.append(("onnx-int8", encode_onnx(onnx_int8)))

    print()
    print(f"{'backend':<12} {'mean_ms':>10} {'p50_ms':>10} {'p95_ms':>10} {'docs/s':>10}")
    print("-" * 56)
    for name, fn in backends:
        stats = benchmark(fn, args.batch_size)
        print(
            f"{name:<12} {stats['mean_ms']:>10.2f} {stats['p50_ms']:>10.2f} "
            f"{stats['p95_ms']:>10.2f} {stats['throughput']:>10.0f}"
        )

    print()
    print("accuracy check: cosine similarity between PyTorch and ONNX FP32 embeddings")
    sample = ["one", "two", "three"]
    pt_vec = encode_pt(sample)
    onnx_vec = encode_onnx(onnx_fp32)(sample)
    sims = (pt_vec * onnx_vec).sum(axis=1)
    print(f"  per-text cosine: {sims.round(4).tolist()}")
    print(f"  mean: {float(np.mean(sims)):.4f} (1.0 is identical)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
