#!/usr/bin/env python
"""Fine-tune a sentence-transformer with MultipleNegativesRankingLoss.

Defaults pull a small slice of `sentence-transformers/all-nli` pair split. For
real training swap in MS MARCO triplets or your own (anchor, positive) data.

Run smoke test (CPU, ~2 minutes):
    python scripts/finetune.py --steps 50 --batch-size 8

Run full fine-tune (GPU recommended):
    python scripts/finetune.py --steps 5000 --batch-size 64

Requires: pip install -e ".[ml]"
"""

import argparse
import logging
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dataset", default="sentence-transformers/all-nli")
    parser.add_argument("--config", default="pair")
    parser.add_argument("--split", default="train[:10000]")
    parser.add_argument("--output", default="./output/finetuned")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
        from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    except ImportError as e:
        print(f"missing dependency: {e}", file=sys.stderr)
        print('install ML extras: pip install -e ".[ml]"', file=sys.stderr)
        return 2

    print(f"base model: {args.base_model}")
    model = SentenceTransformer(args.base_model)

    print(f"loading dataset {args.dataset} / {args.config} / {args.split}")
    train_ds = load_dataset(args.dataset, args.config, split=args.split)
    print(f"train examples: {len(train_ds)}")
    print(f"columns: {list(train_ds.column_names)}")

    loss = losses.MultipleNegativesRankingLoss(model)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to="none",
        fp16=False,
        bf16=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        loss=loss,
    )

    print(f"training: max_steps={args.steps} batch_size={args.batch_size} lr={args.learning_rate}")
    trainer.train()

    model.save_pretrained(str(output_dir))
    print(f"saved fine-tuned model to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
