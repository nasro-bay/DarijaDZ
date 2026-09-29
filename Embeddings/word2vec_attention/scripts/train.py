#!/usr/bin/env python
"""Trains the CBOW+attention model (see ../plan.md and model.py). Requires
../data/{rows.jsonl,token_freq.json,meta.json} from build_training_data.py.

Run via the GPU venv's Python (see ../requirements.txt):

    ".../ai-gpu/Scripts/python.exe" train.py --epochs 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from dataset import (
    CBOWCollator,
    ClusterBatchSampler,
    RowDataset,
    build_negative_sampling_table,
    build_subsample_keep_prob,
)
from model import CBOWAttention

ROOT = Path(__file__).resolve().parents[3]
# Same WORD2VEC_DATA_DIR/WORD2VEC_MODELS_DIR overrides as build_training_data.py --
# point a bigcorpus/new-tokenizer run at separate trees so the checkpoint this
# module's docstring references (checkpoint_step675000.pt) stays untouched.
DATA_DIR = Path(os.environ.get("WORD2VEC_DATA_DIR") or (Path(__file__).resolve().parents[1] / "data"))
MODELS_DIR = Path(os.environ.get("WORD2VEC_MODELS_DIR") or (Path(__file__).resolve().parents[1] / "models"))

sys.path.insert(0, str(ROOT / "Tokenization"))
from tokenizer_utils import load_tokenizer  # noqa: E402


def load_prepared_data():
    rows_path = DATA_DIR / "rows.jsonl"
    freq_path = DATA_DIR / "token_freq.json"
    meta_path = DATA_DIR / "meta.json"
    for p in (rows_path, freq_path, meta_path):
        if not p.exists():
            raise SystemExit(f"{p} not found -- run build_training_data.py first.")

    with rows_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    token_freq = {int(k): v for k, v in json.loads(freq_path.read_text(encoding="utf-8")).items()}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return rows, token_freq, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CBOW+attention word embedding model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--target-pairs-per-batch",
        type=int,
        default=4096,
        help="target (context, center) training pairs per batch -- rows-per-batch is "
        "derived per length-cluster from this budget (avg_tokens_in_cluster rows "
        "generate roughly this many pairs), so long-row clusters get fewer rows per "
        "batch and short-row clusters get more, keeping actual batch size roughly "
        "constant across clusters. Must stay well under PyTorch's efficient-attention "
        "65535 hard limit (see dataset.py's MAX_PAIRS_PER_BATCH backstop).",
    )
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--num-negative", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to a checkpoint .pt (from a previous run) to resume training from -- "
        "restores model/optimizer state and continues from the exact batch it left off "
        "at, since ClusterBatchSampler's per-epoch shuffle is seeded deterministically.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this Python environment. Run this script via the "
            "GPU venv's interpreter (see ../requirements.txt) -- the base environment's "
            "torch build is CPU-only."
        )
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    resume_ckpt = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise SystemExit(f"--resume checkpoint not found: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device)
        ckpt_args = resume_ckpt["args"]
        # Architecture-shaping args must match the checkpoint's or load_state_dict will
        # fail on a shape mismatch -- override rather than error, since these are the
        # only args that actually matter for compatibility.
        for key in ("embed_dim", "num_heads", "ff_dim"):
            if getattr(args, key) != ckpt_args[key]:
                print(
                    f"  --resume: overriding --{key.replace('_', '-')} "
                    f"{getattr(args, key)} -> {ckpt_args[key]} (must match checkpoint)"
                )
                setattr(args, key, ckpt_args[key])

    print("Loading prepared data...")
    rows, token_freq, meta = load_prepared_data()
    vocab_size = meta["tokenizer"]["vocab_size_actual"]
    cluster_ids = [r["cluster_id"] for r in rows]
    token_counts = [r["token_count"] for r in rows]
    print(f"  {len(rows):,} rows, vocab_size={vocab_size:,}")
    print(f"  cluster sizes: {meta['cluster_sizes']}")
    print(f"  script buckets: {meta['script_bucket_counts']}")

    print("Loading tokenizer + building negative-sampling table and subsample probabilities...")
    tok = load_tokenizer(meta["tokenizer"]["key"], meta["tokenizer"]["vocab_size"])
    negative_table = build_negative_sampling_table(token_freq, vocab_size)
    keep_prob = build_subsample_keep_prob(token_freq)

    dataset = RowDataset(rows)
    collator = CBOWCollator(tok, negative_table, keep_prob, num_negative=args.num_negative)
    batch_sampler = ClusterBatchSampler(
        cluster_ids, token_counts, target_pairs_per_batch=args.target_pairs_per_batch
    )
    print(f"  rows per batch by cluster: {batch_sampler.rows_per_batch}")

    model = CBOWAttention(
        vocab_size=vocab_size, embed_dim=args.embed_dim, num_heads=args.num_heads, ff_dim=args.ff_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    start_step = 0
    resume_batch_in_epoch = 0
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        start_step = resume_ckpt["step"]
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            start_epoch = resume_ckpt["epoch"]
            resume_batch_in_epoch = resume_ckpt["batch_in_epoch"]
            print(
                f"  resumed from {args.resume}: step={start_step:,}, epoch={start_epoch + 1}, "
                f"batch_in_epoch={resume_batch_in_epoch:,}"
            )
        else:
            # Pre-dates the --resume feature: only has model_state_dict/step/args.
            # Optimizer state (Adam's moment estimates) and epoch/batch position are
            # lost -- restarts optimizer from scratch and the current epoch from its
            # beginning, since there's no batch_in_epoch to resume from exactly.
            print(
                f"  resumed from {args.resume}: old-format checkpoint (no optimizer/epoch "
                f"state) -- restarting optimizer state and epoch {start_epoch + 1} from its "
                f"beginning; step counter continues from {start_step:,}"
            )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    step = start_step
    t0 = time.time()
    running_loss = 0.0
    running_count = 0

    for epoch in range(start_epoch, args.epochs):
        batch_sampler.set_epoch(epoch)
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        # Resuming mid-epoch: ClusterBatchSampler's shuffle is seeded by
        # (seed + epoch), so re-running set_epoch(epoch) reproduces the exact
        # same batch order. Fast-forward past the first `skip` batches by
        # consuming the sampler's index-lists directly (cheap: just shuffled
        # int lists) WITHOUT running them through the collator -- collating
        # (BPE re-tokenization + augmentation + subsampling per row) is the
        # expensive part, and running it for batches about to be discarded
        # made resuming late in an epoch take nearly as long as just
        # retraining that portion (25k skipped batches took 10+ minutes and
        # counting before this fix).
        skip = resume_batch_in_epoch if epoch == start_epoch else 0
        sampler_iter = iter(batch_sampler)
        if skip:
            print(f"  fast-forwarding {skip:,} batches to resume point (index-only)...")
            for _ in range(skip):
                next(sampler_iter)
        for batch_idx, indices in enumerate(sampler_iter, start=skip):
            batch = collator([dataset[i] for i in indices])
            if batch is None:
                continue
            context_ids, attention_mask, center_ids, negative_ids = batch
            context_ids = context_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            center_ids = center_ids.to(device, non_blocking=True)
            negative_ids = negative_ids.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = model(context_ids, attention_mask, center_ids, negative_ids)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_count += 1
            step += 1

            if step % args.log_every == 0:
                avg_loss = running_loss / running_count
                elapsed = time.time() - t0
                gpu_mem = torch.cuda.memory_allocated() / 1e6
                print(f"  step {step:>7}  loss {avg_loss:.4f}  "
                      f"{(step - start_step) / elapsed:.1f} steps/s  gpu_mem {gpu_mem:.0f}MB")
                running_loss = 0.0
                running_count = 0

            if step % args.checkpoint_every == 0:
                ckpt_path = MODELS_DIR / f"checkpoint_step{step}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_idx + 1,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"  saved {ckpt_path}")

    final_path = MODELS_DIR / "final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "epoch": args.epochs - 1,
            "batch_in_epoch": 0,
            "args": vars(args),
        },
        final_path,
    )
    print(f"\nDone. Wrote {final_path}")


if __name__ == "__main__":
    main()
