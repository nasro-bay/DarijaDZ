#!/usr/bin/env python
"""Trains the classic CBOW model (see ../../plan.md and model.py). Reads
the prepared corpus cache from ../../../word2vec_attention/data/ (shared
with word2vec_attention, not rebuilt here -- see common/data_utils.py).

Run via the GPU venv's Python (see ../../requirements.txt):

    ".../ai-gpu/Scripts/python.exe" train.py --epochs 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import CBOWCollator, RowDataset
from model import CBOW

ROOT = Path(__file__).resolve().parents[3]  # Embeddings/
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

sys.path.insert(0, str(ROOT / "word2vec"))
from common.data_utils import (  # noqa: E402
    build_negative_sampling_table,
    build_subsample_keep_prob,
    load_rows,
    load_tok,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the classic CBOW word embedding model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rows-per-batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--num-negative", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to a checkpoint .pt (from a previous run) to resume training from -- "
        "restores model/optimizer state and continues from the exact batch it left off at.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this Python environment. Run this script via the "
            "GPU venv's interpreter -- the base environment's torch build is CPU-only."
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
        if args.embed_dim != ckpt_args["embed_dim"]:
            print(f"  --resume: overriding --embed-dim {args.embed_dim} -> {ckpt_args['embed_dim']} (must match checkpoint)")
            args.embed_dim = ckpt_args["embed_dim"]

    print("Loading prepared data (shared with word2vec_attention)...")
    rows, token_freq, meta = load_rows()
    vocab_size = meta["tokenizer"]["vocab_size_actual"]
    print(f"  {len(rows):,} rows, vocab_size={vocab_size:,}")

    print("Loading tokenizer + building negative-sampling table and subsample probabilities...")
    tok = load_tok()
    negative_table = build_negative_sampling_table(token_freq, vocab_size)
    keep_prob = build_subsample_keep_prob(token_freq)

    dataset = RowDataset(rows)
    collator = CBOWCollator(tok, negative_table, keep_prob, num_negative=args.num_negative)

    model = CBOW(vocab_size=vocab_size, embed_dim=args.embed_dim).to(device)
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
            print(f"  resumed from {args.resume}: step={start_step:,}, epoch={start_epoch + 1}, batch_in_epoch={resume_batch_in_epoch:,}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    step = start_step
    t0 = time.time()
    running_loss = 0.0
    running_count = 0

    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        # Resuming mid-epoch: DataLoader's shuffle isn't seeded per-epoch
        # the way word2vec_attention's ClusterBatchSampler is, so an exact
        # batch-index resume isn't reproducible here -- skip `resume_batch_
        # in_epoch` batches' worth of *row indices* before collation to land
        # roughly back where we left off instead. Good enough given each
        # epoch is a fresh shuffle anyway; unlike word2vec_attention there's
        # no cluster-order dependency to preserve exactly.
        #
        # This must be a true index-level skip, not "iterate the DataLoader
        # and discard the first N batches" -- that alternative (this
        # project's original approach, until a real resume stalled with no
        # GPU activity for ~80 minutes on a ~15k-batch skip) still pays the
        # full collation cost (tokenization, subsampling, negative-sampling)
        # for every skipped batch, it just throws the result away.
        # torch.randperm + a plain index list passed as `sampler=` skips the
        # already-consumed prefix as an O(1) slice instead.
        skip_batches = resume_batch_in_epoch if epoch == start_epoch else 0
        indices = torch.randperm(len(dataset)).tolist()
        if skip_batches:
            skip_rows = skip_batches * args.rows_per_batch
            indices = indices[skip_rows:]
            print(f"  resuming mid-epoch: skipped {skip_batches:,} batches ({skip_rows:,} rows) instantly, no collation")
        loader = DataLoader(
            dataset, batch_size=args.rows_per_batch, sampler=indices,
            collate_fn=collator, drop_last=False,
        )
        for local_batch_idx, batch in enumerate(loader):
            batch_idx = skip_batches + local_batch_idx
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
                      f"{(step - start_step) / elapsed:.1f} steps/s  "
                      f"pairs/batch {context_ids.shape[0]}  gpu_mem {gpu_mem:.0f}MB")
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
