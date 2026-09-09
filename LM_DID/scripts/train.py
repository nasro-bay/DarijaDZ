#!/usr/bin/env python
"""Trains the decoder-only LM (model.py) on the packed token stream
(build_training_data.py's tokens.bin) via ordinary next-token prediction
(full softmax + z-loss, see model.py's docstring).

Run via the GPU venv's Python:

    ".../ai-gpu/Scripts/python.exe" train.py --steps 2000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import PackedTokenDataset, load_meta
from model import DecoderLM

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LM_DiD decoder-only word-level LM")
    parser.add_argument("--steps", type=int, default=None, help="total steps to run (default: run until Ctrl+C)")
    # Default 64 (at seq_len=128) OOMs this project's 6GB card -- full-vocab
    # softmax cross-entropy over batch_size*seq_len positions x 30,016
    # classes is the real memory cost here, not the model itself (12.5M
    # params). 16 is the verified-working default (real smoke test: loss
    # 8.98->5.98 over 500 steps, ~464MB GPU memory, plenty of headroom
    # left -- 32 is probably also safe, untested).
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--epoch-size", type=int, default=50_000, help="random slices per logged 'epoch' (see dataset.py)")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this Python environment. Run this script via the "
            "GPU venv's interpreter -- the base environment's torch build is CPU-only."
        )
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    meta = load_meta()
    print(f"Loaded tokens.bin: {meta['n_tokens']:,} tokens, vocab_size={meta['vocab_size']:,}")

    dataset = PackedTokenDataset(seq_len=args.seq_len, epoch_size=args.epoch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DecoderLM(
        vocab_size=meta["vocab_size"],
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        pad_id=meta["pad_id"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, padded_vocab_size={model.padded_vocab_size:,} "
          f"(real vocab {model.real_vocab_size:,}), head_dim={model.head_dim}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise SystemExit(f"--resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
        print(f"  resumed from {args.resume}: step={start_step:,}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    step = start_step
    t0 = time.time()
    running_loss = 0.0
    running_count = 0

    print(f"\n=== Training (seq_len={args.seq_len}, batch_size={args.batch_size}) ===")
    done = False
    while not done:
        for input_ids, targets in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            _, loss = model(input_ids, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_count += 1
            step += 1

            if step % args.log_every == 0:
                avg_loss = running_loss / running_count
                elapsed = time.time() - t0
                gpu_mem = torch.cuda.memory_allocated() / 1e6
                print(f"  step {step:>7}  loss {avg_loss:.4f}  ppl {torch.exp(torch.tensor(avg_loss)):.1f}  "
                      f"{(step - start_step) / elapsed:.2f} steps/s  gpu_mem {gpu_mem:.0f}MB")
                running_loss = 0.0
                running_count = 0

            if step % args.checkpoint_every == 0:
                ckpt_path = MODELS_DIR / f"checkpoint_step{step}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "args": vars(args),
                }, ckpt_path)
                print(f"  saved {ckpt_path}")

            if args.steps is not None and step >= start_step + args.steps:
                done = True
                break

    final_path = MODELS_DIR / "final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "args": vars(args),
    }, final_path)
    print(f"\nDone. Wrote {final_path}")


if __name__ == "__main__":
    main()
