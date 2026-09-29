#!/usr/bin/env python
"""Trains one Part-1 data-scaling-law data point: a fresh DecoderLM
(model.py) on the first `--subset-tokens` tokens of `data/train_tokens.bin`,
for a fixed `--epochs` passes over that subset (default 4, per the lab's
repetition policy), then reports held-out loss on the fixed, disjoint
`data/heldout_tokens.bin` set built by `build_data.py`.

Applies GPU_optimization.md's general principles (this is exactly the kind
of many-small-runs workload where they compound): fused AdamW, FP16
autocast (this machine has no BF16 tensor cores), linear-warmup + cosine
LR decay, batch size / seq_len already multiples of 64.

Run via the GPU venv's Python:
    ".../ai-gpu/Scripts/python.exe" train.py --subset-tokens 1000000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DecoderLM  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WORD2VEC_CKPT = ROOT / "Embeddings" / "word2vec" / "skip-gram" / "models" / "final.pt"


class WindowedTokenDataset(Dataset):
    """Non-overlapping (seq_len+1)-token windows over a flat token array --
    deterministic, not LM_DiD's random-sampling scheme, specifically so
    "N epochs" means every token is seen (about) N times, exactly -- the
    lab's repetition policy needs that to be a real, checkable guarantee,
    not an artifact of how many random windows happened to be drawn.
    """

    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.seq_len = seq_len
        n_windows = len(tokens) // (seq_len + 1)
        usable = n_windows * (seq_len + 1)
        self.windows = tokens[:usable].reshape(n_windows, seq_len + 1).astype(np.int64)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        row = self.windows[idx]
        return torch.from_numpy(row[:-1].copy()), torch.from_numpy(row[1:].copy())


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype) -> float:
    model.eval()
    total_loss, total_batches = 0.0, 0
    for input_ids, targets in loader:
        input_ids, targets = input_ids.to(device), targets.to(device)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            _, ce_loss, _ = model(input_ids, targets)
        total_loss += ce_loss.item()
        total_batches += 1
    model.train()
    return total_loss / total_batches


def compute_subset_and_epochs(d_target: float, pool_tokens: int, max_epochs: int = 4) -> tuple[int, int]:
    """For Part 3 (IsoFLOPs): a target token count `d_target` implied by a fixed compute
    budget (C=6*N*D) can exceed the available training pool -- when it does, train on the
    *full* pool for `round(d_target / pool_tokens)` epochs instead (capped at `max_epochs`,
    per the lab's own repetition-limit guidance: repeating data is ~free up to ~4 epochs,
    then rapidly worthless). Returns (subset_tokens, epochs); the actual tokens trained on
    is `subset_tokens * epochs`, which may differ slightly from `d_target` due to rounding
    to a whole number of epochs -- callers should use the actual, not the target, for any
    downstream FLOPs/compute accounting.
    """
    if d_target <= pool_tokens:
        return int(d_target), 1
    epochs = max(1, min(max_epochs, round(d_target / pool_tokens)))
    return pool_tokens, epochs


def make_lr_schedule(optimizer, total_steps: int, warmup_ratio: float = 0.1):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_subset(
    subset_tokens: int,
    *,
    epochs: int = 4,
    seq_len: int = 128,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    init_word2vec: bool = True,
    seed: int = 0,
    d_model: int = 128,
    num_heads: int = 2,
    num_layers: int = 4,
    verbose: bool = True,
) -> dict:
    device = torch.device("cuda")
    torch.manual_seed(seed)

    meta = json.loads((DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    train_tokens = np.fromfile(DATA_DIR / "train_tokens.bin", dtype=np.uint16)
    heldout_tokens = np.fromfile(DATA_DIR / "heldout_tokens.bin", dtype=np.uint16)
    assert subset_tokens <= len(train_tokens), (
        f"subset_tokens={subset_tokens} exceeds available train pool ({len(train_tokens)})"
    )

    subset = train_tokens[:subset_tokens]
    train_ds = WindowedTokenDataset(subset, seq_len)
    heldout_ds = WindowedTokenDataset(heldout_tokens, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    heldout_loader = DataLoader(heldout_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    if total_steps == 0:
        raise ValueError(
            f"subset_tokens={subset_tokens} produces 0 batches at batch_size={batch_size}, "
            f"seq_len={seq_len} -- too small."
        )

    model = DecoderLM(
        vocab_size=meta["tokenizer_vocab_size"],
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=seq_len,
        pad_id=meta["pad_id"],
    ).to(device)

    word2vec_report = None
    if init_word2vec:
        word2vec_report = model.load_word2vec_init(WORD2VEC_CKPT)

    n_params = sum(p.numel() for p in model.parameters())
    n_nonembed = n_params - model.tok_embedding.weight.numel() - model.lm_head.weight.numel()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
    scheduler = make_lr_schedule(optimizer, total_steps)
    scaler = torch.amp.GradScaler("cuda")
    autocast_dtype = torch.float16  # this GPU (Turing) has no BF16 tensor cores -- see GPU_optimization.md

    if verbose:
        print(f"[{subset_tokens:>9,} tokens] {n_params:,} params ({n_nonembed:,} non-embedding), "
              f"{steps_per_epoch:,} steps/epoch x {epochs} epochs = {total_steps:,} steps, "
              f"word2vec_init={word2vec_report}")

    step = 0
    t0 = time.time()
    model.train()
    for epoch in range(epochs):
        for input_ids, targets in train_loader:
            input_ids, targets = input_ids.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                _, ce_loss, total_loss = model(input_ids, targets)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

    elapsed = time.time() - t0
    heldout_loss = evaluate(model, heldout_loader, device, autocast_dtype)

    result = {
        "subset_tokens": subset_tokens,
        "epochs": epochs,
        "total_steps": total_steps,
        "n_params": n_params,
        "n_params_non_embedding": n_nonembed,
        "heldout_ce_loss": heldout_loss,
        "heldout_ppl": math.exp(heldout_loss),
        "train_seconds": elapsed,
        "word2vec_init": word2vec_report,
    }
    if verbose:
        print(f"[{subset_tokens:>9,} tokens] held-out loss={heldout_loss:.4f} "
              f"ppl={result['heldout_ppl']:.1f} ({elapsed:.1f}s)")

    # Explicit cleanup: a sweep calls this repeatedly in one process (run_sweep.py, or a
    # notebook cell loop) -- without this, each successive model/optimizer/dataloader
    # lingers until Python's GC gets to it, and on a 6GB card that accumulation is exactly
    # what turned a real OOM into a confusing one (see plan.md's Part 2 section: the error
    # reported ~11GB "allocated" on a 6GB card, which is cross-run accumulation, not this
    # one run's real footprint).
    del model, optimizer, scheduler, scaler, train_loader, heldout_loader, train_ds, heldout_ds
    torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-tokens", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no-word2vec-init", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available -- run via the ai-gpu venv's python.exe")

    out = train_one_subset(
        args.subset_tokens,
        epochs=args.epochs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        init_word2vec=not args.no_word2vec_init,
        seed=args.seed,
    )
    print(json.dumps(out, indent=2))
