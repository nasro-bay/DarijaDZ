#!/usr/bin/env python
"""Trains GloVe (Pennington et al. 2014) on a co-occurrence matrix built by
`build_cooccurrence.py`.

Objective (paper Eq. 8), summed over the nonzero entries of X only:

    J = sum_{i,j} f(X_ij) * (w_i . w~_j + b_i + b~_j - log X_ij)^2
    f(x) = (x / x_max)^alpha  if x < x_max  else 1        (x_max=100, alpha=3/4)

Parameters: two vector sets W, W~ (V x d) and two bias vectors b, b~. The
released embedding is W + W~ (paper §4.2). Optimizer: AdaGrad, lr 0.05,
50 iterations (paper: 50 for vectors under 300 dims).

One deliberate deviation from the paper's procedure: the paper samples ONE
nonzero entry at a time; here entries are processed in large batches on the
GPU (each iteration still passes over every nonzero entry exactly once, in
a fresh random order). The loss is a SUM over the batch, not a mean, so
each parameter's gradient is the sum of its entries' per-entry gradients --
close to the paper's sequential per-entry updates in effect, and it keeps
AdaGrad's step sizes in the same regime (a mean would shrink gradients far
below AdaGrad's initial accumulator and stall learning). Initialization
(uniform in +-0.5/d) and AdaGrad's initial accumulator of 1.0 follow the
reference C implementation's conventions -- the paper itself doesn't state
them.

Run via the GPU venv's Python:
    ".../ai-gpu/Scripts/python.exe" -u train.py --mode pieces --dim 64
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
RECORD = np.dtype([("i", "<i4"), ("j", "<i4"), ("x", "<f4")])


def load_shards(mode: str, device: torch.device, on_device: bool):
    shards = []
    total = 0
    for path in sorted((DATA_DIR / f"cooc_{mode}").glob("shard_*.bin")):
        rec = np.fromfile(path, dtype=RECORD)
        if len(rec) == 0:
            continue
        tensors = (
            torch.from_numpy(rec["i"].copy()),
            torch.from_numpy(rec["j"].copy()),
            torch.from_numpy(rec["x"].copy()),
        )
        if on_device:
            tensors = tuple(t.to(device) for t in tensors)
        shards.append(tensors)
        total += len(rec)
    return shards, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pieces", "words"], required=True)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--x-max", type=float, default=100.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=131_072)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="", help="suffix for the output filename (e.g. to keep sweeps apart)")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    meta = json.loads((DATA_DIR / f"cooc_{args.mode}_meta.json").read_text(encoding="utf-8"))
    V = meta["V"]
    on_device = meta["nnz"] * 12 < 3.0e9  # keep entries resident on the 6GB GPU only if they leave headroom
    shards, nnz = load_shards(args.mode, device, on_device)
    print(f"mode={args.mode} V={V:,} dim={args.dim} nnz={nnz:,} ({len(shards)} shards, "
          f"{'GPU' if on_device else 'CPU'}-resident), batch={args.batch_size:,}, "
          f"~{nnz // args.batch_size:,} steps/epoch")

    d = args.dim
    params = [((torch.rand(V, d, device=device) - 0.5) / d).requires_grad_() for _ in range(2)]
    params += [((torch.rand(V, device=device) - 0.5) / d).requires_grad_() for _ in range(2)]
    W, Wt, b, bt = params
    # torch's Adagrad has no CUDA-fused path (fused=True raises at the first step); foreach (the
    # default on CUDA) already batches the per-parameter updates.
    opt = torch.optim.Adagrad(params, lr=args.lr, initial_accumulator_value=1.0, foreach=True)

    curve = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        for si in torch.randperm(len(shards)).tolist():
            i_all, j_all, x_all = shards[si]
            perm = torch.randperm(len(x_all), device=i_all.device)
            i_all, j_all, x_all = i_all[perm], j_all[perm], x_all[perm]
            for s in range(0, len(x_all), args.batch_size):
                i = i_all[s:s + args.batch_size].to(device, non_blocking=True)
                j = j_all[s:s + args.batch_size].to(device, non_blocking=True)
                x = x_all[s:s + args.batch_size].to(device, non_blocking=True)
                pred = (W.index_select(0, i) * Wt.index_select(0, j)).sum(1) + b.index_select(0, i) + bt.index_select(0, j)
                diff = pred - torch.log(x)
                f = torch.clamp((x / args.x_max) ** args.alpha, max=1.0)
                loss = (f * diff * diff).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_sum += loss.detach().double()
        mean_loss = loss_sum.item() / nnz
        curve.append(mean_loss)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:>3}/{args.epochs}  mean weighted sq. error per entry {mean_loss:.5f}  "
                  f"({time.time() - t0:.0f}s)")

    MODELS_DIR.mkdir(exist_ok=True)
    out = MODELS_DIR / f"glove_{args.mode}_d{d}{args.tag}.npz"
    W_, Wt_ = W.detach().cpu().numpy(), Wt.detach().cpu().numpy()
    np.savez(out, embeddings=W_ + Wt_, W=W_, Wt=Wt_, b=b.detach().cpu().numpy(), bt=bt.detach().cpu().numpy())
    (out.with_suffix(".json")).write_text(json.dumps({"args": vars(args), "loss_curve": curve, "nnz": nnz,
                                                       "train_seconds": time.time() - t0}, indent=2), encoding="utf-8")
    print(f"Saved {out} (embeddings = W + W~, shape {W_.shape})")


if __name__ == "__main__":
    main()
