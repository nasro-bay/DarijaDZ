#!/usr/bin/env python
"""Runs train_one_subset() across all of Part 1's subset sizes and writes
the results to data/part1_results.json. Separated from train.py so a
single subset can still be run/debugged standalone.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from train import train_one_subset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SUBSET_SIZES = [100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available -- run via the ai-gpu venv's python.exe")

    results = []
    for size in SUBSET_SIZES:
        result = train_one_subset(size, seed=0)
        results.append(result)
        (DATA_DIR / "part1_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    for r in results:
        print(f"{r['subset_tokens']:>10,} tokens -> held-out loss {r['heldout_ce_loss']:.4f} "
              f"(ppl {r['heldout_ppl']:.1f}), {r['train_seconds']:.1f}s")


if __name__ == "__main__":
    main()
