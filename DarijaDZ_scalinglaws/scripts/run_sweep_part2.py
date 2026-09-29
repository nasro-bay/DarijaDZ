#!/usr/bin/env python
"""Runs Part 2 (model scaling law): one fixed data budget (D=50M tokens),
5 models at a fixed aspect ratio (d_model/n_layer=32), random init (NOT
word2vec-initialized -- see plan.md's Part 2 section for why: word2vec's
checkpoint is pinned to embed_dim=128, so it can only fit one point in a
d_model-varying sweep; using it for just that one point would confound
"bigger model" with "this one model also got a pretrained head start").

Writes results to data/part2_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from train import train_one_subset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIXED_D = 50_000_000

# (d_model, num_heads, num_layers) -- head_dim=32 throughout (num_heads=d_model/32),
# aspect ratio d_model/n_layer=32 held constant. Chosen (not the originally-sketched
# {64,128,192,256,320}/head_dim=64 family) specifically so every point stays comfortably
# above the ~20-tokens/param compute-optimal line at D=50M -- see plan.md for the actual
# margins (151.7x down to 1.6x); the wider family would have put the two largest points
# deep in the data-limited regime (0.4x, 0.2x), contaminating the fit for the wrong reason.
MODEL_CONFIGS = [
    (32, 1, 1),
    (64, 2, 2),
    (96, 3, 3),
    (128, 4, 4),
    (160, 5, 5),
]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available -- run via the ai-gpu venv's python.exe")

    results = []
    for d_model, num_heads, num_layers in MODEL_CONFIGS:
        result = train_one_subset(
            FIXED_D,
            epochs=1,
            batch_size=64,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            init_word2vec=False,
            seed=0,
        )
        result["d_model"] = d_model
        result["num_heads"] = num_heads
        result["num_layers"] = num_layers
        results.append(result)
        (DATA_DIR / "part2_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    for r in results:
        print(f"d_model={r['d_model']:4d} n_layer={r['num_layers']:2d}  "
              f"total={r['n_params']:,}  nonembed={r['n_params_non_embedding']:,}  "
              f"held-out loss={r['heldout_ce_loss']:.4f}  ({r['train_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
