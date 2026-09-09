# LM_DiD

A decoder-only Transformer language model, trained word-level on this
project's shared Darija corpus, built with the architectural tweaks that
became standard after "Attention Is All You Need" (2017): RoPE, pre-norm
with RMSNorm, no biases, a gated (SwiGLU) feedforward block, fused/fast
attention via `F.scaled_dot_product_attention`, tiling-friendly shapes,
and a z-loss term stabilizing the output softmax. See `scripts/model.py`'s
module docstring for exactly which paper/technique each choice comes
from, and `plan.md` (internal, gitignored) for the full design rationale.

**Purpose**: eventually the shared next-token predictor for an LM-based
dialect-ID approach discussed for `Dialect_Identification/` (project a
word's known class distribution through this model's predicted
next-token distribution, weighted-sum the result into a class vote) --
this folder is the language model itself; it isn't wired into that
downstream classification yet.

## Layout

```
LM_DiD/
  scripts/
    model.py               # the architecture (see its docstring)
    build_vocab.py          # word-level vocabulary from the shared corpus
    build_training_data.py  # tokenizes + packs the corpus into tokens.bin
    dataset.py              # random fixed-length-slice sampling over tokens.bin
    train.py                # training loop
  data/                     # gitignored: vocab.json, tokens.bin, tokens_meta.json
  models/                   # gitignored: checkpoints
```

See `guide.md` (internal, gitignored) for exact run commands.
