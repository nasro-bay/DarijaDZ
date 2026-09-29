#!/usr/bin/env python
"""Builds Part 1's data: a held-out eval token stream (carved off FIRST,
so it can never leak into any training subset) and a packed training
token stream long enough for the largest (10M-token) subset, both
tokenized with `unigram_20000` -- **not** `bpe_10000`, the scaling-laws
lab doc's original suggestion. Deviation is deliberate: `unigram_20000`
is the exact tokenizer `Embeddings/word2vec/skip-gram`'s reused
embedding checkpoint was trained against (see
`Tokenization/models_bigcorpus/sentencepiece/unigram_20000.model` and
`model.py`'s `load_word2vec_init`) -- using any other tokenizer would
make that checkpoint's vocabulary ids meaningless here.

Source: `Data/youtube_corpus_shuffled.jsonl` (already a seeded, full-corpus
shuffle of YouTube+TikTok -- see `Data/scripts/build_combined_dataset.py`).
Reading it in order and taking prefixes is therefore equivalent to an
independent random sample, with the added property that every subset
size's tokens are a strict prefix of the next larger subset's.

Run via the GPU venv's Python (only needs `sentencepiece`, no torch):
    ".../ai-gpu/Scripts/python.exe" build_data.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Tokenization"))

# Must be set before importing tokenizer_utils -- picks the bigcorpus
# tokenizer tree (see module docstring) instead of the default models/.
os.environ["TOKENIZER_MODELS_DIR"] = str(ROOT / "Tokenization" / "models_bigcorpus")
from tokenizer_utils import load_tokenizer  # noqa: E402

CORPUS_PATH = ROOT / "Data" / "youtube_corpus_shuffled.jsonl"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"

HELDOUT_TARGET_TOKENS = 300_000  # ~3x the smallest training subset (100K)
SUBSET_SIZES = [100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]  # Part 1 (data scaling)
PART2_FIXED_D = 50_000_000  # Part 2 (model scaling) -- one fixed data budget, model size varies
TRAIN_TARGET_TOKENS = PART2_FIXED_D + 50_000  # small buffer past the largest ceiling needed;
# Part 1's subsets are prefixes of this same pool (same corpus, same read order), so
# extending the pool for Part 2 doesn't change Part 1's already-recorded results.

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def _pack_docs(doc_iter, target_tokens: int, encode) -> tuple[np.ndarray, int]:
    """Tokenizes docs from `doc_iter`, wraps each as <s> ids... </s>,
    concatenates into one flat uint16 stream (nanoGPT-style packing --
    same convention as LM_DiD, see its plan.md) until `target_tokens` is
    reached or `doc_iter` is exhausted. Returns (array, n_docs_used)."""
    buf: list[int] = []
    n_docs = 0
    for line in doc_iter:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = rec.get("text")
        if not text:
            continue
        ids = encode(text)
        if not ids:
            continue
        buf.append(BOS_ID)
        buf.extend(ids)
        buf.append(EOS_ID)
        n_docs += 1
        if len(buf) >= target_tokens:
            break
    arr = np.array(buf, dtype=np.uint16)
    return arr, n_docs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = load_tokenizer("unigram", 20_000)
    print(f"Tokenizer: {spec.label}, vocab_size_actual={spec.vocab_size_actual}")

    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        # Held-out set FIRST -- carved off the front of the shuffled stream
        # before any training data is read, so it's structurally impossible
        # for a training subset (all of which start from the same point
        # right after this) to contain a held-out document.
        heldout_arr, heldout_docs = _pack_docs(f, HELDOUT_TARGET_TOKENS, spec.encode)
        print(f"Held-out: {heldout_docs:,} docs, {len(heldout_arr):,} tokens")

        train_arr, train_docs = _pack_docs(f, TRAIN_TARGET_TOKENS, spec.encode)
        print(f"Train pool: {train_docs:,} docs, {len(train_arr):,} tokens")

    if len(train_arr) < SUBSET_SIZES[-1]:
        raise SystemExit(
            f"Train pool ({len(train_arr):,} tokens) is smaller than the largest "
            f"subset ({SUBSET_SIZES[-1]:,}) -- corpus exhausted or target too high."
        )

    heldout_arr.tofile(OUT_DIR / "heldout_tokens.bin")
    train_arr.tofile(OUT_DIR / "train_tokens.bin")

    meta = {
        "tokenizer_key": "unigram",
        "tokenizer_vocab_size": spec.vocab_size_actual,
        "tokenizer_model_dir": "Tokenization/models_bigcorpus/sentencepiece",
        "pad_id": PAD_ID,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "heldout_docs": heldout_docs,
        "heldout_tokens": int(len(heldout_arr)),
        "train_pool_docs": train_docs,
        "train_pool_tokens": int(len(train_arr)),
        "subset_sizes": SUBSET_SIZES,
        "source_corpus": str(CORPUS_PATH.relative_to(ROOT)),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'meta.json'}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
