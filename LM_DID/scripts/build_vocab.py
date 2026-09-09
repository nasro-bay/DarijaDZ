#!/usr/bin/env python
"""Builds a word-level vocabulary (not the project's shared BPE-20K
tokenizer -- "we will work at the word level first", per spec) from the
same prepared corpus cache word2vec_attention/cbow/skip-gram already
share (Embeddings/word2vec_attention/data/rows.jsonl), so this LM trains
on the same ~8.95M-row corpus as everything else in this project rather
than a separate sample.

Whitespace-split words (same convention as Dialect_Identification's
word_cluster/HMM notebooks), top VOCAB_SIZE by frequency kept, everything
else maps to <unk>. Four special tokens reserved at fixed low ids so
`<pad>=0` lines up with the model's `padding_idx`/`ignore_index`
convention:

    0 = <pad>   1 = <unk>   2 = <bos>   3 = <eos>

Run via the base Python environment (no torch needed):

    python build_vocab.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROWS_PATH = ROOT / "Embeddings" / "word2vec_attention" / "data" / "rows.jsonl"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "vocab.json"

VOCAB_SIZE = 30_000  # real words, not counting the 4 specials below
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def main() -> None:
    if not ROWS_PATH.exists():
        raise SystemExit(
            f"{ROWS_PATH} not found -- build the shared corpus cache first "
            f"(Embeddings/word2vec_attention/scripts/build_training_data.py)."
        )

    print(f"Counting word frequencies over {ROWS_PATH}...")
    counts: Counter[str] = Counter()
    n_rows = 0
    with ROWS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            counts.update(row["text"].split())
            n_rows += 1
            if n_rows % 1_000_000 == 0:
                print(f"  ...{n_rows:,} rows scanned, {len(counts):,} unique words so far")

    print(f"Done: {n_rows:,} rows, {len(counts):,} unique whitespace-split words.")

    most_common = [w for w, _ in counts.most_common(VOCAB_SIZE)]
    word_to_id = {tok: i for i, tok in enumerate(SPECIALS)}
    for i, word in enumerate(most_common):
        word_to_id[word] = len(SPECIALS) + i

    total_tokens = sum(counts.values())
    covered_tokens = sum(counts[w] for w in most_common)
    print(f"Vocab size (incl. specials): {len(word_to_id):,}")
    print(f"Coverage: {covered_tokens:,}/{total_tokens:,} tokens ({covered_tokens / total_tokens:.2%}) "
          f"are in-vocabulary; the rest map to <unk>.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump({"word_to_id": word_to_id, "specials": SPECIALS, "vocab_size": len(word_to_id)}, f, ensure_ascii=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
