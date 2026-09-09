#!/usr/bin/env python
"""Tokenizes the shared corpus (word2vec_attention/data/rows.jsonl) with
this LM's own word-level vocabulary (build_vocab.py) and packs every
row's `<bos> word_ids... <eos>` sequence into one flat binary token
stream -- same "pack everything into one array, train on random
fixed-length slices of it" convention as nanoGPT, rather than
padding every row out to a fixed length individually (this corpus's rows
are short -- median ~7-13 words -- so per-row padding would waste most
of every batch on `<pad>`).

Token ids are stored as `uint16` (vocab is ~30,004 < 65,536) and written
incrementally, row by row, so this never holds the whole ~130M-token
stream in Python-list form in RAM.

Run via the base Python environment (no torch needed):

    python build_training_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROWS_PATH = ROOT / "Embeddings" / "word2vec_attention" / "data" / "rows.jsonl"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
VOCAB_PATH = DATA_DIR / "vocab.json"
TOKENS_PATH = DATA_DIR / "tokens.bin"
META_PATH = DATA_DIR / "tokens_meta.json"


def main() -> None:
    if not VOCAB_PATH.exists():
        raise SystemExit(f"{VOCAB_PATH} not found -- run build_vocab.py first.")
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    word_to_id = vocab["word_to_id"]
    pad_id, unk_id, bos_id, eos_id = (word_to_id[s] for s in vocab["specials"])

    print(f"Tokenizing {ROWS_PATH} -> {TOKENS_PATH} (vocab size {vocab['vocab_size']:,})...")
    n_rows = 0
    n_tokens = 0
    n_unk = 0
    with ROWS_PATH.open("r", encoding="utf-8") as f_in, TOKENS_PATH.open("wb") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            row = json.loads(line)
            ids = [bos_id]
            for w in row["text"].split():
                tok_id = word_to_id.get(w, unk_id)
                if tok_id == unk_id:
                    n_unk += 1
                ids.append(tok_id)
            ids.append(eos_id)

            arr = np.array(ids, dtype=np.uint16)
            arr.tofile(f_out)
            n_rows += 1
            n_tokens += len(ids)
            if n_rows % 1_000_000 == 0:
                print(f"  ...{n_rows:,} rows, {n_tokens:,} tokens written so far")

    print(f"Done: {n_rows:,} rows, {n_tokens:,} total tokens "
          f"({n_unk:,} <unk> = {n_unk / n_tokens:.2%}).")

    META_PATH.write_text(json.dumps({
        "n_tokens": n_tokens,
        "n_rows": n_rows,
        "dtype": "uint16",
        "pad_id": pad_id, "unk_id": unk_id, "bos_id": bos_id, "eos_id": eos_id,
        "vocab_size": vocab["vocab_size"],
    }, indent=2), encoding="utf-8")
    print(f"Wrote {TOKENS_PATH} ({TOKENS_PATH.stat().st_size / 1e6:.1f} MB) and {META_PATH}")


if __name__ == "__main__":
    main()
