#!/usr/bin/env python
"""Tokenizes `data/corpus_10m.jsonl` into flat id arrays for the
co-occurrence counter, in one of two modes (both V=20,000 so the pieces-vs-
words comparison isn't confounded by vocabulary size):

- `pieces`: SentencePiece unigram-20K (the `_bigcorpus` tokenizer -- the
  same one the repo's CBOW/skip-gram embeddings use). No filtering: every
  piece id 0..19999 is kept as-is.
- `words`: lowercased `\\w+` tokens (Arabic letters, Latin letters incl.
  accents, digits; punctuation and emoji are dropped), vocabulary = the
  20,000 most frequent. Out-of-vocabulary words are *skipped without
  counting toward window distance* (the reference GloVe implementation's
  behaviour), not replaced by an <unk> id.

Outputs per mode (in `data/`): `tokens_<mode>.npy` (uint16, all comments
concatenated), `doclens_<mode>.npy` (int32 tokens per comment -- windows
never cross a comment boundary), and for `words` also `vocab_words.json`.

Run via the GPU venv's Python:
    ".../ai-gpu/Scripts/python.exe" -u tokenize_corpus.py --mode pieces
    ".../ai-gpu/Scripts/python.exe" -u tokenize_corpus.py --mode words
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CORPUS = DATA_DIR / "corpus_10m.jsonl"
SP_MODEL = ROOT / "Tokenization" / "models_bigcorpus" / "sentencepiece" / "unigram_20000.model"
VOCAB_SIZE = 20_000
WORD_RE = re.compile(r"\w+")


def iter_texts():
    with CORPUS.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def tokenize_pieces() -> None:
    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
    assert sp.vocab_size() == VOCAB_SIZE, sp.vocab_size()
    flat: list[np.ndarray] = []
    doclens: list[np.ndarray] = []
    batch: list[str] = []
    n_docs = 0
    t0 = time.time()

    def flush() -> None:
        encoded = sp.encode(batch, out_type=int, num_threads=8)
        lens = np.fromiter((len(e) for e in encoded), dtype=np.int32, count=len(encoded))
        flat.append(np.fromiter((t for e in encoded for t in e), dtype=np.uint16, count=int(lens.sum())))
        doclens.append(lens)

    for text in iter_texts():
        batch.append(text)
        n_docs += 1
        if len(batch) == 50_000:
            flush()
            batch = []
            if n_docs % 1_000_000 == 0:
                print(f"  {n_docs:,} comments ({time.time() - t0:.0f}s)")
    if batch:
        flush()

    tokens = np.concatenate(flat)
    lens = np.concatenate(doclens)
    np.save(DATA_DIR / "tokens_pieces.npy", tokens)
    np.save(DATA_DIR / "doclens_pieces.npy", lens)
    print(f"pieces: {len(lens):,} comments, {len(tokens):,} tokens, {len(np.unique(tokens)):,} distinct ids")


def tokenize_words() -> None:
    print("Pass 1/2: counting words")
    counts: Counter[str] = Counter()
    t0 = time.time()
    for i, text in enumerate(iter_texts()):
        counts.update(WORD_RE.findall(text.lower()))
        if (i + 1) % 1_000_000 == 0:
            print(f"  {i + 1:,} comments ({time.time() - t0:.0f}s)")
    vocab = [w for w, _ in counts.most_common(VOCAB_SIZE)]
    word_id = {w: i for i, w in enumerate(vocab)}
    total_words = sum(counts.values())
    covered = sum(counts[w] for w in vocab)
    print(f"{len(counts):,} distinct words, {total_words:,} tokens; top-{VOCAB_SIZE:,} cover "
          f"{covered / total_words:.2%} of tokens")

    print("Pass 2/2: encoding")
    flat: list[np.ndarray] = []
    doclens: list[int] = []
    buf: list[int] = []
    t0 = time.time()
    for i, text in enumerate(iter_texts()):
        ids = [word_id[w] for w in WORD_RE.findall(text.lower()) if w in word_id]
        buf.extend(ids)
        doclens.append(len(ids))
        if len(buf) > 20_000_000:
            flat.append(np.asarray(buf, dtype=np.uint16))
            buf = []
        if (i + 1) % 1_000_000 == 0:
            print(f"  {i + 1:,} comments ({time.time() - t0:.0f}s)")
    if buf:
        flat.append(np.asarray(buf, dtype=np.uint16))

    tokens = np.concatenate(flat)
    lens = np.asarray(doclens, dtype=np.int32)
    np.save(DATA_DIR / "tokens_words.npy", tokens)
    np.save(DATA_DIR / "doclens_words.npy", lens)
    (DATA_DIR / "vocab_words.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print(f"words: {len(lens):,} comments, {len(tokens):,} in-vocab tokens "
          f"({int((lens == 0).sum()):,} comments empty after OOV drop)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pieces", "words"], required=True)
    args = parser.parse_args()
    if args.mode == "pieces":
        tokenize_pieces()
    else:
        tokenize_words()


if __name__ == "__main__":
    main()
