#!/usr/bin/env python
"""Selects the GloVe training corpus: the N longest comments (default 10M)
from the big corpus (`Embeddings/word2vec_attention/data_bigcorpus/rows.jsonl`,
16.3M comments, already cleaned/deduped).

"Favor long comments, then short ones": ranks every comment by its
`token_count` (SentencePiece pieces, already stored on each row) descending
and keeps the top N. Short comments (one word, emoji only) contribute almost
no within-window co-occurrence evidence, so dropping the shortest ~40% costs
little and every kept comment can fill a window. Ties at the length cutoff
are broken by file order, so the selection is deterministic.

Two cheap byte-level passes (no JSON parsing): pass 1 reads only each row's
`token_count`; pass 2 copies the selected raw lines, in original file order,
to `data/corpus_10m.jsonl`. Writes `data/corpus_meta.json` with the cutoff
and length statistics.

Run via the GPU venv's Python (needs only numpy):
    ".../ai-gpu/Scripts/python.exe" -u select_corpus.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "Embeddings" / "word2vec_attention" / "data_bigcorpus" / "rows.jsonl"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT = DATA_DIR / "corpus_10m.jsonl"
TC_KEY = b'"token_count": '


def read_token_counts() -> np.ndarray:
    counts: list[int] = []
    t0 = time.time()
    with SRC.open("rb") as f:
        for i, line in enumerate(f):
            pos = line.rfind(TC_KEY)
            if pos < 0:
                raise SystemExit(f"row {i} has no token_count field -- unexpected rows.jsonl format")
            start = pos + len(TC_KEY)
            end = start
            while line[end:end + 1].isdigit():
                end += 1
            counts.append(int(line[start:end]))
            if (i + 1) % 2_000_000 == 0:
                print(f"  pass 1: {i + 1:,} rows ({time.time() - t0:.0f}s)")
    return np.asarray(counts, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000_000, help="comments to keep (longest first)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Pass 1: reading token counts from {SRC}")
    tc = read_token_counts()
    total = len(tc)
    if args.n >= total:
        raise SystemExit(f"--n={args.n:,} >= corpus size {total:,}; nothing to select")

    cutoff = int(np.sort(tc)[::-1][args.n - 1])
    keep = tc > cutoff
    need = args.n - int(keep.sum())
    tie_idx = np.flatnonzero(tc == cutoff)[:need]
    keep[tie_idx] = True
    assert int(keep.sum()) == args.n

    kept_tc, dropped_tc = tc[keep], tc[~keep]
    stats = {
        "source": str(SRC.relative_to(ROOT)),
        "total_rows": int(total),
        "selected_rows": int(args.n),
        "length_cutoff_token_count": cutoff,
        "selected_mean_tokens": float(kept_tc.mean()),
        "selected_total_tokens": int(kept_tc.sum()),
        "selected_min_tokens": int(kept_tc.min()),
        "dropped_rows": int(total - args.n),
        "dropped_mean_tokens": float(dropped_tc.mean()),
        "dropped_max_tokens": int(dropped_tc.max()),
    }
    print(json.dumps(stats, indent=2))

    print(f"Pass 2: writing selected rows to {OUT}")
    t0 = time.time()
    written = 0
    with SRC.open("rb") as fin, OUT.open("wb") as fout:
        for i, line in enumerate(fin):
            if keep[i]:
                fout.write(line)
                written += 1
            if (i + 1) % 4_000_000 == 0:
                print(f"  pass 2: {i + 1:,} rows read, {written:,} written ({time.time() - t0:.0f}s)")
    assert written == args.n, f"wrote {written:,}, expected {args.n:,}"

    (DATA_DIR / "corpus_meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Done: {written:,} comments -> {OUT} ({OUT.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
