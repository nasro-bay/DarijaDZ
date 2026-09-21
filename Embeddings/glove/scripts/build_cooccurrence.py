#!/usr/bin/env python
"""Builds the GloVe co-occurrence matrix X from the tokenized corpus
(`tokenize_corpus.py`'s output) -- paper §4.2: a symmetric window of
`--window` words (default 10) either side of the target word, where a pair
d words apart contributes 1/d. Windows never cross a comment boundary.

Implementation notes:
- Vectorized numpy, no Python-level pair loop: for each offset d in
  1..window, all in-window (position, position+d) pairs of a ~2M-token chunk
  are counted at once (`np.unique` on pair keys) and added, weighted 1/d, to
  a dense V x V float32 accumulator (V=20,000 -> 1.6 GB).
- Only the *upper triangle* is accumulated (key = min(a,b)*V + max(a,b));
  the symmetric matrix is recovered at extraction time (off-diagonal (i,j)
  emitted as both (i,j) and (j,i); a diagonal pair (i,i) appears twice per
  occurrence in the full matrix, so it is doubled). `--selftest` checks this
  against a naive reference implementation.
- float32 is safe here because counts are aggregated per chunk before being
  added (adding a ~1e5 chunk sum to a ~1e8 total loses <0.01%), unlike
  adding 1/d one pair at a time, which would silently stall at large totals.
- Nonzero entries are written to `data/cooc_<mode>/shard_XX.bin` as packed
  (int32 i, int32 j, float32 x) records, randomly assigned to shards, so
  training can draw i.i.d.-like batches without a giant global shuffle.

Run via the GPU venv's Python:
    ".../ai-gpu/Scripts/python.exe" -u build_cooccurrence.py --selftest
    ".../ai-gpu/Scripts/python.exe" -u build_cooccurrence.py --mode pieces
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RECORD = np.dtype([("i", "<i4"), ("j", "<i4"), ("x", "<f4")])


def count_upper(tokens: np.ndarray, doclens: np.ndarray, V: int, window: int,
                chunk_tokens: int = 2_000_000, log_every: int = 10) -> np.ndarray:
    """Dense flat (V*V) float32 accumulator of upper-triangle weighted counts."""
    U = np.zeros(V * V, dtype=np.float32)
    offsets = np.concatenate([[0], np.cumsum(doclens, dtype=np.int64)])
    n_docs = len(doclens)
    doc_start = 0
    chunk_no = 0
    t0 = time.time()
    while doc_start < n_docs:
        target = offsets[doc_start] + chunk_tokens
        doc_end = int(np.searchsorted(offsets, target, side="left"))
        doc_end = min(max(doc_end, doc_start + 1), n_docs)
        s, e = int(offsets[doc_start]), int(offsets[doc_end])
        t = tokens[s:e].astype(np.int64)
        did = np.repeat(np.arange(doc_end - doc_start, dtype=np.int32), doclens[doc_start:doc_end])
        for d in range(1, window + 1):
            if len(t) <= d:
                break
            valid = did[:-d] == did[d:]
            a, b = t[:-d][valid], t[d:][valid]
            keys = np.minimum(a, b) * V + np.maximum(a, b)
            uniq, cnt = np.unique(keys, return_counts=True)
            U[uniq] += cnt.astype(np.float32) / d
        doc_start = doc_end
        chunk_no += 1
        if chunk_no % log_every == 0:
            print(f"  counted {offsets[doc_end]:,}/{offsets[-1]:,} tokens ({time.time() - t0:.0f}s)")
    return U


def iter_triplet_blocks(U: np.ndarray, V: int, block_rows: int = 256):
    """Yields (i, j, x) for the FULL symmetric matrix, block by block."""
    Um = U.reshape(V, V)
    for r0 in range(0, V, block_rows):
        block = Um[r0:r0 + block_rows]
        ri, cj = np.nonzero(block)
        x = block[ri, cj]
        ri = ri + r0
        diag = ri == cj
        off = ~diag
        ii = np.concatenate([ri[off], cj[off], ri[diag]]).astype(np.int32)
        jj = np.concatenate([cj[off], ri[off], ri[diag]]).astype(np.int32)
        xx = np.concatenate([x[off], x[off], 2.0 * x[diag]]).astype(np.float32)
        yield ii, jj, xx


def naive_dense(tokens, doclens, V, window) -> np.ndarray:
    X = np.zeros((V, V), dtype=np.float64)
    pos = 0
    for L in doclens:
        doc = tokens[pos:pos + L]
        pos += L
        for p in range(len(doc)):
            for q in range(p + 1, min(p + window, len(doc) - 1) + 1):
                w = 1.0 / (q - p)
                X[doc[p], doc[q]] += w
                X[doc[q], doc[p]] += w
    return X


def selftest() -> None:
    rng = np.random.default_rng(0)
    V, window = 40, 10
    doclens = rng.integers(1, 30, size=60).astype(np.int32)
    tokens = rng.integers(0, V, size=int(doclens.sum())).astype(np.uint16)
    expected = naive_dense(tokens, doclens, V, window)
    U = count_upper(tokens, doclens, V, window, chunk_tokens=97, log_every=10**9)  # tiny chunks: exercise boundaries
    got = np.zeros((V, V), dtype=np.float64)
    for ii, jj, xx in iter_triplet_blocks(U, V, block_rows=7):
        np.add.at(got, (ii, jj), xx)
    assert np.allclose(got, got.T), "reconstructed matrix is not symmetric"
    assert np.allclose(got, expected, atol=1e-3), f"max abs diff {np.abs(got - expected).max()}"
    print(f"selftest OK: vectorized counter == naive reference "
          f"(V={V}, {len(tokens)} tokens, window={window}, total weight {got.sum():.1f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pieces", "words"])
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--shards", type=int, default=32)
    parser.add_argument("--chunk-tokens", type=int, default=2_000_000)
    parser.add_argument("--max-docs", type=int, default=None, help="use only the first N comments (quick test)")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.mode:
        raise SystemExit("--mode is required (unless --selftest)")

    tokens = np.load(DATA_DIR / f"tokens_{args.mode}.npy")
    doclens = np.load(DATA_DIR / f"doclens_{args.mode}.npy")
    if args.max_docs is not None:
        doclens = doclens[: args.max_docs]
        tokens = tokens[: int(doclens.sum())]
    V = 20_000 if args.mode == "pieces" else len(json.loads((DATA_DIR / "vocab_words.json").read_text(encoding="utf-8")))
    print(f"mode={args.mode}: {len(doclens):,} comments, {len(tokens):,} tokens, V={V:,}, window={args.window}")

    U = count_upper(tokens, doclens, V, args.window, args.chunk_tokens)

    out_dir = DATA_DIR / f"cooc_{args.mode}"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("shard_*.bin"):
        old.unlink()
    files = [open(out_dir / f"shard_{k:02d}.bin", "wb") for k in range(args.shards)]
    rng = np.random.default_rng(0)
    nnz = 0
    x_max = 0.0
    total_weight = 0.0
    t0 = time.time()
    try:
        for ii, jj, xx in iter_triplet_blocks(U, V):
            nnz += len(ii)
            x_max = max(x_max, float(xx.max()))
            total_weight += float(xx.astype(np.float64).sum())
            shard = rng.integers(0, args.shards, size=len(ii))
            order = np.argsort(shard, kind="stable")
            bounds = np.concatenate([[0], np.cumsum(np.bincount(shard, minlength=args.shards))])
            rec = np.empty(len(ii), dtype=RECORD)
            rec["i"], rec["j"], rec["x"] = ii[order], jj[order], xx[order]
            for k in range(args.shards):
                if bounds[k + 1] > bounds[k]:
                    rec[bounds[k]:bounds[k + 1]].tofile(files[k])
    finally:
        for f in files:
            f.close()

    meta = {
        "mode": args.mode, "V": V, "window": args.window, "n_comments": int(len(doclens)),
        "n_tokens": int(len(tokens)), "nnz": nnz, "density": nnz / (V * V),
        "max_x": x_max, "total_weight": total_weight, "shards": args.shards,
    }
    (DATA_DIR / f"cooc_{args.mode}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"extracted in {time.time() - t0:.0f}s")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
