#!/usr/bin/env python
"""Preprocessing pass for the CBOW+attention model (see ../plan.md):
classifies each row's script, BPE-tokenizes it, builds a corpus-wide
token-frequency table (for subsampling + negative sampling), and runs
K-Means (k=3) on token counts for length-bucketed batching.

Run via the GPU venv's Python (see ../requirements.txt) -- this script
itself doesn't need torch/CUDA, but keeping every script in this folder
on the same interpreter avoids environment-mismatch surprises later:

    ".../ai-gpu/Scripts/python.exe" build_training_data.py --limit 50000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[3]
YOUTUBE_DIR = ROOT / "Youtube_scrap" / "data" / "processed"
DJELFA_DIR = ROOT / "Mountada_djelfa_scrap" / "data" / "processed"
# WORD2VEC_DATA_DIR override lets a bigcorpus/new-tokenizer rebuild write
# to a separate tree (e.g. data_bigcorpus/) without touching the data/
# the currently-trained embedding checkpoints were built from.
DATA_DIR = Path(os.environ.get("WORD2VEC_DATA_DIR") or (Path(__file__).resolve().parents[1] / "data"))

sys.path.insert(0, str(ROOT / "Tokenization"))
from tokenizer_utils import load_tokenizer  # noqa: E402

# Same script-classification regexes as N-gram/scripts/build_augmented_corpus.py
# and Youtube_scrap/scripts/build_unified_dataset.py -- kept in sync across
# all three copies (established pattern in this repo).
ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻾]")
LATIN_RE = re.compile(r"[a-zA-ZÀ-ɏ]")

# Switched from "bpe" to "unigram" -- SentencePiece Unigram won the
# tokenization_eval.ipynb sweep on the rebuilt 16.47M-doc corpus (best
# compression/fertility at every vocab size, 0 round-trip errors); see
# Tokenization/DarijaDz_Tokenizers/README.md.
TOKENIZER_KEY = "unigram"
VOCAB_SIZE = 20_000
N_CLUSTERS = 3


def classify_token(token: str) -> str:
    has_arabic = bool(ARABIC_RE.search(token))
    has_latin = bool(LATIN_RE.search(token))
    if has_arabic and not has_latin:
        return "arabic"
    if has_latin and not has_arabic:
        return "latin"
    if has_arabic and has_latin:
        return "mixed"
    return "digits_symbols"


def classify_document(text: str) -> str | None:
    """Buckets a whole row as 'arabic' / 'latin' / 'mixed' script, same
    logic as N-gram/scripts/build_augmented_corpus.py's classify_document().
    """
    counts: dict[str, int] = defaultdict(int)
    for tok in text.split():
        counts[classify_token(tok)] += 1
    alphabetic = counts["arabic"] + counts["latin"] + counts["mixed"]
    if alphabetic == 0:
        return None
    if counts["latin"] == 0:
        return "arabic"
    if counts["arabic"] == 0 and counts["mixed"] == 0:
        return "latin"
    return "mixed"


def iter_raw_docs(limit: int | None):
    batch_files = sorted(YOUTUBE_DIR.glob("batch_*.jsonl")) + sorted(DJELFA_DIR.glob("batch_*.jsonl"))
    seen = 0
    for batch_file in batch_files:
        with batch_file.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                if limit is not None and seen >= limit:
                    return
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    # a few processed-batch lines are truncated by an
                    # interrupted writer (e.g. batch_2026-08-30_Nour.jsonl)
                    # -- skip, don't abort the whole build
                    print(f"  WARNING: skipping malformed JSON at {batch_file.name}:{lineno}")
                    continue
                seen += 1
                yield doc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the prepared training cache for the CBOW+attention model")
    parser.add_argument("--limit", type=int, default=None, help="cap total docs read, for a smoke test")
    args = parser.parse_args()

    print(f"Loading tokenizer: {TOKENIZER_KEY} @ {VOCAB_SIZE:,} ...")
    tok = load_tokenizer(TOKENIZER_KEY, VOCAB_SIZE)
    print(f"  vocab_size_actual = {tok.vocab_size_actual}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    token_freq: Counter[int] = Counter()
    rows: list[dict] = []

    print("Pass: classify script, tokenize, count token frequencies...")
    for doc in iter_raw_docs(args.limit):
        text = doc.get("text")
        if not text:
            continue
        bucket = classify_document(text)
        if bucket is None:
            continue
        ids = tok.encode(text)
        if len(ids) < 2:
            continue  # need at least 2 tokens for any (context, center) pair
        token_freq.update(ids)

        rows.append({
            "id": doc.get("id"),
            "text": text,
            "script_bucket": bucket,
            "word_count": len(text.split()),
            "token_count": len(ids),
        })

        if len(rows) % 200_000 == 0:
            print(f"  ...{len(rows):,} rows processed")

    print(f"Done: {len(rows):,} eligible rows")

    print(f"K-Means (k={N_CLUSTERS}) on token counts...")
    token_counts = np.array([r["token_count"] for r in rows], dtype=np.float64).reshape(-1, 1)
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    cluster_ids = kmeans.fit_predict(token_counts)
    centroids = kmeans.cluster_centers_.flatten().tolist()
    cluster_sizes = Counter(cluster_ids.tolist())
    print(f"  centroids (token count): {[round(c, 2) for c in sorted(centroids)]}")
    print(f"  cluster sizes: {dict(sorted(cluster_sizes.items()))}")
    if len(set(round(c, 1) for c in centroids)) < N_CLUSTERS:
        print("  WARNING: two or more centroids collapsed to nearly the same value -- "
              "token-count distribution may be too narrow for 3 meaningfully distinct length buckets")

    for row, cid in zip(rows, cluster_ids):
        row["cluster_id"] = int(cid)

    rows_path = DATA_DIR / "rows.jsonl"
    print(f"Writing {rows_path} ...")
    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    freq_path = DATA_DIR / "token_freq.json"
    print(f"Writing token frequency table ({len(token_freq):,} distinct tokens) -> {freq_path}")
    with freq_path.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in token_freq.items()}, f)

    meta = {
        "tokenizer": {"key": TOKENIZER_KEY, "vocab_size": VOCAB_SIZE, "vocab_size_actual": tok.vocab_size_actual},
        "total_rows": len(rows),
        "kmeans_centroids_sorted": sorted(round(c, 3) for c in centroids),
        "cluster_sizes": {str(k): v for k, v in cluster_sizes.items()},
        "script_bucket_counts": dict(Counter(r["script_bucket"] for r in rows)),
    }
    meta_path = DATA_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
