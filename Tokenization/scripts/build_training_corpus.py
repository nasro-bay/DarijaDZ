#!/usr/bin/env python
"""Streams cleaned `text` from every processed batch file in both source
projects into a single flat training file for the tokenizer trainers
(one doc per line — the format both SentencePiece and HF `tokenizers`
expect).

Docs whose `id` appears in the cached reservoir samples
(`Data/sample_youtube.jsonl` / `Data/sample_djelfa_info.jsonl`, built by
`01_sample_and_explore.ipynb`) are excluded from the training corpus and
instead written to `data/heldout_docs.jsonl` — this is the tokenizer
eval notebook's held-out set. Note: the *text* in those cached sample
files predates later cleaning fixes (tachkil-stripping, NFKC
normalization), so we only reuse their `id`s as a held-out allowlist and
re-extract fresh text for those ids from the live processed corpus here,
rather than trusting the stale cached text.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/ -> Tokenization/ -> Darija/
OUT_DIR = Path(__file__).resolve().parents[1] / "data"

SOURCES = {
    "youtube": sorted((ROOT / "Youtube_scrap" / "data" / "processed").glob("batch_*.jsonl")),
    "djelfa_info": sorted((ROOT / "Mountada_djelfa_scrap" / "data" / "processed").glob("batch_*.jsonl")),
}
HELDOUT_SAMPLE_FILES = [
    ROOT / "Data" / "sample_youtube.jsonl",
    ROOT / "Data" / "sample_djelfa_info.jsonl",
]


def load_heldout_ids() -> set[str]:
    ids: set[str] = set()
    for path in HELDOUT_SAMPLE_FILES:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run 01_sample_and_explore.ipynb first to build the cached sample."
            )
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(json.loads(line)["id"])
    return ids


def main() -> None:
    for name, files in SOURCES.items():
        if not files:
            raise FileNotFoundError(f"no processed batch files found for {name}")

    heldout_ids = load_heldout_ids()
    print(f"held-out doc ids loaded: {len(heldout_ids):,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train_corpus.txt"
    heldout_path = OUT_DIR / "heldout_docs.jsonl"

    train_count = 0
    heldout_count = 0
    malformed_count = 0
    with train_path.open("w", encoding="utf-8") as train_out, heldout_path.open(
        "w", encoding="utf-8"
    ) as heldout_out:
        for name, files in SOURCES.items():
            for path in files:
                with path.open("r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            doc = json.loads(line)
                        except json.JSONDecodeError:
                            # a few processed-batch lines are truncated by an
                            # interrupted writer (e.g. batch_2026-08-30_Nour.jsonl)
                            # -- skip, don't abort the whole rebuild
                            malformed_count += 1
                            print(f"  WARNING: skipping malformed JSON at {path.name}:{lineno}")
                            continue
                        text = doc["text"].replace("\n", " ").strip()
                        if not text:
                            continue
                        if doc["id"] in heldout_ids:
                            doc["source_bucket"] = name
                            heldout_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                            heldout_count += 1
                        else:
                            train_out.write(text + "\n")
                            train_count += 1

    print(f"train corpus: {train_count:,} docs -> {train_path}")
    print(f"held-out set: {heldout_count:,} docs -> {heldout_path}")
    if malformed_count:
        print(f"skipped {malformed_count} malformed JSON line(s)")


if __name__ == "__main__":
    main()
