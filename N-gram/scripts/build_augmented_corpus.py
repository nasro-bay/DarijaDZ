#!/usr/bin/env python
"""Builds the augmented training corpus for the N-gram language model
(see ../plan.md, section 2): classifies every YouTube-processed document
by script (arabic / latin / mixed), then adds transliterated synthetic
copies alongside a subset of the originals to narrow the Arabic-script /
Arabizi imbalance -- without ever removing or modifying an original
document.

- Arabic-script-majority docs: a random 20% sample, stratified by
  channel (so no single channel dominates the augmented slice), gets a
  transliterated synthetic copy added.
- Mixed-script docs: ALL of them get a transliterated synthetic copy
  (not just 20%) -- plan.md section 2's last bullet asks for full
  coverage here, not the 20% sampling rate used for arabic-majority docs.
  `Arabizi_transliteration.transliterate()` only converts pure-Arabic-
  script word tokens and leaves Latin tokens/punctuation untouched by
  construction (see its own docstring/regex), so this already implements
  the "mask the Arabic words, transliterate them, keep both" requirement
  with no extra masking logic needed here.
- Native Latin/Arabizi-majority docs: untouched, no synthetic copy.

Every doc (original and synthetic) gets `script_bucket` and
`is_transliterated` fields for provenance -- downstream train/dev/test
splitting (prepare_ngram_data.py) excludes is_transliterated=true docs
from dev/test, per plan.md section 5 ("never evaluate on synthetic
data").
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "Youtube_scrap" / "data" / "processed"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "augmented_corpus.jsonl"

sys.path.insert(0, str(ROOT / "DarijaDZ_spaces" / "Arabizi_transliteration"))
from transliterate import transliterate  # noqa: E402

SAMPLE_SEED = 42
ARABIC_SAMPLE_RATE = 0.20

# Same script-classification regexes as Youtube_scrap/scripts/build_unified_dataset.py
# and Tokenization/tokenizer_utils.py's classify_script() -- kept in sync
# across all three copies (established pattern in this repo, not a new one).
ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻾]")
LATIN_RE = re.compile(r"[a-zA-ZÀ-ɏ]")


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
    """Buckets a whole document as 'arabic' / 'latin' / 'mixed' script,
    based only on tokens that carry script signal (ignoring pure
    digit/punctuation/emoji tokens, which say nothing about script).
    Returns None if there's no alphabetic signal at all (rare on the
    already-cleaned corpus, but near-empty/emoji-only docs can slip
    through).
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


def iter_docs(batch_files: list[Path], limit: int | None):
    """Yields raw doc dicts across all batch files, honoring an optional
    total-docs cap for smoke testing (see --limit).
    """
    seen = 0
    for batch_file in batch_files:
        with batch_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if limit is not None and seen >= limit:
                    return
                seen += 1
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the augmented N-gram training corpus")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap total docs read (across all batch files) for a quick smoke test",
    )
    args = parser.parse_args()

    batch_files = sorted(PROCESSED_DIR.glob("batch_*.jsonl"))
    if not batch_files:
        raise SystemExit(f"No batch files found in {PROCESSED_DIR}")

    rng = random.Random(SAMPLE_SEED)

    # Pass 1: classify every doc, group arabic-bucket doc *ids* by channel
    # for stratified sampling -- can't decide who's in the 20% until every
    # channel's arabic-bucket doc count is known, so this can't be a
    # single-pass stream-sample.
    print("Pass 1: classifying documents by script bucket...")
    arabic_ids_by_channel: dict[str, list[str]] = defaultdict(list)
    bucket_counts: dict[str, int] = defaultdict(int)
    total_docs = 0

    for doc in iter_docs(batch_files, args.limit):
        total_docs += 1
        bucket = classify_document(doc["text"])
        if bucket is None:
            continue
        bucket_counts[bucket] += 1
        if bucket == "arabic":
            arabic_ids_by_channel[doc.get("channel") or "_unknown"].append(doc["id"])
        if total_docs % 200_000 == 0:
            print(f"  ...{total_docs:,} docs classified")

    print(f"  {total_docs:,} total docs read")
    for bucket, count in sorted(bucket_counts.items()):
        print(f"  {bucket}: {count:,} ({count / total_docs:.1%})")

    sampled_arabic_ids: set[str] = set()
    for channel_ids in arabic_ids_by_channel.values():
        k = round(len(channel_ids) * ARABIC_SAMPLE_RATE)
        if k:
            sampled_arabic_ids.update(rng.sample(channel_ids, k))
    arabic_total = bucket_counts.get("arabic", 0)
    print(
        f"  sampled {len(sampled_arabic_ids):,} arabic docs for augmentation "
        f"({len(sampled_arabic_ids) / arabic_total:.1%} of arabic bucket)"
        if arabic_total
        else "  no arabic-bucket docs found"
    )

    # Pass 2: write every original doc (with provenance fields) plus a
    # transliterated synthetic copy for sampled-arabic and all-mixed docs.
    print("Pass 2: writing augmented corpus (this re-runs classify_document,"
          " which is cheap -- the expensive step is transliterate(), only"
          " called for docs actually being augmented)...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    synthetic = 0
    processed = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
        for doc in iter_docs(batch_files, args.limit):
            processed += 1
            bucket = classify_document(doc["text"])
            if bucket is None:
                continue

            out_f.write(
                json.dumps(
                    {
                        "id": doc["id"],
                        "text": doc["text"],
                        "channel": doc.get("channel"),
                        "script_bucket": bucket,
                        "is_transliterated": False,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

            augment = (bucket == "arabic" and doc["id"] in sampled_arabic_ids) or bucket == "mixed"
            if augment:
                translit_text = transliterate(doc["text"])
                out_f.write(
                    json.dumps(
                        {
                            "id": f"{doc['id']}_translit",
                            "text": translit_text,
                            "channel": doc.get("channel"),
                            "script_bucket": bucket,
                            "is_transliterated": True,
                            "source_id": doc["id"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
                synthetic += 1

            if processed % 200_000 == 0:
                print(f"  ...{processed:,} docs processed, {synthetic:,} synthetic so far")

    print(f"Done! {written:,} total docs written ({synthetic:,} synthetic) -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
