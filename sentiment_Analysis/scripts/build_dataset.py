#!/usr/bin/env python
"""Builds (and can EXTEND) the sentence pool to be labeled for sentiment
(POS/NEG/NEU/MIX/UNSURE, see ../plan.md). Adapted directly from
Dialect_Identification/scripts/build_dataset.py -- same sampling design,
same reasoning, just a different label task and a third source added.

Sampling design:
- djelfa forum posts: reservoir sample, no script-based stratification
  (djelfa is mostly Arabic-script already).
- YouTube + TikTok comments (pooled together as "social" -- both are
  short-form social video comments, same register): split into three
  equal script-based buckets via regex (not random sampling) so all
  three ~arabic/~latin/~mixed regimes are represented regardless of the
  corpus's natural script mix:
    - "arabic": Arabic-script only
    - "latin": Latin-script only (Arabizi/French/English)
    - "mixed": both scripts present in the same text (code-switched)
- Each bucket is oversampled (~30%) via single-pass reservoir sampling,
  then MinHash/LSH-deduped (fresh index built here, not touching any
  project's persisted pipeline LSH state), then trimmed to the exact
  target count.

This stratification is only about which TEXTS get sampled, for script
diversity -- it does NOT restrict which sentiment label an item can get.
The labeling step (see ../plan.md) is explicitly told not to gate labels
by script.

**Extending an existing pool** (this project works in runs, e.g. 10k ->
20k for run 2): if OUT_PATH already exists, this script does NOT touch
its existing rows at all -- it only samples however many MORE rows each
group needs to reach the new per-group targets below, excluding every id
already in the file, and seeding each group's dedup LSH with its
existing rows first. New rows are appended after the existing ones in
their own freshly-shuffled order.

Run via the base Python environment (no GPU needed -- this is pure data
sampling, not model inference):

    python build_dataset.py
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from datasketch import MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "unlabeled_10k.jsonl"

DJELFA_FILES = sorted((ROOT / "Mountada_djelfa_scrap" / "data" / "processed").glob("batch_*.jsonl"))
SOCIAL_FILES = sorted((ROOT / "Youtube_scrap" / "data" / "processed").glob("batch_*.jsonl")) + sorted(
    (ROOT / "Tiktok_scrap" / "data" / "processed").glob("batch_*.jsonl")
)

# Run 1 targets: 10,000 total, 15% djelfa / 85% social split evenly across
# the three script buckets -- same proportions as Dialect_Identification's
# original 10k pool. main() figures out how many MORE rows each group
# actually needs by subtracting what's already in OUT_PATH, so bumping
# these for a future run (e.g. 20k) and rerunning just extends the pool.
DJELFA_TARGET = 1_500
SOCIAL_BUCKET_TARGETS = {"arabic": 2_834, "latin": 2_833, "mixed": 2_833}
OVERSAMPLE_FACTOR = 1.3
SEED = 42

# Same regex convention reused verbatim across this repo.
ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
LATIN_RE = re.compile(r"[a-zA-ZÀ-ɏ]")

NUM_PERM = 128
SHINGLE_SIZE = 4
DEDUP_THRESHOLD = 0.8
_WHITESPACE_RE = re.compile(r"\s+")


def script_of(text: str) -> str:
    has_ar = bool(ARABIC_RE.search(text))
    has_lat = bool(LATIN_RE.search(text))
    if has_ar and has_lat:
        return "mixed"
    if has_ar:
        return "arabic"
    if has_lat:
        return "latin"
    return "other"


# Same placeholder-stripping convention as Dialect_Identification/build_dataset.py --
# [MENTION]/[URL] are Latin-script tokens that would otherwise mislabel a
# pure-Arabic row as "mixed". Only used to decide the bucket; the stored
# row keeps the original, uncleaned text.
_MENTION_WITH_FRAGMENT_RE = re.compile(r"\[MENTION\](\s*-[^\s]{1,10})?")
_URL_RE = re.compile(r"\[URL\]")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")


def clean_for_classification(text: str) -> str:
    text = _MENTION_WITH_FRAGMENT_RE.sub("", text)
    text = _URL_RE.sub("", text)
    return _INLINE_WHITESPACE_RE.sub(" ", text).strip()


def _shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    normalized = _WHITESPACE_RE.sub(" ", text.strip().lower())
    if not normalized:
        return set()
    if len(normalized) <= k:
        return {normalized}
    return {normalized[i : i + k] for i in range(len(normalized) - k + 1)}


def _minhash(text: str) -> MinHash:
    mh = MinHash(num_perm=NUM_PERM)
    for shingle in _shingles(text):
        mh.update(shingle.encode("utf-8"))
    return mh


def dedup_and_trim(rows: list[dict], target: int, label: str, lsh: MinHashLSH | None = None) -> list[dict]:
    if lsh is None:
        lsh = MinHashLSH(threshold=DEDUP_THRESHOLD, num_perm=NUM_PERM)
    kept: list[dict] = []
    i = -1
    for i, row in enumerate(rows):
        if len(kept) >= target:
            break
        mh = _minhash(row["text"])
        if lsh.query(mh):
            continue
        lsh.insert(row["id"], mh)
        kept.append(row)
    print(f"  {label}: {len(kept)}/{target} after dedup (from {min(i + 1, len(rows))} candidates scanned)")
    if len(kept) < target:
        print(f"  WARNING: {label} came up short -- raise OVERSAMPLE_FACTOR and rerun")
    return kept


def reservoir_sample_djelfa(
    target_pool: int, rng: random.Random, exclude_ids: set[str] = frozenset()
) -> list[dict]:
    reservoir: list[dict] = []
    seen = 0
    for path in DJELFA_FILES:
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  WARNING: skipping malformed JSON at {path.name}:{lineno}")
                    continue
                if not row.get("text", "").strip():
                    continue
                if row.get("id") in exclude_ids:
                    continue
                seen += 1
                if len(reservoir) < target_pool:
                    reservoir.append(row)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < target_pool:
                        reservoir[j] = row
    print(f"djelfa: reservoir-sampled {len(reservoir)} from {seen:,} candidates")
    return reservoir


def reservoir_sample_social_buckets(
    target_pools: dict[str, int], rng: random.Random, exclude_ids: set[str] = frozenset()
) -> dict[str, list[dict]]:
    reservoirs: dict[str, list[dict]] = {k: [] for k in target_pools}
    seen: dict[str, int] = {k: 0 for k in target_pools}
    for i, path in enumerate(SOCIAL_FILES):
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  WARNING: skipping malformed JSON at {path.name}:{lineno}")
                    continue
                text = row.get("text", "")
                if not text.strip():
                    continue
                if row.get("id") in exclude_ids:
                    continue
                bucket = script_of(clean_for_classification(text))
                if bucket not in reservoirs:
                    continue  # "other" -- no real script content, skip
                seen[bucket] += 1
                target = target_pools[bucket]
                res = reservoirs[bucket]
                if len(res) < target:
                    res.append(row)
                else:
                    j = rng.randint(0, seen[bucket] - 1)
                    if j < target:
                        res[j] = row
        if (i + 1) % 5 == 0:
            print(f"  ...{i + 1}/{len(SOCIAL_FILES)} social batch files scanned")
    for bucket, res in reservoirs.items():
        print(f"social[{bucket}]: reservoir-sampled {len(res)} from {seen[bucket]:,} candidates")
    return reservoirs


def _load_existing() -> list[dict]:
    if not OUT_PATH.exists():
        return []
    with OUT_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    from collections import Counter

    rng = random.Random(SEED)

    existing = _load_existing()
    existing_by_group: dict[str, list[dict]] = {}
    for row in existing:
        existing_by_group.setdefault(row["sample_group"], []).append(row)
    existing_ids = {row["id"] for row in existing}
    if existing:
        print(f"Found {len(existing):,} existing rows in {OUT_PATH} -- extending, not rebuilding.")
        print("Existing by sample_group:", {k: len(v) for k, v in existing_by_group.items()})

    djelfa_have = len(existing_by_group.get("djelfa", []))
    djelfa_need = max(0, DJELFA_TARGET - djelfa_have)
    djelfa_lsh = MinHashLSH(threshold=DEDUP_THRESHOLD, num_perm=NUM_PERM)
    for row in existing_by_group.get("djelfa", []):
        djelfa_lsh.insert(row["id"], _minhash(row["text"]))

    djelfa_new: list[dict] = []
    if djelfa_need:
        djelfa_pool = reservoir_sample_djelfa(int(djelfa_need * OVERSAMPLE_FACTOR), rng, existing_ids)
        rng.shuffle(djelfa_pool)
        djelfa_new = dedup_and_trim(djelfa_pool, djelfa_need, "djelfa", lsh=djelfa_lsh)
        for row in djelfa_new:
            row["sample_group"] = "djelfa"
    else:
        print(f"djelfa: already has {djelfa_have}/{DJELFA_TARGET} -- nothing more needed")

    bucket_needs = {}
    bucket_lsh: dict[str, MinHashLSH] = {}
    for bucket, target in SOCIAL_BUCKET_TARGETS.items():
        group = f"social_{bucket}"
        have = len(existing_by_group.get(group, []))
        bucket_needs[bucket] = max(0, target - have)
        lsh = MinHashLSH(threshold=DEDUP_THRESHOLD, num_perm=NUM_PERM)
        for row in existing_by_group.get(group, []):
            lsh.insert(row["id"], _minhash(row["text"]))
        bucket_lsh[bucket] = lsh
        if bucket_needs[bucket] == 0:
            print(f"social[{bucket}]: already has {have}/{target} -- nothing more needed")

    social_pools = reservoir_sample_social_buckets(
        {k: int(v * OVERSAMPLE_FACTOR) for k, v in bucket_needs.items() if v > 0}, rng, existing_ids
    )
    social_new: list[dict] = []
    for bucket, need in bucket_needs.items():
        if need == 0:
            continue
        pool = social_pools[bucket]
        rng.shuffle(pool)
        rows = dedup_and_trim(pool, need, f"social[{bucket}]", lsh=bucket_lsh[bucket])
        for row in rows:
            row["sample_group"] = f"social_{bucket}"
        social_new.extend(rows)

    new_rows = djelfa_new + social_new
    rng.shuffle(new_rows)  # mix sample_groups together so batch/shard assignment isn't biased by group

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("a", encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps({"id": row["id"], "text": row["text"], "sample_group": row["sample_group"]}, ensure_ascii=False) + "\n")

    total = len(existing) + len(new_rows)
    print(f"\nAppended {len(new_rows):,} new rows to {OUT_PATH} (total now {total:,})")
    all_groups = Counter(r["sample_group"] for r in existing) + Counter(r["sample_group"] for r in new_rows)
    print("By sample_group (total):", dict(all_groups))


if __name__ == "__main__":
    main()
