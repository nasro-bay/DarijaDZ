#!/usr/bin/env python
"""Builds the public Hugging Face release of the sentiment-labeled dataset --
`DarijaDZ_SA` -- from the internal, fuller `labeled_10k_final.jsonl` (see
apply_corrections.py): re-derives the final merged labels fresh each run (not reading
`labeled_10k_final.jsonl` directly, so this always reflects the current state of
corrections/disagreement-reviews without needing apply_corrections.py run first),
drops the UNSURE class entirely, and keeps only the public-facing columns (id, text,
label -- no `source`/`original_agent_label`, which are internal bookkeeping).

Also rewrites `DarijaDZ_SA/README.md`'s numeric stats (row count, size_categories
bucket, per-class distribution table) via targeted regex substitution, same pattern as
Youtube_scrap/scripts/build_unified_dataset.py -- so the card never goes stale after a
future relabeling session changes the counts; this script is the one place that knows
the real numbers.

Run via the base Python environment:

    python build_hf_dataset.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LABELED_PATH = DATA_DIR / "labeled_10k.jsonl"
DISAGREEMENTS_PATH = DATA_DIR / "disagreements.jsonl"
DISAGREEMENT_REVIEWS_PATH = DATA_DIR / "disagreement_reviews.jsonl"
CORRECTIONS_PATH = DATA_DIR / "corrections.jsonl"

OUT_DIR = Path(__file__).resolve().parents[1] / "DarijaDZ_SA"
OUT_DATA_PATH = OUT_DIR / "data.jsonl"
README_PATH = OUT_DIR / "README.md"

# HF's standard size_categories buckets.
SIZE_BUCKETS = [
    (1_000, "n<1K"),
    (10_000, "1K<n<10K"),
    (100_000, "10K<n<100K"),
    (1_000_000, "100K<n<1M"),
    (10_000_000, "1M<n<10M"),
]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def size_category(n: int) -> str:
    for bound, label in SIZE_BUCKETS:
        if n < bound:
            return label
    return "10M<n<100M"


def build_final_rows() -> list[dict]:
    """Same merge as apply_corrections.py (labeled_10k.jsonl + resolved disagreements,
    corrections overlay applied last), recomputed here directly rather than imported,
    to keep this script runnable standalone."""
    rows: dict[str, dict] = {}
    for r in _load_jsonl(LABELED_PATH):
        rows[r["id"]] = {"id": r["id"], "text": r["text"], "label": r["label"]}

    reviews = {r["id"]: r["manual_label"] for r in _load_jsonl(DISAGREEMENT_REVIEWS_PATH)}
    for r in _load_jsonl(DISAGREEMENTS_PATH):
        manual_label = reviews.get(r["id"])
        if manual_label is None:
            continue
        rows[r["id"]] = {"id": r["id"], "text": r["text"], "label": manual_label}

    for c in _load_jsonl(CORRECTIONS_PATH):
        if c["id"] in rows:
            rows[c["id"]]["label"] = c["corrected_label"]

    return list(rows.values())


def main() -> None:
    rows = build_final_rows()
    public_rows = [r for r in rows if r["label"] != "UNSURE"]  # UNSURE excluded from the public release
    dropped_unsure = len(rows) - len(public_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_DATA_PATH.open("w", encoding="utf-8") as f:
        for r in public_rows:
            f.write(json.dumps({"id": r["id"], "text": r["text"], "label": r["label"]}, ensure_ascii=False) + "\n")

    n = len(public_rows)
    dist = Counter(r["label"] for r in public_rows)
    class_order = ["POS", "NEU", "NEG", "MIX"]

    print(f"Wrote {n:,} rows -> {OUT_DATA_PATH} (dropped {dropped_unsure} UNSURE rows)")
    print("Distribution:", dict(dist))

    if not README_PATH.exists():
        print(f"NOTE: {README_PATH} doesn't exist yet -- write it once by hand first, "
              f"then rerun this script to sync its stats.")
        return

    text = README_PATH.read_text(encoding="utf-8")

    text = re.sub(
        r"(size_categories:\n  - ).*", lambda m: m.group(1) + size_category(n), text, count=1
    )
    text = re.sub(
        r"(\*\*Documents\*\*\s*\|\s*)[\d,]+", lambda m: m.group(1) + f"{n:,}", text, count=1
    )

    rows_md = "\n".join(
        f"| {label:<23} | {dist.get(label, 0):>6,} | {100 * dist.get(label, 0) / n:>5.1f}% |"
        for label in class_order
    )
    text = re.sub(
        r"(<!-- DISTRIBUTION_TABLE_START -->\n).*?(<!-- DISTRIBUTION_TABLE_END -->)",
        lambda m: m.group(1) + rows_md + "\n" + m.group(2),
        text,
        flags=re.DOTALL,
    )

    README_PATH.write_text(text, encoding="utf-8")
    print(f"Synced stats into {README_PATH}")


if __name__ == "__main__":
    main()
