#!/usr/bin/env python
"""Merges every human correction made in the review UIs (local Flask app + Claude
Artifact) back onto the agent-labeled pool, producing one consolidated, trustworthy
10,000-row file -- `data/labeled_10k_final.jsonl`.

Three overlapping sources needed stitching together (none of them individually is the
full picture, and none of the review UIs writes back into `labeled_10k.jsonl` directly
-- see each UI's module docstring / CLAUDE.md's "corrections are a separate overlay"
note):

  1. `labeled_10k.jsonl` (9,971 rows) -- unique-shard items used as-is, plus
     overlap-set items where >=3/5 agents agreed (see merge_and_score.py).
  2. `disagreements.jsonl` (29 rows, the overlap items with no agent majority) +
     `disagreement_reviews.jsonl` (this session's manual resolutions for them, via the
     "Disagreements" tab) -- together these 29 complete the pool
     (9,971 + 29 = 10,000, the full unlabeled_10k.jsonl).
  3. `corrections.jsonl` (local) -- one-off relabels made via "Browse & correct" or the
     dedicated "resolve UNSURE one by one" queue, keyed by doc id, applied last (highest
     priority) since it represents the most deliberate, most recently reviewed label for
     that specific item.

The Claude Artifact's own `corrections` db collection was checked this run and is
empty (all relabeling so far happened in the local Flask app) -- if that ever changes,
re-run this after pulling the Artifact's `corrections` collection down to a local
JSONL file in the same {id, corrected_label, original_label} shape.

Output row shape: id, text, label (the final, corrected label), source (unchanged from
labeled_10k.jsonl, or "disagreement_resolved" for the 29 merged-in items), and
original_agent_label (only present when a correction changed the label away from what
the agents originally produced -- lets a later consumer tell "was this corrected" and
"what did the agents say" without a second file).

Run via the base Python environment:

    python apply_corrections.py
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LABELED_PATH = DATA_DIR / "labeled_10k.jsonl"
DISAGREEMENTS_PATH = DATA_DIR / "disagreements.jsonl"
DISAGREEMENT_REVIEWS_PATH = DATA_DIR / "disagreement_reviews.jsonl"
CORRECTIONS_PATH = DATA_DIR / "corrections.jsonl"
OUT_PATH = DATA_DIR / "labeled_10k_final.jsonl"

VALID_LABELS = {"POS", "NEG", "NEU", "MIX", "UNSURE"}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    rows: dict[str, dict] = {}

    for r in _load_jsonl(LABELED_PATH):
        rows[r["id"]] = {"id": r["id"], "text": r["text"], "label": r["label"], "source": r["source"]}

    reviews = {r["id"]: r["manual_label"] for r in _load_jsonl(DISAGREEMENT_REVIEWS_PATH)}
    unresolved_disagreements = 0
    for r in _load_jsonl(DISAGREEMENTS_PATH):
        manual_label = reviews.get(r["id"])
        if manual_label is None:
            unresolved_disagreements += 1
            continue  # left out of the final file until reviewed -- same "don't guess" convention
        rows[r["id"]] = {"id": r["id"], "text": r["text"], "label": manual_label, "source": "disagreement_resolved"}

    corrections = _load_jsonl(CORRECTIONS_PATH)
    applied, skipped_unknown_id = 0, 0
    for c in corrections:
        doc_id = c["id"]
        if doc_id not in rows:
            skipped_unknown_id += 1
            continue
        original_label = rows[doc_id]["label"]
        rows[doc_id]["original_agent_label"] = original_label
        rows[doc_id]["label"] = c["corrected_label"]
        applied += 1

    assert all(r["label"] in VALID_LABELS for r in rows.values())

    final_rows = list(rows.values())
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"Wrote {len(final_rows):,} rows -> {OUT_PATH}")
    print(f"  disagreements resolved: {len(reviews)}  (unresolved, left out: {unresolved_disagreements})")
    print(f"  corrections applied: {applied}  (skipped, id not in pool: {skipped_unknown_id})")
    print(f"  final label distribution: {dict(Counter(r['label'] for r in final_rows))}")


if __name__ == "__main__":
    main()
