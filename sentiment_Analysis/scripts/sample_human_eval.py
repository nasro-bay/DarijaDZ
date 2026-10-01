#!/usr/bin/env python
"""Draws the stratified sample for human evaluation of the agent labels (see ../plan.md's "Human
evaluation" section): 250 items, floored per class rather than pure random, so the small MIX/UNSURE
classes get enough items to say anything about, not the ~4-9 a random sample would give them.

Deliberately excludes the 29 disagreements.jsonl items -- those already get separate manual
annotation (see merge_and_score.py); this sample is about checking the *agreed-upon* labels,
the ones no human has looked at yet.

Writes data/human_eval_sample.jsonl: id, text, sample_group, and the agent's final label (the
"answer key" -- the review UIs must not show this to the reviewer until after they've rated the
item, to keep the rating blind).

Run via the base Python environment:

    python sample_human_eval.py
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_PATH = DATA_DIR / "human_eval_sample.jsonl"
SEED = 42

# Deliberately not proportional to each class's real share (that would give MIX ~4-5 items) --
# floored so every class gets enough items for a meaningful per-class read, while still spending
# most of the budget on the three large classes. UNSURE excluded per explicit instruction (its
# "I can't tell" label isn't a judgment call a human rater can agree/disagree with the same way).
TARGET_PER_CLASS = {"POS": 50, "NEG": 50, "NEU": 50, "MIX": 60}


def main() -> None:
    final = [json.loads(l) for l in (DATA_DIR / "labeled_10k.jsonl").open(encoding="utf-8")]
    disagreement_ids = {json.loads(l)["id"] for l in (DATA_DIR / "disagreements.jsonl").open(encoding="utf-8")}
    pool_by_id = {r["id"]: r for r in (json.loads(l) for l in (DATA_DIR / "unlabeled_10k.jsonl").open(encoding="utf-8"))}

    final = [r for r in final if r["id"] not in disagreement_ids]
    by_label: dict[str, list[dict]] = {}
    for r in final:
        by_label.setdefault(r["label"], []).append(r)

    rng = random.Random(SEED)
    sample = []
    for label, target in TARGET_PER_CLASS.items():
        pool = by_label.get(label, [])
        n = min(target, len(pool))
        if n < target:
            print(f"WARNING: {label} pool only has {len(pool)} items, wanted {target}")
        picked = rng.sample(pool, n)
        for r in picked:
            sample.append({
                "id": r["id"], "text": r["text"], "label": r["label"],
                "sample_group": pool_by_id.get(r["id"], {}).get("sample_group", "?"),
            })

    rng.shuffle(sample)  # mix classes together so the reviewer can't infer the label from item order

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sample)} items -> {OUT_PATH}")
    print("By label:", dict(Counter(r["label"] for r in sample)))


if __name__ == "__main__":
    main()
