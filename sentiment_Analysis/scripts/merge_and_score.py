#!/usr/bin/env python
"""Merges the 5 agents' output files (data/agent{1..5}_labels.jsonl, one
{"id":..., "label":...} per line, written by the labeling agents per
../plan.md) into the final labeled dataset + a disagreement list for
manual annotation.

- Unique-shard items: exactly one agent labeled them -- used as-is.
- Overlap items: all 5 agents labeled them -- majority vote. A clear
  majority (>=3/5 agree) is used directly; anything else (e.g. 2-2-1,
  no label reaching 3) is a disagreement, written to disagreements.jsonl
  for manual annotation instead of being auto-resolved.
- Reports the overlap-set exact-agreement rate (fraction of overlap items
  where the majority label's vote count / 5) as this run's QA metric --
  see ../plan.md's "Open questions" for why this is the simple version,
  not Fleiss' kappa, for now.

Writes:
  data/labeled_10k.jsonl     -- id, text, label, source ("unique" or
                                 "majority_vote"), and, for overlap items,
                                 the full vote breakdown for traceability.
  data/disagreements.jsonl   -- overlap items with no clear majority,
                                 for manual annotation; each gets a
                                 "manual_label": null field to fill in.

Run via the base Python environment:

    python merge_and_score.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
NUM_AGENTS = 5
MAJORITY_THRESHOLD = 3  # out of 5


def load_agent_labels(i: int) -> dict[str, str]:
    path = DATA_DIR / f"agent{i}_labels.jsonl"
    out = {}
    for line in path.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["id"]] = row["label"]
    return out


def load_batch(i: int) -> dict[str, dict]:
    path = DATA_DIR / f"agent{i}_batch.jsonl"
    return {json.loads(line)["id"]: json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()}


def main() -> None:
    agent_labels = {i: load_agent_labels(i) for i in range(1, NUM_AGENTS + 1)}
    agent_batches = {i: load_batch(i) for i in range(1, NUM_AGENTS + 1)}

    # every row, keyed by id, regardless of which agent(s) saw it
    rows_by_id: dict[str, dict] = {}
    unique_owner: dict[str, int] = {}
    overlap_ids: set[str] = set()
    for i in range(1, NUM_AGENTS + 1):
        for doc_id, row in agent_batches[i].items():
            rows_by_id[doc_id] = row
            if row["shard"] == "unique":
                unique_owner[doc_id] = i
            else:
                overlap_ids.add(doc_id)

    missing = []
    final_rows = []
    disagreements = []

    for doc_id, owner in unique_owner.items():
        label = agent_labels[owner].get(doc_id)
        if label is None:
            missing.append((doc_id, owner))
            continue
        final_rows.append({"id": doc_id, "text": rows_by_id[doc_id]["text"], "label": label, "source": "unique"})

    agreement_scores = []
    for doc_id in overlap_ids:
        votes = [agent_labels[i].get(doc_id) for i in range(1, NUM_AGENTS + 1)]
        if any(v is None for v in votes):
            missing.append((doc_id, "overlap"))
            continue
        counts = Counter(votes)
        top_label, top_count = counts.most_common(1)[0]
        agreement_scores.append(top_count / NUM_AGENTS)
        if top_count >= MAJORITY_THRESHOLD:
            final_rows.append({
                "id": doc_id, "text": rows_by_id[doc_id]["text"], "label": top_label,
                "source": "majority_vote", "votes": dict(counts),
            })
        else:
            disagreements.append({
                "id": doc_id, "text": rows_by_id[doc_id]["text"], "votes": dict(counts), "manual_label": None,
            })

    with (DATA_DIR / "labeled_10k.jsonl").open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (DATA_DIR / "disagreements.jsonl").open("w", encoding="utf-8") as f:
        for row in disagreements:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Unique-shard items labeled: {len(unique_owner) - sum(1 for _, o in missing if o != 'overlap'):,}")
    print(f"Overlap items with a clear majority (>= {MAJORITY_THRESHOLD}/5): "
          f"{sum(1 for r in final_rows if r['source'] == 'majority_vote'):,}")
    print(f"Overlap items flagged as disagreements (no clear majority): {len(disagreements):,}")
    if agreement_scores:
        print(f"Overlap-set mean agreement rate (top-label votes / 5): {sum(agreement_scores) / len(agreement_scores):.1%}")
    if missing:
        print(f"WARNING: {len(missing)} item(s) missing a label from at least one expected agent -- check agent output files")
    print(f"\nWrote {len(final_rows):,} rows -> data/labeled_10k.jsonl")
    print(f"Wrote {len(disagreements):,} rows -> data/disagreements.jsonl (fill in manual_label, then re-merge)")


if __name__ == "__main__":
    main()
