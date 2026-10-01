#!/usr/bin/env python
"""Splits data/unlabeled_10k.jsonl into the 5-agent partial-overlap layout
(see ../plan.md's "Labeling architecture" section):

  - a shared overlap set (OVERLAP_SIZE docs), labeled by all 5 agents
  - 5 unique shards (the rest, split evenly), one per agent

Writes data/agent{1..5}_batch.jsonl -- each file is that agent's unique
shard rows followed by the overlap rows, every row tagged with
"shard": "unique" | "overlap" so merge_and_score.py can tell them apart
later without re-deriving the split. Deterministic (fixed seed), so
rerunning after unlabeled_10k.jsonl is extended (a later run) reassigns
cleanly rather than needing manual bookkeeping -- existing agent batch
files are simply overwritten.

Run via the base Python environment:

    python assign_shards.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
IN_PATH = DATA_DIR / "unlabeled_10k.jsonl"

NUM_AGENTS = 5
OVERLAP_SIZE = 1_000
SEED = 42


def main() -> None:
    rows = [json.loads(line) for line in IN_PATH.open("r", encoding="utf-8") if line.strip()]
    rng = random.Random(SEED)
    rng.shuffle(rows)

    overlap = rows[:OVERLAP_SIZE]
    unique_pool = rows[OVERLAP_SIZE:]

    shard_size = len(unique_pool) // NUM_AGENTS
    remainder = len(unique_pool) % NUM_AGENTS
    print(f"Total pool: {len(rows):,} | overlap: {len(overlap):,} | unique pool: {len(unique_pool):,}")
    if remainder:
        print(f"  ({remainder} leftover unique row(s) go to the last agent's shard)")

    start = 0
    for i in range(1, NUM_AGENTS + 1):
        size = shard_size + (remainder if i == NUM_AGENTS else 0)
        shard = unique_pool[start : start + size]
        start += size

        out_rows = [{**r, "shard": "unique"} for r in shard] + [{**r, "shard": "overlap"} for r in overlap]
        rng.shuffle(out_rows)  # don't let the agent see all-unique-then-all-overlap in order

        out_path = DATA_DIR / f"agent{i}_batch.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"agent{i}: {len(shard):,} unique + {len(overlap):,} overlap = {len(out_rows):,} total -> {out_path.name}")


if __name__ == "__main__":
    main()
