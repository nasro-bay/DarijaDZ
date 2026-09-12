#!/usr/bin/env python
"""Class-balancing step 3/3 -- merge the staged new rows
(_balance_stage/corpus_new.jsonl + _balance_stage/external_new.jsonl)
into data/labeled_10k.jsonl.

The pre-balance file is backed up to
_balance_stage/labeled_10k.pre_balance.jsonl first. Dedup is by row id
and by normalised text; existing pool rows always win a tie.

Run:  python assemble_balanced.py
"""
from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

DID = Path(__file__).resolve().parents[2]
DATA = DID / "data"
POOL = DATA / "labeled_10k.jsonl"
STAGE = DATA / "_balance_stage"
BACKUP = STAGE / "labeled_10k.pre_balance.jsonl"

_WS = re.compile(r"\s+")


def norm_key(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        print(f"  (missing: {p.name})")
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def main() -> None:
    pool = read_jsonl(POOL)
    new_rows = read_jsonl(STAGE / "corpus_new.jsonl") + read_jsonl(STAGE / "external_new.jsonl")
    print(f"pool={len(pool):,}  new(staged)={len(new_rows):,}")

    merged: list[dict] = []
    seen_ids, seen_keys = set(), set()
    dup_id = dup_text = 0
    for r in pool + new_rows:
        rid, key = r["id"], norm_key(r["text"])
        if rid in seen_ids:
            dup_id += 1
            continue
        if key in seen_keys:
            dup_text += 1
            continue
        seen_ids.add(rid)
        seen_keys.add(key)
        merged.append(r)

    print(f"dropped {dup_id} dup-id, {dup_text} dup-text -> {len(merged):,} rows")
    counts = Counter(r["label"] for r in merged)
    for c in ("darija", "arabize", "msa", "french", "english", "code_switch"):
        print(f"  {c:<12} {counts[c]:>6,}")

    shutil.copy2(POOL, BACKUP)
    with POOL.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nbacked up old pool -> {BACKUP}")
    print(f"wrote {len(merged):,} rows -> {POOL}")


if __name__ == "__main__":
    main()
