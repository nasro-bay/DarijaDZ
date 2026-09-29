#!/usr/bin/env python
"""Incrementally builds Youtube_scrap/data/unified_corpus.jsonl (id + text
only) from processed YouTube batches.

**Incremental, not a full rebuild**: tracks which `data/processed/
batch_*.jsonl` files have already been folded in
(`data/state/unified_build_state.json`), and each run only reads/appends
the ones that are new since the last run -- unlike the old version of
this script (see git history), which re-read every batch file from
scratch every time. At this corpus's scale (tens of millions of lines)
that was wasted, repeated work; this only ever processes the delta.

This script does **not** shuffle, sync release copies, or touch any
README -- those are now `Data/scripts/build_combined_dataset.py`'s job,
since they operate on the *combined* YouTube+TikTok corpus, not a single
source. Run that script after this one (and after
Tiktok_scrap/scripts/build_unified_dataset.py, its TikTok-side twin) to
actually publish an update.

id/text only, no `source` field: YouTube ids are always `yt_...`,
TikTok ids are always `tt_...` (see each source's schema.py) --
guaranteed never to collide, so nothing needs a source tag to stay
unambiguous once combined.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "Youtube_scrap" / "data" / "processed"
UNIFIED_PATH = ROOT / "Youtube_scrap" / "data" / "unified_corpus.jsonl"
STATE_PATH = ROOT / "Youtube_scrap" / "data" / "state" / "unified_build_state.json"

_DEFAULT_STATE = {
    "included_batch_files": [],
    "total_docs": 0,
    "total_tokens": 0,
    "channels": [],
    "video_ids": [],
}


def _load_state() -> dict:
    if STATE_PATH.exists():
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        for key, default in _DEFAULT_STATE.items():
            data.setdefault(key, default if not isinstance(default, list) else [])
        return data
    return json.loads(json.dumps(_DEFAULT_STATE))


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(STATE_PATH)


def main() -> None:
    if not PROCESSED_DIR.exists():
        raise SystemExit(f"Processed directory not found: {PROCESSED_DIR}")

    state = _load_state()
    included = set(state["included_batch_files"])
    channels = set(state["channels"])
    video_ids = set(state["video_ids"])

    all_batches = sorted(PROCESSED_DIR.glob("batch_*.jsonl"))
    new_batches = [b for b in all_batches if b.name not in included]

    if not new_batches:
        print(f"Nothing new -- {len(all_batches):,} batch files already included. "
              f"Unified corpus stands at {state['total_docs']:,} documents.")
        return

    print(f"{len(new_batches)} new batch file(s) to fold in (of {len(all_batches):,} total)...")
    UNIFIED_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_docs = 0
    new_tokens = 0

    with UNIFIED_PATH.open("a", encoding="utf-8") as out_f:
        for batch_file in new_batches:
            print(f"  Processing {batch_file.name}...")
            with batch_file.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"  Error parsing line: {e}")
                        continue

                    doc_id = record.get("id")
                    text = record.get("text")
                    if not doc_id or text is None:
                        continue

                    out_f.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")
                    new_docs += 1
                    new_tokens += len(text.split())

                    channel = record.get("channel")
                    if channel:
                        channels.add(channel)
                    video_id = record.get("video_id")
                    if video_id:
                        video_ids.add(video_id)

            included.add(batch_file.name)

    state["included_batch_files"] = sorted(included)
    state["total_docs"] += new_docs
    state["total_tokens"] += new_tokens
    state["channels"] = sorted(channels)
    state["video_ids"] = sorted(video_ids)
    _save_state(state)

    print(f"\nAdded {new_docs:,} documents ({new_tokens:,} tokens) this run.")
    print(f"Unified corpus now: {state['total_docs']:,} documents, "
          f"{len(channels):,} channels, {len(video_ids):,} raw video files -- {UNIFIED_PATH}")


if __name__ == "__main__":
    main()
