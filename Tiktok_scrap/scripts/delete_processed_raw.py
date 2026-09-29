#!/usr/bin/env python
"""Deletes raw comment files that have already been processed -- no Drive
backup, no archive copy, just removes them from local disk. Use
`Data/scripts/push_to_drive.py --category raw` instead if a Drive-archived
copy should be kept before deleting.

"Already processed" means the filename appears in
`data/state/scrape_state.json`'s `pipeline.processed_raw_files` list --
the pipeline has already turned it into processed/cleaned output, so the
raw copy is no longer needed for anything except a future re-clean (see
CLAUDE.md's "rebuilding processed output" convention -- if that's ever
needed again for files deleted here, they'd have to be re-scraped, there
is no backup).

Defaults to a dry run; pass --execute to actually delete. Every run
appends a summary to data/state/deleted_raw_log.json (audit trail).

Run via the base env: `python scripts/delete_processed_raw.py --execute`
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
STATE_PATH = ROOT / "data" / "state" / "scrape_state.json"
LOG_PATH = ROOT / "data" / "state" / "deleted_raw_log.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                         help="Actually delete files (default: dry run, prints what would be deleted)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only delete the first N eligible files (for testing)")
    args = parser.parse_args()

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    processed_raw_names = set(state["pipeline"]["processed_raw_files"])
    names = sorted(n for n in processed_raw_names if (RAW_DIR / n).exists())
    if args.limit is not None:
        names = names[: args.limit]

    total_bytes = sum((RAW_DIR / n).stat().st_size for n in names)
    print(f"{len(names):,} already-processed raw file(s) found ({total_bytes / 1e9:.2f} GB)")

    deleted = 0
    deleted_bytes = 0
    for n in names:
        p = RAW_DIR / n
        size = p.stat().st_size
        if args.execute:
            try:
                p.unlink()
                deleted += 1
                deleted_bytes += size
            except OSError as e:
                print(f"  WARNING: failed to delete {p}: {e}")

    if args.execute:
        print(f"Deleted {deleted:,} files, freed {deleted_bytes / 1e9:.2f} GB.")
    else:
        print(f"[dry run] Would delete {len(names):,} files, freeing {total_bytes / 1e9:.2f} GB. "
              f"Pass --execute to actually delete.")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = json.loads(LOG_PATH.read_text(encoding="utf-8")) if LOG_PATH.exists() else []
    log.append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "execute": args.execute,
        "count": len(names),
        "bytes": total_bytes,
        "deleted": deleted,
        "deleted_bytes": deleted_bytes,
    })
    LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Logged to {LOG_PATH}")


if __name__ == "__main__":
    main()
