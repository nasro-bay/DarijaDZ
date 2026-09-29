#!/usr/bin/env python
"""CLI: counts how many posts (= forum comments), threads and subforums were scraped *today*,
straight from the raw JSONL files -- no need to run run_pipeline.py first.

Works because every post record is stamped with `scrape_date` (the date
it was actually scraped) at scrape time, in scrape.py's post-record
builder -- independent of which raw file it ends up in (raw files are
nested per subforum under data/raw/djelfa/), and independent of
run_pipeline's own counters (which only exist after a pipeline run).
Same technique as Youtube_scrap/scripts/count_today_scraped.py.

    python scripts/count_today_scraped.py
    python scripts/count_today_scraped.py --date 2026-08-30
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date", default=None,
        help="date to count (YYYY-MM-DD), defaults to today",
    )
    args = parser.parse_args()
    target_date = args.date or date.today().isoformat()

    raw_dir = ROOT / "data" / "raw" / "djelfa"
    count = 0
    files_touched = 0
    threads: set[str] = set()
    subforums: set[str] = set()
    for path in sorted(raw_dir.rglob("*.jsonl")):
        file_count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("scrape_date") == target_date:
                    file_count += 1
                    threads.add(str(rec.get("thread_id")))
                    subforums.add(str(rec.get("subforum_id")))
        if file_count:
            files_touched += 1
            count += file_count

    print(f"{target_date}:")
    print(f"  comments (posts): {count:,}")
    print(f"  threads:          {len(threads):,}")
    print(f"  subforums:        {len(subforums):,}")
    print(f"  raw thread files: {files_touched:,}")

if __name__ == "__main__":
    main()
