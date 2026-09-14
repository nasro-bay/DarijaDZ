#!/usr/bin/env python
"""CLI: counts how many comments (+ replies) were scraped *today*,
straight from the raw JSONL files -- no need to run run_pipeline.py
first. Ported from Youtube_scrap's count_today_scraped.py -- same
`scrape_date`-per-record convention.

    python scripts/count_today_scraped.py
    python scripts/count_today_scraped.py --date 2026-09-12
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="date to count (YYYY-MM-DD), defaults to today")
    args = parser.parse_args()
    target_date = args.date or date.today().isoformat()

    raw_dir = ROOT / "data" / "raw"
    count = 0
    files_touched = 0
    for path in sorted(raw_dir.glob("*.jsonl")):
        file_count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("scrape_date") == target_date:
                    file_count += 1
        if file_count:
            files_touched += 1
            count += file_count

    print(f"{target_date}: {count:,} comments scraped, across {files_touched:,} raw video file(s)")


if __name__ == "__main__":
    main()
