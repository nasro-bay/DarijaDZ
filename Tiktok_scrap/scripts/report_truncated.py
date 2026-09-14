#!/usr/bin/env python
"""Summarizes TikTok's comment-pagination cap (see PLAN.md's "Known
limitation: comment pagination cap") across everything scraped so far --
per channel, how many videos are known-truncated and roughly how many
comments that cost.

Videos scraped before this field existed have no `truncated`/
`api_reported_total` recorded (state predates the fix) -- reported
separately as "unknown", not silently counted as complete.

    python scripts/report_truncated.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    state = json.loads((ROOT / "data" / "state" / "scrape_state.json").read_text(encoding="utf-8"))
    videos = state.get("videos", {})

    per_channel = defaultdict(lambda: {"total": 0, "truncated": 0, "unknown": 0, "missing_comments": 0})

    for vid, info in videos.items():
        if info.get("status") != "done":
            continue
        ch = info.get("channel") or "?"
        row = per_channel[ch]
        row["total"] += 1
        if "truncated" not in info:
            row["unknown"] += 1
        elif info["truncated"]:
            row["truncated"] += 1
            api_total = info.get("api_reported_total") or 0
            row["missing_comments"] += max(0, api_total - info.get("comment_count", 0))

    print(f"{'channel':<24}{'videos':>8}{'truncated':>11}{'unknown':>9}{'~missing comments':>20}")
    grand = {"total": 0, "truncated": 0, "unknown": 0, "missing_comments": 0}
    for ch, row in sorted(per_channel.items(), key=lambda kv: -kv[1]["total"]):
        print(f"{ch:<24}{row['total']:>8}{row['truncated']:>11}{row['unknown']:>9}{row['missing_comments']:>20,}")
        for k in grand:
            grand[k] += row[k]
    print(f"{'TOTAL':<24}{grand['total']:>8}{grand['truncated']:>11}{grand['unknown']:>9}{grand['missing_comments']:>20,}")

    if grand["unknown"]:
        print(
            f"\n{grand['unknown']:,} video(s) were scraped before this tracking existed -- "
            "unknown whether they're truncated. Not re-scraped automatically (would add load "
            "for metadata only); rerun run_scrape.py to pick this up on any video re-touched."
        )


if __name__ == "__main__":
    main()
