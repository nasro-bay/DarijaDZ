#!/usr/bin/env python
"""Scrapes comments for every channel in config/seed_channels.yaml.
Resumable: discovery tops up known video ids rather than re-discovering
from scratch, and already-scraped videos (State.is_video_done) are
skipped.

    python run_scrape.py
    python run_scrape.py --max-videos 20 --max-comments 50   # smoke test
    python run_scrape.py --no-headless                       # watch the browser (e.g. to solve a captcha)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darija_tiktok.scrape import scrape_channel  # noqa: E402
from darija_tiktok.state import State  # noqa: E402

CONFIG_PATH = ROOT / "config" / "seed_channels.yaml"
DATA_DIR = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape TikTok channel comments")
    parser.add_argument("--max-videos", type=int, default=None, help="cap videos discovered per channel")
    parser.add_argument("--max-comments", type=int, default=None, help="cap comments scraped per video")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reply-workers", type=int, default=4)
    parser.add_argument("--no-replies", action="store_true", help="skip nested replies (faster)")
    parser.add_argument("--no-headless", action="store_true", help="show the browser window")
    args = parser.parse_args()

    channels = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["channels"]

    state_path = DATA_DIR / "state" / "scrape_state.json"
    state = State(state_path)
    raw_dir = DATA_DIR / "raw"
    browser_data_dir = DATA_DIR / "state" / ".browser_data"

    for username in channels:
        result = scrape_channel(
            username,
            state,
            raw_dir,
            browser_data_dir,
            workers=args.workers,
            reply_workers=args.reply_workers,
            include_replies=not args.no_replies,
            max_videos=args.max_videos,
            max_comments_per_video=args.max_comments,
            headless=not args.no_headless,
        )
        print(
            f"@{username}: {result['videos_scraped']} videos, "
            f"{result['comments_written']} comments in {result['elapsed_seconds']:.1f}s"
        )
        if result["videos_truncated"]:
            print(
                f"  {result['videos_truncated']} video(s) hit TikTok's pagination cap -- "
                f"~{result['estimated_missing_comments']:,} comments known-missing on those "
                f"(see PLAN.md's 'Known limitation: comment pagination cap')"
            )


if __name__ == "__main__":
    main()
