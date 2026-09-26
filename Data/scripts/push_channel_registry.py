#!/usr/bin/env python
"""Publishes a consolidated snapshot of "which channels have already been
scraped" to Google Drive (`DarijaDZ:DarijaDZ/channel_registry/`) -- so a
future collaborator can pull it *before* they start scraping and skip
channels already covered, instead of duplicating work.

One-way publish, not a two-way merge: this script only ever reads this
project's own local state and overwrites the Drive snapshot with it. It
never reads anything back from Drive -- there's no merge-conflict handling
here because there's exactly one writer (you). A future collaborator's own
setup would pull this file and diff against their own planned channel list
themselves; this script doesn't need to know how they do that.

Sources covered: YouTube (`channel_names.json`, keyed by channel id) and
TikTok (`scrape_state.json`'s `channels` dict, keyed by handle). djelfa.info
isn't included -- it's a single forum site organized by subforum, not a list
of independent channels/creators, so "which channels are already covered"
doesn't apply there the same way.

Run via the base env: `python Data/scripts/push_channel_registry.py`
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE_PATH = "DarijaDZ:DarijaDZ/channel_registry/scraped_channels.json"

YOUTUBE_CHANNEL_NAMES = ROOT / "Youtube_scrap" / "data" / "state" / "channel_names.json"
TIKTOK_SCRAPE_STATE = ROOT / "Tiktok_scrap" / "data" / "state" / "scrape_state.json"


def collect_youtube() -> list[dict]:
    if not YOUTUBE_CHANNEL_NAMES.exists():
        return []
    names = json.loads(YOUTUBE_CHANNEL_NAMES.read_text(encoding="utf-8"))
    return [
        {
            "source": "youtube",
            "id": channel_id,
            "handle": info.get("custom_url"),
            "title": info.get("title"),
        }
        for channel_id, info in names.items()
    ]


def collect_tiktok() -> list[dict]:
    if not TIKTOK_SCRAPE_STATE.exists():
        return []
    state = json.loads(TIKTOK_SCRAPE_STATE.read_text(encoding="utf-8"))
    return [
        {
            "source": "tiktok",
            "handle": handle,
            "videos_found": info.get("videos_found"),
            "completed": info.get("completed"),
            "last_discovered_at": info.get("last_discovered_at"),
        }
        for handle, info in state.get("channels", {}).items()
    ]


def main() -> None:
    channels = collect_youtube() + collect_tiktok()
    registry = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "channel_count": len(channels),
        "channels": channels,
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
        tmp_path = f.name

    try:
        print(f"Publishing {len(channels)} channels "
              f"({sum(1 for c in channels if c['source'] == 'youtube')} youtube, "
              f"{sum(1 for c in channels if c['source'] == 'tiktok')} tiktok) -> {REMOTE_PATH}")
        result = subprocess.run(
            ["rclone", "copyto", tmp_path, REMOTE_PATH],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print("--- STDERR ---")
            print(result.stderr)
            raise SystemExit("rclone copyto failed -- registry NOT updated on Drive.")
        print("Done.")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
