"""Persisted, resumable state: per-channel video discovery progress,
per-video comment-scrape progress, and pipeline progress. Backed by a
single JSON file so scraping and the pipeline can be interrupted (crash,
soft-block, captcha) and resumed without redoing work. Mirrors
Youtube_scrap's/Mountada_djelfa_scrap's state.py pattern.

No quota block, unlike Youtube_scrap -- there's no official API budget
here (TikTok's endpoints are reverse-engineered, not a sanctioned quota-
metered API), same as Mountada_djelfa_scrap's forum crawl. `session_health`
tracks consecutive request failures instead, as a soft-block/captcha
signal for the scraper to back off on, since that's the real failure mode
this source has that the other two don't.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._atomic import replace_with_retry

_DEFAULT_STATE = {
    "channels": {},
    "videos": {},
    "session_health": {"consecutive_failures": 0},
    "pipeline": {"processed_raw_files": []},
}


class State:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = self._load()
        # See Youtube_scrap's state.py -- cached to avoid an O(n) list scan
        # per is_raw_file_processed() call against a growing raw-file count.
        self._processed_raw_files_set: set[str] = set(self.data["pipeline"]["processed_raw_files"])

    def _load(self) -> dict:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(json.dumps(_DEFAULT_STATE))
        for key, default in _DEFAULT_STATE.items():
            data.setdefault(key, json.loads(json.dumps(default)))
        return data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        replace_with_retry(tmp_path, self.path)

    # --- channel (video discovery) progress ---
    def channel_state(self, username: str) -> dict:
        return self.data["channels"].setdefault(
            username,
            {
                "video_ids": [],
                "videos_found": 0,
                "last_discovered_at": None,
                "completed": False,
            },
        )

    # --- video (comment scrape) progress ---
    def video_state(self, video_id: str) -> dict:
        return self.data["videos"].setdefault(
            video_id,
            {
                "status": "pending",
                "channel": None,
                "comment_count": 0,
                # See scrape.py's _scrape_one_video docstring -- TikTok's
                # comment/list endpoint stops pagination early (has_more=0)
                # before `total` is reached, for unauthenticated requests.
                # `api_reported_total` is what TikTok said the true count
                # was at scrape time; `truncated` flags whether
                # comment_count fell short of it.
                "api_reported_total": None,
                "truncated": False,
            },
        )

    def is_video_done(self, video_id: str) -> bool:
        return self.data["videos"].get(video_id, {}).get("status") == "done"

    # --- session health (soft-block / captcha backoff signal) ---
    def record_request_result(self, ok: bool) -> int:
        """Returns the current consecutive-failure count after recording
        this result -- caller decides what to do with it (e.g. pause and
        retry with --no-headless past some threshold)."""
        if ok:
            self.data["session_health"]["consecutive_failures"] = 0
        else:
            self.data["session_health"]["consecutive_failures"] += 1
        return self.data["session_health"]["consecutive_failures"]

    # --- pipeline progress ---
    def is_raw_file_processed(self, filename: str) -> bool:
        return filename in self._processed_raw_files_set

    def mark_raw_file_processed(self, filename: str) -> None:
        self.data["pipeline"]["processed_raw_files"].append(filename)
        self._processed_raw_files_set.add(filename)
