"""Scrapes comments (+ nested replies) for a channel's videos, resuming
via State rather than the upstream tool's "skip if output file exists"
check. Writes one raw file per video, `data/raw/<video_id>.jsonl` --
matching Youtube_scrap's raw-file convention (one file per video id) so
pipeline.py's clean -> schema -> processed-batch shape carries over
almost unchanged.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from . import discover
from .state import State
from .tiktokcomment import TiktokComment


def _write_video_raw_file(
    raw_path: Path,
    video_id: str,
    username: str,
    comments,  # tiktokcomment.typing.Comments
) -> int:
    """One JSON line per top-level comment AND one per reply (flattened,
    each carrying its own parent_comment_id) -- matches the "one flat
    record per retained unit" shape pipeline.py expects, same as
    Youtube_scrap's raw comment records."""
    scrape_date = date.today().isoformat()
    n_written = 0
    with raw_path.open("w", encoding="utf-8") as f:
        for c in comments.comments:
            if not c.comment:
                continue
            f.write(json.dumps({
                "comment_id": c.comment_id,
                "text": c.comment,
                "video_id": video_id,
                "channel": username,
                "is_reply": False,
                "parent_comment_id": None,
                "scrape_date": scrape_date,
            }, ensure_ascii=False) + "\n")
            n_written += 1
            for r in c.replies:
                if not r.comment:
                    continue
                f.write(json.dumps({
                    "comment_id": r.comment_id,
                    "text": r.comment,
                    "video_id": video_id,
                    "channel": username,
                    "is_reply": True,
                    "parent_comment_id": c.comment_id,
                    "scrape_date": scrape_date,
                }, ensure_ascii=False) + "\n")
                n_written += 1
    return n_written


def _scrape_one_video(
    video_id: str,
    username: str,
    raw_dir: Path,
    max_comments: Optional[int],
    include_replies: bool,
    reply_workers: int,
) -> tuple[str, int, bool, Optional[int], bool]:
    """Returns (video_id, comments_written, ok, api_reported_total, truncated).

    `api_reported_total` / `truncated` record a confirmed, structural
    TikTok limit for unauthenticated requests: `comment/list` reports
    `has_more=0` at some point during pagination even though `total`
    says there's much more -- verified empirically (2026-09-12) to
    happen regardless of page size or retry, and even with real browser
    cookies attached, on ~45% of a random sample, with the loss ranging
    from a few percent up to 80%+ on popular videos. Not fixable without
    an authenticated account session (a materially bigger, riskier step
    -- real account, ban exposure -- not taken here without explicit
    sign-off). So: recorded, not silently swallowed -- `truncated=True`
    means "this video's comments are a known-partial sample," which
    matters for any later analysis that assumes comment counts reflect
    real engagement.

    ok=False on a fetch exception -- caller feeds this into
    State.record_request_result() for soft-block/captcha backoff
    detection.
    """
    try:
        client = TiktokComment(reply_workers=reply_workers, include_replies=include_replies)
        comments = client.get_all_comments(aweme_id=video_id, max_comments=max_comments, delay=0.05)
        raw_path = raw_dir / f"{video_id}.jsonl"
        n = _write_video_raw_file(raw_path, video_id, username, comments)
        top_saved = len(comments.comments)
        api_total = comments.total
        truncated = bool(api_total and api_total > top_saved)
        return video_id, n, True, api_total, truncated
    except Exception as exc:
        logger.warning(f"Failed to scrape video {video_id}: {exc}")
        return video_id, 0, False, None, False


def scrape_channel(
    username: str,
    state: State,
    raw_dir: Path,
    browser_data_dir: Path,
    *,
    workers: int = 8,
    reply_workers: int = 4,
    include_replies: bool = True,
    max_videos: Optional[int] = None,
    max_comments_per_video: Optional[int] = None,
    headless: bool = True,
    consecutive_failure_pause_threshold: int = 8,
) -> dict:
    """Discovers a channel's videos (merged with any previously
    discovered ids in State, so re-running tops up rather than
    re-discovering from scratch), then scrapes comments for every video
    not already marked done in State. Resumable at both the discovery
    and per-video level.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    channel = state.channel_state(username)

    discovered_ids, _meta = discover.discover_channel_videos(
        username=username,
        browser_data_dir=browser_data_dir,
        max_videos=max_videos,
        headless=headless,
    )
    known_ids = set(channel["video_ids"])
    new_ids = [v for v in discovered_ids if v not in known_ids]
    channel["video_ids"].extend(new_ids)
    channel["videos_found"] = len(channel["video_ids"])
    channel["last_discovered_at"] = datetime.now().isoformat()
    state.save()
    logger.info(f"@{username}: {len(channel['video_ids'])} known videos ({len(new_ids)} newly discovered)")

    pending = [v for v in channel["video_ids"] if not state.is_video_done(v)]
    logger.info(f"@{username}: {len(pending)}/{len(channel['video_ids'])} videos pending comment scrape")

    videos_scraped = 0
    comments_written = 0
    videos_truncated = 0
    estimated_missing_comments = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _scrape_one_video, vid, username, raw_dir, max_comments_per_video, include_replies, reply_workers
            ): vid
            for vid in pending
        }
        for i, future in enumerate(as_completed(futures), start=1):
            video_id, n, ok, api_total, truncated = future.result()
            failures = state.record_request_result(ok)
            if ok:
                video_state = state.video_state(video_id)
                video_state["status"] = "done"
                video_state["channel"] = username
                video_state["comment_count"] = n
                video_state["api_reported_total"] = api_total
                video_state["truncated"] = truncated
                videos_scraped += 1
                comments_written += n
                if truncated:
                    videos_truncated += 1
                    estimated_missing_comments += max(0, api_total - video_state.get("comment_count", 0))
            if failures >= consecutive_failure_pause_threshold:
                logger.warning(
                    f"{failures} consecutive failures -- likely soft-blocked/captcha'd. "
                    "Stopping this run early; rerun later (--no-headless helps solve captchas)."
                )
                state.save()
                break
            if i % 20 == 0:
                state.save()

    state.save()
    if videos_truncated:
        logger.warning(
            f"@{username}: {videos_truncated}/{videos_scraped} videos hit TikTok's unauthenticated "
            f"pagination limit (has_more=0 before total is reached) -- comments for those videos are "
            f"known-partial, not complete. See PLAN.md's 'Known limitation: comment pagination cap'."
        )
    elapsed = time.time() - start
    return {
        "channel": username,
        "videos_scraped": videos_scraped,
        "comments_written": comments_written,
        "videos_truncated": videos_truncated,
        "estimated_missing_comments": estimated_missing_comments,
        "elapsed_seconds": elapsed,
    }
