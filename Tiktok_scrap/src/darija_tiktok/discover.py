"""Full-channel TikTok video discovery, ported from the external
TikTok_scrap tool's channel_comments.py (same repo the tiktokcomment/
library here also came from), adapted to this project's layout.

Three layers, tried in order (same strategy as the original tool):
  1. Playwright: load the profile page, scroll to trigger TikTok's own
     `item_list`/`creator/post` API calls, intercept those responses.
  2. Direct API pagination using the browser's own cookies + the
     profile's `secUid`, once Playwright has established a session --
     usually the fastest/most complete path once available.
  3. A static-HTTP fallback (no browser) that only reads the profile
     page's initial server-rendered rehydration data -- much shallower
     (first page only), used only if Playwright discovery fails outright
     (e.g. no Playwright browsers installed, or an early crash).

Not an official, sanctioned API -- these are TikTok's own internal
endpoints, reverse-engineered by the upstream tool, not a documented/
rate-limited-by-contract API the way YouTube Data API v3 is. Same kind
of load-bearing assumption as Mountada_djelfa_scrap's robots.txt
situation: flagged here, not silently glossed over, and worth revisiting
if TikTok's frontend changes break these internal endpoints.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger

VIDEO_ID_PATTERN = re.compile(r"/video/(\d+)")


def normalize_username(raw: str) -> str:
    """Extract a clean username from a bare handle, @handle, or full
    profile URL."""
    cleaned = raw.strip()
    url_match = re.search(r"tiktok\.com/@([^/?#]+)", cleaned)
    if url_match:
        return url_match.group(1).lstrip("@")
    return cleaned.lstrip("@").strip("/")


def extract_video_items_from_json(value: Any) -> tuple[list[str], dict[str, Any]]:
    """Recursively search a JSON structure (e.g. the profile page's
    rehydration blob) for video IDs and item metadata, avoiding user-
    profile objects that happen to share some field names."""
    found_ids: list[str] = []
    metadata_map: dict[str, Any] = {}

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            is_user_profile = "uniqueId" in obj and "nickname" in obj and "desc" not in obj
            if not is_user_profile:
                item_id = obj.get("aweme_id") or obj.get("itemId") or obj.get("awemeId")
                if not item_id and "desc" in obj and ("stats" in obj or "video" in obj or "author" in obj):
                    item_id = obj.get("id")
                if item_id and str(item_id).isdigit() and len(str(item_id)) >= 15:
                    vid = str(item_id)
                    if vid not in found_ids:
                        found_ids.append(vid)
                    if vid not in metadata_map:
                        metadata_map[vid] = {
                            "video_id": vid,
                            "desc": obj.get("desc", ""),
                            "create_time": obj.get("createTime"),
                            "author": (
                                obj.get("author", {}).get("uniqueId")
                                if isinstance(obj.get("author"), dict)
                                else None
                            ),
                        }
            if "itemList" in obj and isinstance(obj["itemList"], list):
                for sub in obj["itemList"]:
                    if isinstance(sub, (str, int)) and str(sub).isdigit() and len(str(sub)) >= 15:
                        vid = str(sub)
                        if vid not in found_ids:
                            found_ids.append(vid)
                    elif isinstance(sub, dict):
                        _walk(sub)
            for key, val in obj.items():
                if key != "userInfo":
                    _walk(val)
        elif isinstance(obj, list):
            for sub in obj:
                _walk(sub)

    _walk(value)
    return found_ids, metadata_map


def _safe_get_dom_video_ids(page) -> list[str]:
    """Extract video IDs from the rendered DOM, without crashing on
    SPA client-side navigation mid-scrape."""
    vids: list[str] = []
    try:
        links = page.locator('a[href*="/video/"]').evaluate_all(
            "elements => elements.map(el => el.href)"
        )
        for link in links:
            for match in VIDEO_ID_PATTERN.findall(str(link)):
                if match not in vids:
                    vids.append(match)
    except Exception:
        pass
    return vids


def _extract_sec_uid_from_page(page) -> str:
    """Extract the profile's TikTok `secUid` from the rehydration script
    tag -- needed for direct API pagination (layer 2)."""
    try:
        html_content = page.content()
        soup = BeautifulSoup(html_content, "html.parser")
        tag = soup.find("script", attrs={"id": "__UNIVERSAL_DATA_FOR_REHYDRATION__"})
        if tag and tag.string:
            data = json.loads(tag.string)

            def _find_sec_uid(obj):
                if isinstance(obj, dict):
                    if "secUid" in obj and isinstance(obj["secUid"], str) and len(obj["secUid"]) > 30:
                        return obj["secUid"]
                    for v in obj.values():
                        r = _find_sec_uid(v)
                        if r:
                            return r
                elif isinstance(obj, list):
                    for item in obj:
                        r = _find_sec_uid(item)
                        if r:
                            return r
                return None

            return _find_sec_uid(data) or ""
    except Exception:
        pass
    return ""


def _api_paginate_user_videos(
    sec_uid: str,
    username: str,
    session: requests.Session,
    max_videos: Optional[int] = None,
) -> tuple[list[str], dict[str, Any]]:
    """Directly paginate TikTok's `/api/post/item_list/` endpoint using
    the profile's secUid + the active Playwright session's cookies."""
    discovered_ids: list[str] = []
    videos_metadata: dict[str, Any] = {}
    cursor = 0
    has_more = True
    page_num = 0

    logger.info(f"Direct API pagination for @{username} using secUid...")
    while has_more:
        if max_videos and len(discovered_ids) >= max_videos:
            break
        params = {"aid": 1988, "secUid": sec_uid, "count": 35, "cursor": cursor, "coverFormat": 2}
        try:
            resp = session.get("https://www.tiktok.com/api/post/item_list/", params=params, timeout=15)
            if resp.status_code != 200:
                break
            data = resp.json()
        except Exception as exc:
            logger.debug(f"API pagination error: {exc}")
            break

        items = data.get("itemList") or []
        has_more = bool(data.get("hasMore", False))
        new_cursor = int(data.get("cursor", 0) or 0)
        page_num += 1

        for item in items:
            if isinstance(item, dict):
                vid = str(item.get("id") or item.get("aweme_id") or "")
                if vid and vid not in discovered_ids:
                    discovered_ids.append(vid)
                    videos_metadata[vid] = {
                        "video_id": vid,
                        "desc": item.get("desc", ""),
                        "create_time": item.get("createTime"),
                        "author": (
                            item.get("author", {}).get("uniqueId")
                            if isinstance(item.get("author"), dict)
                            else username
                        ),
                    }
        if items:
            logger.info(f"API page {page_num}: +{len(items)} videos | Total: {len(discovered_ids)}")
        else:
            break
        if not has_more or new_cursor == 0 or new_cursor <= cursor:
            break
        cursor = new_cursor
        time.sleep(0.4)

    return discovered_ids, videos_metadata


def discover_videos_playwright(
    username: str,
    browser_data_dir: Path,
    max_videos: Optional[int] = None,
    headless: bool = True,
    wait_per_scroll: float = 1.5,
) -> tuple[list[str], dict[str, Any]]:
    """Full-channel video discovery: load the profile, scroll to trigger
    TikTok's own API calls, intercept the responses; then fall back to
    direct API pagination via the resulting session's cookies."""
    from playwright.sync_api import sync_playwright

    clean_username = normalize_username(username)
    profile_url = f"https://www.tiktok.com/@{clean_username}"
    logger.info(f"Launching Playwright (headless={headless}) to discover videos for @{clean_username}...")

    discovered_ids: list[str] = []
    videos_metadata: dict[str, Any] = {}
    has_more_feed = True
    browser_data_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(browser_data_dir),
            headless=headless,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"],
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        def handle_response(response):
            nonlocal has_more_feed
            url = response.url
            if "item_list" in url or "creator/post" in url:
                try:
                    res_json = response.json()
                    has_more_feed = bool(res_json.get("hasMore", True))
                    items = res_json.get("itemList") or res_json.get("items") or []
                    new_count = 0
                    for item in items:
                        if isinstance(item, dict):
                            vid = str(item.get("id") or item.get("aweme_id") or "")
                            if vid and vid not in discovered_ids:
                                discovered_ids.append(vid)
                                videos_metadata[vid] = {
                                    "video_id": vid,
                                    "desc": item.get("desc", ""),
                                    "create_time": item.get("createTime"),
                                    "author": (
                                        item.get("author", {}).get("uniqueId")
                                        if isinstance(item.get("author"), dict)
                                        else clean_username
                                    ),
                                }
                                new_count += 1
                    if items:
                        logger.info(f"Intercepted API batch: {len(items)} items (+{new_count} new) | Total: {len(discovered_ids)}")
                except Exception as e:
                    logger.debug(f"Response parse note: {e}")

        page.on("response", handle_response)

        try:
            logger.info(f"Navigating to {profile_url}...")
            page.goto(profile_url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(5000)

            sec_uid = _extract_sec_uid_from_page(page)

            try:
                html_content = page.content()
                soup = BeautifulSoup(html_content, "html.parser")
                rehydration_tag = soup.find("script", attrs={"id": "__UNIVERSAL_DATA_FOR_REHYDRATION__"})
                if rehydration_tag and rehydration_tag.string:
                    init_data = json.loads(rehydration_tag.string)
                    init_ids, init_meta = extract_video_items_from_json(init_data)
                    for vid in init_ids:
                        if vid not in discovered_ids:
                            discovered_ids.append(vid)
                    videos_metadata.update(init_meta)
            except Exception as e:
                logger.debug(f"Rehydration parse: {e}")

            for vid in _safe_get_dom_video_ids(page):
                if vid not in discovered_ids:
                    discovered_ids.append(vid)

            logger.info(f"Initial render: {len(discovered_ids)} videos found. Scrolling feed...")

            consecutive_no_new = 0
            max_scroll_attempts = 150
            import random

            for _ in range(max_scroll_attempts):
                if max_videos and len(discovered_ids) >= max_videos:
                    break
                prev_count = len(discovered_ids)

                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(int(wait_per_scroll * 1000 + random.randint(200, 600)))
                page.evaluate("window.scrollBy(0, -400)")
                page.wait_for_timeout(300)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(800)

                for vid in _safe_get_dom_video_ids(page):
                    if vid not in discovered_ids:
                        discovered_ids.append(vid)

                if len(discovered_ids) > prev_count:
                    consecutive_no_new = 0
                else:
                    consecutive_no_new += 1
                if consecutive_no_new >= 6 or (not has_more_feed and consecutive_no_new >= 2):
                    logger.info("Reached end of profile feed.")
                    break
        except Exception as exc:
            logger.warning(f"Browser interaction notice: {exc}")

        if sec_uid:
            try:
                session = requests.Session()
                session.headers.update({
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Referer": "https://www.tiktok.com/",
                    "Origin": "https://www.tiktok.com",
                })
                for ck in context.cookies():
                    session.cookies.set(ck["name"], ck["value"], domain=ck.get("domain", ".tiktok.com"))
                api_ids, api_meta = _api_paginate_user_videos(sec_uid, clean_username, session, max_videos)
                for vid in api_ids:
                    if vid not in discovered_ids:
                        discovered_ids.append(vid)
                videos_metadata.update(api_meta)
            except Exception as exc:
                logger.debug(f"Direct API pagination attempt failed: {exc}")

        context.close()

    if max_videos and max_videos > 0:
        discovered_ids = discovered_ids[:max_videos]
    return discovered_ids, videos_metadata


def discover_videos_static(username: str) -> tuple[list[str], dict[str, Any]]:
    """Fast static-HTTP fallback: only the profile page's first-render
    rehydration data, no browser -- shallower than Playwright discovery,
    used only if that fails outright."""
    clean_username = normalize_username(username)
    profile_url = f"https://www.tiktok.com/@{clean_username}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        response = requests.get(profile_url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        script = soup.find("script", attrs={"id": "__UNIVERSAL_DATA_FOR_REHYDRATION__"})
        if script and script.string:
            return extract_video_items_from_json(json.loads(script.string))
    except Exception as exc:
        logger.warning(f"Static discovery fallback notice: {exc}")
    return [], {}


def discover_videos_ytdlp(username: str, max_videos: Optional[int] = None) -> list[str]:
    """Primary discovery method: yt-dlp's own TikTok user-page extractor.

    yt-dlp is a large, actively-maintained project with a TikTok
    extractor that gets patched continuously against TikTok's anti-bot
    changes -- something a small hand-rolled scraper can't keep up with
    on its own. Verified empirically (2026-09-12) to succeed where this
    project's own Playwright-based interception
    (discover_videos_playwright) was blocked outright on the same
    network/session: yt-dlp found 131 videos for @anya.hamimed while
    Playwright interception returned 0. Shells out to the `yt-dlp` CLI
    (not its Python API, which churns faster/is less stable across
    versions) with `--flat-playlist` (list only, no per-video metadata
    fetch -- fast) and `--print "%(id)s"`.

    Requires `yt-dlp` on PATH (`pip install yt-dlp`) and, for its best
    anti-detection mode (browser TLS/HTTP2 impersonation), `curl_cffi`
    installed too (`pip install curl_cffi`) -- yt-dlp works without it,
    just with a one-line warning, and still succeeded in testing either
    way.
    """
    import subprocess

    clean_username = normalize_username(username)
    profile_url = f"https://www.tiktok.com/@{clean_username}"
    cmd = ["yt-dlp", "--flat-playlist", "--print", "%(id)s"]
    if max_videos:
        cmd += ["--playlist-end", str(max_videos)]
    cmd.append(profile_url)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except FileNotFoundError:
        logger.warning("yt-dlp not found on PATH -- falling back to Playwright discovery")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("yt-dlp discovery timed out")
        return []

    if result.returncode != 0:
        logger.warning(f"yt-dlp discovery failed (exit {result.returncode}): {result.stderr[:300]}")

    ids = [line.strip() for line in result.stdout.splitlines() if line.strip().isdigit()]
    logger.info(f"yt-dlp discovered {len(ids)} video(s) for @{clean_username}")
    return ids


def discover_channel_videos(
    username: str,
    browser_data_dir: Path,
    max_videos: Optional[int] = None,
    headless: bool = True,
    wait_time: float = 1.5,
) -> tuple[list[str], dict[str, Any]]:
    """Top-level entry point: yt-dlp first (see discover_videos_ytdlp --
    the actively-maintained, more robust option), then this project's own
    Playwright interception, then a static-HTTP fallback if that raises
    too. Caller (scrape.py) is responsible for merging with previously-
    discovered ids via State -- no on-disk caching here, unlike the
    upstream tool (State already is that cache, one layer up)."""
    clean_username = normalize_username(username)

    ytdlp_ids = discover_videos_ytdlp(clean_username, max_videos=max_videos)
    if ytdlp_ids:
        return ytdlp_ids, {}

    logger.warning("yt-dlp discovery returned nothing -- falling back to Playwright interception")
    try:
        video_ids, videos_meta = discover_videos_playwright(
            username=clean_username,
            browser_data_dir=browser_data_dir,
            max_videos=max_videos,
            headless=headless,
            wait_per_scroll=wait_time,
        )
    except Exception as exc:
        logger.warning(f"Playwright discovery issue ({exc}). Trying static fallback...")
        video_ids, videos_meta = discover_videos_static(clean_username)

    if not video_ids:
        video_ids, videos_meta = discover_videos_static(clean_username)

    logger.info(f"Discovered {len(video_ids)} video(s) for @{clean_username}")
    return video_ids, videos_meta
