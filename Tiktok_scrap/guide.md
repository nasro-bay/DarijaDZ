# Running Tiktok_scrap

Commands below are PowerShell/bash-agnostic (plain `python` -- this
package needs no GPU/torch, runs in the base env like the other two
sources' pipelines).

## 1. Setup

```bash
cd Tiktok_scrap
pip install -r requirements.txt
playwright install chromium   # one-time; only exercised as a discovery fallback -- see below
```

`yt-dlp` is the primary video-discovery mechanism (see `PLAN.md`'s
"Fix: yt-dlp as the primary discovery method") -- make sure it's on
PATH and reasonably current:

```bash
yt-dlp --version
python -m pip install -U yt-dlp   # if it's more than a few months old
```

## 2. Scrape

```bash
python scripts/run_scrape.py
```

Scrapes every channel in `config/seed_channels.yaml` (currently just
`anya.hamimed`). Resumable at two levels:
- **Discovery**: re-running tops up the channel's known video-id list
  (via yt-dlp) rather than re-discovering from scratch.
- **Per-video**: videos already marked `"status": "done"` in
  `data/state/scrape_state.json` are skipped.

Useful flags:
- `--max-videos N` -- cap videos discovered per channel (smoke-test
  before a full run).
- `--max-comments N` -- cap comments scraped per video.
- `--workers` / `--reply-workers` -- concurrency (defaults 8 / 4).
- `--no-replies` -- skip nested replies, ~faster.
- `--no-headless` -- show the browser window; only matters if yt-dlp
  discovery fails and it falls back to this project's own Playwright
  interception (see PLAN.md) -- lets you watch/solve a captcha manually.

Watch `data/state/scrape_state.json`'s `session_health.
consecutive_failures` if a run seems to be failing a lot -- rising
consecutive failures across per-video comment fetches is the signal
TikTok may be soft-blocking this session (see PLAN.md's scalability
section for what to do about it: mainly, don't run back-to-back bursts).

## 3. Pipeline

```bash
python scripts/run_pipeline.py
```

Cleans raw comments (`clean_text.py` -- byte-identical rules to
Youtube_scrap's/Mountada_djelfa_scrap's), drops near-empty results
(TikTok's comment culture runs much higher on pure-emoji reactions than
YouTube's -- expect a meaningfully higher drop rate here, that's real,
not a bug, see PLAN.md), and writes
`data/processed/batch_<date>.jsonl` + updates `data/logs/log.json`.
Safe to rerun -- resumable via the same `State`, skips already-processed
raw files.

Dedup (`dedup.py`, MinHash/LSH) is **not** run by default -- same
reasoning as the other two sources (sequential LSH cost dominates at
scale; kept as an optional standalone pass here, not discovered the
hard way this time).

## 4. Check today's numbers without running the pipeline

```bash
python scripts/count_today_scraped.py
python scripts/count_today_scraped.py --date 2026-09-12
```

Counts straight from `data/raw/*.jsonl`'s `scrape_date` field --
independent of whether the pipeline has been run yet.

## 5. Check the comment-pagination-cap limitation

```bash
python scripts/report_truncated.py
```

Summarizes, per channel, how many already-scraped videos are known to
have hit TikTok's unauthenticated pagination cap (see `PLAN.md`'s
"Known limitation: comment pagination cap") -- comment *counts* on
those videos are known-partial; comment *text* collected is still real
and usable. Videos scraped before this tracking existed show up as
"unknown" rather than being assumed complete.

## 6. Archive raw/processed data to Drive (frees local disk space)

```bash
python ../Data/scripts/push_to_drive.py --source Tiktok_scrap --execute
```

Archives raw files already folded into a processed batch, and processed
batches already folded into `data/unified_corpus.jsonl`, to the `DarijaDZ`
rclone remote (`DarijaDZ:DarijaDZ/data_archive/Tiktok_scrap/`) -- only
deletes the local copy after a checksummed upload is independently
re-verified (`rclone check`).

Options (all optional, combine freely):
- `--execute` -- actually delete local files after a verified copy.
  **Omit this for a dry run** (prints what would be archived/deleted
  without touching anything) -- always sanity-check a dry run first after
  a pipeline change.
- `--source Youtube_scrap` / `--source Tiktok_scrap` -- only archive that
  one source (default: both).
- `--category raw` / `--category processed` -- only archive that one
  category for the chosen source(s) (default: both). E.g. to push just
  this source's processed batches: `--source Tiktok_scrap --category
  processed --execute`.
- `--limit N` -- cap it to the first N eligible files per category (for
  testing the pipeline on a small batch before trusting it with everything).

Safe to interrupt (Ctrl+C) and rerun -- nothing gets deleted locally until
its whole category's copy+check both pass, and `rclone copy --checksum`
skips files already uploaded, so a rerun continues rather than
re-uploading from scratch.

## 7. Publish the channel registry (so collaborators don't re-scrape your channels)

```bash
python ../Data/scripts/push_channel_registry.py
```

Publishes a snapshot of every channel already scraped (YouTube from
`channel_names.json`, TikTok from `scrape_state.json`'s `channels` dict) to
`DarijaDZ:DarijaDZ/channel_registry/scraped_channels.json`. No options --
always publishes everything from both sources, overwriting the previous
snapshot (one-way publish, not a merge -- see the script's own docstring).

Not wired into the scrape/pipeline commands above, so it goes stale
whenever new channels get scraped -- **rerun this after adding channels**,
not just after archiving data. A future collaborator would pull this file
and check it before picking which channels to scrape next, so it doing
nothing on your end just means their view goes stale too.

## Current state (as of 2026-09-12)

**Known limitation** (see `PLAN.md` for the full investigation):
TikTok's comment API caps how many comments an unauthenticated request
can page through per video -- confirmed on ~45% of a random sample,
losses ranging from a few percent to 80%+ on popular videos. Not
fixable without an authenticated account session (a bigger, riskier
step not taken without sign-off). Every scrape now records this per
video (`report_truncated.py` above); the 2,768 videos scraped before
this fix existed are marked "unknown," not silently assumed complete.

**Not yet done**:
- `session_health` only tracks per-video comment-fetch failures, not
  yt-dlp discovery failures yet.
- A genuine uncapped 1-hour timed run hasn't been done -- the two
  scalability data points in `PLAN.md` (46.6 min vs. 90.2s for similar
  volumes) disagree enough that neither should be treated as the
  final sustained-rate number.
- Whether to pursue an authenticated-session fix for the pagination
  cap -- explicitly left as a decision for the user, not made here.

See `PLAN.md` for the full design (why this shape, what was ported
from the external tool vs. built new) and the discovery-block
investigation + fix.
