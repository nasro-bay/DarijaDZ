# YouTube Comment Scraping — Execution Guide

How to run the pipeline in `src/darija_corpus/` end to end: pick videos →
scrape comments → dedupe → check output. See `PLAN.md` for the
architecture and `Readme.md` for the overall corpus project.

## 0. Prerequisites (one-time)

1. **Python deps**: `pip install -r requirements.txt`
2. **API key**: Google Cloud Console → create/select a project → enable
   "YouTube Data API v3" → create an API key.
3. **`.env`**: copy `.env.example` to `.env` (if not already done) and set:
   ```
   YOUTUBE_API_KEY=your_key_here
   ```
   `.env` is gitignored — never commit a real key. Keep `.env.example`
   as a blank template (`YOUTUBE_API_KEY=`) since that file **is**
   committed.

## 1. Pick what to scrape

Two ways to target content — use either or both:

**A. Specific video links** — edit `config/seed_videos.yaml`:
```yaml
videos:
  - "https://www.youtube.com/watch?v=XXXXXXXXXXX"
  - "https://youtu.be/XXXXXXXXXXX"
  - "XXXXXXXXXXX"   # a bare video ID also works
```
Any common YouTube URL shape works (`watch?v=`, `youtu.be/`, `/shorts/`,
`/embed/`, `/live/`), with or without extra query params (`&t=`, `?si=`,
etc.) — `extract_video_id()` strips those.

**B. Whole channels, newest videos first** — edit `config/seed_channels.yaml`:
```yaml
channels:
  - handle: "@example_channel"
    max_videos: 20   # optional cap on the N newest videos; omit for the whole channel
  - channel_id: "UCxxxxxxxxxxxxxxxxxxxxxx"
  - "https://www.youtube.com/@another_example"   # a plain URL/handle/ID string also works
```
Walks the channel's uploads playlist newest-first (cheap — no need for
the expensive `search.list(order=date)`), feeding each video into the
same per-video scraper as option A. A video reached both ways (direct
link and via a channel walk) is only ever scraped once. Entries can be a
dict (`channel_id:`/`handle:`/`max_videos:`) or a plain string — a
channel URL (`/@handle`, `/channel/UCxxx`, `/c/Name`, `/user/Name`), a
bare `@handle`, or a bare channel ID; string entries have no cap.

> **Careful with `max_videos`.** Omitting it means "eventually scrape the
> whole channel" — fine for a small vlogger, but a large news/media
> channel can have thousands of videos, which could take days of quota to
> fully walk. Set an explicit cap for anything you're not sure is small,
> or use `--max-videos` below to override it per run without editing the
> file.

**Don't know which channels to target?** Run the channel-discovery helper
to find Algerian channels worth reviewing:
```
python scripts/discover_channels.py
python scripts/discover_channels.py --keywords "كوميدي جزائري,vlog algérien" --max-results 15
```
This only prints candidates — it never adds anything automatically. Costs
100 quota units per keyword, so don't rerun it repeatedly.

## 2. Scrape comments

```
python scripts/run_scrape.py            # video links from seed_videos.yaml
python scripts/run_scrape_channels.py    # channels from seed_channels.yaml, newest-first
```

Both resolve each target's channel/videos, page through `commentThreads`
(top-level comments + replies), and write raw JSONL to
`data/raw/<video_id>.jsonl`. Each prints per-item results and the quota
used at the end (out of the 10,000/day budget).

- **Resumable**: if either stops mid-run (quota exhausted, network issue,
  Ctrl-C), just rerun the same command later — it picks up exactly where
  it left off (mid comment-page for a video, or mid channel walk).
- **Skips gracefully**: videos with comments disabled are logged and
  skipped, not treated as a failure.
- A channel already fully walked (or that hit its `max_videos` cap) is a
  no-op on rerun — no API calls spent re-checking it, including channels
  configured by `handle` (the handle → channel-id lookup is cached too).
- Optional: `--config path/to/other_list.yaml` on either script to use a
  different config file than the default.
- `run_scrape_channels.py` also takes `--max-videos N`, which caps *every*
  channel in that run to at most N newest videos (using the stricter of
  this and each entry's own `max_videos`, if both are set) — without
  editing the config. Handy for a cheap test run on a channel you haven't
  capped yet:
  ```
  python scripts/run_scrape_channels.py --max-videos 2
  ```

## 3. Run the clean + dedup pipeline

```
python scripts/run_pipeline.py
```

Reads every raw file not yet processed, runs each comment through
`clean_text.clean()` (URL/mention placeholders, elongation/punctuation-run
collapsing, emoji-run collapsing, tachkil stripping, NFKC normalization —
see `src/darija_corpus/clean_text.py`'s module docstring for the full
rule list and rationale), dropping near-empty results
(`comments_dropped_empty` in the log). Cleaning runs **before** dedup so
noise variation doesn't hide near-duplicates from each other. Applies
near-duplicate filtering on the cleaned text (MinHash/LSH — persisted
across runs, so a repost seen in an earlier batch is still caught later),
and writes the schema-conformant corpus to
`data/processed/batch_<date>.jsonl`. Appends one entry to the single
global `data/logs/log.json` with this run's stats and running cumulative
totals.

`script` and `darija_confidence` are `null` for now — the language/dialect
filter is a deferred phase (see `PLAN.md`).

**Known caveat**: `_append_log()` only writes the run's log entry once, at
the very end of the whole per-file loop. If the process is killed after
finishing (and per-file-marking) all raw files but before that final
append, the batch file and `pipeline.processed_raw_files` are already
correct, but the run's stats never make it into `log.json` — that run's
contribution silently reads as "never happened" in the log, even though
the data is real and won't be reprocessed. This actually happened once
(the 2026-08-14 run entry logged all zeros despite
`batch_2026-08-14.jsonl` holding 15,903 real retained comments — confirmed
by cross-checking batch line counts against the log). **Don't trust
`log.json`'s cumulative totals as the sole source of truth for corpus
size** — cross-check against `data/processed/batch_*.jsonl` line counts
(or the unified dataset's row count, step 5) if the numbers need to be
exact.

## 4. Check the output

```
python -c "import json; [print(json.loads(l)['text']) for l in list(open('data/processed/batch_2026-08-16.jsonl', encoding='utf-8'))[:10]]"
```

or open the file directly. Check `data/logs/log.json` for the running
totals (`videos_scraped`, `comments_collected`, `comments_retained`)
toward the 1M → 10M → 50M token milestones from `Readme.md` — but see the
caveat above before treating it as exact.

## 5. Build the unified dataset (optional, for downstream use)

```
python scripts/resolve_channel_names.py   # optional: resolve channel IDs -> titles/handles
python scripts/build_unified_dataset.py   # build + shuffle + README stat sync (see below)
```

- `resolve_channel_names.py` collects every channel ID seen in
  `scrape_state.json` and `data/processed/batch_*.jsonl`, looks up any not
  already cached, and writes/updates
  `data/state/channel_names.json` (title, custom URL, description,
  published date per channel) via one `channels.list` call per 50 IDs
  (cheap — 1 unit/chunk). Safe to rerun; only fetches IDs not already
  cached. Not required for the pipeline itself — it's for
  human-readable channel attribution (e.g. in a dataset card).
- `build_unified_dataset.py` does three things in one run, in order:
  1. Concatenates every `data/processed/batch_*.jsonl` file into
     `Data/youtube_corpus.jsonl`, keeping only `id` and `text` per
     document (source/schema metadata dropped — this is the flat format
     used for tokenizer training). While streaming the batches it also
     tallies the stats needed for step 3 below (word-token counts per
     script, distinct channels, distinct video files) — this only works
     because it reads the *full* processed records (with `channel`/
     `video_id`), before flattening them down to `id`+`text`.
  2. Shuffles that file into `Data/youtube_corpus_shuffled.jsonl` via
     external chunk-shuffle (100K-doc chunks, fixed `SEED = 42`, temp
     chunks written to `Data/_shuffle_chunks/` and cleaned up after) —
     avoids loading the ~2.8M-doc file into RAM at once. Deterministic
     given the same input and seed.
  3. Updates the numeric statistics already present in
     `DarijaDZ/README.md` and `Kaggle_DarijaDz/README.md` (documents,
     word-level tokens, mean tokens/doc, the script-distribution table,
     channel count, raw video files processed) via targeted regex
     substitution keyed to each row's label text — only the numbers
     change, wording/structure untouched. If a README's phrasing around a
     stat ever changes, the matching substitution stops firing and prints
     a `WARNING: pattern not found` instead of silently corrupting the
     file — check for those if the READMEs don't update as expected after
     an edit to their wording.

  Always rewrites the unified/shuffled files and README numbers from
  scratch on every run (not incremental) — rerun after any pipeline run
  that added new batches. There used to be a separate
  `shuffle_unified_dataset.py` script for step 2; it's been folded into
  this one so a single command keeps everything (unified file, shuffled
  file, and the two dataset-card READMEs) in sync.

## 6. Archive raw/processed data to Drive (frees local disk space)

```bash
python ../Data/scripts/push_to_drive.py --source Youtube_scrap --execute
```

Archives raw files already folded into a processed batch, and processed
batches already folded into `data/unified_corpus.jsonl`, to the `DarijaDZ`
rclone remote (`DarijaDZ:DarijaDZ/data_archive/Youtube_scrap/`) -- only
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
  this source's processed batches: `--source Youtube_scrap --category
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

## Repeating the cycle

Add more entries to `seed_videos.yaml` / `seed_channels.yaml`, then rerun
steps 2 and 3 (and step 5 if you need the unified/shuffled dataset and
README stats refreshed). All scripts are additive/incremental —
already-scraped videos, already-walked channels, and already-processed
raw files are skipped automatically, so it's safe to run them as often as
you like. (`build_unified_dataset.py` is the exception — it always
regenerates its output files from scratch.)

## Quota notes

- Daily budget: 10,000 units. `commentThreads.list` (comments) and
  `comments.list` (extra replies) cost 1 unit/page (100 comments/page).
  Resolving a video's channel, or a channel's uploads playlist, costs
  1 unit, done once and cached in state thereafter. Paging a channel's
  video list costs 1 unit per 50 videos. `discover_channels.py` costs
  100 units per keyword — use sparingly.
- If a run stops with a quota message, nothing is lost — state is saved
  incrementally. Just rerun the same script after the daily quota resets
  (resets at midnight Pacific time, per Google's quota policy).

## Troubleshooting

- **`YOUTUBE_API_KEY not set`**: `.env` is missing or empty — see step 0.
- **`No videos configured` / `No channels configured`**: the relevant
  config file has an empty list.
- **A specific video errors out**: check `data/state/scrape_state.json`
  → `videos.<video_id>.status`. `"comments_disabled"` means it was
  skipped intentionally; `"error"` means something else went wrong —
  rerunning the scrape script will retry it.
- **I want more videos than my original `max_videos` cap**: just raise
  `max_videos` (in the config, or via `--max-videos` on the command line)
  and rerun — a channel that stopped because it hit a cap resumes and
  scrapes further, up to the new cap, without re-scraping videos it
  already has. A rerun with the *same or lower* cap stays a true no-op
  (zero API calls). Only a channel that's been walked to the very end
  (no cap, or cap never reached) is marked fully `completed` — that one
  won't resume with a higher cap because there's nothing more to fetch.
- **A channel seems stuck / won't pick up brand-new uploads**: once a
  channel is marked `completed` in state (genuinely fully walked, not
  just capped), reruns are a no-op for it — there's no automatic "check
  for new videos since last time" yet.
