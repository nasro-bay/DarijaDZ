"""Runs newly-scraped raw comments through clean_text.clean() (dropping
near-empty results) and writes the schema-conformant corpus -- appending
one entry to the single global JSON log (data/logs/log.json) per run.
Ported from Youtube_scrap's pipeline.py; same structure, adjusted for
this source's raw-record fields and nested schema.

Dedup (dedup.py, MinHash/LSH) is NOT run here by default, same reasoning
as Youtube_scrap: it's a standalone opt-in pass over
data/processed/*.jsonl, not part of the default per-run pipeline (its
sequential LSH insert/query cost dominates at scale once the corpus
grows -- see Youtube_scrap/src/darija_corpus/pipeline.py's docstring for
the full story of why that got disabled there).

Cleaning is parallelized across a process pool, same as Youtube_scrap.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from multiprocessing import Pool
from pathlib import Path

from . import clean_text, schema
from .state import State


def _clean(text: str) -> str | None:
    """Worker-process entry point (module-level so it's picklable, incl.
    Windows' spawn start method)."""
    return clean_text.clean(text)


def _append_log(log_path: Path, run_entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = {
            "runs": [],
            "cumulative": {
                "videos_scraped": 0,
                "comments_collected": 0,
                "comments_dropped_empty": 0,
                "comments_retained": 0,
            },
        }
    log["runs"].append(run_entry)
    for key in log["cumulative"]:
        log["cumulative"][key] += run_entry[key]

    tmp_path = log_path.with_suffix(log_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    tmp_path.replace(log_path)


def run_pipeline(
    *,
    raw_dir: Path,
    processed_dir: Path,
    state: State,
    log_path: Path,
    workers: int | None = None,
) -> dict:
    raw_files = sorted(p for p in raw_dir.glob("*.jsonl") if not state.is_raw_file_processed(p.name))

    comments_collected = 0
    comments_dropped_empty = 0
    comments_retained = 0

    today = date.today().isoformat()
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / f"batch_{today}.jsonl"

    num_workers = workers if workers is not None else (os.cpu_count() or 1)
    STATE_SAVE_INTERVAL = 50  # see Youtube_scrap's pipeline.py -- avoids a full-state rewrite per file

    with processed_path.open("a", encoding="utf-8") as out, Pool(processes=num_workers) as pool:
        for i, raw_file in enumerate(raw_files, start=1):
            comments = []
            with raw_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    comments.append(json.loads(line))

            comments_collected += len(comments)

            texts = [c["text"] for c in comments]
            chunksize = max(1, len(texts) // (num_workers * 4)) if texts else 1
            results = pool.imap(_clean, texts, chunksize=chunksize)

            for comment, cleaned in zip(comments, results):
                if cleaned is None:
                    comments_dropped_empty += 1
                    continue
                doc_id = f"tt_{comment['video_id']}_{comment['comment_id']}"
                doc = schema.build_document(
                    doc_id=doc_id,
                    text=cleaned,
                    video_id=comment["video_id"],
                    channel=comment["channel"],
                    is_reply=comment["is_reply"],
                    parent_comment_id=comment["parent_comment_id"],
                    scrape_date=comment["scrape_date"],
                    dedup_hash=None,
                )
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                comments_retained += 1

            state.mark_raw_file_processed(raw_file.name)
            if i % STATE_SAVE_INTERVAL == 0:
                state.save()

        state.save()

    run_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "videos_scraped": len(raw_files),
        "comments_collected": comments_collected,
        "comments_dropped_empty": comments_dropped_empty,
        "comments_retained": comments_retained,
    }
    _append_log(log_path, run_entry)
    return run_entry
