#!/usr/bin/env python
"""Archives raw/processed scraper files that are no longer needed locally to
Google Drive (via the `DarijaDZ` rclone remote), then frees the local disk
space -- but only after a checksum-verified copy, never on a bare "the copy
command didn't error" assumption.

**What counts as "no longer needed locally"** (mirrors this repo's own
pipeline state, not a separate tracking system):
- A **raw** file is eligible once it appears in that source's
  `data/state/scrape_state.json` -> `pipeline.processed_raw_files` -- i.e.
  the pipeline has already turned it into processed output. Note: per
  CLAUDE.md's "rebuilding processed output" convention, a future cleaning-
  rule change re-processes raw files from scratch -- archiving them to Drive
  (not deleting outright) keeps that path open, just via a pull-from-Drive
  step first instead of assuming they're still on local disk.
- A **processed** batch file is eligible once it appears in that source's
  `data/state/unified_build_state.json` -> `included_batch_files` -- i.e.
  it has already been folded into `data/unified_corpus.jsonl`, so the
  standalone batch file is redundant with data already captured elsewhere.

**Safety model**: `rclone copy` (checksummed) -> `rclone check --one-way`
(independent re-verification, not just trusting copy's own exit code) ->
only then delete local files, one by one, only the ones that passed.
Defaults to a dry run (`--execute` required to actually delete anything).
Every run appends a summary to `data/state/drive_archive_log.json` in the
repo root (audit trail of what was archived and when).

Run via the base env: `python Data/scripts/push_to_drive.py --execute`
(add `--source Youtube_scrap`/`Tiktok_scrap` and/or `--category raw`/
`processed` to scope a single source/category, e.g. just the processed
batches for TikTok: `--source Tiktok_scrap --category processed --execute`)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "DarijaDZ:DarijaDZ/data_archive"
LOG_PATH = ROOT / "Data" / "state" / "drive_archive_log.json"

SOURCES = {
    "Youtube_scrap": {
        "raw_dir": ROOT / "Youtube_scrap" / "data" / "raw",
        "processed_dir": ROOT / "Youtube_scrap" / "data" / "processed",
        "scrape_state": ROOT / "Youtube_scrap" / "data" / "state" / "scrape_state.json",
        "unified_state": ROOT / "Youtube_scrap" / "data" / "state" / "unified_build_state.json",
    },
    "Tiktok_scrap": {
        "raw_dir": ROOT / "Tiktok_scrap" / "data" / "raw",
        "processed_dir": ROOT / "Tiktok_scrap" / "data" / "processed",
        "scrape_state": ROOT / "Tiktok_scrap" / "data" / "state" / "scrape_state.json",
        "unified_state": ROOT / "Tiktok_scrap" / "data" / "state" / "unified_build_state.json",
    },
}


def eligible_files(local_dir: Path, names: set[str]) -> list[str]:
    """Names that both appear in the state-file's list AND still exist
    locally (already-archived-and-deleted files naturally drop out here on
    a rerun, no separate "already archived" tracking needed)."""
    return sorted(n for n in names if (local_dir / n).exists())


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout)
    if result.returncode != 0:
        print("--- STDERR ---")
        print(result.stderr)
    return result


def archive_category(source: str, category: str, local_dir: Path, remote_subdir: str,
                      names: list[str], execute: bool, limit: int | None) -> dict:
    if limit is not None:
        names = names[:limit]
    if not names:
        return {"category": category, "count": 0, "bytes": 0, "deleted": 0}

    total_bytes = sum((local_dir / n).stat().st_size for n in names)
    remote_dir = f"{REMOTE}/{source}/{category}"
    print(f"\n--- {source}/{category}: {len(names):,} files eligible ({total_bytes / 1e9:.2f} GB) ---")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(names))
        list_path = f.name

    try:
        copy_result = run([
            "rclone", "copy", str(local_dir), remote_dir,
            "--files-from", list_path, "--checksum",
            "--transfers", "8", "--checkers", "16",
        ])
        if copy_result.returncode != 0:
            print(f"  COPY FAILED for {source}/{category} -- skipping delete, nothing removed locally.")
            return {"category": category, "count": len(names), "bytes": total_bytes, "deleted": 0, "error": "copy_failed"}

        check_result = run([
            "rclone", "check", str(local_dir), remote_dir,
            "--files-from", list_path, "--one-way",
        ])
        if check_result.returncode != 0:
            print(f"  VERIFICATION FAILED for {source}/{category} -- skipping delete, nothing removed locally.")
            return {"category": category, "count": len(names), "bytes": total_bytes, "deleted": 0, "error": "check_failed"}

        print(f"  Verified OK: {len(names):,} files match on Drive.")
        deleted = 0
        deleted_bytes = 0
        if execute:
            for n in names:
                p = local_dir / n
                try:
                    size = p.stat().st_size
                    p.unlink()
                    deleted += 1
                    deleted_bytes += size
                except OSError as e:
                    print(f"  WARNING: failed to delete {p}: {e}")
            print(f"  Deleted {deleted:,} local files, freed {deleted_bytes / 1e9:.2f} GB.")
        else:
            print(f"  [dry run] Would delete {len(names):,} local files, freeing {total_bytes / 1e9:.2f} GB. "
                  f"Pass --execute to actually delete.")
        return {"category": category, "count": len(names), "bytes": total_bytes, "deleted": deleted}
    finally:
        Path(list_path).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=list(SOURCES), default=None,
                         help="Only archive this source (default: all)")
    parser.add_argument("--execute", action="store_true",
                         help="Actually delete local files after verified copy (default: dry run)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N eligible files per category (for testing)")
    parser.add_argument("--category", choices=["raw", "processed"], default=None,
                         help="Only archive this category (default: both raw and processed)")
    args = parser.parse_args()

    sources = [args.source] if args.source else list(SOURCES)
    categories = [args.category] if args.category else ["raw", "processed"]
    run_summary = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "execute": args.execute, "sources": {}}

    for source in sources:
        cfg = SOURCES[source]
        results = []

        if "raw" in categories:
            scrape_state = json.loads(cfg["scrape_state"].read_text(encoding="utf-8"))
            processed_raw_names = set(scrape_state["pipeline"]["processed_raw_files"])
            raw_names = eligible_files(cfg["raw_dir"], processed_raw_names)
            results.append(archive_category(source, "raw", cfg["raw_dir"], "raw", raw_names, args.execute, args.limit))

        if "processed" in categories:
            unified_state = json.loads(cfg["unified_state"].read_text(encoding="utf-8"))
            included_batch_names = set(unified_state["included_batch_files"])
            processed_names = eligible_files(cfg["processed_dir"], included_batch_names)
            results.append(archive_category(source, "processed", cfg["processed_dir"], "processed", processed_names, args.execute, args.limit))

        run_summary["sources"][source] = results

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = json.loads(LOG_PATH.read_text(encoding="utf-8")) if LOG_PATH.exists() else []
    log.append(run_summary)
    LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"\nLogged to {LOG_PATH}")


if __name__ == "__main__":
    main()
