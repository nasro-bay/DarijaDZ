#!/usr/bin/env python
"""Runs the clean -> schema -> processed-batch pipeline over any raw
files not yet processed. Safe to rerun (resumable via State).

    python run_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darija_tiktok.pipeline import run_pipeline  # noqa: E402
from darija_tiktok.state import State  # noqa: E402

DATA_DIR = ROOT / "data"


def main() -> None:
    state = State(DATA_DIR / "state" / "scrape_state.json")
    result = run_pipeline(
        raw_dir=DATA_DIR / "raw",
        processed_dir=DATA_DIR / "processed",
        state=state,
        log_path=DATA_DIR / "logs" / "log.json",
    )
    print(result)


if __name__ == "__main__":
    main()
