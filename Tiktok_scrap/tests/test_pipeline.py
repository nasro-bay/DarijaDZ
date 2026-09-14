"""Pipeline tests: fake hand-written raw JSONL fixtures, no live scraping
-- same convention as Youtube_scrap's/Mountada_djelfa_scrap's PipelineTests.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_tiktok.pipeline import run_pipeline  # noqa: E402
from darija_tiktok.state import State  # noqa: E402


def _write_raw(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.raw_dir.mkdir()
        self.state = State(self.root / "state.json")
        self.log_path = self.root / "log.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self):
        return run_pipeline(
            raw_dir=self.raw_dir, processed_dir=self.processed_dir,
            state=self.state, log_path=self.log_path, workers=1,
        )

    def test_schema_and_near_empty_drop(self):
        _write_raw(self.raw_dir / "123.jsonl", [
            {"comment_id": "c1", "text": "wach rak khouya", "video_id": "123",
             "channel": "anya.hamimed", "is_reply": False, "parent_comment_id": None,
             "scrape_date": "2026-09-12"},
            {"comment_id": "c2", "text": "😂😂😂", "video_id": "123",  # near-empty after cleaning -> dropped
             "channel": "anya.hamimed", "is_reply": False, "parent_comment_id": None,
             "scrape_date": "2026-09-12"},
            {"comment_id": "c3", "text": "haha true", "video_id": "123",
             "channel": "anya.hamimed", "is_reply": True, "parent_comment_id": "c1",
             "scrape_date": "2026-09-12"},
        ])
        result = self._run()
        self.assertEqual(result["comments_collected"], 3)
        self.assertEqual(result["comments_dropped_empty"], 1)
        self.assertEqual(result["comments_retained"], 2)

        batch_path = next(self.processed_dir.glob("batch_*.jsonl"))
        docs = [json.loads(l) for l in batch_path.open(encoding="utf-8")]
        self.assertEqual(len(docs), 2)

        top = next(d for d in docs if d["id"] == "tt_123_c1")
        self.assertEqual(top["source"], "tiktok")
        self.assertEqual(top["source_type"], "comment")
        self.assertEqual(top["source_metadata"]["video_id"], "123")
        self.assertEqual(top["source_metadata"]["channel"], "anya.hamimed")
        self.assertIsNone(top["source_metadata"]["parent_comment_id"])
        self.assertIsNone(top["dedup_hash"])

        reply = next(d for d in docs if d["id"] == "tt_123_c3")
        self.assertEqual(reply["source_type"], "reply")
        self.assertEqual(reply["source_metadata"]["parent_comment_id"], "c1")

    def test_resume_skips_already_processed_raw_files(self):
        _write_raw(self.raw_dir / "111.jsonl", [
            {"comment_id": "c1", "text": "sahit khouya", "video_id": "111",
             "channel": "anya.hamimed", "is_reply": False, "parent_comment_id": None,
             "scrape_date": "2026-09-12"},
        ])
        first = self._run()
        self.assertEqual(first["comments_retained"], 1)

        second = self._run()
        self.assertEqual(second["videos_scraped"], 0)  # no new raw files left to process
        self.assertEqual(second["comments_retained"], 0)


if __name__ == "__main__":
    unittest.main()
