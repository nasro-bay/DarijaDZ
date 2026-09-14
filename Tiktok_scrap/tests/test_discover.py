"""Unit tests for discover_videos_ytdlp's output parsing -- mocked
subprocess, no live network/yt-dlp call needed."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_tiktok import discover  # noqa: E402


class YtdlpDiscoveryTests(unittest.TestCase):
    def _fake_run(self, stdout: str, returncode: int = 0):
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")

    def test_parses_video_ids_from_stdout(self):
        with patch("subprocess.run", return_value=self._fake_run("111\n222\n333\n")):
            ids = discover.discover_videos_ytdlp("someuser")
        self.assertEqual(ids, ["111", "222", "333"])

    def test_ignores_blank_and_non_numeric_lines(self):
        with patch("subprocess.run", return_value=self._fake_run("111\n\nWARNING: something\n222\n")):
            ids = discover.discover_videos_ytdlp("someuser")
        self.assertEqual(ids, ["111", "222"])

    def test_max_videos_passes_playlist_end_flag(self):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return self._fake_run("111\n")

        with patch("subprocess.run", side_effect=fake_run):
            discover.discover_videos_ytdlp("someuser", max_videos=5)
        self.assertIn("--playlist-end", captured["cmd"])
        self.assertIn("5", captured["cmd"])

    def test_missing_ytdlp_binary_returns_empty_not_raises(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            ids = discover.discover_videos_ytdlp("someuser")
        self.assertEqual(ids, [])

    def test_timeout_returns_empty_not_raises(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="yt-dlp", timeout=180)):
            ids = discover.discover_videos_ytdlp("someuser")
        self.assertEqual(ids, [])


if __name__ == "__main__":
    unittest.main()
