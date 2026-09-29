"""Unit tests for ForumHttpClient's cookie hygiene and error handling -- no live requests (the
underlying requests.Session.get is replaced by a fake).

Regression: vBulletin's `bbforum_view` cookie grows with every distinct forum viewed; once the Cookie
header passed ~8 KB the server answered HTTP 400 for every request, and the old client parsed those 400
pages as empty subforums (a whole dialect-labelling run came out 'unlabelled')."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_forum.http_client import (  # noqa: E402
    VIEW_TRACKING_COOKIES,
    BadRequestError,
    ForumHttpClient,
    SessionExpiredError,
)


def _make_client(tmp: str) -> ForumHttpClient:
    path = Path(tmp) / "session.json"
    path.write_text(json.dumps({
        "cookies": [
            {"name": "cf_clearance", "value": "abc", "domain": ".djelfa.info", "path": "/"},
            {"name": "bbsessionhash", "value": "def", "domain": ".djelfa.info", "path": "/"},
        ],
        "user_agent": "test-agent", "solved_at": "2026-01-01T00:00:00+00:00", "source": "manual",
    }), encoding="utf-8")
    return ForumHttpClient(path)


def _fake_get(client: ForumHttpClient, *, status: int = 200, text: str = "<html>ok</html>", grow_marker: bool = True):
    calls = []

    def fake(url, **kwargs):
        calls.append(url)
        if grow_marker:
            existing = client._session.cookies.get("bbforum_view", "")
            client._session.cookies.set("bbforum_view", existing + "9." * 20, domain=".djelfa.info", path="/")
            client._session.cookies.set("bbthread_view", "1.2.3", domain=".djelfa.info", path="/")
        resp = requests.Response()
        resp.status_code, resp._content, resp.url = status, text.encode("utf-8"), url
        resp.request = requests.Request("GET", url).prepare()
        return resp

    client._session.get = fake
    return calls


class CookieHygieneTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.client = _make_client(self._tmp.name)

    def names(self):
        return {c.name for c in self.client._session.cookies}

    def test_read_marker_cookies_are_dropped_after_each_response(self):
        _fake_get(self.client)
        self.client.get("https://www.djelfa.info/vb/forumdisplay.php?f=1")
        for name in VIEW_TRACKING_COOKIES:
            self.assertNotIn(name, self.names())

    def test_session_cookies_the_crawl_needs_are_kept(self):
        _fake_get(self.client)
        self.client.get("https://www.djelfa.info/vb/forumdisplay.php?f=1")
        self.assertIn("cf_clearance", self.names())
        self.assertIn("bbsessionhash", self.names())

    def test_cookie_jar_does_not_grow_across_many_distinct_forums(self):
        _fake_get(self.client)
        size_before = sum(len(c.name) + len(c.value or "") for c in self.client._session.cookies)
        for forum_id in range(600):  # more than the ~390 that broke the real session
            self.client.get(f"https://www.djelfa.info/vb/forumdisplay.php?f={forum_id}")
        size_after = sum(len(c.name) + len(c.value or "") for c in self.client._session.cookies)
        self.assertEqual(size_after, size_before)


class ErrorHandlingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.client = _make_client(self._tmp.name)

    def test_ok_response_is_returned(self):
        _fake_get(self.client, text="<html>forum page</html>")
        self.assertEqual(self.client.get("https://www.djelfa.info/vb/").text, "<html>forum page</html>")

    def test_http_400_raises_instead_of_being_parsed_as_an_empty_page(self):
        _fake_get(self.client, status=400, text="<h1>Bad Request</h1>")
        with self.assertRaises(BadRequestError):
            self.client.get("https://www.djelfa.info/vb/forumdisplay.php?f=1")

    def test_http_403_is_still_a_session_expiry(self):
        _fake_get(self.client, status=403, text="<html>blocked</html>")
        with self.assertRaises(SessionExpiredError):
            self.client.get("https://www.djelfa.info/vb/")


if __name__ == "__main__":
    unittest.main()
