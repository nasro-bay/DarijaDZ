"""Unit tests for dialect.py: majority-dialect voting, subforum post sampling, label_forums() (skips
categories/private, resumes, relabels, saves on a mid-run session expiry) and select_targets_by_dialect().
Uses a fake HTTP client and a fake classifier -- no live requests. One extra test runs the real shipped
classifier when `skops` and DarijaDZ_DialectID_Classifier/ are available."""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_forum.dialect import (  # noqa: E402
    DEFAULT_CLASSIFIER_DIR,
    DIALECT_CLASSES,
    DialectClassifier,
    clean_for_classification,
    label_forums,
    label_subforum,
    majority_dialect,
    sample_subforum_posts,
    select_targets_by_dialect,
)
from darija_forum.http_client import SessionExpiredError  # noqa: E402


class FakeResponse:
    def __init__(self, text: str):
        self.text = text
        self.status_code = 200


class FakeClient:
    """threads: {forum_id: [thread_id, ...]}; posts: {thread_id: [text, ...]} (first pages only)."""

    def __init__(self, threads: dict, posts: dict, fail_after_calls: int = None):
        self.threads, self.posts, self.calls, self.fail_after_calls = threads, posts, [], fail_after_calls

    def get(self, url: str) -> FakeResponse:
        self.calls.append(url)
        if self.fail_after_calls is not None and len(self.calls) > self.fail_after_calls:
            raise SessionExpiredError("simulated session expiry")
        if "forumdisplay.php" in url:
            forum_id = re.search(r"f=(\w+)", url).group(1)
            links = "".join(
                f'<a id="thread_title_{t}" href="showthread.php?t={t}"><b>thread {t}</b></a>'
                for t in self.threads.get(forum_id, [])
            )
            return FakeResponse(f"<html><body>{links}</body></html>")
        thread_id = re.search(r"t=(\w+)", url).group(1)
        tables = "".join(
            f'<table id="post{thread_id}{i}"><tr><td class="thead">2020-01-01, 00:00</td></tr>'
            f'<tr><td><a class="bigusername">u{i}</a><div id="post_message_{thread_id}{i}">{text}</div></td></tr></table>'
            for i, text in enumerate(self.posts.get(thread_id, []))
        )
        return FakeResponse(f"<html><body>{tables}</body></html>")


class FakeClassifier:
    """'DARIJA' in the text -> darija, 'MSA' -> msa, 'LAT' -> french, anything else -> other."""

    def classify_many(self, texts):
        out = []
        for t in texts:
            out.append("darija" if "DARIJA" in t else "msa" if "MSA" in t else "french" if "LAT" in t else "other")
        return out


def _forum(is_category=False, is_private=False, **extra):
    return {"title": "t", "url": "u", "parent_id": None, "category_id": "1", "category_title": "c",
            "is_private": is_private, "is_category": is_category, **extra}


class MajorityDialectTests(unittest.TestCase):
    def test_most_common_class_wins(self):
        self.assertEqual(majority_dialect(["msa", "darija", "darija", "french"]), "darija")

    def test_other_is_ignored_even_when_most_common(self):
        self.assertEqual(majority_dialect(["other", "other", "other", "msa"]), "msa")

    def test_no_classifiable_posts_gives_none(self):
        self.assertIsNone(majority_dialect([]))
        self.assertIsNone(majority_dialect(["other", "other"]))

    def test_tie_goes_to_the_earlier_class_in_dialect_classes(self):
        self.assertEqual(majority_dialect(["darija", "msa"]), "msa")
        self.assertEqual(majority_dialect(["english", "code_switch"]), "code_switch")
        self.assertLess(DIALECT_CLASSES.index("msa"), DIALECT_CLASSES.index("darija"))


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient(
            threads={"10": ["1", "2", "3"]},
            posts={"1": ["DARIJA post number one", "DARIJA post number two"],
                   "2": ["MSA post number three", "MSA post number four"],
                   "3": ["LAT post number five", "LAT post number six"]},
        )

    def test_takes_at_most_max_threads_first_pages(self):
        texts = sample_subforum_posts(self.client, "10", max_threads=2, max_posts=100)
        self.assertEqual(len(texts), 4)
        self.assertEqual(len(self.client.calls), 1 + 2)  # thread list + two thread pages
        self.assertFalse(any("LAT" in t for t in texts))

    def test_stops_fetching_once_max_posts_is_reached(self):
        texts = sample_subforum_posts(self.client, "10", max_threads=10, max_posts=3)
        self.assertEqual(len(texts), 3)
        self.assertEqual(len(self.client.calls), 1 + 2)  # 3 posts reached after the second thread

    def test_posts_that_clean_to_nothing_are_skipped(self):
        client = FakeClient({"10": ["1"]}, {"1": ["!!", "DARIJA a real post"]})
        self.assertEqual(sample_subforum_posts(client, "10"), ["DARIJA a real post"])

    def test_empty_subforum_gives_no_texts_and_no_label(self):
        client = FakeClient({"10": []}, {})
        self.assertEqual(sample_subforum_posts(client, "10"), [])
        self.assertIsNone(label_subforum(client, FakeClassifier(), "10"))


class LabelSubforumTests(unittest.TestCase):
    def test_label_is_the_majority_of_the_sample(self):
        client = FakeClient({"10": ["1", "2"]},
                            {"1": ["DARIJA one", "DARIJA two", "DARIJA three"], "2": ["MSA four"]})
        self.assertEqual(label_subforum(client, FakeClassifier(), "10"), "darija")


class LabelForumsTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient(
            threads={"10": ["1"], "20": ["2"], "30": ["3"]},
            posts={"1": ["DARIJA post one"], "2": ["MSA post two"], "3": ["LAT post three"]},
        )
        self.forums = {
            "1": _forum(is_category=True),
            "10": _forum(), "20": _forum(), "30": _forum(),
            "99": _forum(is_private=True),
        }

    def test_labels_subforums_only_not_categories_or_private(self):
        n = label_forums(self.client, self.forums, FakeClassifier())
        self.assertEqual(n, 3)
        self.assertEqual([self.forums[k]["dialect"] for k in ("10", "20", "30")], ["darija", "msa", "french"])
        self.assertNotIn("dialect", self.forums["1"])
        self.assertNotIn("dialect", self.forums["99"])

    def test_existing_labels_are_kept_unless_relabel_but_null_is_retried(self):
        self.forums["10"]["dialect"] = "english"
        self.forums["20"]["dialect"] = None  # what a failed/empty earlier sample leaves behind -> not done yet
        self.assertEqual(label_forums(self.client, self.forums, FakeClassifier()), 2)
        self.assertEqual(self.forums["10"]["dialect"], "english")
        self.assertEqual(self.forums["20"]["dialect"], "msa")
        self.assertEqual(self.forums["30"]["dialect"], "french")
        self.assertEqual(label_forums(self.client, self.forums, FakeClassifier(), relabel=True), 3)
        self.assertEqual(self.forums["10"]["dialect"], "darija")

    def test_a_forum_that_stays_empty_is_retried_on_every_run(self):
        client = FakeClient({"10": []}, {})
        forums = {"10": _forum()}
        self.assertEqual(label_forums(client, forums, FakeClassifier()), 1)
        self.assertIsNone(forums["10"]["dialect"])
        self.assertEqual(label_forums(client, forums, FakeClassifier()), 1)

    def test_saves_periodically_and_progress_survives_a_session_expiry(self):
        saved = []
        self.client.fail_after_calls = 4  # forum 10 = 2 calls, forum 20 = 2 calls, forum 30 fails
        with self.assertRaises(SessionExpiredError):
            label_forums(self.client, self.forums, FakeClassifier(),
                         save=lambda: saved.append({k: v.get("dialect", "-") for k, v in self.forums.items()}),
                         save_every=100)
        self.assertEqual(len(saved), 1)  # nothing hit save_every, but the finally-save ran on the error
        self.assertEqual((saved[-1]["10"], saved[-1]["20"], saved[-1]["30"]), ("darija", "msa", "-"))

    def test_progress_callback_fires_once_per_labelled_forum(self):
        ticks = []
        label_forums(self.client, self.forums, FakeClassifier(), on_progress=lambda: ticks.append(1))
        self.assertEqual(len(ticks), 3)


class SelectTargetsTests(unittest.TestCase):
    def test_filters_by_label_keeps_order_and_skips_unlabelled(self):
        tree = {"5": {"dialect": "msa"}, "6": {"dialect": "darija"}, "7": {"dialect": None},
                "8": {}, "9": {"dialect": "darija"}}
        targets = ["9", "5", "6", "7", "8", "404"]
        self.assertEqual(select_targets_by_dialect(targets, tree, {"darija"}), ["9", "6"])
        self.assertEqual(select_targets_by_dialect(targets, tree, {"msa", "darija"}), ["9", "5", "6"])
        self.assertEqual(select_targets_by_dialect(targets, tree, {"french"}), [])


class CleanForClassificationTests(unittest.TestCase):
    def test_strips_placeholders_and_collapses_spaces(self):
        self.assertEqual(clean_for_classification("[MENTION] wach   [URL]  rak"), "wach rak")


def _real_classifier_available() -> bool:
    try:
        import skops.io  # noqa: F401
    except ImportError:
        return False
    return (DEFAULT_CLASSIFIER_DIR / "arabic_svm.skops").exists()


@unittest.skipUnless(_real_classifier_available(), "skops or DarijaDZ_DialectID_Classifier/ not available")
class RealClassifierTests(unittest.TestCase):
    def test_script_gating_and_shipped_examples(self):
        clf = DialectClassifier()
        labels = clf.classify_many(["wach rak khouya", "Bonjour tout le monde", "مرحبا wach", "12345 !!", "[URL]"])
        self.assertEqual(labels[0], "arabize")
        self.assertEqual(labels[1], "french")
        self.assertEqual(labels[2], "code_switch")
        self.assertEqual(labels[3], "other")
        self.assertEqual(labels[4], "other")  # only a placeholder -> nothing left after cleaning

    def test_arabic_script_goes_to_the_msa_darija_model(self):
        clf = DialectClassifier()
        (label,) = clf.classify_many(["الحمد لله رب العالمين والصلاة والسلام على أشرف المرسلين"])
        self.assertIn(label, ("msa", "darija"))


if __name__ == "__main__":
    unittest.main()
