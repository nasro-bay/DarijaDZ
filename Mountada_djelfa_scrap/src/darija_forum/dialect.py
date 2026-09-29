"""Labels each subforum with its majority dialect while the forum tree is fetched, and selects scrape
targets by that label.

Per subforum: fetch the first page of its thread list (most recently active first), fetch the first
post page of up to `max_threads` of those threads, clean the collected posts with the same
`clean_text.clean()` the pipeline uses, classify each post with the shipped DarijaDZ dialect classifier
(`DarijaDZ_DialectID_Classifier/`, script-gated char n-gram TF-IDF + SVM) and keep only the most common
class. That single label is all that gets stored (`dialect` in forum_tree.json); `None` means no
classifiable post was sampled.

skops/scikit-learn are imported lazily inside `DialectClassifier`, so scraping without a dialect
filter doesn't need them.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from . import clean_text
from .parse import list_posts, list_threads

BASE_URL = "https://www.djelfa.info/vb/"
DEFAULT_CLASSIFIER_DIR = Path(__file__).resolve().parents[3] / "DarijaDZ_DialectID_Classifier"

ARABIC_CLASSES = ["msa", "darija"]
LATIN_CLASSES = ["arabize", "french", "english"]
# Order doubles as the tie-break when two classes have the same number of sampled posts.
DIALECT_CLASSES = ("msa", "darija", "code_switch", "arabize", "french", "english")

_ARABIC_RE = re.compile("[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN_RE = re.compile("[a-zA-ZÀ-ɏ]")
_MENTION_RE = re.compile(r"\[MENTION\](\s*-[^\s]{1,10})?")
_URL_RE = re.compile(r"\[URL\]")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")


def clean_for_classification(text: str) -> str:
    """The classifier's own training-time cleaning: without it the `[URL]`/`[MENTION]` placeholders'
    Latin letters would push Arabic-script posts into the code_switch rule."""
    text = _MENTION_RE.sub("", text)
    text = _URL_RE.sub("", text)
    return _INLINE_WHITESPACE_RE.sub(" ", text).strip()


class DialectClassifier:
    """The shipped script-gated classifier: both scripts present -> code_switch (rule), Arabic only ->
    msa/darija SVM, Latin only -> arabize/french/english SVM, neither -> other."""

    def __init__(self, classifier_dir: Path = DEFAULT_CLASSIFIER_DIR):
        import skops.io as sio

        def load(name: str):
            path = Path(classifier_dir) / name
            return sio.load(path, trusted=sio.get_untrusted_types(file=path))

        self._arabic = (load("arabic_vectorizer.skops"), load("arabic_svm.skops"))
        self._latin = (load("latin_vectorizer.skops"), load("latin_svm.skops"))

    def classify_many(self, texts: list[str]) -> list[str]:
        cleaned = [clean_for_classification(t) for t in texts]
        labels: list[Optional[str]] = [None] * len(cleaned)
        arabic_idx: list[int] = []
        latin_idx: list[int] = []
        for i, text in enumerate(cleaned):
            has_arabic, has_latin = bool(_ARABIC_RE.search(text)), bool(_LATIN_RE.search(text))
            if has_arabic and has_latin:
                labels[i] = "code_switch"
            elif has_arabic:
                arabic_idx.append(i)
            elif has_latin:
                latin_idx.append(i)
            else:
                labels[i] = "other"
        for idxs, (vectorizer, svm), classes in (
            (arabic_idx, self._arabic, ARABIC_CLASSES),
            (latin_idx, self._latin, LATIN_CLASSES),
        ):
            if idxs:
                predicted = svm.predict(vectorizer.transform([cleaned[i] for i in idxs]))
                for i, p in zip(idxs, predicted):
                    labels[i] = classes[int(p)]
        return labels  # type: ignore[return-value]


def majority_dialect(labels: list[str]) -> Optional[str]:
    """Most common class among the classifiable posts (`other` is ignored); ties go to the class that
    comes first in DIALECT_CLASSES; None if nothing was classifiable."""
    counts = Counter(label for label in labels if label in DIALECT_CLASSES)
    if not counts:
        return None
    return max(DIALECT_CLASSES, key=lambda c: (counts[c], -DIALECT_CLASSES.index(c)))


def sample_subforum_posts(client, forum_id: str, *, max_threads: int = 10, max_posts: int = 60) -> list[str]:
    """Cleaned texts of up to `max_posts` posts taken from the first page of up to `max_threads` of the
    subforum's most recently active threads (1 + max_threads requests at most). Posts that clean to
    nothing are skipped and don't count."""
    threads, _ = list_threads(client.get(f"{BASE_URL}forumdisplay.php?f={forum_id}&page=1").text)
    texts: list[str] = []
    for thread in threads[:max_threads]:
        posts, _ = list_posts(client.get(f"{BASE_URL}showthread.php?t={thread.thread_id}&page=1").text)
        for post in posts:
            cleaned = clean_text.clean(post.text)
            if cleaned is not None:
                texts.append(cleaned)
        if len(texts) >= max_posts:
            break
    return texts[:max_posts]


def label_subforum(client, classifier, forum_id: str, *, max_threads: int = 10, max_posts: int = 60) -> Optional[str]:
    texts = sample_subforum_posts(client, forum_id, max_threads=max_threads, max_posts=max_posts)
    if not texts:
        return None
    return majority_dialect(classifier.classify_many(texts))


def label_forums(
    client,
    forums: dict[str, dict],
    classifier,
    *,
    max_threads: int = 10,
    max_posts: int = 60,
    relabel: bool = False,
    save: Optional[Callable[[], None]] = None,
    save_every: int = 10,
    on_progress: Optional[Callable[[], None]] = None,
) -> int:
    """Sets `forums[id]["dialect"]` for every non-category, non-private entry whose label is missing or
    null (all of them with `relabel`). A null label always counts as "not labelled yet" -- it is what a
    failed or empty sample leaves behind -- so a rerun retries it (a genuinely empty subforum just costs
    one listing request again). Progress is saved every `save_every` forums and on any exit (including
    SessionExpiredError, which propagates), so a rerun resumes where it stopped."""
    todo = [
        forum_id
        for forum_id, info in forums.items()
        if not info["is_category"] and not info["is_private"] and (relabel or info.get("dialect") is None)
    ]
    done = 0
    try:
        for forum_id in todo:
            forums[forum_id]["dialect"] = label_subforum(
                client, classifier, forum_id, max_threads=max_threads, max_posts=max_posts
            )
            done += 1
            if on_progress:
                on_progress()
            if save and done % save_every == 0:
                save()
    finally:
        if save and done:
            save()
    return done


def select_targets_by_dialect(targets: list[str], tree: dict[str, dict], wanted: set[str]) -> list[str]:
    """Targets whose labelled majority dialect is in `wanted`, in the original target order. Unlabelled
    subforums (no `dialect` key, or null) never match."""
    return [forum_id for forum_id in targets if tree.get(forum_id, {}).get("dialect") in wanted]
