#!/usr/bin/env python
"""Class-balancing step 1/2 -- expand `darija`, `arabize`, `code_switch`
by sampling fresh rows from the DarijaDZ YouTube corpus (rows not already
in data/labeled_10k.jsonl) and labelling them with the deployed SVM
classifier (models/best_model/*.pkl), per the plan agreed in-session:

  darija       -- keep every row the SVM's arabic model calls `darija`
                  (trusted, no second check)
  arabize      -- keep rows the SVM's latin model calls `arabize`, then
                  drop any that fastText lid.176 calls `fr`/`en` with
                  score >= 0.85 (a light sanity filter -- validated on
                  400 real arabize rows: a naive "drop all fr/en" would
                  delete ~65% of genuine Arabizi since lid.176 has no
                  Arabizi label; >=0.85 removes ~5%, ~85% of which is
                  genuinely French-dominant)
  code_switch  -- rows whose cleaned text contains BOTH scripts; assigned
                  deterministically by the same regex the rest of the
                  project uses, no model involved

Corpus rows have already passed DarijaDZ's collection cleaning
(`clean_text.py`), so only the classifier's `clean_for_classification`
is applied here. Output: data/_balance_stage/corpus_new.jsonl.

Run via the base env (sklearn + ftlangdetect, no torch):

    python expand_corpus_classes.py
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

from ftlangdetect import detect

ROOT = Path(__file__).resolve().parents[3]
DID = ROOT / "Dialect_Identification"
CORPUS = ROOT / "Data" / "youtube_corpus_shuffled.jsonl"
POOL = DID / "data" / "labeled_10k.jsonl"
MODELS = DID / "models" / "best_model"
OUT = DID / "data" / "_balance_stage" / "corpus_new.jsonl"

TARGETS = {"darija": 1548, "arabize": 5371, "code_switch": 1059}
FT_DROP_SCORE = 0.85          # drop SVM-arabize rows fastText calls fr/en at >= this
MIN_WORDS = 3
MAX_WORDS = 100              # matches the classifier's own truncation

_MENTION_WITH_FRAGMENT_RE = re.compile(r"\[MENTION\](\s*-[^\s]{1,10})?")
_URL_RE = re.compile(r"\[URL\]")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN_RE = re.compile(r"[a-zA-ZÀ-ɏ]")

ARABIC_CLASSES = ["msa", "darija"]
LATIN_CLASSES = ["arabize", "french", "english"]


def clean_for_classification(text: str) -> str:
    text = _MENTION_WITH_FRAGMENT_RE.sub("", text)
    text = _URL_RE.sub("", text)
    return _INLINE_WHITESPACE_RE.sub(" ", text).strip()


def script_of(text: str) -> str:
    a, l = bool(_ARABIC_RE.search(text)), bool(_LATIN_RE.search(text))
    if a and l:
        return "mixed"
    if a:
        return "arabic"
    if l:
        return "latin"
    return "other"


def norm_key(text: str) -> str:
    return _INLINE_WHITESPACE_RE.sub(" ", text.lower()).strip()


def ft_lang(text: str) -> tuple[str, float]:
    d = detect(" ".join(text.split())[:200])
    return d["lang"], d["score"]


def main() -> None:
    pool_ids = set()
    pool_keys = set()
    with POOL.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pool_ids.add(r["id"])
            pool_keys.add(norm_key(r["text"]))
    print(f"pool: {len(pool_ids):,} ids to exclude")

    vecs, svms = {}, {}
    for g in ("arabic", "latin"):
        vecs[g] = pickle.loads((MODELS / f"{g}_vectorizer.pkl").read_bytes())
        svms[g] = pickle.loads((MODELS / f"{g}_svm.pkl").read_bytes())
    print("loaded SVM models")

    kept: dict[str, list[dict]] = {c: [] for c in TARGETS}
    seen_keys = set(pool_keys)
    n_scanned = 0
    n_arabize_svm = 0
    n_arabize_ft_dropped = 0

    def done() -> bool:
        return all(len(kept[c]) >= TARGETS[c] for c in TARGETS)

    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            if done():
                break
            n_scanned += 1
            if n_scanned % 50000 == 0:
                print(f"  scanned {n_scanned:,} | "
                      + " ".join(f"{c}={len(kept[c])}/{TARGETS[c]}" for c in TARGETS))
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row["id"] in pool_ids:
                continue
            cleaned = clean_for_classification(row["text"])
            nw = len(cleaned.split())
            if nw < MIN_WORDS or nw > MAX_WORDS:
                continue
            key = norm_key(cleaned)
            if key in seen_keys:
                continue

            grp = script_of(cleaned)
            if grp == "mixed":
                if len(kept["code_switch"]) >= TARGETS["code_switch"]:
                    continue
                seen_keys.add(key)
                kept["code_switch"].append({"row": row, "text": row["text"],
                                            "label": "code_switch", "raw": "regex:mixed"})
                continue
            if grp == "arabic":
                if len(kept["darija"]) >= TARGETS["darija"]:
                    continue
                pred = ARABIC_CLASSES[int(svms["arabic"].predict(vecs["arabic"].transform([cleaned]))[0])]
                if pred != "darija":
                    continue
                seen_keys.add(key)
                kept["darija"].append({"row": row, "text": row["text"],
                                       "label": "darija", "raw": "svm:darija"})
                continue
            if grp == "latin":
                if len(kept["arabize"]) >= TARGETS["arabize"]:
                    continue
                pred = LATIN_CLASSES[int(svms["latin"].predict(vecs["latin"].transform([cleaned]))[0])]
                if pred != "arabize":
                    continue
                n_arabize_svm += 1
                lang, score = ft_lang(cleaned)
                if lang in ("fr", "en") and score >= FT_DROP_SCORE:
                    n_arabize_ft_dropped += 1
                    continue
                seen_keys.add(key)
                kept["arabize"].append({"row": row, "text": row["text"], "label": "arabize",
                                        "raw": f"svm:arabize;ft:{lang}:{score:.2f}"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for cls, items in kept.items():
            for it in items[:TARGETS[cls]]:
                f.write(json.dumps({
                    "id": it["row"]["id"],
                    "text": it["text"],
                    "label": it["label"],
                    "raw_model_output": it["raw"],
                    "sample_group": f"balance_corpus_{cls}",
                }, ensure_ascii=False) + "\n")

    print(f"\nscanned {n_scanned:,} corpus rows")
    print(f"arabize: {n_arabize_svm:,} SVM-labelled arabize, "
          f"{n_arabize_ft_dropped:,} dropped by fastText >= {FT_DROP_SCORE} "
          f"({n_arabize_ft_dropped / max(n_arabize_svm, 1):.1%})")
    for c in TARGETS:
        print(f"  {c:<12} kept {min(len(kept[c]), TARGETS[c]):,}/{TARGETS[c]:,}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
