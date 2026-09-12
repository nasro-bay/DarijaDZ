#!/usr/bin/env python
"""Class-balancing step 2/2 -- expand `msa`, `french`, `english` from
public datasets (the DarijaDZ corpus barely contains these classes, so
sampling more of it can't fix the imbalance).

Sources (informal / mixed sentence length, closest public match to the
YouTube-comment domain that's reliably loadable):

  english  -- google/civil_comments      (online news-site comments)
  french   -- tblard/allocine            (film reviews, short + long)
  msa      -- wikimedia/wikipedia 20231101.ar, sentence-split
              (encyclopedic, but unambiguously MSA -- and the classifier
              only has to tell msa from darija, a wide margin)

Each row is put through BOTH cleaning pipelines, in order:
  1. DarijaDZ collection cleaning  (darija_corpus.clean_text.clean)
  2. classifier cleaning           (clean_for_classification)
then script-checked and language-verified with fastText lid.176 (msa->ar,
french->fr, english->en); disagreements are dropped. Long sentences are
kept on purpose (robustness), only extreme outliers are capped.

Output: data/_balance_stage/external_new.jsonl

Run via the base env:  python expand_external_classes.py
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset
from ftlangdetect import detect

ROOT = Path(__file__).resolve().parents[3]
DID = ROOT / "Dialect_Identification"
POOL = DID / "data" / "labeled_10k.jsonl"
OUT = DID / "data" / "_balance_stage" / "external_new.jsonl"

sys.path.insert(0, str(ROOT / "Youtube_scrap" / "src"))
from darija_corpus.clean_text import clean as collection_clean  # noqa: E402

TARGETS = {"msa": 5953, "french": 6824, "english": 9245}
LANG_OK = {"msa": "ar", "french": "fr", "english": "en"}
SEED = 42
MIN_WORDS = 3
MAX_WORDS = 140                      # allow some long ones; drop only extreme outliers
OVERSAMPLE = 4                       # raw rows to consider per wanted row

_MENTION_WITH_FRAGMENT_RE = re.compile(r"\[MENTION\](\s*-[^\s]{1,10})?")
_URL_RE = re.compile(r"\[URL\]")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN_RE = re.compile(r"[a-zA-ZÀ-ɏ]")
_AR_SENT_SPLIT = re.compile(r"(?<=[.!؟\n])\s+")


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


def ft_lang(text: str) -> str:
    return detect(" ".join(text.split())[:200])["lang"]


def raw_text_iter(cls: str):
    """Yield candidate raw strings for a class, already lightly chunked."""
    if cls == "english":
        ds = load_dataset("google/civil_comments", split="train", streaming=True)
        for r in ds:
            yield r["text"]
    elif cls == "french":
        ds = load_dataset("tblard/allocine", split="train", streaming=True)
        for r in ds:
            yield r["review"]
    elif cls == "msa":
        ds = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train", streaming=True)
        for r in ds:
            parts = [p.strip() for p in _AR_SENT_SPLIT.split(r["text"]) if p.strip()]
            # mix: individual sentences + occasional 2-3 sentence spans
            i = 0
            while i < len(parts):
                span = 1 if (i % 3) else min(3, len(parts) - i)
                yield " ".join(parts[i:i + span])
                i += span


def build_class(cls: str, pool_keys: set[str], rng: random.Random) -> list[dict]:
    want = TARGETS[cls]
    target_script = "arabic" if cls == "msa" else "latin"
    kept: list[dict] = []
    seen = set()
    n_seen = 0
    for raw in raw_text_iter(cls):
        if len(kept) >= want:
            break
        n_seen += 1
        if n_seen > want * OVERSAMPLE * 8:      # hard stop, avoid runaway
            break
        if not raw or not raw.strip():
            continue
        collected = collection_clean(raw)
        if not collected:                       # clean_text drops near-empty / all-junk -> None
            continue
        t = clean_for_classification(collected)
        nw = len(t.split())
        if nw < MIN_WORDS or nw > MAX_WORDS:
            continue
        if script_of(t) != target_script:
            continue
        key = norm_key(t)
        if key in seen or key in pool_keys:
            continue
        if ft_lang(t) != LANG_OK[cls]:
            continue
        seen.add(key)
        kept.append({
            "id": f"ext_{cls}_{hashlib.sha1(key.encode()).hexdigest()[:16]}",
            "text": t,
            "label": cls,
            "raw_model_output": f"external:{cls}",
            "sample_group": f"balance_external_{cls}",
        })
        if len(kept) % 1000 == 0:
            print(f"  {cls}: {len(kept)}/{want}  (scanned {n_seen:,})")
    print(f"{cls}: kept {len(kept)}/{want} from {n_seen:,} raw candidates")
    return kept


def main() -> None:
    rng = random.Random(SEED)
    pool_keys = set()
    with POOL.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pool_keys.add(norm_key(json.loads(line)["text"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # per-class: <class>.part.jsonl, written on success; OUT is the concat
    for cls in TARGETS:
        part = OUT.with_name(f"external_{cls}.part.jsonl")
        if part.exists() and sum(1 for _ in part.open(encoding="utf-8")) >= TARGETS[cls]:
            print(f"{cls}: {part.name} already complete, skipping")
            continue
        try:
            rows = build_class(cls, pool_keys, rng)
        except Exception as e:                     # network hiccup etc -- keep other classes
            print(f"{cls}: FAILED ({type(e).__name__}: {str(e)[:120]}) -- rerun to retry this class")
            continue
        with part.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{cls}: wrote {part.name}")

    with OUT.open("w", encoding="utf-8") as out:
        for cls in TARGETS:
            part = OUT.with_name(f"external_{cls}.part.jsonl")
            if part.exists():
                out.write(part.read_text(encoding="utf-8"))
    print(f"\nconcatenated -> {OUT}")


if __name__ == "__main__":
    main()
