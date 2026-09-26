#!/usr/bin/env python
"""Combines Youtube_scrap's and Tiktok_scrap's per-source unified
corpora (see each source's own scripts/build_unified_dataset.py -- both
incremental, id+text only) into the single published DarijaDZ dataset:
concatenates them, shuffles the result, syncs it into the DarijaDZ/ and
Kaggle_DarijaDz/ release folders, and updates the numeric statistics in
those READMEs (plus DarijaDz_Tokenizers/README.md's YouTube-only figure)
to match.

**Filenames kept as `youtube_corpus[_shuffled].jsonl`** even though the
corpus is now YouTube+TikTok combined -- renaming would break the
already-published HF/Kaggle dataset's known file name; not done without
being asked. Worth reconsidering later, flagged here rather than done
silently.

**No `source` field in the combined records**: YouTube ids are always
`yt_...`, TikTok ids are always `tt_...` (see each source's own
schema.py) -- guaranteed never to collide, so a source tag would be
redundant, not disambiguating.

Run *after* both per-source build_unified_dataset.py scripts (each one
tops up its own `data/unified_corpus.jsonl` incrementally); this script
itself always rebuilds the combined+shuffled output from the two
(already-incremental) source files -- concatenating and shuffling two
flat files is cheap, unlike the old full-corpus-from-raw-batches rebuild
this replaced.

Run via the base env: `python Data/scripts/build_combined_dataset.py`
"""
from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

YOUTUBE_UNIFIED = ROOT / "Youtube_scrap" / "data" / "unified_corpus.jsonl"
YOUTUBE_STATE = ROOT / "Youtube_scrap" / "data" / "state" / "unified_build_state.json"
TIKTOK_UNIFIED = ROOT / "Tiktok_scrap" / "data" / "unified_corpus.jsonl"
TIKTOK_STATE = ROOT / "Tiktok_scrap" / "data" / "state" / "unified_build_state.json"
DJELFA_PROCESSED_DIR = ROOT / "Mountada_djelfa_scrap" / "data" / "processed"

COMBINED_PATH = ROOT / "Data" / "youtube_corpus.jsonl"
SHUFFLED_PATH = ROOT / "Data" / "youtube_corpus_shuffled.jsonl"
README_PATHS = [ROOT / "DarijaDZ" / "README.md", ROOT / "Kaggle_DarijaDz" / "README.md"]
TOKENIZER_README_PATH = ROOT / "DarijaDz_Tokenizers" / "README.md"
RELEASE_CORPUS_PATHS = [ROOT / "DarijaDZ" / "youtube_corpus.jsonl", ROOT / "Kaggle_DarijaDz" / "youtube_corpus.jsonl"]

SHUFFLE_SEED = 42
SHUFFLE_CHUNK_SIZE = 100_000

_EMPTY_STATE = {"total_docs": 0, "total_tokens": 0, "channels": [], "video_ids": []}


def _load_state(path: Path) -> dict:
    if not path.exists():
        print(f"  WARNING: {path} not found -- treating as zero (run that source's "
              f"build_unified_dataset.py first)")
        return dict(_EMPTY_STATE)
    data = json.loads(path.read_text(encoding="utf-8"))
    for key, default in _EMPTY_STATE.items():
        data.setdefault(key, default)
    return data


def concat_sources() -> None:
    COMBINED_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Concatenating {YOUTUBE_UNIFIED.relative_to(ROOT)} + {TIKTOK_UNIFIED.relative_to(ROOT)}...")
    with COMBINED_PATH.open("wb") as out_f:
        for src in (YOUTUBE_UNIFIED, TIKTOK_UNIFIED):
            if not src.exists():
                print(f"  WARNING: {src} not found -- skipping (run its build_unified_dataset.py first)")
                continue
            with src.open("rb") as in_f:
                shutil.copyfileobj(in_f, out_f)
    print(f"Done! Combined dataset written to: {COMBINED_PATH}")


def shuffle_combined() -> int:
    """External chunk-shuffle of COMBINED_PATH -> SHUFFLED_PATH: shuffle
    within fixed-size chunks, shuffle chunk order, concatenate -- avoids
    loading the multi-million-line file into RAM at once. Same algorithm
    this project has always used for this step."""
    rng = random.Random(SHUFFLE_SEED)
    temp_dir = ROOT / "Data" / "_shuffle_chunks"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("\nShuffling combined dataset...")
    chunk: list[dict] = []
    chunk_files: list[Path] = []
    chunk_number = 0
    total_count = 0

    def _flush_chunk() -> None:
        nonlocal chunk, chunk_number
        rng.shuffle(chunk)
        chunk_path = temp_dir / f"chunk_{chunk_number:05d}.jsonl"
        with chunk_path.open("w", encoding="utf-8") as out_f:
            for record in chunk:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        chunk_files.append(chunk_path)
        chunk = []
        chunk_number += 1

    with COMBINED_PATH.open("r", encoding="utf-8") as in_f:
        for line in in_f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Error parsing line: {e}")
                continue
            chunk.append(record)
            total_count += 1
            if len(chunk) >= SHUFFLE_CHUNK_SIZE:
                _flush_chunk()
    if chunk:
        _flush_chunk()

    print(f"  Created {len(chunk_files)} chunks ({total_count:,} documents).")
    rng.shuffle(chunk_files)

    print("Writing final shuffled dataset...")
    with SHUFFLED_PATH.open("w", encoding="utf-8") as out_f:
        for i, chunk_path in enumerate(chunk_files, start=1):
            print(f"  Merging chunk {i}/{len(chunk_files)}...")
            with chunk_path.open("r", encoding="utf-8") as chunk_f:
                for line in chunk_f:
                    out_f.write(line)
            chunk_path.unlink()
    temp_dir.rmdir()

    print(f"Done! Shuffled dataset written to: {SHUFFLED_PATH}")
    print(f"Total documents: {total_count:,}")
    return total_count


def sync_release_copies() -> None:
    for dest in RELEASE_CORPUS_PATHS:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SHUFFLED_PATH, dest)
        print(f"  Synced {dest.relative_to(ROOT)}")


def _sub_bold_number(text: str, label_pattern: str, value: str) -> str:
    pattern = re.compile(rf"({label_pattern}.*?\*\*)[\d,.]+%?(\*\*)")
    new_text, n = pattern.subn(lambda m: m.group(1) + value + m.group(2), text, count=1)
    if n == 0:
        print(f"  WARNING: pattern not found for {label_pattern!r} -- a README line was left unchanged")
    return new_text


def _sub_intro_prose(text: str, total_docs: int, total_tokens: int) -> str:
    pattern = re.compile(
        r"approximately \*\*[\d.]+ million comment documents\*\* "
        r"and \*\*[\d.]+ million word-level tokens\*\*"
    )
    replacement = (
        f"approximately **{total_docs / 1_000_000:.1f} million comment documents** "
        f"and **{total_tokens / 1_000_000:.2f} million word-level tokens**"
    )
    new_text, n = pattern.subn(replacement, text, count=1)
    if n == 0:
        print("  WARNING: pattern not found for the intro 'approximately X million...' sentence")
    return new_text


def _sub_in_section(text: str, section_heading: str, label_pattern: str, value: str) -> str:
    """Like _sub_bold_number, but scoped to only the text from
    `section_heading` up to the next `###`/`##` heading -- needed since
    both the YouTube and TikTok collection blocks share the same
    `Channels:`/`Raw video files processed:` labels, so an unscoped
    substitution would only ever hit the first (YouTube's) occurrence."""
    heading_idx = text.find(section_heading)
    if heading_idx == -1:
        print(f"  WARNING: section {section_heading!r} not found -- a README section was left unchanged")
        return text
    next_heading = re.search(r"\n#{2,3} ", text[heading_idx + len(section_heading):])
    section_end = (heading_idx + len(section_heading) + next_heading.start()) if next_heading else len(text)
    before, section, after = text[:heading_idx], text[heading_idx:section_end], text[section_end:]
    section = _sub_bold_number(section, label_pattern, value)
    return before + section + after


def _count_djelfa_docs() -> int:
    total = 0
    for batch_file in DJELFA_PROCESSED_DIR.glob("batch_*.jsonl"):
        with batch_file.open("r", encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    return total


def update_tokenizer_readme(youtube_docs: int) -> None:
    """DarijaDz_Tokenizers/README.md's training corpus is YouTube +
    djelfa only (see Tokenization/scripts/build_training_corpus.py's
    SOURCES) -- TikTok is not part of it. Only the combined total-docs
    figure is synced here; the "docs used for training" / held-out
    split depends on actually rerunning build_training_corpus.py (which
    resamples the held-out allowlist), so that sentence is left as-is."""
    if not TOKENIZER_README_PATH.exists():
        print(f"  WARNING: {TOKENIZER_README_PATH} not found, skipping")
        return
    if not DJELFA_PROCESSED_DIR.exists():
        print(f"  WARNING: {DJELFA_PROCESSED_DIR} not found, skipping {TOKENIZER_README_PATH.name}")
        return
    total_docs = youtube_docs + _count_djelfa_docs()
    text = TOKENIZER_README_PATH.read_text(encoding="utf-8")
    pattern = re.compile(r"\*\*[\d,]+ documents\*\* from both Algerian YouTube comments")
    new_text, n = pattern.subn(f"**{total_docs:,} documents** from both Algerian YouTube comments", text, count=1)
    if n == 0:
        print("  WARNING: pattern not found for the training-data doc-count sentence")
        return
    TOKENIZER_README_PATH.write_text(new_text, encoding="utf-8")
    print(f"  Updated {TOKENIZER_README_PATH.relative_to(ROOT)}")


def update_readmes(yt_state: dict, tt_state: dict) -> None:
    total_docs = yt_state["total_docs"] + tt_state["total_docs"]
    total_tokens = yt_state["total_tokens"] + tt_state["total_tokens"]
    mean_tokens = total_tokens / total_docs if total_docs else 0.0

    for readme_path in README_PATHS:
        if not readme_path.exists():
            print(f"  WARNING: {readme_path} not found, skipping")
            continue
        text = readme_path.read_text(encoding="utf-8")

        text = _sub_intro_prose(text, total_docs, total_tokens)
        text = _sub_bold_number(text, r"\|\s*Documents\s*\|", f"{total_docs:,}")
        text = _sub_bold_number(text, r"\|\s*Word-level tokens\s*\|", f"{total_tokens:,}")
        text = _sub_bold_number(text, r"\|\s*Mean tokens / document\s*\|", f"{mean_tokens:.2f}")

        text = _sub_in_section(text, "### YouTube Collection", r"\*\s*Channels:", f"{len(yt_state['channels'])}")
        text = _sub_in_section(text, "### YouTube Collection", r"\*\s*Raw video files processed:",
                                f"{len(yt_state['video_ids']):,}")
        text = _sub_in_section(text, "### TikTok Collection", r"\*\s*Channels:", f"{len(tt_state['channels'])}")
        text = _sub_in_section(text, "### TikTok Collection", r"\*\s*Raw video files processed:",
                                f"{len(tt_state['video_ids']):,}")

        readme_path.write_text(text, encoding="utf-8")
        print(f"  Updated {readme_path.relative_to(ROOT)}")


def main() -> None:
    yt_state = _load_state(YOUTUBE_STATE)
    tt_state = _load_state(TIKTOK_STATE)

    concat_sources()
    shuffle_combined()

    print("\nSyncing release copies...")
    sync_release_copies()

    print("\nUpdating README statistics...")
    update_readmes(yt_state, tt_state)
    update_tokenizer_readme(yt_state["total_docs"])


if __name__ == "__main__":
    main()
