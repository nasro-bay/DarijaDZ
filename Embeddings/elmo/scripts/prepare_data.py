#!/usr/bin/env python
"""Builds the ELMo biLM training data (plan.md section 4).

Corpus: the N longest comments (by `token_count`, ties by file order) of the big corpus
`Embeddings/word2vec_attention/data_bigcorpus/rows.jsonl` -- the same selection rule as
`Embeddings/glove/scripts/select_corpus.py`. A seeded random subset of them is carved off as the
**held-out** set first and is excluded from all counting/vocabulary building (so held-out OOV rates
are realistic).

Tokenization: regex `\\w+|[^\\w\\s]` (words + single punctuation/emoji chars), case preserved.
Each comment is split into chunks of <= 64 tokens (nothing is truncated), and every chunk becomes
`[<S>, t_1..t_n, </S>]`.

Representation (all integer ids, so training batches are just slices):
- input **type ids**: every distinct cased token (sorted by frequency) gets an id (0=<pad>, 1=<S>,
  2=</S>, real types from 3); `char_table` [n_types, 20] uint8 holds each type's characters
  (BOW + <=18 codepoint ids + EOW). The char-CNN reads this table on the GPU.
- output **word ids**: lowercased form -> top-19,996 vocab (+ <pad>=0 <unk>=1 <S>=2 </S>=3);
  `type_to_out` maps input type id -> output id (targets are derived on the GPU, never stored).
- chunks are stored **sorted by exact length** (random order within a length): every training batch is
  one contiguous slice of same-length chunks -> zero padding, no masks, static shapes per length.

Outputs in `--out-dir` (default `../data`): `{train,heldout}_tokens.i32`, `{...}_lens.npy` (uint8),
`{...}_offs.npy` (int64), `char_table.npy`, `type_to_out.npy`, `vocab_out.json`, `char_vocab.json`,
`meta.json`.

Run via the GPU venv's Python (numpy only):
    ".../ai-gpu/Scripts/python.exe" -u prepare_data.py
"""
from __future__ import annotations

import argparse
import json
import re
import time
from array import array
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "Embeddings" / "word2vec_attention" / "data_bigcorpus" / "rows.jsonl"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data"

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
TC_KEY = b'"token_count": '
MAX_CHUNK = 64
MAX_CHARS = 20            # BOW + up to 18 chars + EOW
N_CHAR_KEEP = 248         # char ids 4..251 (0=pad 1=unk 2=BOW 3=EOW 252=<S> 253=</S>)
V_OUT_WORDS = 19_996      # + 4 specials = 20,000; padded to 20,032 in the model
SEED = 0


def read_token_counts(limit: int | None) -> np.ndarray:
    counts: list[int] = []
    t0 = time.time()
    with SRC.open("rb") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            pos = line.rfind(TC_KEY)
            start = pos + len(TC_KEY)
            end = start
            while line[end:end + 1].isdigit():
                end += 1
            counts.append(int(line[start:end]))
            if (i + 1) % 4_000_000 == 0:
                print(f"  token counts: {i + 1:,} rows ({time.time() - t0:.0f}s)", flush=True)
    return np.asarray(counts, dtype=np.int32)


def iter_selected(split: np.ndarray, want: set[int]):
    """Yields (split_code, text) for rows whose split code is in `want`, in file order."""
    with SRC.open("rb") as f:
        for i, line in enumerate(f):
            if i >= len(split):
                break
            code = split[i]
            if code in want:
                yield int(code), json.loads(line)["text"]


def sort_and_write(unsorted_path: Path, lens: np.ndarray, out_prefix: Path, rng: np.random.Generator) -> int:
    """Reorders chunks by (length, random) and writes the flat token file + lens + offsets."""
    n_chunks = len(lens)
    starts = np.concatenate([[0], np.cumsum(lens, dtype=np.int64)[:-1]])
    order = np.lexsort((rng.permutation(n_chunks), lens))
    s_lens = lens[order]
    s_starts = starts[order]
    total = int(s_lens.sum())
    src = np.memmap(unsorted_path, dtype=np.int32, mode="r", shape=(total,))
    dst = np.memmap(out_prefix.with_name(out_prefix.name + "_tokens.i32"), dtype=np.int32, mode="w+", shape=(total,))
    pos = 0
    block = 400_000
    for a in range(0, n_chunks, block):
        L = s_lens[a:a + block].astype(np.int64)
        st = s_starts[a:a + block]
        n = int(L.sum())
        idx = np.repeat(st - (np.cumsum(L) - L), L) + np.arange(n, dtype=np.int64)
        dst[pos:pos + n] = src[idx]
        pos += n
    dst.flush()
    del dst, src
    offs = np.concatenate([[0], np.cumsum(s_lens, dtype=np.int64)[:-1]])
    np.save(out_prefix.with_name(out_prefix.name + "_lens.npy"), s_lens.astype(np.uint8))
    np.save(out_prefix.with_name(out_prefix.name + "_offs.npy"), offs)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000_000, help="comments to select (longest first)")
    parser.add_argument("--heldout", type=int, default=50_000)
    parser.add_argument("--limit-rows", type=int, default=None, help="only read the first N rows of the source (tests)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    T0 = time.time()

    # ---- pass A: selection + held-out split -------------------------------------------------------------
    print("Pass A: reading token counts / selecting", flush=True)
    tc = read_token_counts(args.limit_rows)
    n_sel = min(args.n, len(tc))
    cutoff = int(np.sort(tc)[::-1][n_sel - 1])
    keep = tc > cutoff
    keep[np.flatnonzero(tc == cutoff)[: n_sel - int(keep.sum())]] = True
    sel_idx = np.flatnonzero(keep)
    held_idx = rng.choice(sel_idx, size=min(args.heldout, len(sel_idx) // 2), replace=False)
    split = np.full(len(tc), -1, dtype=np.int8)     # -1 unselected, 0 train, 1 held-out
    split[sel_idx] = 0
    split[held_idx] = 1
    print(f"  selected {len(sel_idx):,} (length cutoff {cutoff} pieces); held-out {len(held_idx):,}", flush=True)
    del tc, keep

    # ---- pass B: counts (train rows only) -------------------------------------------------------------
    print("Pass B: counting tokens and chars (train rows)", flush=True)
    cnt: Counter[str] = Counter()
    chars: Counter[str] = Counter()
    t0 = time.time()
    for i, (_, text) in enumerate(iter_selected(split, {0})):
        cnt.update(TOKEN_RE.findall(text))
        chars.update(text)
        if (i + 1) % 1_000_000 == 0:
            print(f"  {i + 1:,} comments ({time.time() - t0:.0f}s), {len(cnt):,} types", flush=True)
    total_tokens = sum(cnt.values())
    print(f"  {total_tokens:,} tokens, {len(cnt):,} distinct cased types", flush=True)

    # ---- vocabularies ---------------------------------------------------------------------------------
    types = [w for w, _ in cnt.most_common()]
    lc: Counter[str] = Counter()
    for w, c in cnt.items():
        lc[w.lower()] += c
    out_words = [w for w, _ in lc.most_common(V_OUT_WORDS)]
    del lc
    out_id = {w: i + 4 for i, w in enumerate(out_words)}
    vocab_out = ["<pad>", "<unk>", "<S>", "</S>"] + out_words

    keep_chars = [c for c, _ in chars.most_common() if not c.isspace()][:N_CHAR_KEEP]
    char_id = {c: i + 4 for i, c in enumerate(keep_chars)}

    type_id = {w: i + 3 for i, w in enumerate(types)}
    type_to_out = [0, 2, 3]
    unk_positions = 0
    for w in types:
        o = out_id.get(w.lower(), 1)
        type_to_out.append(o)
        if o == 1:
            unk_positions += cnt[w]
    del cnt
    print(f"  output vocab {len(vocab_out):,}; <unk> covers {100 * unk_positions / total_tokens:.2f}% of train tokens; "
          f"{len(keep_chars)} chars kept", flush=True)

    # ---- pass C: encode ---------------------------------------------------------------------------------
    print("Pass C: encoding", flush=True)
    streams = {0: {"path": out / "train_unsorted.i32", "lens": array("B"), "buf": array("i"), "fh": None},
               1: {"path": out / "heldout_unsorted.i32", "lens": array("B"), "buf": array("i"), "fh": None}}
    for s in streams.values():
        s["fh"] = s["path"].open("wb")
    extra_types: list[str] = []
    t0 = time.time()
    n_rows = 0
    for code, text in iter_selected(split, {0, 1}):
        toks = TOKEN_RE.findall(text)
        if not toks:
            continue
        if code == 0:
            ids = [type_id[t] for t in toks]
        else:                                        # held-out: types unseen in training get fresh ids
            ids = []
            for t in toks:
                tid = type_id.get(t)
                if tid is None:
                    tid = len(type_to_out)
                    type_id[t] = tid
                    type_to_out.append(out_id.get(t.lower(), 1))
                    extra_types.append(t)
                ids.append(tid)
        st = streams[code]
        for s in range(0, len(ids), MAX_CHUNK):
            piece = ids[s:s + MAX_CHUNK]
            st["buf"].append(1)
            st["buf"].extend(piece)
            st["buf"].append(2)
            st["lens"].append(len(piece) + 2)
        if len(st["buf"]) > 8_000_000:
            st["buf"].tofile(st["fh"])
            st["buf"] = array("i")
        n_rows += 1
        if n_rows % 1_000_000 == 0:
            print(f"  {n_rows:,} comments ({time.time() - t0:.0f}s)", flush=True)
    for s in streams.values():
        s["buf"].tofile(s["fh"])
        s["fh"].close()

    # ---- char table -------------------------------------------------------------------------------------
    print("Building char table", flush=True)
    all_types = types + extra_types

    def row(w: str) -> bytes:
        ids = [2] + [char_id.get(c, 1) for c in w[:MAX_CHARS - 2]] + [3]
        return bytes(ids).ljust(MAX_CHARS, b"\0")

    specials = [bytes(MAX_CHARS), bytes([2, 252, 3]).ljust(MAX_CHARS, b"\0"), bytes([2, 253, 3]).ljust(MAX_CHARS, b"\0")]
    char_table = np.frombuffer(b"".join(specials + [row(w) for w in all_types]), dtype=np.uint8).reshape(-1, MAX_CHARS)
    assert len(char_table) == len(type_to_out), (len(char_table), len(type_to_out))
    np.save(out / "char_table.npy", char_table)
    np.save(out / "type_to_out.npy", np.asarray(type_to_out, dtype=np.uint16))
    (out / "vocab_out.json").write_text(json.dumps(vocab_out, ensure_ascii=False), encoding="utf-8")
    (out / "char_vocab.json").write_text(json.dumps(keep_chars, ensure_ascii=False), encoding="utf-8")

    # ---- sort chunks by exact length and write final files ------------------------------------------------
    stats = {}
    for code, name in ((0, "train"), (1, "heldout")):
        st = streams[code]
        lens = np.frombuffer(st["lens"], dtype=np.uint8)
        print(f"Sorting {name}: {len(lens):,} chunks", flush=True)
        total = sort_and_write(st["path"], lens, out / name, np.random.default_rng(SEED + code))
        st["path"].unlink()
        stats[name] = {"chunks": int(len(lens)), "positions": total,
                       "mean_len": float(lens.mean()), "max_len": int(lens.max())}

    meta = {"source": str(SRC.relative_to(ROOT)), "selected_comments": int(len(sel_idx)), "length_cutoff": cutoff,
            "heldout_comments": int(len(held_idx)), "train_tokens": total_tokens, "n_types": len(type_to_out),
            "extra_heldout_types": len(extra_types), "out_vocab": len(vocab_out),
            "unk_rate_train": unk_positions / total_tokens, "n_chars_kept": len(keep_chars),
            "max_chunk": MAX_CHUNK, "max_chars": MAX_CHARS, "splits": stats, "seconds": time.time() - T0}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
