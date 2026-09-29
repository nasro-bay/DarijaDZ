"""Shared data-loading pieces for both cbow/ and skip-gram/ -- moved here
verbatim from word2vec_attention/scripts/{build_training_data,dataset}.py
rather than duplicated per algorithm, since none of this is
attention-specific: loading the prepared corpus cache, the negative-
sampling distribution, frequent-word subsampling probabilities, and the
per-row Arabizi-transliteration augmentation.

Neither cbow/ nor skip-gram/ has its own preprocessing script or data/
cache -- both read directly from word2vec_attention/data/ (see
load_rows()), reusing the prepared cache built there instead of paying
for another full pass over the corpus. `cluster_id` (K-Means length
bucket) is present in that cache but unused here -- it existed only for
word2vec_attention's attention-batching needs (see load_rows()'s
docstring for why cbow/skip-gram don't need equivalent batching logic).
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # project root
# Same WORD2VEC_DATA_DIR override as word2vec_attention/scripts/build_training_data.py
# -- must point at the same tree that script was run with.
WORD2VEC_ATTENTION_DATA = Path(
    os.environ.get("WORD2VEC_DATA_DIR") or (ROOT / "Embeddings" / "word2vec_attention" / "data")
)

sys.path.insert(0, str(ROOT / "Tokenization"))
sys.path.insert(0, str(ROOT / "DarijaDZ_spaces" / "Arabizi_transliteration"))
from tokenizer_utils import load_tokenizer  # noqa: E402
from transliterate import transliterate_word  # noqa: E402

# Switched from "bpe" to "unigram" -- see build_training_data.py's comment.
TOKENIZER_KEY = "unigram"
VOCAB_SIZE = 20_000
PAD_ID = 0  # SentencePiece trained with pad_id=0 explicitly (train_sentencepiece.py) -- same as the old BPE-20K vocab's <pad>=0

SUBSAMPLE_THRESHOLD = 1e-3
AUGMENT_RATE = 0.20
AUGMENT_MIN_WORDS = 5
NEGATIVE_TABLE_SIZE = 10_000_000


def load_tok():
    """Same BPE-20K tokenizer as word2vec_attention -- kept as a thin
    wrapper (not just calling load_tokenizer directly at each call site)
    so the key/vocab size only need to be right in one place."""
    return load_tokenizer(TOKENIZER_KEY, VOCAB_SIZE)


def load_rows() -> tuple[list[dict], dict[int, int], dict]:
    """Reads word2vec_attention's prepared corpus cache directly --
    rows.jsonl, token_freq.json, meta.json -- rather than re-running a
    preprocessing pass (see module docstring). Raises a clear error
    (not a bare FileNotFoundError) if the cache doesn't exist yet, since
    "go build word2vec_attention's cache first" is a one-time setup step
    a future session could easily forget.
    """
    rows_path = WORD2VEC_ATTENTION_DATA / "rows.jsonl"
    freq_path = WORD2VEC_ATTENTION_DATA / "token_freq.json"
    meta_path = WORD2VEC_ATTENTION_DATA / "meta.json"
    for p in (rows_path, freq_path, meta_path):
        if not p.exists():
            raise SystemExit(
                f"{p} not found -- build word2vec_attention's prepared cache first "
                f"(word2vec_attention/scripts/build_training_data.py). cbow/ and "
                f"skip-gram/ read from it directly rather than building their own."
            )

    with rows_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    token_freq = {int(k): v for k, v in json.loads(freq_path.read_text(encoding="utf-8")).items()}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return rows, token_freq, meta


def build_negative_sampling_table(
    token_freq: dict[int, int], vocab_size: int, table_size: int = NEGATIVE_TABLE_SIZE
) -> np.ndarray:
    """Unigram^0.75 sampling table (standard word2vec negative-sampling
    distribution) -- identical to word2vec_attention/scripts/dataset.py's
    version. Fills a large array proportionally to freq^0.75, then samples
    by drawing random indices into it (O(1) per sample).
    """
    freqs = np.zeros(vocab_size, dtype=np.float64)
    for tok_id, count in token_freq.items():
        tok_id = int(tok_id)
        if tok_id < vocab_size:
            freqs[tok_id] = count
    freqs = np.power(freqs, 0.75)
    total = freqs.sum()
    probs = freqs / total if total > 0 else np.full(vocab_size, 1.0 / vocab_size)
    counts_per_id = np.round(probs * table_size).astype(np.int64)
    counts_per_id[counts_per_id == 0] = 0  # words that never occurred stay at 0 -- never sampled as negatives
    table = np.repeat(np.arange(vocab_size), counts_per_id)
    return table


def build_subsample_keep_prob(token_freq: dict[int, int], threshold: float = SUBSAMPLE_THRESHOLD) -> dict[int, float]:
    """P(keep token) = sqrt(threshold / freq_ratio), capped at 1.0 --
    Mikolov et al.'s subsampling formula, identical to
    word2vec_attention/scripts/dataset.py's version.
    """
    total = sum(token_freq.values())
    keep_prob: dict[int, float] = {}
    for tok_id, count in token_freq.items():
        freq_ratio = count / total
        keep_prob[int(tok_id)] = min(1.0, (threshold / freq_ratio) ** 0.5)
    return keep_prob


def maybe_augment(row: dict, rng: random.Random) -> str:
    """Same 20%-chance single-word Arabizi transliteration as
    word2vec_attention/scripts/dataset.py's CBOWCollator._maybe_augment --
    kept for consistency so a later comparison isolates architecture as
    the only real variable, not also the data pipeline (see plan.md).
    """
    text = row["text"]
    if (
        row["script_bucket"] == "arabic"
        and row["word_count"] > AUGMENT_MIN_WORDS
        and rng.random() < AUGMENT_RATE
    ):
        words = text.split()
        i = rng.randrange(len(words))
        words[i] = transliterate_word(words[i])
        text = " ".join(words)
    return text
