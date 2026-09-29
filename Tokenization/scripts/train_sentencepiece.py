#!/usr/bin/env python
"""Train a SentencePiece tokenizer (Unigram) on data/train_corpus.txt.

Unigram uses the probabilistic subword model from Kudo et al. (2018). Subword
Regularization is applied at encode time (see tokenizer_utils.py), not here.

`normalization_rule_name="identity"` keeps the corpus exactly as clean_text.py
left it — SentencePiece's default NFKC pass is disabled.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CORPUS = ROOT / "data" / "train_corpus.txt"
MODEL_DIR = Path(os.environ.get("TOKENIZER_MODELS_DIR") or (ROOT / "models")) / "sentencepiece"

# Reduce SentencePiece's very verbose INFO logs.
# 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = FATAL
spm.set_min_log_level(1)


def train(*, vocab_size: int) -> None:
    """Train a SentencePiece Unigram tokenizer."""
    if not TRAIN_CORPUS.exists():
        raise FileNotFoundError(
            f"{TRAIN_CORPUS} not found — "
            "run build_training_corpus.py first."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    prefix = MODEL_DIR / f"unigram_{vocab_size}"

    spm.SentencePieceTrainer.train(
        input=str(TRAIN_CORPUS),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        byte_fallback=True,
        character_coverage=0.9995,
        normalization_rule_name="identity",
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    print(f"saved {prefix}.model / {prefix}.vocab")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece Unigram tokenizer"
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=20_000,
        help="vocabulary size (default: 20,000)",
    )

    args = parser.parse_args()
    train(vocab_size=args.vocab_size)


if __name__ == "__main__":
    main()
