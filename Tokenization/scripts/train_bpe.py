#!/usr/bin/env python
"""Train a byte-level BPE tokenizer (GPT-2/RoBERTa-style) on data/train_corpus.txt."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CORPUS = ROOT / "data" / "train_corpus.txt"

SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]


def train(*, vocab_size: int) -> None:
    if not TRAIN_CORPUS.exists():
        raise FileNotFoundError(f"{TRAIN_CORPUS} not found — run build_training_corpus.py first.")

    models_root = Path(os.environ.get("TOKENIZER_MODELS_DIR") or (ROOT / "models"))
    model_dir = models_root / "bpe" / f"bpe_{vocab_size}"
    model_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train([str(TRAIN_CORPUS)], trainer)

    tokenizer.save(str(model_dir / "tokenizer.json"))
    tokenizer.model.save(str(model_dir))
    print(f"saved {model_dir / 'tokenizer.json'}, vocab.json, merges.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train byte-level BPE tokenizer")
    parser.add_argument("--vocab-size", type=int, default=20_000, help="vocabulary size")
    args = parser.parse_args()
    train(vocab_size=args.vocab_size)


if __name__ == "__main__":
    main()
