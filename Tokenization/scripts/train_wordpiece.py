from __future__ import annotations

import argparse
import os
from pathlib import Path

from tokenizers import Tokenizer, decoders
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordPieceTrainer

ROOT = Path(__file__).resolve().parents[1]

TRAIN_CORPUS = ROOT / "data" / "train_corpus.txt"
MODEL_DIR = Path(os.environ.get("TOKENIZER_MODELS_DIR") or (ROOT / "models")) / "wordpiece"


SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    # Sentinel for real newlines. WhitespaceSplit() treats "\n" as
    # ordinary whitespace, indistinguishable from a space, so the decoder
    # can't tell them apart -- tokenizer_utils.py's _load_wordpiece()
    # substitutes "\n" <-> " [NEWLINE] " around encode/decode to recover
    # it. Registering it as a special token guarantees a vocab slot
    # regardless of training-corpus frequency (train_corpus.txt has real
    # newlines already stripped to spaces upstream, by design, for the
    # other tokenizers that don't need this).
    "[NEWLINE]",
]


def train(*, vocab_size: int) -> None:
    """Train a WordPiece tokenizer on the Darija training corpus."""

    if not TRAIN_CORPUS.exists():
        raise FileNotFoundError(
            f"{TRAIN_CORPUS} not found — "
            "run build_training_corpus.py first."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    output_dir = MODEL_DIR / f"wordpiece_{vocab_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(
        WordPiece(
            unk_token="[UNK]",
            continuing_subword_prefix="##",
            # Default is 100 -- WhitespaceSplit() no longer splits on
            # punctuation the way the old Whitespace() did, so a long
            # punctuation-only-separated run (no real whitespace) can
            # exceed the default and get silently replaced whole with
            # [UNK] (which then vanishes entirely on decode). Confirmed on
            # a real held-out doc (a 115-char no-space run). 1000
            # comfortably covers any realistic comment/post length here.
            max_input_chars_per_word=1000,
        )
    )

    # WhitespaceSplit() (splits only on actual whitespace) rather than
    # Whitespace() (which also splits off punctuation/emoji into their own
    # pre-tokens) -- the latter throws away zero-gap adjacency info (e.g.
    # "[MENTION]" -> "[", "MENTION", "]"), which the decoder can never
    # recover since it always rejoins pre-tokens with a single space.
    tokenizer.pre_tokenizer = WhitespaceSplit()

    # Bake the decoder into the saved file itself (not just attached at
    # load time in tokenizer_utils.py) -- a bare tokenizer.json with no
    # decoder loaded via plain Tokenizer.from_file()/AutoTokenizer decodes
    # by space-joining raw tokens, leaving literal "##" markers in the
    # output. cleanup=False for the same reason as tokenizer_utils.py: the
    # default cleanup=True strips real spaces before punctuation in this
    # corpus (confirmed regression, see PLAN.md).
    tokenizer.decoder = decoders.WordPiece(prefix="##", cleanup=False)

    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    print(
        f"\n=== Training WordPiece vocab={vocab_size:,} ==="
    )

    tokenizer.train(
        files=[str(TRAIN_CORPUS)],
        trainer=trainer,
    )

    tokenizer.save(
        str(output_dir / "tokenizer.json")
    )

    print(
        f"saved {output_dir / 'tokenizer.json'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a WordPiece tokenizer"
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