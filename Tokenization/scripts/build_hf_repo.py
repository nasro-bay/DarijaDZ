#!/usr/bin/env python
"""Builds ../DarijaDz_Tokenizers/ -- a HF Hub-ready model repo staging
folder -- from the trained models in models/{bpe,wordpiece,sentencepiece}/.

Adds `AutoTokenizer.from_pretrained(...)` support for all three algorithm
families:
  - BPE, WordPiece: already self-contained `tokenizer.json` files (byte-level
    BPE has always embedded its decoder; WordPiece has since the newline/
    max_input_chars_per_word fixes, see PLAN.md) -- just need a
    tokenizer_config.json + special_tokens_map.json alongside them.
  - SentencePiece Unigram: converted to the same `tokenizers`-library
    tokenizer.json format via a `transformers.convert_slow_tokenizer
    .SpmConverter` subclass with byte-fallback decoding fixed (the base
    class's decoder doesn't reconstruct byte-fallback tokens correctly --
    see DarijaUnigramConverter below). The raw .model/.vocab files are also
    kept alongside, since Subword-Regularization sampling (enable_sampling=
    True) only exists in the `sentencepiece` library's own API, not in the
    Rust `tokenizers` port used by the fast/AutoTokenizer path.

Re-run after retraining any of the underlying models.
"""
from __future__ import annotations

import json
import os
import shutil
import warnings
from pathlib import Path

from tokenizers import decoders
from transformers.convert_slow_tokenizer import SpmConverter, _get_prepend_scheme

ROOT = Path(__file__).resolve().parents[1]  # Tokenization/
MODELS_DIR = Path(os.environ.get("TOKENIZER_MODELS_DIR") or (ROOT / "models"))
OUT_DIR = ROOT.parent / "DarijaDz_Tokenizers"

VOCAB_SIZES = (1_000, 5_000, 10_000, 20_000, 30_000)


class DarijaUnigramConverter(SpmConverter):
    """SpmConverter subclass with byte-fallback DECODING actually wired up.

    The base class's `handle_byte_fallback` flag controls whether the
    *model* is built with byte_fallback=True (encode side, which already
    works correctly on the base class) -- but its `decoder()` method
    doesn't include the ByteFallback+Fuse steps needed to reconstruct the
    original bytes on decode. Confirmed empirically: without this override,
    decode() left raw "<0xF0><0x9D>..." token strings in the output instead
    of the actual character they represent.
    """

    handle_byte_fallback = True

    def decoder(self, replacement, add_prefix_space):
        prepend_scheme = _get_prepend_scheme(add_prefix_space, self.original_tokenizer)
        return decoders.Sequence(
            [
                decoders.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme),
                decoders.ByteFallback(),
                decoders.Fuse(),
            ]
        )


class _SlowTokenizerShim:
    """Minimal duck-typed stand-in for the `original_tokenizer` SpmConverter
    expects -- it only ever reads `.vocab_file` (path to the .model file)
    and uses `getattr(..., "legacy", True)` / `getattr(..., "add_prefix_space", True)`
    defaults for everything else, so nothing further is needed."""

    def __init__(self, vocab_file: str):
        self.vocab_file = vocab_file


def convert_unigram(model_path: Path, out_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the byte-fallback warning is expected -- we handle it
        tok = DarijaUnigramConverter(_SlowTokenizerShim(str(model_path))).converted()
    tok.save(str(out_path))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_bpe() -> None:
    for vs in VOCAB_SIZES:
        src = MODELS_DIR / "bpe" / f"bpe_{vs}"
        dst = OUT_DIR / "bpe" / f"bpe_{vs}"
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("tokenizer.json", "vocab.json", "merges.txt"):
            shutil.copy2(src / name, dst / name)
        write_json(
            dst / "tokenizer_config.json",
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "clean_up_tokenization_spaces": False,
            },
        )
        write_json(
            dst / "special_tokens_map.json",
            {"unk_token": "<unk>", "pad_token": "<pad>", "bos_token": "<s>", "eos_token": "</s>"},
        )
        print(f"  bpe_{vs}: tokenizer.json + config copied")


def build_wordpiece() -> None:
    for vs in VOCAB_SIZES:
        src = MODELS_DIR / "wordpiece" / f"wordpiece_{vs}" / "tokenizer.json"
        dst = OUT_DIR / "wordpiece" / f"wordpiece_{vs}"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / "tokenizer.json")
        write_json(
            dst / "tokenizer_config.json",
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
                "clean_up_tokenization_spaces": False,
            },
        )
        write_json(
            dst / "special_tokens_map.json",
            {
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
            },
        )
        print(f"  wordpiece_{vs}: tokenizer.json + config copied")


def build_sentencepiece() -> None:
    for vs in VOCAB_SIZES:
        model_path = MODELS_DIR / "sentencepiece" / f"unigram_{vs}.model"
        vocab_path = MODELS_DIR / "sentencepiece" / f"unigram_{vs}.vocab"
        dst = OUT_DIR / "sentencepiece" / f"unigram_{vs}"
        dst.mkdir(parents=True, exist_ok=True)

        # Raw files -- needed for Subword Regularization (enable_sampling=True
        # only exists in the sentencepiece library's API).
        shutil.copy2(model_path, dst / f"unigram_{vs}.model")
        shutil.copy2(vocab_path, dst / f"unigram_{vs}.vocab")

        # Converted -- for AutoTokenizer (deterministic Unigram only).
        convert_unigram(model_path, dst / "tokenizer.json")
        write_json(
            dst / "tokenizer_config.json",
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "clean_up_tokenization_spaces": False,
            },
        )
        write_json(
            dst / "special_tokens_map.json",
            {"unk_token": "<unk>", "pad_token": "<pad>", "bos_token": "<s>", "eos_token": "</s>"},
        )
        print(f"  unigram_{vs}: converted tokenizer.json + raw .model/.vocab + config written")


def main() -> None:
    print("Building bpe/ ...")
    build_bpe()
    print("Building wordpiece/ ...")
    build_wordpiece()
    print("Building sentencepiece/ ...")
    build_sentencepiece()
    print(f"\nDone. Staged at {OUT_DIR}")


if __name__ == "__main__":
    main()
