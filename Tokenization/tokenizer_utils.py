"""Shared tokenizer paths, loading, and evaluation metrics."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import sentencepiece as spm
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders

ROOT = Path(__file__).resolve().parent

# Where trained tokenizer artifacts live. Defaults to ROOT/models; set
# TOKENIZER_MODELS_DIR to point training / eval / HF-repo build at an
# alternate tree (e.g. models_bigcorpus/) without disturbing the models/
# tree the embedding pipelines are pinned to.
MODELS_DIR = Path(os.environ.get("TOKENIZER_MODELS_DIR") or (ROOT / "models"))

VOCAB_SIZES = [1_000, 5_000, 10_000, 20_000, 30_000]

# Four tokenizer variants: three trained models + SR uses the unigram model.
TOKENIZER_TYPES = ("unigram", "unigram_sr", "wordpiece", "bpe")

TOKENIZER_LABELS = {
    "unigram": "SentencePiece Unigram",
    "unigram_sr": "Unigram + Subword Regularization",
    "wordpiece": "SentencePiece WordPiece",
    "bpe": "Byte-level BPE",
}

SR_ALPHA = 0.1
SR_NBest = -1

UNK_TOKENS = ("<unk>",)


@dataclass(frozen=True)
class TokenizerSpec:
    key: str
    label: str
    vocab_size: int
    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]
    pieces: Callable[[str], list[str]]
    vocab_size_actual: int


LEGACY_UNIGRAM = ROOT / "models" / "sentencepiece" / "darija_unigram.model"
LEGACY_BPE = ROOT / "models" / "bpe" / "tokenizer.json"


def sp_model_path(model_type: str, vocab_size: int) -> Path:
    if model_type != "unigram":
        raise ValueError(f"not a SentencePiece model type: {model_type}")
    path = MODELS_DIR / "sentencepiece" / f"{model_type}_{vocab_size}.model"
    if path.exists():
        return path
    # Pre-refactor naming: single 20K unigram model only (default tree only).
    if vocab_size == 20_000 and MODELS_DIR == ROOT / "models" and LEGACY_UNIGRAM.exists():
        return LEGACY_UNIGRAM
    return path


def bpe_model_path(vocab_size: int) -> Path:
    path = MODELS_DIR / "bpe" / f"bpe_{vocab_size}" / "tokenizer.json"
    if path.exists():
        return path
    if vocab_size == 20_000 and MODELS_DIR == ROOT / "models" and LEGACY_BPE.exists():
        return LEGACY_BPE
    return path


def wordpiece_model_path(vocab_size: int) -> Path:
    # train_wordpiece.py trains via the HF `tokenizers` library (WordPieceTrainer),
    # not SentencePiece -- despite PLAN.md's original design calling for
    # `spm.SentencePieceTrainer(model_type="wordpiece")`. The trained artifact is
    # an HF tokenizer.json under models/wordpiece/, not a .model file under
    # models/sentencepiece/ -- this path must match train_wordpiece.py's actual
    # output, not the original (unimplemented) design.
    return MODELS_DIR / "wordpiece" / f"wordpiece_{vocab_size}" / "tokenizer.json"


def load_heldout_docs(path: Path | None = None) -> list[dict]:
    path = path or ROOT / "data" / "heldout_docs.jsonl"
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tokenizer(key: str, vocab_size: int) -> TokenizerSpec:
    if key == "unigram":
        return _load_sentencepiece("unigram", vocab_size, sampling=False)
    if key == "unigram_sr":
        return _load_sentencepiece("unigram", vocab_size, sampling=True)
    if key == "wordpiece":
        return _load_wordpiece(vocab_size)
    if key == "bpe":
        return _load_bpe(vocab_size)
    raise ValueError(f"unknown tokenizer key: {key}")


def _load_sentencepiece(model_type: str, vocab_size: int, *, sampling: bool) -> TokenizerSpec:
    path = sp_model_path(model_type, vocab_size)
    if not path.exists():
        raise FileNotFoundError(f"missing model: {path} — run scripts/train_all.py first")

    sp = spm.SentencePieceProcessor(model_file=str(path))
    sr_key = "unigram_sr" if sampling else model_type
    label = TOKENIZER_LABELS[sr_key]

    def encode(text: str) -> list[int]:
        if sampling:
            return sp.encode(text, enable_sampling=True, alpha=SR_ALPHA, nbest_size=SR_NBest)
        return sp.encode(text)

    def decode(ids: list[int]) -> str:
        return sp.decode(ids)

    def pieces(text: str) -> list[str]:
        if sampling:
            return sp.encode(
                text, out_type=str, enable_sampling=True, alpha=SR_ALPHA, nbest_size=SR_NBest
            )
        return sp.encode(text, out_type=str)

    return TokenizerSpec(
        key=sr_key,
        label=label,
        vocab_size=vocab_size,
        encode=encode,
        decode=decode,
        pieces=pieces,
        vocab_size_actual=sp.vocab_size(),
    )


def _load_wordpiece(vocab_size: int) -> TokenizerSpec:
    path = wordpiece_model_path(vocab_size)
    if not path.exists():
        raise FileNotFoundError(f"missing model: {path} — run scripts/train_all.py first")

    tok = HFTokenizer.from_file(str(path))
    # train_wordpiece.py never attaches a decoder to the saved tokenizer.json,
    # so a fresh load has none by default -- decode() would otherwise just
    # space-join raw tokens (leaving literal "##" continuation markers in the
    # output). Attach it here at load time rather than rewriting the saved
    # model files.
    # cleanup=False: the default cleanup=True applies English-oriented
    # heuristics (e.g. stripping the space before "!", "?") that incorrectly
    # eat real spaces in this multilingual corpus -- confirmed via a real
    # heldout mismatch: "نبداو !!" round-tripped to "نبداو!!" with cleanup=True.
    tok.decoder = decoders.WordPiece(prefix="##", cleanup=False)

    # WhitespaceSplit() treats "\n" as ordinary whitespace, indistinguishable
    # from a space once collapsed, so the decoder can never restore it --
    # substitute it for a dedicated special token (registered in
    # train_wordpiece.py's SPECIAL_TOKENS) around encode, and reverse the
    # substitution around decode. The regex (not a plain .replace) matters:
    # it strips up to one space independently on *each* side of "[NEWLINE]"
    # rather than requiring both, which is what makes it correctly reverse
    # consecutive newlines (two markers sharing one decoder-inserted space
    # between them) and newlines at the very start/end of a doc (no space
    # on one side to match at all) -- a naive symmetric-padding replace
    # fails both of those. skip_special_tokens=False is required too, since
    # tok.decode() silently drops special tokens by default.
    _NEWLINE_TOKEN = "[NEWLINE]"
    _NEWLINE_DECODE_RE = re.compile(r"\s?\[NEWLINE\]\s?")

    def _to_sentinel(text: str) -> str:
        return text.replace("\n", f" {_NEWLINE_TOKEN} ")

    def encode(text: str) -> list[int]:
        return tok.encode(_to_sentinel(text)).ids

    def decode(ids: list[int]) -> str:
        decoded = tok.decode(ids, skip_special_tokens=False)
        return _NEWLINE_DECODE_RE.sub("\n", decoded)

    def pieces(text: str) -> list[str]:
        return tok.encode(_to_sentinel(text)).tokens

    return TokenizerSpec(
        key="wordpiece",
        label=TOKENIZER_LABELS["wordpiece"],
        vocab_size=vocab_size,
        encode=encode,
        decode=decode,
        pieces=pieces,
        vocab_size_actual=tok.get_vocab_size(),
    )


def _load_bpe(vocab_size: int) -> TokenizerSpec:
    path = bpe_model_path(vocab_size)
    if not path.exists():
        raise FileNotFoundError(f"missing model: {path} — run scripts/train_all.py first")

    tok = HFTokenizer.from_file(str(path))

    def encode(text: str) -> list[int]:
        return tok.encode(text).ids

    def decode(ids: list[int]) -> str:
        return tok.decode(ids)

    def pieces(text: str) -> list[str]:
        return tok.encode(text).tokens

    return TokenizerSpec(
        key="bpe",
        label=TOKENIZER_LABELS["bpe"],
        vocab_size=vocab_size,
        encode=encode,
        decode=decode,
        pieces=pieces,
        vocab_size_actual=tok.get_vocab_size(),
    )


def discover_available_models() -> list[tuple[str, int]]:
    """Return (tokenizer_key, vocab_size) pairs with trained artifacts on disk."""
    found: set[tuple[str, int]] = set()
    sp_dir = MODELS_DIR / "sentencepiece"
    if sp_dir.exists():
        for path in sp_dir.glob("unigram_*.model"):
            vocab_size = int(path.stem.rsplit("_", 1)[-1])
            found.add(("unigram", vocab_size))
            found.add(("unigram_sr", vocab_size))
    wordpiece_dir = MODELS_DIR / "wordpiece"
    if wordpiece_dir.exists():
        for path in wordpiece_dir.glob("wordpiece_*/tokenizer.json"):
            vocab_size = int(path.parent.name.rsplit("_", 1)[-1])
            found.add(("wordpiece", vocab_size))
    bpe_dir = MODELS_DIR / "bpe"
    if bpe_dir.exists():
        for path in bpe_dir.glob("bpe_*/tokenizer.json"):
            vocab_size = int(path.parent.name.rsplit("_", 1)[-1])
            found.add(("bpe", vocab_size))
        if LEGACY_BPE.exists():
            found.add(("bpe", 20_000))
    if LEGACY_UNIGRAM.exists():
        found.add(("unigram", 20_000))
        found.add(("unigram_sr", 20_000))
    return sorted(found, key=lambda x: (x[1], TOKENIZER_TYPES.index(x[0])))


_ARABIC_RE = re.compile(r"[؀-ۿ]")
_LATIN_RE = re.compile(r"[a-zA-Z]")


def classify_script(word: str) -> str:
    has_ar = bool(_ARABIC_RE.search(word))
    has_lat = bool(_LATIN_RE.search(word))
    if has_ar and has_lat:
        return "mixed"
    if has_ar:
        return "arabic"
    if has_lat:
        return "latin"
    return "other"


def effective_word_cost(word: str, pieces: list[str], unk_tokens: tuple[str, ...] = UNK_TOKENS) -> int:
    if any(p in unk_tokens for p in pieces):
        return len(word) + 1
    return len(pieces)


def compression_factor(
    text: str,
    pieces_fn: Callable[[str], list[str]],
    unk_tokens: tuple[str, ...] = UNK_TOKENS,
) -> float:
    """CF = total effective tokens / (total characters + total words).

    Lower is better (fewer splits / less UNK inflation).
    """
    words = [w for w in text.split() if w]
    if not words:
        return 0.0
    total_chars = sum(len(w) for w in words)
    total_words = len(words)
    effective = sum(effective_word_cost(w, pieces_fn(w), unk_tokens) for w in words)
    return effective / (total_chars + total_words)


def fertility_for_words(words: list[tuple[str, str]], pieces_fn: Callable[[str], list[str]]) -> dict:
    per_script_tokens: dict[str, int] = {}
    per_script_words: dict[str, int] = {}
    for word, script in words:
        n = len(pieces_fn(word))
        per_script_tokens[script] = per_script_tokens.get(script, 0) + n
        per_script_words[script] = per_script_words.get(script, 0) + 1
    rows = []
    for script in sorted(per_script_words):
        rows.append(
            {
                "script": script,
                "words": per_script_words[script],
                "tokens": per_script_tokens[script],
                "fertility": per_script_tokens[script] / per_script_words[script],
            }
        )
    overall_tokens = sum(per_script_tokens.values())
    overall_words = sum(per_script_words.values())
    rows.append(
        {
            "script": "ALL",
            "words": overall_words,
            "tokens": overall_tokens,
            "fertility": overall_tokens / overall_words if overall_words else 0.0,
        }
    )
    return {"rows": rows, "overall_fertility": overall_tokens / overall_words if overall_words else 0.0}


def time_encode_decode(tok: TokenizerSpec, texts: list[str], *, rounds: int = 3) -> dict:
    """Wall-clock encode/decode timing averaged over `rounds` passes."""
    if not texts:
        return {"encode_ms_per_doc": 0.0, "decode_ms_per_doc": 0.0, "roundtrip_ms_per_doc": 0.0}

    encode_t = 0.0
    decode_t = 0.0
    roundtrip_t = 0.0
    n = len(texts)

    for _ in range(rounds):
        t0 = time.perf_counter()
        encoded = [tok.encode(t) for t in texts]
        encode_t += time.perf_counter() - t0

        t0 = time.perf_counter()
        for ids in encoded:
            tok.decode(ids)
        decode_t += time.perf_counter() - t0

        t0 = time.perf_counter()
        for t in texts:
            tok.decode(tok.encode(t))
        roundtrip_t += time.perf_counter() - t0

    scale = 1000.0 / (rounds * n)
    return {
        "encode_ms_per_doc": encode_t * scale,
        "decode_ms_per_doc": decode_t * scale,
        "roundtrip_ms_per_doc": roundtrip_t * scale,
    }


def piece_label(tokenizer_key: str, piece: str, tok: TokenizerSpec | None = None) -> str:
    """Human-readable label for visualization."""
    if tokenizer_key == "bpe" and tok is not None:
        # Re-decode single id when possible; fall back to raw piece string.
        enc = tok.encode(piece) if len(piece) > 1 else None
        if enc:
            decoded = tok.decode([enc[0]]) if len(enc) == 1 else tok.decode(enc)
            if decoded.strip():
                return decoded
    return piece.replace("\u2581", " ").strip() or piece
