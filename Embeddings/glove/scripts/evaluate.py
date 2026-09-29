#!/usr/bin/env python
"""Evaluates trained GloVe embeddings with the repo's existing intrinsic
evaluation (`Embeddings/intrinsic_eval`: word-similarity Spearman + analogy
accuracy on Darija word sets), using exactly the protocol the CBOW/skip-gram
eval notebooks use so numbers are directly comparable:

- `pieces` model: a word's vector = mean of the vectors of its SentencePiece
  pieces (a single-piece word is a direct lookup) -- `get_word_vector` from
  `Embeddings/word2vec/*/word2vec_eval.ipynb`.
- `words` model: direct lookup of the lowercased word; a word outside the
  20K vocabulary is OOV (`None`), which the intrinsic-eval harness skips and
  counts -- so read the OOV/coverage columns alongside the scores, since a
  model can score higher just by being evaluated on fewer, easier items.

Also evaluates the repo's skip-gram checkpoint (128-d, pieces) as a
reference row. Writes `models/eval_results.json`.

Run via the GPU venv's Python:
    ".../ai-gpu/Scripts/python.exe" -u evaluate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SP_MODEL = ROOT / "Tokenization" / "models_bigcorpus" / "sentencepiece" / "unigram_20000.model"
SKIPGRAM_CKPT = ROOT / "Embeddings" / "word2vec" / "skip-gram" / "models" / "final.pt"

sys.path.insert(0, str(ROOT / "Embeddings" / "intrinsic_eval" / "scripts"))
from evaluate_embeddings import evaluate_analogies, evaluate_similarity  # noqa: E402


def pieces_embed_fn(E: np.ndarray):
    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
    V = E.shape[0]

    def embed(word: str):
        ids = [i for i in sp.encode(word) if i != 0 and i < V]
        return E[ids].mean(axis=0) if ids else None

    return embed


def words_embed_fn(E: np.ndarray):
    vocab = json.loads((DATA_DIR / "vocab_words.json").read_text(encoding="utf-8"))
    word_id = {w: i for i, w in enumerate(vocab)}

    def embed(word: str):
        i = word_id.get(word.lower())
        return None if i is None else E[i]

    return embed


def run_eval(name: str, embed) -> dict:
    sim = evaluate_similarity(embed)
    ana = evaluate_analogies(embed)
    total_pairs = sim["n_pairs_scored"] + sim["n_pairs_oov"]
    result = {
        "name": name,
        "similarity_spearman": sim["overall_spearman"],
        "similarity_pairs_scored": sim["n_pairs_scored"],
        "similarity_pairs_oov": sim["n_pairs_oov"],
        "similarity_coverage": sim["n_pairs_scored"] / total_pairs if total_pairs else float("nan"),
        "analogy_accuracy": ana["overall_accuracy"],
        "analogy_questions": ana["overall_n_questions"],
        "analogy_skipped_oov": sum(c["n_skipped_oov"] for c in ana["by_category"].values()),
        "similarity_by_category": {k: v["spearman"] for k, v in sim["by_category"].items()},
        "analogy_by_category": {k: v["accuracy"] for k, v in ana["by_category"].items()},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None,
                        help="glove .npz filenames in models/ (default: every glove_*.npz)")
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()

    paths = [MODELS_DIR / m for m in args.models] if args.models else sorted(
        p for p in MODELS_DIR.glob("glove_*.npz") if "smoke" not in p.name)
    word_vocab = set(json.loads((DATA_DIR / "vocab_words.json").read_text(encoding="utf-8")))
    results = []
    for path in paths:
        mode = "words" if "_words_" in path.name else "pieces"
        E = np.load(path)["embeddings"]
        embed = words_embed_fn(E) if mode == "words" else pieces_embed_fn(E)
        results.append(run_eval(f"{path.stem} ({mode}, d={E.shape[1]})", embed))
        if mode == "pieces":
            # Apples-to-apples with the words model: a pieces model can embed ANY word, but the words
            # model only the 20K in its vocabulary -- so score the pieces model on exactly the items
            # (pairs / analogy questions / candidate pool) the words model can answer.
            def restricted(word: str, _embed=embed):
                return _embed(word) if word.lower() in word_vocab else None
            results.append(run_eval(f"{path.stem} (pieces, words-model coverage only)", restricted))

    if not args.no_baseline and SKIPGRAM_CKPT.exists():
        sd = torch.load(SKIPGRAM_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
        E = sd["input_embeddings.weight"].numpy()
        results.append(run_eval(f"skip-gram final.pt (pieces, d={E.shape[1]}) [reference]", pieces_embed_fn(E)))

    print(f"\n{'model':<52} {'sim rho':>8} {'scored/OOV':>11} {'analogy acc':>12} {'questions/skipped':>18}")
    for r in results:
        print(f"{r['name']:<52} {r['similarity_spearman']:>8.4f} "
              f"{r['similarity_pairs_scored']:>5}/{r['similarity_pairs_oov']:<5} "
              f"{r['analogy_accuracy']:>12.4f} {r['analogy_questions']:>9}/{r['analogy_skipped_oov']:<8}")

    # A run restricted with --models must not overwrite the full table.
    out_name = "eval_results_subset.json" if args.models else "eval_results.json"
    (MODELS_DIR / out_name).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {MODELS_DIR / out_name}")


if __name__ == "__main__":
    main()
