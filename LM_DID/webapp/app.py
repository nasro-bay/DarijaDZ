#!/usr/bin/env python
"""Tiny local web page for LM_DiD: type text, see the model's predicted
next-token distribution.

Loads the highest-step checkpoint present in `LM_DiD/models/` (same
auto-select convention as `Dialect_Identification/notebooks/12_lm_weighted_vote.ipynb`
and `13_lm_hypothesis_confirmation.ipynb` -- `final.pt` is deliberately
not considered here, since it can be a stale, earlier/shorter run left
over from a previous training session, not necessarily the most-trained
checkpoint).

Run via the GPU venv (this needs torch + cuda, same as everything else
in this project):

    ".../ai-gpu/Scripts/python.exe" app.py

Then open http://127.0.0.1:5000 in a browser.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request

LM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LM_DIR / "scripts"))
from model import DecoderLM  # noqa: E402

MAX_WORDS = 100  # keeps <bos> + words safely under the LM's max_seq_len (128)
DEFAULT_TOP_K = 20
MAX_TOP_K = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = json.loads((LM_DIR / "data" / "vocab.json").read_text(encoding="utf-8"))
word_to_id: dict[str, int] = vocab["word_to_id"]
id_to_word: dict[int, str] = {i: w for w, i in word_to_id.items()}
pad_id, unk_id, bos_id, eos_id = (word_to_id[s] for s in vocab["specials"])

checkpoints = sorted(
    (LM_DIR / "models").glob("checkpoint_step*.pt"),
    key=lambda p: int(p.stem.split("step")[-1]),
)
if not checkpoints:
    raise FileNotFoundError(f"No checkpoint_step*.pt found in {LM_DIR / 'models'}")
CHECKPOINT = checkpoints[-1]
ckpt = torch.load(CHECKPOINT, map_location=device)
ckpt_args = ckpt["args"]
print(f"Using checkpoint: {CHECKPOINT.name} (step={ckpt['step']:,})")

lm = DecoderLM(
    vocab_size=vocab["vocab_size"],
    d_model=ckpt_args["d_model"],
    num_heads=ckpt_args["num_heads"],
    num_layers=ckpt_args["num_layers"],
    max_seq_len=ckpt_args["seq_len"],
    dropout=0.0,
    pad_id=pad_id,
).to(device)
lm.load_state_dict(ckpt["model_state_dict"])
lm.eval()
for p in lm.parameters():
    p.requires_grad = False

PADDED_VOCAB_SIZE = lm.padded_vocab_size

# Positions the LM's output head can technically produce a logit for but
# that are never real next-word predictions: <pad> itself, and the dummy
# slots padded_vocab_size adds beyond the real vocab (round-up to a
# multiple of 64) that have no word mapped to them at all.
_INVALID_IDS = [pad_id] + list(range(vocab["vocab_size"], PADDED_VOCAB_SIZE))

app = Flask(__name__)


@torch.no_grad()
def predict_next_token(text: str, top_k: int) -> dict:
    words = text.split()
    truncated = len(words) > MAX_WORDS
    words = words[:MAX_WORDS]
    oov_words = sorted({w for w in words if w not in word_to_id})

    ids = [bos_id] + [word_to_id.get(w, unk_id) for w in words]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = lm(input_ids)  # (1, seq_len, padded_vocab_size)
    next_logits = logits[0, -1].clone()  # prediction for the position right after the input
    next_logits[_INVALID_IDS] = float("-inf")
    probs = F.softmax(next_logits, dim=-1)

    top_probs, top_ids = torch.topk(probs, top_k)
    predictions = [
        {"word": id_to_word[int(i)], "prob": float(p)}
        for p, i in zip(top_probs.tolist(), top_ids.tolist())
    ]
    return {
        "input_words": words,
        "truncated": truncated,
        "oov_words": oov_words,
        "predictions": predictions,
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        checkpoint_name=CHECKPOINT.name,
        checkpoint_step=f"{ckpt['step']:,}",
        vocab_size=f"{vocab['vocab_size']:,}",
        default_top_k=DEFAULT_TOP_K,
        max_top_k=MAX_TOP_K,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400

    top_k = body.get("top_k", DEFAULT_TOP_K)
    try:
        top_k = max(1, min(int(top_k), MAX_TOP_K))
    except (TypeError, ValueError):
        top_k = DEFAULT_TOP_K

    return jsonify(predict_next_token(text, top_k))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
