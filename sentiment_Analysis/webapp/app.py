#!/usr/bin/env python
"""Local web review tool for Sentiment_Analysis's run-1 labels -- a local counterpart to the
published Claude Artifact review page (same content/design, kept separately: this one persists to
local JSONL files instead of the artifact's db capability, so it works offline and its output feeds
straight back into merge_and_score.py).

Three views, same as the Artifact page:
  - Disagreements: the ../data/disagreements.jsonl items (no 3/5 agent majority) -- pick a final
    label for each, one at a time, with keyboard shortcuts.
  - Browse & correct: all of ../data/labeled_10k.jsonl, filterable by label/source, with a
    flag/correct action on any row.
  - Human eval: a blind rating pass over ../data/human_eval_sample.jsonl (see
    scripts/sample_human_eval.py -- a 250-item stratified sample of the *agreed-upon* labels, to
    check whether they're actually trustworthy, not just self-consistent across agents). The
    agent's label is never sent to the browser until after a rating is submitted for that item --
    `/api/human_eval` strips it out -- so the rating is genuinely blind, not just visually hidden.

Reads and writes the real files directly (atomic write: tmp file + os.replace, same convention as
Embeddings/intrinsic_eval/app.py), so it's safe to stop and restart at any point -- nothing is lost
mid-review.

Run: python app.py, then open http://127.0.0.1:5057
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from sklearn.metrics import cohen_kappa_score, confusion_matrix

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent / "data"
POOL_PATH = DATA_DIR / "unlabeled_10k.jsonl"
LABELED_PATH = DATA_DIR / "labeled_10k.jsonl"
DISAGREEMENTS_PATH = DATA_DIR / "disagreements.jsonl"
REVIEWS_PATH = DATA_DIR / "disagreement_reviews.jsonl"
CORRECTIONS_PATH = DATA_DIR / "corrections.jsonl"
HUMAN_EVAL_SAMPLE_PATH = DATA_DIR / "human_eval_sample.jsonl"
HUMAN_EVAL_RATINGS_PATH = DATA_DIR / "human_eval_ratings.jsonl"

LABELS = ["POS", "NEG", "NEU", "MIX", "UNSURE"]

app = Flask(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def _load_keyed(path: Path) -> dict[str, dict]:
    """Reviews/corrections files: one JSON object per line, each with an "id" field -- keyed by id,
    last line for a given id wins (matches how a rewrite-the-whole-file save behaves)."""
    return {row["id"]: row for row in _load_jsonl(path)}


def _save_keyed(path: Path, keyed: dict[str, dict]) -> None:
    _atomic_write_jsonl(path, list(keyed.values()))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    pool_by_id = {r["id"]: r for r in _load_jsonl(POOL_PATH)}
    labeled = []
    for r in _load_jsonl(LABELED_PATH):
        row = {"id": r["id"], "text": r["text"], "label": r["label"], "source": r["source"]}
        row["sample_group"] = pool_by_id.get(r["id"], {}).get("sample_group", "?")
        if "votes" in r:
            row["votes"] = r["votes"]
        labeled.append(row)

    disagreements = []
    for r in _load_jsonl(DISAGREEMENTS_PATH):
        disagreements.append({
            "id": r["id"], "text": r["text"], "votes": r["votes"],
            "sample_group": pool_by_id.get(r["id"], {}).get("sample_group", "?"),
        })

    reviews = {k: {"manual_label": v["manual_label"]} for k, v in _load_keyed(REVIEWS_PATH).items()}
    corrections = {
        k: {"corrected_label": v["corrected_label"], "original_label": v.get("original_label")}
        for k, v in _load_keyed(CORRECTIONS_PATH).items()
    }
    return jsonify({"labeled": labeled, "disagreements": disagreements, "reviews": reviews, "corrections": corrections})


@app.route("/api/review", methods=["POST"])
def api_review():
    body = request.get_json(force=True)
    doc_id, label = body.get("id"), body.get("label")
    if not doc_id or label not in LABELS:
        return jsonify({"error": "id and a valid label are required"}), 400
    keyed = _load_keyed(REVIEWS_PATH)
    keyed[doc_id] = {"id": doc_id, "manual_label": label}
    _save_keyed(REVIEWS_PATH, keyed)
    return jsonify({"ok": True})


@app.route("/api/correction", methods=["POST"])
def api_correction():
    body = request.get_json(force=True)
    doc_id = body.get("id")
    if not doc_id:
        return jsonify({"error": "id is required"}), 400
    keyed = _load_keyed(CORRECTIONS_PATH)
    label = body.get("label")
    if label is None:
        keyed.pop(doc_id, None)  # clearing a correction
    else:
        if label not in LABELS:
            return jsonify({"error": "invalid label"}), 400
        keyed[doc_id] = {"id": doc_id, "corrected_label": label, "original_label": body.get("original_label")}
    _save_keyed(CORRECTIONS_PATH, keyed)
    return jsonify({"ok": True})


@app.route("/api/human_eval")
def api_human_eval():
    """Items without the agent label -- the rating must be blind. Also returns which ids this
    reviewer has already rated (label hidden there too; the frontend only shows its own past
    rating, not the agent's)."""
    sample = _load_jsonl(HUMAN_EVAL_SAMPLE_PATH)
    ratings = _load_keyed(HUMAN_EVAL_RATINGS_PATH)
    items = [{"id": r["id"], "text": r["text"], "sample_group": r["sample_group"]} for r in sample]
    my_ratings = {k: {"human_label": v["human_label"]} for k, v in ratings.items()}
    return jsonify({"items": items, "ratings": my_ratings, "total": len(sample)})


@app.route("/api/human_eval/rate", methods=["POST"])
def api_human_eval_rate():
    body = request.get_json(force=True)
    doc_id, label = body.get("id"), body.get("label")
    if not doc_id or label not in LABELS:
        return jsonify({"error": "id and a valid label are required"}), 400
    sample_by_id = {r["id"]: r for r in _load_jsonl(HUMAN_EVAL_SAMPLE_PATH)}
    if doc_id not in sample_by_id:
        return jsonify({"error": "unknown id"}), 404
    keyed = _load_keyed(HUMAN_EVAL_RATINGS_PATH)
    keyed[doc_id] = {"id": doc_id, "human_label": label}
    _save_keyed(HUMAN_EVAL_RATINGS_PATH, keyed)
    # reveal the agent's label now that the rating is locked in, so the UI can show agree/disagree
    return jsonify({"ok": True, "agent_label": sample_by_id[doc_id]["label"]})


@app.route("/api/human_eval/results")
def api_human_eval_results():
    sample = {r["id"]: r for r in _load_jsonl(HUMAN_EVAL_SAMPLE_PATH)}
    ratings = _load_keyed(HUMAN_EVAL_RATINGS_PATH)
    rated_ids = [i for i in ratings if i in sample]
    if not rated_ids:
        return jsonify({"n": 0})

    human = [ratings[i]["human_label"] for i in rated_ids]
    agent = [sample[i]["label"] for i in rated_ids]

    agree = sum(h == a for h, a in zip(human, agent))
    kappa = cohen_kappa_score(human, agent, labels=LABELS)
    cm = confusion_matrix(human, agent, labels=LABELS).tolist()

    # per-class recall from the agent's side: of the items the HUMAN calls class L, what share did
    # the agent also call L -- this is the number that says whether a given class's labels in the
    # full dataset can be trusted, not just the overall accuracy.
    per_class = {}
    for lab in LABELS:
        human_said_lab = [i for i in rated_ids if ratings[i]["human_label"] == lab]
        if human_said_lab:
            agent_agrees = sum(sample[i]["label"] == lab for i in human_said_lab)
            per_class[lab] = {"n": len(human_said_lab), "agent_agreement_rate": agent_agrees / len(human_said_lab)}
        else:
            per_class[lab] = {"n": 0, "agent_agreement_rate": None}

    return jsonify({
        "n": len(rated_ids), "total": len(sample), "accuracy": agree / len(rated_ids), "kappa": kappa,
        "confusion_matrix": cm, "labels": LABELS, "per_class": per_class,
        "human_label_counts": dict(Counter(human)), "agent_label_counts": dict(Counter(agent)),
    })


if __name__ == "__main__":
    print(f"Reading from {DATA_DIR}")
    app.run(host="127.0.0.1", port=5057, debug=True)
