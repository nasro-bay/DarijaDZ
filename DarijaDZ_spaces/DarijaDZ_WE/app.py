"""Gradio Space for the DarijaDZ GloVe 128-d word embeddings: nearest neighbours, analogies, word
similarity and a 2D map of the most frequent words.

Everything is read from files next to this script (model.safetensors, vocab.json, analogy_pairs.jsonl,
map2d.npz -- rebuild the last with build_assets.py), so the Space has no other dependency on the Hub.
"""
from __future__ import annotations

import base64
import json
import re
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from safetensors.numpy import load_file

try:
    import spaces  # preinstalled on Hugging Face; ZeroGPU Spaces need at least one @spaces.GPU function
    gpu = spaces.GPU
except ImportError:  # running locally
    def gpu(fn):
        return fn

HERE = Path(__file__).parent

E = load_file(HERE / "model.safetensors")["embeddings"].astype("float32")
VOCAB: list[str] = json.loads((HERE / "vocab.json").read_text(encoding="utf-8"))
INDEX = {w: i for i, w in enumerate(VOCAB)}
NORMED = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)

_map = np.load(HERE / "map2d.npz")
MAP_COORDS, MAP_N = _map["coords"], int(_map["n"])

ARABIC_RE = re.compile("[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
LATIN_RE = re.compile("[a-zA-ZÀ-ɏ]")
LOGO_B64 = base64.b64encode((HERE / "logo.png").read_bytes()).decode()


def iso(word: str) -> str:
    """Bidi-isolate a word so Arabic words keep their order inside an English/LTR line."""
    return "⁨" + word + "⁩"


def ltr(text: str) -> str:
    return "⁦" + text + "⁩"


def norm_word(word: str) -> str:
    return (word or "").strip().lower()


def script_of(word: str) -> str:
    has_ar, has_lat = bool(ARABIC_RE.search(word)), bool(LATIN_RE.search(word))
    if has_ar and has_lat:
        return "Mixed"
    return "Arabic script" if has_ar else "Latin script" if has_lat else "Other"


def lookup(word: str):
    """(index, message) -- message is set (and index None) when the word has no vector."""
    w = norm_word(word)
    if not w:
        return None, "Type a word first."
    if w not in INDEX:
        return None, f"«{w}» is not in the 20,000-word vocabulary."
    return INDEX[w], ""


def neighbours(idx: int, k: int, exclude: set[int] = frozenset()) -> list[tuple[int, float]]:
    sims = NORMED @ NORMED[idx]
    order = [int(i) for i in np.argsort(-sims) if int(i) not in exclude and int(i) != idx]
    return [(i, float(sims[i])) for i in order[:k]]


# ---- tab 1: nearest neighbours -----------------------------------------------------------------
@gpu
def nearest(word: str, k: int):
    idx, msg = lookup(word)
    if idx is None:
        return gr.update(value=[]), msg
    rows = [[VOCAB[i], round(s, 3)] for i, s in neighbours(idx, int(k))]
    return rows, f"Top {len(rows)} neighbours of «{VOCAB[idx]}» (cosine similarity)."


# ---- tab 2: analogies --------------------------------------------------------------------------
def analogy(a: str, b: str, c: str, k: int = 5):
    ids = []
    for w in (a, b, c):
        i, msg = lookup(w)
        if i is None:
            return [], msg
        ids.append(i)
    target = NORMED[ids[1]] - NORMED[ids[0]] + NORMED[ids[2]]
    sims = NORMED @ (target / np.linalg.norm(target))
    order = [int(i) for i in np.argsort(-sims) if int(i) not in ids][: int(k)]
    rows = [[VOCAB[i], round(float(sims[i]), 3)] for i in order]
    a_, b_, c_ = (f"<bdi>{VOCAB[i]}</bdi>" for i in ids)  # <bdi> + dir=ltr keep Arabic words in reading order
    return rows, (f'<div dir="ltr"><b>{a_} : {b_} :: {c_} : ?</b></div>\n\n'
                  f'<div dir="ltr"><code>target = vec({b_}) − vec({a_}) + vec({c_})</code>, then every word is '
                  f'ranked by its cosine similarity to <code>target</code> ({a_}, {b_} and {c_} are excluded).</div>')


# Question types follow the standard analogy benchmarks (Google/Mikolov: family, plural, nationality,
# opposite, verb tense, city-country; Grave et al.'s French set: gender/articles) plus the Darija-specific
# Arabic-script -> Arabizi type from analogy_pairs.jsonl. Each triple was checked to give the intended
# answer as the top-1 result; build_analogy_groups() re-checks them at startup and drops any that don't.
ANALOGY_CANDIDATES = {
    "Gender": [("ولد", "بنت", "رجل", "امرأة"),
               ("ولد", "بنت", "كبير", "كبيرة"),
               ("ولد", "بنت", "جار", "جارة"),
               ("خويا", "ختي", "ولد", "بنت")],
    "Singular → plural": [("ولد", "ولاد", "بنت", "بنات"),
                            ("ولد", "ولاد", "سيارة", "سيارات"),
                            ("يوم", "أيام", "سنة", "سنوات")],
    "Nationality": [("الجزائر", "جزائري", "المغرب", "مغربي"),
                    ("الجزائر", "جزائري", "تونس", "تونسي"),
                    ("الجزائر", "جزائري", "فرنسا", "فرنسي")],
    "City → country": [("وهران", "الجزائر", "باريس", "فرنسا")],
    "Opposites": [("كبير", "صغير", "طويل", "قصير")],
    "Verb form": [("كتب", "يكتب", "قرا", "يقرا")],
    "Numbers": [("واحد", "اثنين", "ثلاثة", "أربعة")],
    "Arabic script → Arabizi": [("واش", "wach", "راني", "rani"),
                                    ("صح", "sah", "غير", "ghir"),
                                    ("واحد", "wahed", "والو", "walou")],
    "French": [("le", "la", "un", "une")],
}


def build_analogy_groups() -> dict[str, list[list[str]]]:
    groups = {}
    for name, questions in ANALOGY_CANDIDATES.items():
        ok = []
        for a, b, c, d in questions:
            rows, _ = analogy(a, b, c, k=1)
            if rows and rows[0][0] == d:
                ok.append([a, b, c])
        if ok:
            groups[name] = ok
    return groups


# ---- tab 3: similarity -------------------------------------------------------------------------
def similarity(a: str, b: str):
    ia, ma = lookup(a)
    ib, mb = lookup(b)
    if ia is None or ib is None:
        return ma or mb
    return (f'<div dir="ltr"><b><bdi>{VOCAB[ia]}</bdi></b> ↔ <b><bdi>{VOCAB[ib]}</bdi></b></div>\n\n'
            f"cosine similarity: **{float(NORMED[ia] @ NORMED[ib]):.3f}**")


# ---- tab 4: 2D map ----------------------------------------------------------------------------
SCRIPT_COLORS = {"Arabic script": "#04663a", "Latin script": "#c8102e", "Mixed": "#b8922f", "Other": "#74827b"}
VIEW_W, VIEW_H, CHAR_PX, LABEL_H = 930, 780, 8, 19
PLACEHOLDER_WORDS = {"mention", "url"}  # the [MENTION]/[URL] anonymisation tokens, not real words


@lru_cache(maxsize=16)
def label_layout(n_words: int) -> np.ndarray:
    """Positions for the first n_words words such that no two labels overlap. Points are scaled onto a
    canvas (the view size, growing with n_words), then each word in frequency order takes the free spot
    closest to its t-SNE position (spiralling outwards until its label box fits)."""
    scale = max(1.0, (n_words / 200) ** 0.5)
    W, H = VIEW_W * scale, VIEW_H * scale
    raw = MAP_COORDS[:n_words].astype("float64")
    lo, hi = np.percentile(raw, 2, axis=0), np.percentile(raw, 98, axis=0)
    home = (np.clip(raw, lo, hi) - lo) / (hi - lo) * np.array([W * 0.94, H * 0.94]) + np.array([W * 0.03, H * 0.03])
    hw = (np.array([len(VOCAB[i]) for i in range(n_words)]) * CHAR_PX + 8) / 2
    hh = LABEL_H / 2
    placed = np.zeros((n_words, 2))
    step, angles = 6.0, np.linspace(0, 2 * np.pi, 16, endpoint=False)
    for i in range(n_words):
        cand = home[i]
        r = 0.0
        while True:
            if i == 0:
                break
            dx = np.abs(placed[:i, 0] - cand[0])
            dy = np.abs(placed[:i, 1] - cand[1])
            if not ((dx < hw[:i] + hw[i]) & (dy < 2 * hh)).any():
                break
            r += step
            k = int(r / step) % len(angles)
            ang = angles[k] + r * 0.37
            cand = home[i] + r * np.array([np.cos(ang) * 1.6, np.sin(ang)])  # wider than tall, like the labels
            if r > 600:
                break
        placed[i] = cand
    return placed


def build_map(n_words: int, highlight: str):
    n_words = int(min(max(n_words, 100), MAP_N))
    coords = label_layout(n_words)
    fig = go.Figure()
    w = norm_word(highlight)
    near: list[int] = []
    if w in INDEX and INDEX[w] < n_words:
        near = [INDEX[w]] + [i for i, _ in neighbours(INDEX[w], 25) if i < n_words][:10]
    groups: dict[str, list[int]] = {}
    for i in range(n_words):
        if VOCAB[i] in PLACEHOLDER_WORDS or i in near:
            continue
        groups.setdefault(script_of(VOCAB[i]), []).append(i)
    for name, idxs in groups.items():
        c = coords[idxs]
        fig.add_trace(go.Scatter(
            x=c[:, 0], y=c[:, 1], mode="text", name=name, text=[VOCAB[i] for i in idxs],
            textposition="middle center", textfont=dict(size=13, color=SCRIPT_COLORS[name]),
            hovertemplate="%{text}<extra>" + name + "</extra>",
        ))
    note = (f"The {n_words:,} most frequent words, placed by a 2D t-SNE of their 128-d vectors: words used in similar "
            "contexts sit close together. Positions are nudged slightly so labels don't overlap. Drag to pan, "
            "scroll or box-select to zoom, double-click to reset.")
    if w:
        if not near:
            note += f"  \u00ab{w}\u00bb is not among the plotted words."
        else:
            c = coords[near]
            fig.add_trace(go.Scatter(
                x=c[:, 0], y=c[:, 1], mode="markers+text", name=f"\u00ab{w}\u00bb + neighbours",
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                text=[f"<b>{VOCAB[i]}</b>" for i in near], textposition="middle center",
                textfont=dict(size=15, color="#000000"), hovertemplate="%{text}<extra></extra>",
            ))
    axis = dict(showgrid=False, zeroline=False, showticklabels=False, title="", range=None)
    fig.update_layout(
        xaxis=axis, yaxis=dict(axis, scaleanchor="x"), height=VIEW_H, margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.03), dragmode="pan",
    )
    return fig, note


# ---- layout ------------------------------------------------------------------------------------
# Same Algeria-flag palette + Cairo font + topbar/flag-strip identity as the other DarijaDZ Spaces.
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700&display=swap');
:root { --dz-green:#04663a; --dz-red:#c8102e; --dz-gold:#b8922f; --dz-cream:#f7f4ee; --ink:#1e2723; --muted:#74827b; --border:#e2e0d8; }
.gradio-container { font-family:"Cairo","Segoe UI",Arial,sans-serif !important; background:var(--dz-cream) !important; max-width:1000px !important; }
#dz-topbar { display:flex; align-items:center; gap:14px; padding:18px 4px 6px; }
#dz-topbar img { width:48px; height:48px; border-radius:50%; border:1px solid var(--border); object-fit:cover; flex-shrink:0; }
#dz-topbar h1 { margin:0; font-size:1.25rem; font-weight:700; color:var(--ink); }
#dz-topbar p { margin:2px 0 0; font-size:0.8rem; color:var(--muted); }
#dz-flag-strip { height:4px; display:flex; margin:0 0 18px; border-radius:2px; overflow:hidden; }
#dz-flag-strip span { flex:1; display:block; }
#dz-flag-strip span:nth-child(1) { background:var(--dz-green); }
#dz-flag-strip span:nth-child(2) { background:#fff; }
#dz-flag-strip span:nth-child(3) { background:var(--dz-red); }
html, body, gradio-app, .gradio-container { background:var(--dz-cream) !important; }
.gradio-container { margin-left:auto !important; margin-right:auto !important; }
.dz-desc { font-size:0.85rem; color:var(--muted); }
.dz-chips { flex-wrap:wrap !important; gap:6px !important; align-items:center; }
.dz-chips > * { flex:0 0 auto !important; min-width:0 !important; width:auto !important; }
.dz-chip-title { font-size:0.8rem; color:var(--muted); min-width:170px !important; }
.dz-chip-title p { margin:0 !important; }
button.dz-chip { background:#fff !important; border:1px solid var(--border) !important; color:var(--ink) !important; padding:2px 10px !important; height:30px !important; font-size:0.9rem !important; }
.cell-wrap, .cell-wrap * { font-family:"Cairo","Segoe UI",Arial,sans-serif !important; font-size:1.05rem !important; }
button.dz-chip:hover { border-color:var(--dz-green) !important; color:var(--dz-green) !important; }
button.primary { background:var(--dz-green) !important; border:none !important; }
button.primary:hover { background:#054d2c !important; }
"""

TOPBAR_HTML = f"""
<div id="dz-topbar">
  <img src="data:image/png;base64,{LOGO_B64}" alt="DarijaDZ">
  <div><h1>DarijaDZ Word Embeddings &mdash; GloVe 128-d</h1>
  <p>Part of DarijaDZ &mdash; an NLP ecosystem for Algerian Darija</p></div>
</div>
<div id="dz-flag-strip"><span></span><span></span><span></span></div>
"""

# Themes read off the t-SNE map of the frequent words (clusters visible there: cities, animals, food,
# question words, institutions, Arabizi, French...). Every seed word was checked to have sensible neighbours.
NN_THEMES = {
    "Intensity and particles": ["\u0628\u0632\u0627\u0641", "\u0628\u0635\u062d"],
    "Question words": ["\u0648\u0627\u0634", "\u0643\u064a\u0641\u0627\u0634"],
    "Cities and places": ["\u0648\u0647\u0631\u0627\u0646", "\u0645\u0637\u0627\u0631"],
    "Animals": ["\u0643\u0644\u0628"],
    "Food and drink": ["\u0628\u0631\u062a\u0642\u0627\u0644", "\u0628\u0637\u0627\u0637\u0627", "\u0642\u0647\u0648\u0629"],
    "Time": ["\u0633\u0646\u0629"],
    "Family and feelings": ["\u0645\u0627\u0645\u0627", "\u062e\u0648\u064a\u0627", "\u062d\u0632\u064a\u0646"],
    "State and institutions": ["\u0648\u0632\u0627\u0631\u0629", "\u0634\u0631\u0637\u0629", "\u0628\u0646\u0643", "\u062c\u064a\u0634"],
    "Education, health, sport": ["\u0645\u062f\u0631\u0633\u0629", "\u0637\u0628\u064a\u0628", "\u0643\u0631\u0629"],
    "Religion": ["\u0631\u0628\u064a"],
    "Cars and technology": ["\u0633\u064a\u0627\u0631\u0629", "\u062a\u0644\u064a\u0641\u0648\u0646"],
    "Arabizi": ["wach", "khouya", "mabrouk"],
    "French": ["monsieur", "bonjour", "merci", "match"],
}
SIM_THEMES = {
    "Close in meaning": [["\u0644\u064a\u0645\u0648\u0646", "\u0628\u0631\u062a\u0642\u0627\u0644"],
                         ["\u0648\u0647\u0631\u0627\u0646", "\u0642\u0633\u0646\u0637\u064a\u0646\u0629"],
                         ["\u0627\u0644\u0644\u0647", "\u0631\u0628\u064a"], ["\u0643\u0631\u0629", "\u0627\u0644\u0642\u062f\u0645"],
                         ["\u0645\u0627\u0645\u0627", "\u0628\u0627\u0628\u0627"], ["\u0648\u0627\u0634", "\u0643\u064a\u0641\u0627\u0634"]],
    "Related, less close": [["\u062c\u064a\u0634", "\u0634\u0631\u0637\u0629"], ["\u0645\u062f\u0631\u0633\u0629", "\u062c\u0627\u0645\u0639\u0629"],
                            ["monsieur", "ministre"], ["bonjour", "bonsoir"], ["\u062d\u0632\u064a\u0646", "\u0641\u0631\u062d\u0627\u0646"]],
    "Unrelated (expect ~0)": [["\u0643\u0644\u0628", "\u0648\u0632\u0627\u0631\u0629"], ["\u0642\u0647\u0648\u0629", "\u0634\u0631\u0637\u0629"],
                              ["\u0648\u0647\u0631\u0627\u0646", "\u0628\u0631\u062a\u0642\u0627\u0644"]],
    "Same word, different script (expect low)": [["wach", "\u0648\u0627\u0634"], ["mabrouk", "\u0645\u0628\u0631\u0648\u0643"]],
}
ANALOGY_GROUPS = build_analogy_groups()


def chip_rows(themes: dict[str, list[tuple[str, list[str]]]], fill: list, run, run_inputs: list, outputs: list):
    """One compact row per theme; clicking a chip fills the `fill` components with its values, then runs `run`."""
    for theme, items in themes.items():
        with gr.Row(elem_classes="dz-chips"):
            gr.Markdown(f"**{theme}**", elem_classes="dz-chip-title")
            for label, values in items:
                chip = gr.Button(label, size="sm", elem_classes="dz-chip", min_width=0)
                chip.click(lambda v=tuple(values): v if len(v) > 1 else v[0], None, fill, show_progress="hidden").then(run, run_inputs, outputs)


with gr.Blocks(title="DarijaDZ Word Embeddings") as demo:
    gr.HTML(TOPBAR_HTML)
    gr.Markdown(
        "GloVe word vectors (128 dimensions, 20,000-word vocabulary) trained on the DarijaDZ corpus. "
        "Lookups are lowercased. Arabic-script and Latin (Arabizi) spellings of the same word are separate "
        "entries, so their neighbours differ. Click any example to run it.",
        elem_classes="dz-desc",
    )
    with gr.Tabs():
        with gr.Tab("Nearest neighbours"):
            with gr.Row():
                nn_word = gr.Textbox(label="Word", placeholder="\u0628\u0632\u0627\u0641", scale=3)
                nn_k = gr.Slider(5, 30, value=10, step=1, label="Neighbours", scale=2)
            nn_btn = gr.Button("Find neighbours", variant="primary")
            nn_msg = gr.Markdown()
            nn_out = gr.Dataframe(headers=["word", "cosine"], interactive=False)
            nn_run_inputs, nn_run_outputs = [nn_word, nn_k], [nn_out, nn_msg]
            for trigger in (nn_btn.click, nn_word.submit):
                trigger(nearest, nn_run_inputs, nn_run_outputs)
            chip_rows({t: [(w, [w]) for w in ws if w in INDEX] for t, ws in NN_THEMES.items()},
                      [nn_word], nearest, nn_run_inputs, nn_run_outputs)

        with gr.Tab("Analogies"):
            gr.Markdown(
                "**A is to B as C is to ?** The answer is computed by vector arithmetic: "
                "`target = vec(B) \u2212 vec(A) + vec(C)`. `B \u2212 A` captures the relation between A and B "
                "(boy \u2192 girl); adding it to C applies the same shift. Every vocabulary word is then ranked by "
                "cosine similarity to `target`, leaving out A, B and C. It works for regular relations (gender, "
                "plural, nationality) and often fails on others, so the examples below are ones this model gets right.",
                elem_classes="dz-desc",
            )
            with gr.Row():
                an_a = gr.Textbox(label="A")
                an_b = gr.Textbox(label="B")
                an_c = gr.Textbox(label="C")
            an_btn = gr.Button("Solve", variant="primary")
            an_msg = gr.Markdown()
            an_out = gr.Dataframe(headers=["answer", "cosine to target"], interactive=False)
            an_inputs, an_outputs = [an_a, an_b, an_c], [an_out, an_msg]
            an_btn.click(analogy, an_inputs, an_outputs)
            chip_rows({t: [(ltr(f"{iso(a)} : {iso(b)} :: {iso(c)}"), [a, b, c]) for a, b, c in qs] for t, qs in ANALOGY_GROUPS.items()},
                      an_inputs, analogy, an_inputs, an_outputs)

        with gr.Tab("Word similarity"):
            with gr.Row():
                sim_a = gr.Textbox(label="Word 1")
                sim_b = gr.Textbox(label="Word 2")
            sim_btn = gr.Button("Compare", variant="primary")
            sim_out = gr.Markdown()
            sim_btn.click(similarity, [sim_a, sim_b], sim_out)
            chip_rows({t: [(f"{a} \u2013 {b}", [a, b]) for a, b in ps] for t, ps in SIM_THEMES.items()},
                      [sim_a, sim_b], similarity, [sim_a, sim_b], [sim_out])

        with gr.Tab("Explore map (2D)"):
            with gr.Row():
                map_n = gr.Slider(100, MAP_N, value=200, step=100, label="Most frequent words to show")
                map_hl = gr.Textbox(label="Highlight a word and its neighbours", placeholder="\u0628\u0632\u0627\u0641")
            map_btn = gr.Button("Update map", variant="primary")
            map_note = gr.Markdown(elem_classes="dz-desc")
            map_plot = gr.Plot()
            map_btn.click(build_map, [map_n, map_hl], [map_plot, map_note])
            map_hl.submit(build_map, [map_n, map_hl], [map_plot, map_note])
            demo.load(build_map, [map_n, map_hl], [map_plot, map_note])

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)
