#!/usr/bin/env python
"""Renders the dialect-ID method-comparison figure for the
DarijaDZ-DialectID dataset card.

Analogue of FineWeb2's "Comparison of Multilingual Datasets" curve, but
for this dataset: instead of downstream score vs. training tokens across
datasets, it plots held-out test accuracy across the twelve dialect-ID
methods benchmarked on THIS dataset's `data/test.jsonl` split, ordered
weakest -> strongest.

Numbers are copied verbatim from plan.md's "All twelve methods, side by
side" table (search that heading) -- that table is the source of truth;
this script only draws it. If plan.md's table changes, update RESULTS
below to match.

Two methods (word-cluster baseline, flat fastText-style) report a single
"cluster-scored" number spanning both script groups rather than one per
group -- they're drawn as unconnected grey markers, not points on either
curve, and called out in the legend.

Run via the base Python environment (matplotlib only, no GPU):

    python plot_method_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "DarijaDZ_DialectID"
    / "method_comparison.png"
)

# (short label, arabic-group acc, latin-group acc). `None` where the method
# only reports one combined score -- given instead as `combined` below.
# Order = weakest -> strongest by mean, matching plan.md's table.
RESULTS = [
    ("regex + fastText lid.176",          0.340, 0.630, None),
    ("word-cluster baseline",             None,  None,  0.562),
    ("flat fastText (no gating)",         None,  None,  0.582),
    ("regex / marker-word list",          0.699, 0.748, None),
    ("cluster + 200 human prototypes",    0.701, 0.810, None),
    ("Cavnar & Trenkle n-gram profiles",  0.754, 0.844, None),
    ("KNN on sentence embeddings",        0.770, 0.835, None),
    ("LM-weighted vote (LM_DiD step 2k)", 0.776, 0.792, None),
    ("cluster + majority vote (best k)",  0.796, 0.846, None),
    ("HMM sequential (sticky trans.)",    0.808, 0.817, None),
    ("classifier head (transformer)",     0.819, 0.863, None),
    ("char n-gram + SVM-RBF",             0.833, 0.881, None),
]

ARABIC_C = "#c02a2a"   # msa vs darija
LATIN_C = "#2a6fc0"    # arabize / french / english
MEAN_C = "#111111"     # headline "average" curve (the FineWeb2 black line)
COMBINED_C = "#8a8a8a"  # methods with one score spanning both groups


def main() -> None:
    x = np.arange(len(RESULTS))
    labels = [r[0] for r in RESULTS]

    ar = np.array([r[1] if r[1] is not None else np.nan for r in RESULTS])
    la = np.array([r[2] if r[2] is not None else np.nan for r in RESULTS])
    comb_x = [i for i, r in enumerate(RESULTS) if r[3] is not None]
    comb_y = [r[3] for r in RESULTS if r[3] is not None]

    # Headline "average" curve, kept continuous: the per-group mean where
    # both groups are reported, the lone combined score where they aren't.
    mean = np.array([
        r[3] if r[3] is not None else (r[1] + r[2]) / 2
        for r in RESULTS
    ])

    fig, ax = plt.subplots(figsize=(13.5, 6.8))

    ax.plot(x, mean, color=MEAN_C, lw=2.6, zorder=5,
            label="Mean of both groups")
    ax.plot(x, ar, color=ARABIC_C, lw=1.8, marker="o", ms=6, zorder=4,
            label="Arabic-script group  (msa vs darija)")
    ax.plot(x, la, color=LATIN_C, lw=1.8, marker="s", ms=6, zorder=4,
            label="Latin-script group  (arabize / french / english)")
    ax.scatter(comb_x, comb_y, color=COMBINED_C, marker="D", s=55, zorder=4,
               label="Combined score only (not split by group)")

    # Filled endpoint dots, FineWeb2 style.
    ax.scatter([x[-1]], [ar[-1]], color=ARABIC_C, s=90, zorder=6)
    ax.scatter([x[-1]], [la[-1]], color=LATIN_C, s=90, zorder=6)
    ax.scatter([x[-1]], [mean[-1]], color=MEAN_C, s=90, zorder=6)

    # Call out the shipped classifier (last method).
    ax.axvline(x[-1], color="#000000", lw=0.8, ls=":", alpha=0.35, zorder=1)
    ax.annotate(
        "shipped as\nDarijaDZ-DialectID-Classifier",
        xy=(x[-1], la[-1]), xytext=(x[-1] - 3.2, 0.915),
        fontsize=9.5, ha="left", va="center", color="#333333",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1),
    )

    ax.set_title(
        "Comparison of Dialect-ID Methods on DarijaDZ-DialectID\n"
        "(held-out test accuracy, 4,001 rows, methods ordered weakest → strongest)",
        fontsize=13, pad=14,
    )
    ax.set_ylabel("Held-out test accuracy", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=25, ha="right")
    ax.set_ylim(0.30, 0.94)
    ax.set_xlim(-0.4, len(RESULTS) - 0.4)
    ax.grid(True, ls="--", color="#bbbbbb", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
