"""Layer-wise linear probe of the biLM on NArabizi POS (UPOS), the paper's "which layer encodes what"
experiment. The biLM is frozen; only a linear softmax layer per feature set is trained, so accuracy
differences measure what the representation already contains. Also reports the learned scalar mix
(ELMo eq. 1) and sample efficiency (train on 10% / 25% / 100% of the sentences).

    python probe_pos.py --checkpoint ../models/elmo_M_best.pt [--out ../models/probe_pos.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from elmo import ElmoEncoder, ScalarMix

ROOT = Path(__file__).resolve().parents[3]
POS_DIR = ROOT / "Benchmarks" / "NArabizi-main" / "data" / "Narabizi" / "pos"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


MAX_LEN = 64   # must match ElmoEncoder.encode()'s default -- sentences longer than this get silently
               # truncated by the biLM encoder, so truncate the tags identically here or features/labels desync


def read_conllu(path: Path):
    """This corpus's CoNLL-U columns are ID, FORM, GLOSS_FR, ARABIC_TRANSLIT, ARABIC_FORM, UPOS, XPOS, FEATS,
    HEAD, DEPREL, DEPS, MISC (12 columns, not the standard 10 -- UPOS is cols[5], confirmed empirically: only
    that column has the expected ~19-value UD-tagset cardinality; cols[3]/[4] are free-text transliterations).
    Sentences are truncated to MAX_LEN (words and tags together) to match the encoder's own truncation."""
    sents, words, tags = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if words:
                sents.append((words[:MAX_LEN], tags[:MAX_LEN]))
            words, tags = [], []
        elif not line.startswith("#"):
            cols = line.split("\t")
            if len(cols) < 6:      # the raw file has at least one corrupted non-blank line with no tabs at all
                continue
            if "-" in cols[0] or "." in cols[0]:      # multiword-token range / empty node: keep the syntactic words only
                continue
            words.append(cols[1])
            tags.append(cols[5])
    if words:
        sents.append((words[:MAX_LEN], tags[:MAX_LEN]))
    return sents


def featurize(enc: ElmoEncoder, sents):
    reps = enc.encode([w for w, _ in sents], max_len=MAX_LEN)   # per sentence (3, n, D)
    X = torch.cat([r.transpose(0, 1) for r in reps])            # (N, 3, D)
    return X


def train_probe(Xtr, ytr, Xdv, ydv, Xte, yte, n_classes, mix=False, epochs=150, lr=3e-3, wd=1e-4, device="cuda"):
    """Linear softmax probe; if `mix`, the input is (N, 3, D) and a ScalarMix is learned jointly.
    The epoch with the best dev accuracy is the one reported on test."""
    D = Xtr.shape[-1]
    lin = nn.Linear(D, n_classes).to(device)
    sm = ScalarMix().to(device) if mix else None
    params = list(lin.parameters()) + (list(sm.parameters()) if sm else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    def logits(X):
        if sm is None:
            return lin(X)
        return lin(sm(X.transpose(0, 1).unsqueeze(2)).squeeze(1))        # (3,N,1,D) -> (N,D)

    Xtr, Xdv, Xte = Xtr.to(device), Xdv.to(device), Xte.to(device)
    ytr, ydv, yte = ytr.to(device), ydv.to(device), yte.to(device)
    best_dv, best_te, best_w, best_pred = -1.0, 0.0, None, None
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(perm), 512):
            idx = perm[i:i + 512]
            loss = nn.functional.cross_entropy(logits(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            dv = (logits(Xdv).argmax(-1) == ydv).float().mean().item()
            if dv > best_dv:
                best_dv = dv
                pred = logits(Xte).argmax(-1)
                best_te = (pred == yte).float().mean().item()
                best_pred = pred.cpu().numpy()
                best_w = torch.softmax(sm.w, 0).tolist() if sm else None
    return {"dev": best_dv, "test": best_te, "mix_weights": best_w, "test_pred": best_pred}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    enc = ElmoEncoder(args.checkpoint, args.data_dir)
    splits = {s: read_conllu(POS_DIR / f"{s}_NArabizi.conllu") for s in ("train", "dev", "test")}
    tagset = sorted({t for s in splits.values() for _, ts in s for t in ts})
    tid = {t: i for i, t in enumerate(tagset)}
    y = {s: torch.tensor([tid[t] for _, ts in v for t in ts]) for s, v in splits.items()}
    X = {s: featurize(enc, v) for s, v in splits.items()}
    print({s: len(v) for s, v in y.items()}, "tokens;", len(tagset), "tags", flush=True)

    D = enc.dim // 2
    sets = {                                               # name -> (slice of the (N, 3, 2D) tensor, needs mix)
        "layer0_charcnn": lambda t: t[:, 0, :D],
        "layer1_fwd": lambda t: t[:, 1, :D], "layer1_bwd": lambda t: t[:, 1, D:], "layer1_both": lambda t: t[:, 1],
        "layer2_fwd": lambda t: t[:, 2, :D], "layer2_bwd": lambda t: t[:, 2, D:], "layer2_both": lambda t: t[:, 2],
    }
    results: dict = {}
    for name, f in sets.items():
        r = train_probe(f(X["train"]), y["train"], f(X["dev"]), y["dev"], f(X["test"]), y["test"], len(tagset))
        results[name] = {k: v for k, v in r.items() if k != "test_pred"}
        print(f"{name:16s} dev {r['dev']:.4f}  test {r['test']:.4f}", flush=True)
    r = train_probe(X["train"], y["train"], X["dev"], y["dev"], X["test"], y["test"], len(tagset), mix=True)
    results["scalar_mix"] = {k: v for k, v in r.items() if k != "test_pred"}
    print(f"{'scalar_mix':16s} dev {r['dev']:.4f}  test {r['test']:.4f}  weights {np.round(r['mix_weights'], 3).tolist()}", flush=True)

    n_tr = len(splits["train"])
    sizes = {}
    for frac in (0.1, 0.25, 1.0):
        keep = np.random.default_rng(args.seed).permutation(n_tr)[: max(1, int(n_tr * frac))]
        sel = torch.zeros(len(y["train"]), dtype=torch.bool)
        pos = 0
        for i, (w, _) in enumerate(splits["train"]):
            if i in set(keep.tolist()):
                sel[pos:pos + len(w)] = True
            pos += len(w)
        r = train_probe(X["train"][sel], y["train"][sel], X["dev"], y["dev"], X["test"], y["test"], len(tagset), mix=True)
        sizes[str(frac)] = r["test"]
        print(f"sample efficiency: {int(frac * 100):>3d}% of train sentences -> test {r['test']:.4f}", flush=True)
    results["sample_efficiency_scalar_mix"] = sizes
    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
