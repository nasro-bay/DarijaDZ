#!/usr/bin/env python
"""Pretrains the ELMo biLM (plan.md sections 3 and 5).

Optimization techniques applied (each traced to this project's own work, see plan.md):
FP16 autocast + GradScaler (Turing has no BF16 tensor cores), fused AdamW, warmup + cosine LR, grad
clipping, dims that are multiples of 64, no host syncs in the loop (statistics accumulate on the
device and are read every `--log-every` steps), exact-length batches with zero padding, char-CNN once
per unique word type, memmapped data with a prefetch thread, atomic + disk-guarded
+ resumable checkpoints.

Run via the GPU venv's Python:
    ".../ai-gpu/Scripts/python.exe" -u train.py --config M --epochs 4 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import model as M
from dataset import DATA_DIR, Prefetcher, Split

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def lr_at(step: int, total: int, peak: float, warmup_frac: float, min_frac: float) -> float:
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return peak * (step + 1) / warm
    progress = min(1.0, (step - warm) / max(1, total - warm))
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model: M.BiLM, split: Split, device: torch.device) -> dict:
    """Held-out perplexity per direction, with and without <unk> targets (a 20K vocabulary makes
    <unk> a large class, so both are reported)."""
    model.eval()
    torch.cuda.empty_cache()
    acc = torch.zeros(8, dtype=torch.float64, device=device)   # [sum,n,sum_nounk,n_nounk] x (fwd, bwd)
    for i in range(len(split)):
        ids = torch.from_numpy(split.get(i).astype(np.int64)).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            lf, tf, lb, tb = model(ids, detailed=True)
        for k, (loss, tgt) in enumerate(((lf, tf), (lb, tb))):
            nounk = tgt != M.UNK_OUT
            acc[4 * k + 0] += loss.double().sum()
            acc[4 * k + 1] += loss.numel()
            acc[4 * k + 2] += loss.double()[nounk].sum()
            acc[4 * k + 3] += nounk.sum()
    a = acc.tolist()
    model.train()
    out = {}
    for k, name in enumerate(("fwd", "bwd")):
        out[f"ppl_{name}"] = math.exp(a[4 * k] / a[4 * k + 1])
        out[f"ppl_{name}_nounk"] = math.exp(a[4 * k + 2] / a[4 * k + 3])
    out["ppl_mean"] = (out["ppl_fwd"] + out["ppl_bwd"]) / 2
    out["ppl_mean_nounk"] = (out["ppl_fwd_nounk"] + out["ppl_bwd_nounk"]) / 2
    return out


def save_atomic(obj: dict, path: Path, min_free_gb: float = 3.0) -> bool:
    free = shutil.disk_usage(path.parent).free / 1e9
    if free < min_free_gb:
        print(f"  WARNING: only {free:.1f} GB free on the checkpoint drive -- skipping save to avoid a corrupt file", flush=True)
        return False
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return True


def fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="M", choices=list(M.CONFIGS))
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-positions", type=int, default=8192)
    p.add_argument("--fraction", type=float, default=1.0, help="fixed random subset of batches (LR sweeps)")
    p.add_argument("--max-steps", type=int, default=None, help="stop after N steps (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=3000)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--resume", action="store_true", help="continue from models/<name>_last.pt if present")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False   # char-CNN batch size = #unique types varies per step -> benchmark would re-run every step
    # Varying batch shapes fragment the allocator; uncapped, 'reserved' grew past the 6 GB card (Windows spills to
    # shared memory -> ~2 s steps). A cap forces cache reuse: 32.7K words/s vs a collapse to <3K.
    # Steady-state usage during a full run measured at ~4.40-4.41 GB, so keep clear of that: 0.85 (5.1 GB) is
    # comfortable when nothing else is on the GPU. (0.75/4.5 GB, tried after an Epic-Games-caused crash, turned out
    # to be the wrong fix -- see the empty_cache() call after resume-loading below for the actual cause it exposed.)
    torch.cuda.set_per_process_memory_fraction(0.85)
    MODELS_DIR.mkdir(exist_ok=True)
    name = f"elmo_{args.config}{args.tag}"

    print("loading dataset index...", flush=True)
    train = Split("train", args.data_dir, args.batch_positions)
    heldout = Split("heldout", args.data_dir, args.batch_positions)
    print(f"train: {len(train):,} batches / {train.n_positions:,} positions; held-out: {len(heldout):,} batches", flush=True)

    print(f"building {args.config} model on {torch.cuda.get_device_name(device)}...", flush=True)
    model = M.build(args.config, dropout=args.dropout).to(device)
    model.set_tables(np.load(args.data_dir / "char_table.npy"), np.load(args.data_dir / "type_to_out.npy"))
    pc = model.count_params()
    print(f"{args.config}: {pc['total'] / 1e6:.1f}M params ({pc['non_softmax'] / 1e6:.1f}M non-softmax)", flush=True)

    decay = [q for q in model.parameters() if q.dim() >= 2]
    no_decay = [q for q in model.parameters() if q.dim() < 2]
    groups = [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}]
    try:
        opt = torch.optim.AdamW(groups, lr=args.lr, fused=True)
    except (RuntimeError, ValueError):
        opt = torch.optim.AdamW(groups, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    per_epoch = len(train.order(0, args.seed, args.fraction))
    total_steps = per_epoch * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    step, start_epoch, start_batch, best, history = 0, 0, 0, float("inf"), []
    last_path, best_path = MODELS_DIR / f"{name}_last.pt", MODELS_DIR / f"{name}_best.pt"
    if args.resume and last_path.exists():
        print(f"resuming training: found {last_path.name}, loading checkpoint...", flush=True)
        t_load = time.time()
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, start_epoch, start_batch, best, history = ck["step"], ck["epoch"], ck["batch_in_epoch"], ck["best"], ck["history"]
        del ck
        # load_state_dict deserializes straight onto the GPU (Adam's exp_avg/exp_avg_sq included) and leaves the
        # loading tensors and any stale cached blocks from `ck` behind; without this the very first post-resume
        # step can OOM even on an otherwise-empty GPU (confirmed: crashed on step 1 with 0 other GPU usage).
        torch.cuda.empty_cache()
        print(f"resumed from {last_path.name} in {time.time() - t_load:.1f}s: step {step:,}, epoch {start_epoch}, "
              f"batch {start_batch:,}/{per_epoch:,} into that epoch, "
              f"best held-out mean ppl so far {best if best != float('inf') else 'n/a'}", flush=True)
        print(f"{total_steps - step:,} steps remaining", flush=True)
    elif args.resume:
        print(f"--resume given but no checkpoint found at {last_path} -- starting from scratch", flush=True)
    else:
        print("starting training from scratch (no checkpoint loaded)", flush=True)

    def checkpoint(epoch: int, batch_in_epoch: int, path: Path) -> bool:
        if args.no_save:
            return False
        return save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                             "cfg": model.cfg.to_dict(), "args": vars(args) | {"data_dir": str(args.data_dir)},
                             "step": step, "epoch": epoch, "batch_in_epoch": batch_in_epoch, "best": best,
                             "history": history}, path)

    print(f"{per_epoch:,} steps/epoch x {args.epochs} epochs = {total_steps:,} steps, peak lr {args.lr}", flush=True)
    print(f"config: {args.config}  batch~{args.batch_positions} positions  dropout {args.dropout}  "
          f"wd {args.wd}  clip {args.clip}  warmup {args.warmup_frac:.0%}  fraction {args.fraction:.0%}", flush=True)
    print(f"checkpoints: {last_path.name} every {args.ckpt_every:,} steps"
          + ("" if args.no_save else f"; best -> {best_path.name}"), flush=True)
    print("entering training loop...", flush=True)

    model.train()
    acc = torch.zeros(4, dtype=torch.float64, device=device)      # sum fwd loss, sum bwd loss, sum grad norm, n steps
    pos_since, t_log, t_start = 0, time.time(), time.time()
    steps_at_start = step
    done = False
    oom_retries = 0
    for epoch in range(start_epoch, args.epochs):
        order = train.order(epoch, args.seed, args.fraction)
        skip = start_batch if epoch == start_epoch else 0
        for j, ids in enumerate(Prefetcher(train, order[skip:], device), start=skip):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, args.lr, args.warmup_frac, args.min_lr_frac)
            for attempt in range(3):
                try:
                    with torch.autocast("cuda", dtype=torch.float16):
                        lf, lb = model(ids)
                        loss = lf + lb
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    break
                except torch.OutOfMemoryError:
                    # fragmentation near the 6 GB limit: drop cached blocks and retry the same batch
                    lf = lb = loss = None
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    oom_retries += 1
                    print(f"  [warning] CUDA OOM at step {step + 1}, batch shape {tuple(ids.shape)} "
                          f"(attempt {attempt + 1}/3) -- cleared cache, retrying", flush=True)
                    if attempt == 2:
                        print(f"  [fatal] CUDA OOM at step {step + 1} after 3 attempts -- giving up", flush=True)
                        raise
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()

            acc[0] += lf.detach()
            acc[1] += lb.detach()
            acc[2] += grad_norm.detach()
            acc[3] += 1
            pos_since += ids.numel()
            step += 1

            if step % args.log_every == 0:
                sf, sb, gn, n = acc.tolist()          # the only sync point in the loop
                dt = time.time() - t_log
                elapsed = time.time() - t_start
                rate = (step - steps_at_start) / elapsed
                eta = (total_steps - step) / rate if rate > 0 else float("inf")
                pct = 100 * step / total_steps
                print(f"  ep {epoch} step {step:>7,}/{total_steps:,} ({pct:4.1f}%)  ppl fwd {math.exp(sf / n):8.1f} bwd {math.exp(sb / n):8.1f}"
                      f"  lr {opt.param_groups[0]['lr']:.2e}  grad-norm {gn / n:.2f}  {pos_since / dt:>8,.0f} words/s  "
                      f"mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB  oom-retries {oom_retries}  "
                      f"elapsed {fmt_hms(elapsed)}  eta {fmt_hms(eta)}  scale {scaler.get_scale():.0f}", flush=True)
                acc.zero_()
                pos_since, t_log = 0, time.time()
            if step % args.ckpt_every == 0:
                saved = checkpoint(epoch, j + 1, last_path)
                if saved:
                    print(f"  [checkpoint] saved {last_path.name} at step {step:,} (epoch {epoch}, batch {j + 1:,}/{per_epoch:,})", flush=True)
            if step >= total_steps:
                done = True
                break
        print(f"epoch {epoch + 1}/{args.epochs} training done, running held-out evaluation...", flush=True)
        eval_t0 = time.time()
        ev = evaluate(model, heldout, device)
        history.append({"epoch": epoch + 1, "step": step, **ev})
        improved = ev["ppl_mean"] < best
        if improved:
            best = ev["ppl_mean"]
        elapsed = time.time() - t_start
        print(f"== epoch {epoch + 1}/{args.epochs} done  elapsed {fmt_hms(elapsed)}  held-out eval {time.time() - eval_t0:.0f}s  "
              f"held-out ppl: fwd {ev['ppl_fwd']:.1f} bwd {ev['ppl_bwd']:.1f} mean {ev['ppl_mean']:.1f} "
              f"(best so far: {best:.1f}) | excluding <unk> targets: mean {ev['ppl_mean_nounk']:.1f}"
              f"{'  <- new best, saving elmo_' + args.config + '_best.pt' if improved else ''}", flush=True)
        saved = checkpoint(epoch + 1, 0, last_path)
        if saved:
            print(f"  [checkpoint] saved {last_path.name} at end of epoch {epoch + 1}", flush=True)
        if improved and not args.no_save:
            shutil.copyfile(last_path, best_path)
            print(f"  [checkpoint] copied to {best_path.name} (new best)", flush=True)
        (MODELS_DIR / f"{name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if done:
            print("reached --max-steps, stopping early", flush=True)
            break
    print(f"finished  total elapsed {fmt_hms(time.time() - t_start)}  best held-out mean ppl {best:.1f}", flush=True)


if __name__ == "__main__":
    main()
