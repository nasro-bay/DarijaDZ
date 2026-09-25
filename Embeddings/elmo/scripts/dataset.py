"""Exact-length batching over the memmapped id streams built by prepare_data.py.

Chunks are stored sorted by length, so a batch is one contiguous slice of same-length chunks: no
padding, no masks, static shapes per length (<= 65 distinct shapes, which suits
`cudnn.benchmark`). Batch size per length L is ~`batch_positions / L` rounded down to a multiple of
32 (clamped to [32, 1024]) so every batch holds ~8K positions -- the shape the size benchmark used.
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class Split:
    def __init__(self, name: str, data_dir: Path = DATA_DIR, batch_positions: int = 8192,
                 min_batch: int = 32, max_batch: int = 1024):
        self.name = name
        self.tokens = np.memmap(data_dir / f"{name}_tokens.i32", dtype=np.int32, mode="r")
        self.lens = np.load(data_dir / f"{name}_lens.npy")
        self.offs = np.load(data_dir / f"{name}_offs.npy")
        uniq, first, counts = np.unique(self.lens, return_index=True, return_counts=True)
        batches = []
        for L, a, n in zip(uniq.tolist(), first.tolist(), counts.tolist()):
            B = int(np.clip((batch_positions // L) // 32 * 32, min_batch, max_batch))
            for s in range(0, n, B):
                batches.append((a + s, a + min(s + B, n), L))
        self.batches = np.asarray(batches, dtype=np.int64)          # (n_batches, 3): start, end, length
        self.n_positions = int(len(self.tokens))

    def __len__(self) -> int:
        return len(self.batches)

    def get(self, i: int) -> np.ndarray:
        start, end, L = self.batches[i]
        a = int(self.offs[start])
        return np.asarray(self.tokens[a:a + int(end - start) * int(L)]).reshape(int(end - start), int(L))

    def order(self, epoch: int, seed: int = 0, fraction: float = 1.0) -> np.ndarray:
        """Batch order for an epoch. `fraction < 1` keeps a FIXED (seed-only) random subset of batches
        across epochs -- used for cheap LR sweeps -- reshuffled each epoch."""
        n = len(self.batches)
        if fraction < 1.0:
            keep = np.random.default_rng(seed).permutation(n)[: max(1, int(n * fraction))]
        else:
            keep = np.arange(n)
        return np.random.default_rng((seed + 1) * 100_003 + epoch).permutation(keep)


class Prefetcher:
    """One background thread turns batch indices into pinned int64 tensors; `__iter__` moves them to
    the GPU asynchronously. The data path is just memmap slices, so a single thread keeps up."""

    def __init__(self, split: Split, order: np.ndarray, device: torch.device, depth: int = 8):
        self.split, self.order, self.device = split, order, device
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self) -> None:
        for i in self.order:
            arr = torch.from_numpy(self.split.get(int(i)).astype(np.int64))
            self.q.put(arr.pin_memory() if self.device.type == "cuda" else arr)
        self.q.put(None)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            yield item.to(self.device, non_blocking=True)
