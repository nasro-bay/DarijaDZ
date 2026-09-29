"""Data pipeline for the CBOW+attention model (see ../plan.md):

- `ClusterBatchSampler`: groups rows into batches from a single K-Means
  length-cluster at a time (minimizes padding waste), shuffling within
  each cluster and shuffling batch order across clusters every epoch.
- `CBOWCollator`: per batch (not precomputed once), for eligible
  arabic-script rows with >5 words, transliterates one random word with
  20% probability, THEN BPE-tokenizes, applies frequent-word
  subsampling, and expands every row into its (context, center) training
  pairs with a radius-8 window, padded/masked to a fixed width.

Run via the GPU venv's Python -- see ../requirements.txt.
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "DarijaDZ_spaces" / "Arabizi_transliteration"))
from transliterate import transliterate_word  # noqa: E402

PAD_ID = 0  # verified against Tokenization/models/bpe/bpe_20000/vocab.json: "<pad>" -> 0
WINDOW_RADIUS = 8
WINDOW_SIZE = WINDOW_RADIUS * 2
SUBSAMPLE_THRESHOLD = 1e-3
AUGMENT_RATE = 0.20
AUGMENT_MIN_WORDS = 5
NEGATIVE_TABLE_SIZE = 10_000_000
# Hard backstop on generated (context, center) pairs per batch, independent
# of ClusterBatchSampler's pair-budget sizing -- PyTorch's efficient-
# attention backend hard-errors above 65535; this stays well under that
# with headroom for a 6GB card. ClusterBatchSampler should keep batches
# near its target_pairs_per_batch already, but within-cluster token-count
# variance (a cluster's average doesn't bound its max) could still produce
# an oversized batch, so this is a safety net, not the primary control.
MAX_PAIRS_PER_BATCH = 8192


def build_negative_sampling_table(
    token_freq: dict[int, int], vocab_size: int, table_size: int = NEGATIVE_TABLE_SIZE
) -> np.ndarray:
    """Unigram^0.75 sampling table (standard word2vec negative-sampling
    distribution) -- same technique as the original word2vec.c: fill a
    large array proportionally to freq^0.75, then sample by drawing
    random indices into it (O(1) per sample instead of a fresh weighted
    draw each time).
    """
    freqs = np.zeros(vocab_size, dtype=np.float64)
    for tok_id, count in token_freq.items():
        tok_id = int(tok_id)
        if tok_id < vocab_size:
            freqs[tok_id] = count
    freqs = np.power(freqs, 0.75)
    total = freqs.sum()
    probs = freqs / total if total > 0 else np.full(vocab_size, 1.0 / vocab_size)
    counts_per_id = np.round(probs * table_size).astype(np.int64)
    counts_per_id[counts_per_id == 0] = 0  # words that never occurred stay at 0 -- never sampled as negatives
    table = np.repeat(np.arange(vocab_size), counts_per_id)
    return table


def build_subsample_keep_prob(token_freq: dict[int, int], threshold: float = SUBSAMPLE_THRESHOLD) -> dict[int, float]:
    """P(keep token) = sqrt(threshold / freq_ratio), capped at 1.0 --
    Mikolov et al.'s subsampling formula (the original-paper form, not
    word2vec.c's slightly different reformulation), per plan.md.
    """
    total = sum(token_freq.values())
    keep_prob: dict[int, float] = {}
    for tok_id, count in token_freq.items():
        freq_ratio = count / total
        keep_prob[int(tok_id)] = min(1.0, (threshold / freq_ratio) ** 0.5)
    return keep_prob


class RowDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


class ClusterBatchSampler(Sampler[list[int]]):
    """Yields index batches drawn from a single cluster_id at a time --
    every batch is length-homogeneous by construction, which is the point
    of the K-Means length clustering (minimize intra-batch padding).
    Shuffled within each cluster and across batch order, freshly each
    epoch (call set_epoch() before each epoch for a different shuffle).

    Rows per batch are sized PER CLUSTER from a target (context, center)
    PAIR budget rather than a flat row count: each row of token_count T
    expands (roughly) into T pairs at collate time (one per center
    position), so a flat row count means the "long" cluster (djelfa forum
    posts, avg ~500 tokens) can generate 100k+ pairs from a single batch
    of rows -- this blew past PyTorch's efficient-attention 65535
    batch-size ceiling in real training (short YouTube-comment clusters
    never hit it, since a row there is only ~10-20 tokens). Sizing
    rows-per-batch as target_pairs_per_batch / avg_tokens_in_cluster keeps
    the actual pair count roughly constant across clusters.
    """

    def __init__(
        self,
        cluster_ids: list[int],
        token_counts: list[int],
        target_pairs_per_batch: int,
        seed: int = 42,
        min_rows_per_batch: int = 4,
    ):
        self.seed = seed
        self.epoch = 0
        self.by_cluster: dict[int, list[int]] = defaultdict(list)
        for idx, cid in enumerate(cluster_ids):
            self.by_cluster[cid].append(idx)

        self.rows_per_batch: dict[int, int] = {}
        for cid, indices in self.by_cluster.items():
            avg_tokens = sum(token_counts[i] for i in indices) / len(indices)
            rows = max(min_rows_per_batch, round(target_pairs_per_batch / max(avg_tokens, 1.0)))
            self.rows_per_batch[cid] = rows

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for cid, indices in self.by_cluster.items():
            idxs = list(indices)
            rng.shuffle(idxs)
            bs = self.rows_per_batch[cid]
            for i in range(0, len(idxs), bs):
                batches.append(idxs[i : i + bs])
        rng.shuffle(batches)  # shuffle batch presentation order -- never mixes clusters within one batch
        yield from batches

    def __len__(self) -> int:
        return sum(
            (len(v) + self.rows_per_batch[cid] - 1) // self.rows_per_batch[cid]
            for cid, v in self.by_cluster.items()
        )


class CBOWCollator:
    def __init__(self, tok, negative_table: np.ndarray, keep_prob: dict[int, float], num_negative: int = 5):
        self.tok = tok
        self.negative_table = negative_table
        self.keep_prob = keep_prob
        self.num_negative = num_negative
        self.rng = random.Random()  # intentionally unseeded here: fresh randomness per batch/epoch
        # is the whole point of doing augmentation+subsampling at collate
        # time instead of precomputing once (see plan.md's stated tradeoff).

    def _maybe_augment(self, row: dict) -> str:
        text = row["text"]
        if (
            row["script_bucket"] == "arabic"
            and row["word_count"] > AUGMENT_MIN_WORDS
            and self.rng.random() < AUGMENT_RATE
        ):
            words = text.split()
            i = self.rng.randrange(len(words))
            words[i] = transliterate_word(words[i])
            text = " ".join(words)
        return text

    def __call__(self, batch_rows: list[dict]):
        context_list: list[list[int]] = []
        center_list: list[int] = []

        for row in batch_rows:
            text = self._maybe_augment(row)
            ids = self.tok.encode(text)
            if len(ids) < 2:
                continue

            kept = [t for t in ids if self.rng.random() < self.keep_prob.get(t, 1.0)]
            if len(kept) < 2:
                kept = ids  # don't let subsampling wipe out an already-short row entirely

            n = len(kept)
            for center_pos in range(n):
                start = max(0, center_pos - WINDOW_RADIUS)
                end = min(n, center_pos + WINDOW_RADIUS + 1)
                context = kept[start:center_pos] + kept[center_pos + 1 : end]
                if not context:
                    continue
                context_list.append(context[:WINDOW_SIZE])
                center_list.append(kept[center_pos])

        if not context_list:
            return None  # caller skips empty batches (can happen if every row's whole
            # sequence got subsampled away, rare but possible for very short rows)

        if len(context_list) > MAX_PAIRS_PER_BATCH:
            keep = self.rng.sample(range(len(context_list)), MAX_PAIRS_PER_BATCH)
            context_list = [context_list[i] for i in keep]
            center_list = [center_list[i] for i in keep]

        batch_size = len(context_list)
        context_ids = torch.full((batch_size, WINDOW_SIZE), PAD_ID, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, WINDOW_SIZE), dtype=torch.bool)
        for i, ctx in enumerate(context_list):
            context_ids[i, : len(ctx)] = torch.tensor(ctx, dtype=torch.long)
            attention_mask[i, : len(ctx)] = True

        center_ids = torch.tensor(center_list, dtype=torch.long)
        neg_idx = np.random.randint(0, len(self.negative_table), size=(batch_size, self.num_negative))
        negative_ids = torch.from_numpy(self.negative_table[neg_idx]).long()

        return context_ids, attention_mask, center_ids, negative_ids
