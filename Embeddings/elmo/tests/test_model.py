"""Correctness tests for the ELMo biLM (plan.md section 9). Tiny CPU models, stdlib unittest --
run:  python -m unittest discover -s Embeddings/elmo/tests -v   (GPU venv or any env with torch)."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import model as M  # noqa: E402

N_TYPES, N_CHARS, V_OUT = 50, 20, 64


def tiny_model(seed=0, dropout=0.0) -> M.BiLM:
    torch.manual_seed(seed)
    m = M.build({"hidden": 32, "proj": 16, "filters": (4, 4, 8), "n_chars": N_CHARS, "vocab_out": V_OUT,
                 "dropout": dropout})
    rng = np.random.default_rng(seed)
    chars = rng.integers(0, N_CHARS, size=(N_TYPES, M.MAX_CHARS)).astype(np.uint8)
    t2o = rng.integers(2, V_OUT, size=N_TYPES)
    m.set_tables(chars, t2o)
    return m.eval()


def ids(B=3, T=10, seed=1):
    return torch.from_numpy(np.random.default_rng(seed).integers(3, N_TYPES, size=(B, T)))


class BiLMTests(unittest.TestCase):
    def test_full_size_param_counts_match_plan(self):
        m = M.build("M")
        c = m.count_params()
        self.assertAlmostEqual(c["total"] / 1e6, 28.6, delta=0.4)
        self.assertAlmostEqual(c["non_softmax"] / 1e6, 23.5, delta=0.4)
        for name in ("hidden", "proj", "vocab_out"):
            self.assertEqual(getattr(m.cfg, name) % 64, 0, f"{name} should be a multiple of 64")
        self.assertEqual(sum(m.cfg.filters) % 64, 0)

    def test_forward_stack_is_causal(self):
        m, x = tiny_model(), ids()
        k = 4
        _, (_, f_a), _ = m.layers(x)
        y = x.clone()
        y[:, k + 1:] = torch.from_numpy(np.random.default_rng(9).integers(3, N_TYPES, size=y[:, k + 1:].shape))
        _, (_, f_b), _ = m.layers(y)
        self.assertTrue(torch.allclose(f_a[: k + 1], f_b[: k + 1], atol=1e-6), "forward states at <=k changed")
        self.assertFalse(torch.allclose(f_a[k + 1:], f_b[k + 1:], atol=1e-6), "perturbation had no effect at all")

    def test_backward_stack_is_anticausal(self):
        m, x = tiny_model(), ids()
        k = 5
        _, _, (_, b_a) = m.layers(x)
        y = x.clone()
        y[:, :k] = torch.from_numpy(np.random.default_rng(8).integers(3, N_TYPES, size=y[:, :k].shape))
        _, _, (_, b_b) = m.layers(y)
        self.assertTrue(torch.allclose(b_a[k:], b_b[k:], atol=1e-6), "backward states at >=k changed")
        self.assertFalse(torch.allclose(b_a[:k], b_b[:k], atol=1e-6))

    def test_sequence_output_independent_of_batch_mates(self):
        m, x = tiny_model(), ids(B=4)
        _, (f1, f2), (b1, b2) = m.layers(x)
        _, (g1, g2), (c1, c2) = m.layers(x[1:2])
        for full, alone in ((f2, g2), (b2, c2), (f1, g1), (b1, c1)):
            self.assertTrue(torch.allclose(full[:, 1:2], alone, atol=1e-5))

    def test_unique_dedupe_matches_per_token_charcnn_including_gradients(self):
        m, x = tiny_model(), ids(B=4, T=12).t().contiguous()      # (T, B) with repeated types
        self.assertLess(len(torch.unique(x)), x.numel(), "test needs repeated types")
        m.zero_grad()
        via_unique = m.token_repr(x)
        via_unique.pow(2).sum().backward()
        g_unique = [p.grad.clone() for p in m.cnn.parameters()]
        m.zero_grad()
        direct = m.cnn(m.char_table[x.reshape(-1)].long()).view(*x.shape, -1)
        direct.pow(2).sum().backward()
        g_direct = [p.grad.clone() for p in m.cnn.parameters()]
        self.assertTrue(torch.allclose(via_unique, direct, atol=1e-6))
        for a, b in zip(g_unique, g_direct):
            self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_target_alignment(self):
        m, x = tiny_model(), ids(B=2, T=7)
        lf, tf, lb, tb = m(x, detailed=True)
        out = m.type_to_out[x].t()                                  # (T, B)
        self.assertTrue(torch.equal(tf.view(6, 2), out[1:]), "forward must predict the NEXT token")
        self.assertTrue(torch.equal(tb.view(6, 2), out[:-1]), "backward must predict the PREVIOUS token")
        self.assertEqual(lf.shape, tf.shape)
        self.assertEqual(lb.shape, tb.shape)

    def test_initial_loss_is_about_log_vocab(self):
        m, x = tiny_model(), ids(B=8, T=16)
        lf, lb = m(x)
        for loss in (lf, lb):
            self.assertAlmostEqual(float(loss), math.log(V_OUT), delta=0.3)

    def test_can_overfit_a_tiny_batch(self):
        m, x = tiny_model(dropout=0.0).train(), ids(B=4, T=8)
        opt = torch.optim.Adam(m.parameters(), lr=5e-3)
        first = None
        for step in range(200):     # random targets: memorizing 28 positions x 2 directions needs a few hundred steps
            lf, lb = m(x)
            loss = lf + lb
            first = float(loss.detach()) if first is None else first
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(float(loss.detach()), 0.25 * first)


if __name__ == "__main__":
    unittest.main()
