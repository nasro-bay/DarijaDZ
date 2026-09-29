"""Precomputes the 2D t-SNE coordinates behind the Explore map tab (map2d.npz). Run once after the
vectors change: python build_assets.py [--n 1500]"""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
from sklearn.manifold import TSNE

HERE = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1500, help="number of most frequent words to project")
n = parser.parse_args().n

E = load_file(HERE / "model.safetensors")["embeddings"]
vocab = json.loads((HERE / "vocab.json").read_text(encoding="utf-8"))
X = E[:n]  # vocab.json is ordered by frequency
X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
coords = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X)
np.savez_compressed(HERE / "map2d.npz", coords=coords.astype("float32"), n=n)
print(f"wrote map2d.npz: {coords.shape}")
