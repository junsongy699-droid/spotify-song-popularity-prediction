"""Unsupervised learning (R2/O3): k-means on audio features with elbow + silhouette."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .load_data import AUDIO_FEATURES, load_raw, preprocess

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run_clustering(k_range: range = range(2, 11)) -> dict:
    """Cluster standardised audio features; return elbow/silhouette curves and best k."""
    df = preprocess(load_raw())
    X = df[AUDIO_FEATURES].astype(float)
    Xs = StandardScaler().fit_transform(X)

    inertias: list[float] = []
    sil_scores: list[float] = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        inertias.append(float(km.inertia_))
        sil_scores.append(float(silhouette_score(Xs, labels)))

    best_k = int(k_range.start + int(np.argmax(sil_scores)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ks = list(k_range)
    axes[0].plot(ks, inertias, marker="o")
    axes[0].set_title("Elbow method")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("inertia")
    axes[1].plot(ks, sil_scores, marker="o")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("score")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "clustering.png", dpi=150)
    plt.close(fig)

    return {
        "best_k": best_k,
        "inertias": {str(k): round(v, 1) for k, v in zip(ks, inertias)},
        "silhouettes": {str(k): round(v, 3) for k, v in zip(ks, sil_scores)},
    }
