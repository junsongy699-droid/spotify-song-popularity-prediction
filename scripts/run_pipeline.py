"""End-to-end reproduction of the whole coursework pipeline (R2–R4).

Usage:
    python scripts/run_pipeline.py

Outputs:
    - results/eda_*.png          EDA figures (R2)
    - results/clustering.png     elbow + silhouette (R2/O3)
    - console metric table       baselines (R3) and MLP (R4)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sklearn.model_selection import train_test_split

from src.baseline_models import run_baselines
from src.clustering import run_clustering
from src.eda import run_eda
from src.evaluate import format_row
from src.load_data import load_data, load_raw
from src.mlp_model import run_mlp

SEED = 42


def main() -> None:
    print("=" * 60)
    print("R2 · Preprocessing + EDA")
    print("=" * 60)
    eda_summary = run_eda(load_raw())
    print("popularity mean/median/range:",
          eda_summary["popularity_mean"], "/", eda_summary["popularity_median"],
          "/", eda_summary["popularity_min"], "-", eda_summary["popularity_max"])
    print("missing genres:", eda_summary["missing_genres"], "| explicit ratio:",
          eda_summary["explicit_ratio"])
    print("figures:", eda_summary["figures"])

    print()
    print("=" * 60)
    print("R2 · Clustering (O3)")
    print("=" * 60)
    clust = run_clustering()
    print("best_k:", clust["best_k"])
    print("silhouettes:", clust["silhouettes"])

    for task in ("regression", "classification"):
        print()
        print("=" * 60)
        print(f"R3 + R4 · Task: {task}")
        print("=" * 60)
        X, y, _ = load_data(task)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=(y if task == "classification" else None)
        )
        run_baselines(X_train, y_train, X_test, y_test, task)
        run_mlp(X_train, y_train, X_test, y_test, task)

    print()
    print("Done. Figures saved under results/.")


if __name__ == "__main__":
    main()
