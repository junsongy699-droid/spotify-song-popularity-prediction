"""Exploratory data analysis (R2): summaries and figures saved under results/."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .load_data import AUDIO_FEATURES, preprocess

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run_eda(df_raw: pd.DataFrame) -> dict:
    """Produce key EDA outputs and figures. Returns summary statistics."""
    df = preprocess(df_raw)
    results = {}

    # 1. Target distribution
    results["popularity_mean"] = round(df["track_popularity"].mean(), 2)
    results["popularity_median"] = int(df["track_popularity"].median())
    results["popularity_min"] = int(df["track_popularity"].min())
    results["popularity_max"] = int(df["track_popularity"].max())

    # 2. Missing values
    results["missing_genres"] = int((df["genres"] == "unknown").sum())

    # 3. Class balance
    counts = df["explicit"].value_counts(normalize=True)
    results["explicit_ratio"] = {str(k): round(float(v), 3) for k, v in counts.items()}

    # 4. Figures
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["track_popularity"], bins=30, ax=axes[0])
    axes[0].set_title("Distribution of track popularity")
    sns.countplot(x="explicit", data=df, ax=axes[1])
    axes[1].set_title("Explicit vs non-explicit")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "eda_targets.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        df[AUDIO_FEATURES].corr(),
        cmap="RdBu_r",
        center=0,
        annot=False,
        ax=ax,
    )
    ax.set_title("Correlation of audio features")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "eda_corr.png", dpi=150)
    plt.close(fig)

    results["figures"] = [p.name for p in RESULTS_DIR.glob("*.png")]
    return results
