"""Load and preprocess the Spotify dataset (R2: data cleaning & feature engineering)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "spotifydataset.csv"

# Numeric features used across all tasks (11 audio features + useful metadata).
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]
METADATA_FEATURES = [
    "duration_ms",
    "release_year",
    "followers",
    # NOTE: "artist_popularity" is deliberately EXCLUDED — it is perfectly
    # correlated with the target track_popularity (r = 1.0 in this dataset),
    # i.e. a target-leakage feature. Removing it is part of R2 preprocessing
    # and is documented in the report/EDA.
]

TARGET_REGRESSION = "track_popularity"
TARGET_CLASSIFICATION = "explicit"


def load_raw() -> pd.DataFrame:
    """Load the raw CSV."""
    df = pd.read_csv(DATA_FILE)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer features.

    - fill missing genres (163 rows) with 'unknown';
    - derive release_year from release_date;
    - encode explicit as 0/1;
    - cap extreme followers (log1p) to stabilise the scale.
    """
    df = df.copy()
    df["genres"] = df["genres"].fillna("unknown")

    if "release_date" in df.columns:
        df["release_year"] = (
            pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(1970).astype(int)
        )

    df["explicit"] = df["explicit"].map({True: 1, False: 0}).fillna(0).astype(int)
    df["followers"] = np.log1p(df["followers"].astype(float))

    return df


def build_xy(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for 'regression' (popularity) or 'classification' (explicit)."""
    features = AUDIO_FEATURES + METADATA_FEATURES
    X = df[features].copy()
    if task == "regression":
        y = df[TARGET_REGRESSION].astype(float)
    elif task == "classification":
        y = df[TARGET_CLASSIFICATION].astype(int)
    else:
        raise ValueError(f"unknown task: {task}")
    return X, y


def load_data(task: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """One-shot loader: returns (X, y, full_preprocessed_df)."""
    df = preprocess(load_raw())
    X, y = build_xy(df, task)
    return X, y, df
