"""Evaluation utilities (shared by R3 baselines and R4 MLP)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 3),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
    }


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
    }


def format_row(name: str, metrics: dict) -> str:
    parts = " · ".join(f"{k}={v}" for k, v in metrics.items())
    return f"{name:24s} {parts}"
