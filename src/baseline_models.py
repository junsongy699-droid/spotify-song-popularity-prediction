"""R3: at least three traditional baselines covering the course families.

Course families covered here: Decision Tree, Naive Bayes, Linear models,
Perceptron, kNN, Ensembles. Each task (regression / classification) runs
the applicable models under the same train/test split and seed.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Perceptron,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .evaluate import classification_metrics, format_row, regression_metrics


def build_models(task: str) -> dict[str, object]:
    """Return a dict {model_name: estimator} for the given task."""
    if task == "regression":
        return {
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "LinearRegression": LinearRegression(),
            "kNN": Pipeline(
                [("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=5))]
            ),
            "RandomForest(Ensemble)": RandomForestRegressor(n_estimators=100, random_state=42),
        }
    if task == "classification":
        return {
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "NaiveBayes": GaussianNB(),
            "LogisticRegression": Pipeline(
                [("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=42))]
            ),
            "Perceptron": Pipeline(
                [("scaler", StandardScaler()), ("perceptron", Perceptron(random_state=42))]
            ),
            "kNN": Pipeline(
                [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))]
            ),
            "RandomForest(Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42),
        }
    raise ValueError(f"unknown task: {task}")


def run_baselines(X_train, y_train, X_test, y_test, task: str) -> list[str]:
    """Train all baselines and print one metric line per model."""
    rows: list[str] = []
    for name, model in build_models(task).items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if task == "regression":
            metrics = regression_metrics(y_test, y_pred)
        else:
            metrics = classification_metrics(y_test, y_pred)
        row = format_row(f"{name}", metrics)
        rows.append(row)
        print(row)
    return rows
