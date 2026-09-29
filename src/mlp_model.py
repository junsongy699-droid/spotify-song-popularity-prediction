"""R4: neural network — Multi-Layer Perceptron (MLP) for both tasks.

scikit-learn MLP is used for reproducibility on tabular data; a note for
the report: CNNs are not applicable to this tabular feature set.
"""
from __future__ import annotations

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import classification_metrics, format_row, regression_metrics


def run_mlp(X_train, y_train, X_test, y_test, task: str) -> str:
    """Train an MLP and print its metrics. Returns the formatted row."""
    hidden = (64, 32)
    if task == "regression":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(hidden_layer_sizes=hidden, max_iter=500, random_state=42)),
            ]
        )
    else:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(hidden_layer_sizes=hidden, max_iter=500, random_state=42)),
            ]
        )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = (
        regression_metrics(y_test, y_pred)
        if task == "regression"
        else classification_metrics(y_test, y_pred)
    )
    row = format_row("MLP (R4)", metrics)
    print(row)
    return row
