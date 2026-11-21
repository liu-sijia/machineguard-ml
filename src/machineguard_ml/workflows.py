from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import BPRegressor


@dataclass
class WorkflowResult:
    metrics: dict[str, object]
    predictions: pd.DataFrame


def chronological_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    cut = max(1, min(len(features) - 1, int(len(features) * (1 - test_ratio))))
    return features.iloc[:cut], features.iloc[cut:], target.iloc[:cut], target.iloc[cut:]


def run_classification(
    features: pd.DataFrame,
    target: pd.Series,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> WorkflowResult:
    x_train, x_test, y_train, y_test = chronological_split(features, target, test_ratio)
    if y_train.nunique() < 2:
        raise ValueError("Classification training data must contain at least two classes")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=random_state)),
        ]
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    average = "binary" if target.nunique() == 2 else "weighted"
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, average=average, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, average=average, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
    }
    return WorkflowResult(
        metrics=metrics,
        predictions=pd.DataFrame({"actual": y_test.to_numpy(), "predicted": predicted}),
    )


def run_regression(
    features: pd.DataFrame,
    target: pd.Series,
    test_ratio: float = 0.2,
    hidden_size: int = 16,
    epochs: int = 500,
    learning_rate: float = 0.05,
    batch_size: int = 32,
    random_state: int = 42,
) -> WorkflowResult:
    x_train, x_test, y_train, y_test = chronological_split(features, target, test_ratio)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    test_scaled = scaler.transform(x_test)
    target_mean = float(y_train.mean())
    target_scale = float(y_train.std()) or 1.0
    train_target = (y_train.to_numpy(dtype=float) - target_mean) / target_scale
    model = BPRegressor(
        input_size=train_scaled.shape[1],
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
    ).fit(train_scaled, train_target)
    predicted = model.predict(test_scaled) * target_scale + target_mean
    actual = y_test.to_numpy(dtype=float)
    nonzero = np.abs(actual) > np.finfo(float).eps
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else None
    mse = float(mean_squared_error(actual, predicted))
    metrics: dict[str, object] = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predicted)),
        "mape_percent": mape,
        "final_training_loss": model.loss_history_[-1],
        "train_rows": len(x_train),
        "test_rows": len(x_test),
    }
    return WorkflowResult(
        metrics=metrics,
        predictions=pd.DataFrame({"actual": actual, "predicted": predicted}),
    )
