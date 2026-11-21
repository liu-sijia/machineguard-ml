from pathlib import Path

import numpy as np
import pandas as pd

from machineguard_ml.data import load_table, split_features_target
from machineguard_ml.models import BPRegressor
from machineguard_ml.workflows import run_classification, run_regression


def make_frame(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    return pd.DataFrame(
        {
            "sensor_a": x1,
            "sensor_b": x2,
            "fault": (x1 + x2 > 0).astype(int),
            "target": 3 * x1 - 2 * x2 + rng.normal(0, 0.05, rows),
        }
    )


def test_load_csv(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    make_frame().to_csv(path, index=False)
    assert len(load_table(path)) == 120


def test_split_selects_numeric_features() -> None:
    frame = make_frame()
    features, target, names = split_features_target(frame, "fault", ["sensor_a", "sensor_b"])
    assert features.shape == (120, 2)
    assert target.name == "fault"
    assert names == ["sensor_a", "sensor_b"]


def test_bp_regressor_predicts_expected_shape() -> None:
    x = np.arange(40, dtype=float).reshape(20, 2) / 40
    y = x[:, 0] - x[:, 1]
    model = BPRegressor(2, epochs=5, batch_size=5).fit(x, y)
    assert model.predict(x).shape == (20,)
    assert len(model.loss_history_) == 5


def test_classification_workflow() -> None:
    frame = make_frame()
    result = run_classification(frame[["sensor_a", "sensor_b"]], frame["fault"])
    assert 0 <= result.metrics["accuracy"] <= 1
    assert list(result.predictions.columns) == ["actual", "predicted"]


def test_regression_workflow() -> None:
    frame = make_frame()
    result = run_regression(
        frame[["sensor_a", "sensor_b"]],
        frame["target"],
        epochs=80,
        learning_rate=0.08,
    )
    assert result.metrics["mae"] >= 0
    assert result.metrics["r2"] > 0.5
