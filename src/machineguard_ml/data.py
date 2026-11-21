from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_table(path: Path, sheet: str | int | None = None) -> pd.DataFrame:
    """Load a CSV or Excel table without assuming a machine-specific path."""
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    else:
        raise ValueError("Supported input formats are .csv, .xlsx, and .xls")
    if frame.empty:
        raise ValueError("Input table is empty")
    return frame


def resolve_columns(frame: pd.DataFrame, columns: list[str] | None) -> list[str]:
    if columns:
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise ValueError(f"Unknown feature columns: {', '.join(missing)}")
        return columns
    numeric = frame.select_dtypes(include="number").columns.tolist()
    if not numeric:
        raise ValueError("No numeric feature columns were found")
    return numeric


def split_features_target(
    frame: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if target not in frame.columns:
        raise ValueError(f"Target column does not exist: {target}")
    selected = resolve_columns(frame.drop(columns=[target]), features)
    subset = frame[selected + [target]].dropna().reset_index(drop=True)
    if len(subset) < 4:
        raise ValueError("At least four complete rows are required")
    return subset[selected], subset[target], selected
