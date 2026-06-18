"""Evaluation metrics for yield prediction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "correlation": float("nan")}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    corr = float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "correlation": corr}


def save_metrics(metrics: dict, path: Path | str, model_name: str = "model") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_name, **metrics}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_training_history(history: dict, path: Path | str, model_name: str = "model") -> None:
    """Save per-epoch or per-step training curves as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_name, **history}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_predictions(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path | str,
    model_name: str = "model",
) -> None:
    """Save y_true / y_pred per split for visualization."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_name, (y_true, y_pred) in splits.items():
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        for yt, yp in zip(y_true, y_pred):
            rows.append({"model": model_name, "split": split_name, "y_true": yt, "y_pred": yp})
    pd.DataFrame(rows).to_csv(path, index=False)

