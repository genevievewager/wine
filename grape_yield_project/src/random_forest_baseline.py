"""Random Forest baseline on original or PCA features."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from . import config as cfg
from .evaluate import compute_metrics, save_metrics, save_predictions, save_training_history
from .pca_baseline import run_pca_baseline
from .preprocessing import preprocess_dataframe, year_splits


def _xy(split_df: pd.DataFrame, feature_cols: list[str]):
    X = split_df[feature_cols].values
    y = split_df[cfg.TARGET_COLUMN].values
    return X, y


def train_random_forest(
    df: pd.DataFrame,
    use_pca: bool = False,
    n_estimators: int = 200,
    random_state: int = 42,
    model_name: str = "rf",
) -> dict:
    train_df, val_df, test_df = year_splits(df)
    result = {}

    if use_pca:
        pca_out = run_pca_baseline(train_df, n_components=min(10, train_df.shape[0] - 1))
        feature_cols = [c for c in pca_out["pca_df"].columns if c.startswith("PC")]
        train_pca = pca_out["pca_df"]
        # Transform val/test with same numeric cols
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        X_df = train_df[pca_out["columns"]]
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        pipe.fit(X_df)
        def transform_split(split):
            X = pipe.transform(split[pca_out["columns"]])
            Z = pca_out["pca"].transform(X)
            out = split[cfg.ID_COLUMNS + [cfg.COL_YEAR, cfg.TARGET_COLUMN]].copy()
            for i in range(Z.shape[1]):
                out[f"PC{i+1}"] = Z[:, i]
            return out
        train_pca = transform_split(train_df)
        val_pca = transform_split(val_df)
        test_pca = transform_split(test_df)
        model_name = "pca_rf"
    else:
        prep = preprocess_dataframe(train_df, fit=True)
        feature_cols = prep.feature_columns
        train_pca = prep.df
        val_pca = preprocess_dataframe(val_df, fit=False, preprocessor=prep.preprocessor, feature_columns=prep.metadata["raw_feature_columns"]).df
        test_pca = preprocess_dataframe(test_df, fit=False, preprocessor=prep.preprocessor, feature_columns=prep.metadata["raw_feature_columns"]).df

    X_train, y_train = _xy(train_pca, feature_cols)
    X_val, y_val = _xy(val_pca, feature_cols)
    X_test, y_test = _xy(test_pca, feature_cols)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)

    metrics = {}
    predictions = {}
    for split_name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        pred = model.predict(X)
        metrics[split_name] = compute_metrics(y, pred)
        predictions[split_name] = (y, pred)

    # Learning curve: RMSE vs n_estimators
    curve_steps = sorted(set(list(range(25, min(n_estimators, 200) + 1, 25)) + [n_estimators]))
    curve_hist = {"model": model_name, "n_estimators": [], "train_rmse": [], "val_rmse": [], "test_rmse": []}
    for n in curve_steps:
        lc_model = RandomForestRegressor(n_estimators=n, random_state=random_state, n_jobs=-1)
        lc_model.fit(X_train, y_train)
        curve_hist["n_estimators"].append(n)
        for key, X, y in [("train_rmse", X_train, y_train), ("val_rmse", X_val, y_val), ("test_rmse", X_test, y_test)]:
            curve_hist[key].append(compute_metrics(y, lc_model.predict(X))["rmse"])

    model_path = cfg.MODELS / f"{model_name}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols, "use_pca": use_pca}, model_path)

    metrics_path = cfg.METRICS / f"{model_name}_metrics.json"
    save_metrics({k: v for split, m in metrics.items() for k, v in {f"{split}_{kk}": vv for kk, vv in m.items()}.items()}, metrics_path, model_name)
    save_predictions(predictions, cfg.METRICS / f"{model_name}_predictions.csv", model_name)
    save_training_history(curve_hist, cfg.METRICS / f"{model_name}_learning_curve.json", model_name)

    result["model"] = model
    result["metrics"] = metrics
    result["model_path"] = model_path
    result["learning_curve"] = curve_hist
    return result
