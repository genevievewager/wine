"""XGBoost baseline (optional if xgboost is installed)."""

from __future__ import annotations

from pathlib import Path

import joblib

from . import config as cfg
from .evaluate import compute_metrics, save_metrics
from .preprocessing import preprocess_dataframe, year_splits

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def train_xgboost(
    df,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> dict:
    if not HAS_XGBOOST:
        raise ImportError("xgboost is not installed. pip install xgboost")

    train_df, val_df, test_df = year_splits(df)
    prep = preprocess_dataframe(train_df, fit=True)
    feature_cols = prep.feature_columns
    train_p = prep.df
    val_p = preprocess_dataframe(val_df, fit=False, preprocessor=prep.preprocessor, feature_columns=prep.metadata["raw_feature_columns"]).df
    test_p = preprocess_dataframe(test_df, fit=False, preprocessor=prep.preprocessor, feature_columns=prep.metadata["raw_feature_columns"]).df

    X_train, y_train = train_p[feature_cols].values, train_p[cfg.TARGET_COLUMN].values
    X_val, y_val = val_p[feature_cols].values, val_p[cfg.TARGET_COLUMN].values
    X_test, y_test = test_p[feature_cols].values, test_p[cfg.TARGET_COLUMN].values

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        objective="reg:squarederror",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    metrics = {}
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        metrics[name] = compute_metrics(y, model.predict(X))

    model_path = cfg.MODELS / "xgboost.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols, "preprocessor": prep.preprocessor}, model_path)

    flat = {f"{split}_{k}": v for split, m in metrics.items() for k, v in m.items()}
    save_metrics(flat, cfg.METRICS / "xgboost_metrics.json", "xgboost")
    return {"model": model, "metrics": metrics, "model_path": model_path}
