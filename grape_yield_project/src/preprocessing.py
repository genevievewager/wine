"""Preprocessing: filtering, encoding, scaling, and feature engineering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg


@dataclass
class PreprocessResult:
    df: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    preprocessor: ColumnTransformer | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def filter_cs_variety(df: pd.DataFrame) -> pd.DataFrame:
    if cfg.COL_VARIETY not in df.columns:
        return df
    v = df[cfg.COL_VARIETY].astype(str).str.strip()
    mask = v.isin(cfg.VARIETY_FILTER) | v.str.contains("Cabernet", case=False, na=False)
    return df.loc[mask].copy()


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if cfg.COL_SHOOTS in out.columns and cfg.COL_CANOPY in out.columns:
        out["shoots_per_canopy_area"] = out[cfg.COL_SHOOTS] / out[cfg.COL_CANOPY].replace(0, np.nan)
    if cfg.COL_CANOPY in out.columns and "previous_year_canopy_size" in out.columns:
        out["canopy_change_from_previous_year"] = out[cfg.COL_CANOPY] - out["previous_year_canopy_size"]
    if cfg.COL_ACTUAL_YIELD in out.columns and cfg.COL_PRED_YIELD in out.columns:
        out[cfg.COL_YIELD_ERROR] = out[cfg.COL_ACTUAL_YIELD] - out[cfg.COL_PRED_YIELD]
    return out


def year_splits(
    df: pd.DataFrame,
    train_years: list[int] | None = None,
    val_years: list[int] | None = None,
    test_years: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_years = train_years or cfg.TRAIN_YEARS
    val_years = val_years or cfg.VAL_YEARS
    test_years = test_years or cfg.TEST_YEARS
    y = df[cfg.COL_YEAR]
    train = df[y.isin(train_years)].copy()
    val = df[y.isin(val_years)].copy()
    test = df[y.isin(test_years)].copy()
    return train, val, test


def _select_feature_columns(df: pd.DataFrame, exclude_leakage: bool = True) -> list[str]:
    exclude = set(cfg.ID_COLUMNS + [cfg.TARGET_COLUMN, "harvest_date"])
    if exclude_leakage:
        exclude |= {cfg.COL_PRED_YIELD, cfg.COL_YIELD_ERROR, "harvest_yield_t_per_d"}
    numeric = []
    categorical = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        elif col in cfg.CATEGORICAL_COLUMNS or df[col].dtype == object:
            categorical.append(col)
    return numeric + categorical


def build_preprocessor(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    feature_columns = feature_columns or _select_feature_columns(df)
    numeric_cols = [c for c in feature_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_columns if c in df.columns and c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, cat_cols


def preprocess_dataframe(
    df: pd.DataFrame,
    fit: bool = True,
    preprocessor: ColumnTransformer | None = None,
    feature_columns: list[str] | None = None,
) -> PreprocessResult:
    df = filter_cs_variety(df)
    df = add_derived_features(df)
    feature_columns = feature_columns or _select_feature_columns(df)

    if fit or preprocessor is None:
        preprocessor, num_cols, cat_cols = build_preprocessor(df, feature_columns)
        X = preprocessor.fit_transform(df[feature_columns])
    else:
        num_cols = preprocessor.transformers_[0][2]
        cat_cols = preprocessor.transformers_[1][2] if len(preprocessor.transformers_) > 1 else []
        X = preprocessor.transform(df[feature_columns])

    feature_names = preprocessor.get_feature_names_out().tolist()
    processed = df[cfg.ID_COLUMNS + [cfg.COL_YEAR, cfg.TARGET_COLUMN]].copy()
    for i, name in enumerate(feature_names):
        processed[name] = X[:, i]

    return PreprocessResult(
        df=processed,
        feature_columns=feature_names,
        target_column=cfg.TARGET_COLUMN,
        preprocessor=preprocessor,
        metadata={"raw_feature_columns": feature_columns, "numeric_cols": num_cols, "cat_cols": cat_cols},
    )


def get_feature_group_indices(
    feature_names: list[str],
    groups: dict[str, list[str]] | None = None,
) -> dict[str, list[int]]:
    """Map PG-AN feature groups to column indices in the processed matrix."""
    groups = groups or cfg.FEATURE_GROUPS
    indices: dict[str, list[int]] = {}
    for group, cols in groups.items():
        idxs = []
        for i, fname in enumerate(feature_names):
            fn = fname.lower()
            for col in cols:
                col_l = col.lower().replace("weather_", "")
                if col_l in fn or fn.endswith(col_l) or col.lower() in fn:
                    idxs.append(i)
                    break
        indices[group] = sorted(set(idxs))
    return indices
