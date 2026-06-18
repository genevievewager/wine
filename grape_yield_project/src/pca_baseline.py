"""PCA baseline: explained variance, loadings, and reduced feature matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from . import config as cfg


def _numeric_matrix(df: pd.DataFrame, exclude: set[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    exclude = exclude or set(cfg.ID_COLUMNS + [cfg.TARGET_COLUMN, cfg.COL_YEAR])
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return df[cols].copy(), cols


def run_pca_baseline(
    df: pd.DataFrame,
    n_components: int = 10,
    output_dir: Path | str | None = None,
) -> dict:
    """Standardize numeric features, fit PCA, save plots and reduced matrix."""
    output_dir = Path(output_dir or cfg.FIGURES)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = cfg.DATA_PROCESSED
    processed_dir.mkdir(parents=True, exist_ok=True)

    X_df, cols = _numeric_matrix(df)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(X_df))

    n_components = min(n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Explained variance plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
    ax.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), "ro-")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA explained variance")
    fig.tight_layout()
    var_path = output_dir / "pca_explained_variance.png"
    fig.savefig(var_path, dpi=150)
    plt.close(fig)

    # Loadings table
    loadings = pd.DataFrame(
        pca.components_.T,
        index=cols,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    loadings_path = processed_dir / "pca_loadings.csv"
    loadings.to_csv(loadings_path)

    # PC1 vs PC2 colored by actual yield
    if cfg.TARGET_COLUMN in df.columns and n_components >= 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df[cfg.TARGET_COLUMN], cmap="viridis", alpha=0.7)
        plt.colorbar(sc, ax=ax, label=cfg.TARGET_COLUMN)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PC1 vs PC2 colored by actual yield")
        fig.tight_layout()
        pc_path = output_dir / "pca_pc1_pc2_yield.png"
        fig.savefig(pc_path, dpi=150)
        plt.close(fig)

    # Reduced feature matrix for downstream models
    pca_df = df[cfg.ID_COLUMNS + [cfg.COL_YEAR, cfg.TARGET_COLUMN]].copy()
    for i in range(n_components):
        pca_df[f"PC{i+1}"] = X_pca[:, i]
    pca_matrix_path = processed_dir / "pca_features.csv"
    pca_df.to_csv(pca_matrix_path, index=False)

    return {
        "pca": pca,
        "scaler": scaler,
        "imputer": imputer,
        "columns": cols,
        "X_pca": X_pca,
        "pca_df": pca_df,
        "explained_variance_path": var_path,
        "loadings_path": loadings_path,
        "pca_matrix_path": pca_matrix_path,
    }
