"""Unified training entry point for baselines and grape PG-AN / PG-GNN models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import config as cfg
from .data_loading import load_merged_vine_year_table
from .evaluate import compute_metrics, save_metrics, save_predictions, save_training_history
from .graph_builder import GraphStrategy, build_edge_index
from .grape_pg_an import GrapePGAN, GrapePhysicsLoss, PhysicsLossConfig, split_features_by_group

try:
    from .grape_pg_gnn import GrapeGraphSAGE
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
from .pca_baseline import run_pca_baseline
from .preprocessing import get_feature_group_indices, preprocess_dataframe, year_splits
from .random_forest_baseline import train_random_forest
from .xgboost_baseline import HAS_XGBOOST, train_xgboost


def _prepare_splits(df):
    train_df, val_df, test_df = year_splits(df)
    prep = preprocess_dataframe(train_df, fit=True)
    raw_cols = prep.metadata["raw_feature_columns"]
    val_prep = preprocess_dataframe(val_df, fit=False, preprocessor=prep.preprocessor, feature_columns=raw_cols)
    test_prep = preprocess_dataframe(test_df, fit=False, preprocessor=prep.preprocessor, feature_columns=raw_cols)
    return prep, val_prep, test_prep, train_df, val_df, test_df


def _group_dims_from_features(feature_names, group_indices):
    dims = {}
    for g, idxs in group_indices.items():
        dims[g] = len(idxs)
    # ensure at least one dim
    if sum(dims.values()) == 0:
        dims["all"] = len(feature_names)
        group_indices = {"all": list(range(len(feature_names)))}
    return dims, group_indices


def _tensor_groups(X: np.ndarray, group_indices: dict) -> dict[str, torch.Tensor]:
    t = torch.tensor(X, dtype=torch.float32)
    return split_features_by_group(t, group_indices)


def train_pg_an(df, epochs: int = 100, lr: float = 1e-3, batch_size: int = 64) -> dict:
    prep, val_prep, test_prep, train_df, val_df, test_df = _prepare_splits(df)
    feat = prep.feature_columns
    group_idx = get_feature_group_indices(feat)
    group_dims, group_idx = _group_dims_from_features(feat, group_idx)

    X_train = prep.df[feat].values.astype(np.float32)
    y_train = prep.df[cfg.TARGET_COLUMN].values.astype(np.float32).reshape(-1, 1)
    X_val = val_prep.df[feat].values.astype(np.float32)
    y_val = val_prep.df[cfg.TARGET_COLUMN].values.astype(np.float32).reshape(-1, 1)
    X_test = test_prep.df[feat].values.astype(np.float32)
    y_test = test_prep.df[cfg.TARGET_COLUMN].values.astype(np.float32).reshape(-1, 1)

    prev_train = (
        train_df[cfg.COL_PREV_ACTUAL].fillna(train_df[cfg.COL_ACTUAL_YIELD].median()).values.astype(np.float32).reshape(-1, 1)
        if cfg.COL_PREV_ACTUAL in train_df.columns
        else np.zeros_like(y_train)
    )
    shoots_train = (
        train_df[cfg.COL_SHOOTS].fillna(0).values.astype(np.float32).reshape(-1, 1)
        if cfg.COL_SHOOTS in train_df.columns
        else np.zeros_like(y_train)
    )
    canopy_train = (
        train_df[cfg.COL_CANOPY].fillna(0).values.astype(np.float32).reshape(-1, 1)
        if cfg.COL_CANOPY in train_df.columns
        else np.zeros_like(y_train)
    )

    model = GrapePGAN(group_dims)
    loss_fn = GrapePhysicsLoss(PhysicsLossConfig())
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    y_mean, y_std = float(y_train.mean()), max(float(y_train.std()), 1e-6)
    y_train_n = (y_train - y_mean) / y_std

    ds = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(prev_train),
        torch.tensor(y_train_n),
        torch.tensor(shoots_train),
        torch.tensor(canopy_train),
    )
    loader = DataLoader(ds, batch_size=min(batch_size, len(X_train)), shuffle=True)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_rmse": [], "val_rmse": [], "test_rmse": []}
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for xb, prev_b, yb, shoots_b, canopy_b in loader:
            g = split_features_by_group(xb, group_idx)
            opt.zero_grad()
            pred_n = model(g)
            pred = pred_n * y_std + y_mean
            y_true = yb * y_std + y_mean
            loss, _ = loss_fn(y_true, pred, prev_b * y_std + y_mean, shoots_b, canopy_b)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            tr_pred = model(_tensor_groups(torch.tensor(X_train), group_idx)).numpy() * y_std + y_mean
            va_pred = model(_tensor_groups(torch.tensor(X_val), group_idx)).numpy() * y_std + y_mean
            te_pred = model(_tensor_groups(torch.tensor(X_test), group_idx)).numpy() * y_std + y_mean
        model.train()

        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_loss"].append(float(np.mean((va_pred - y_val) ** 2)))
        history["train_rmse"].append(compute_metrics(y_train, tr_pred)["rmse"])
        history["val_rmse"].append(compute_metrics(y_val, va_pred)["rmse"])
        history["test_rmse"].append(compute_metrics(y_test, te_pred)["rmse"])

    def predict(X):
        model.eval()
        with torch.no_grad():
            pred_n = model(_tensor_groups(X, group_idx)).numpy()
        return pred_n * y_std + y_mean

    metrics = {}
    predictions = {}
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        pred = predict(X)
        metrics[name] = compute_metrics(y, pred)
        predictions[name] = (y, pred)

    ckpt = {
        "model_state": model.state_dict(),
        "group_dims": group_dims,
        "group_idx": group_idx,
        "features": feat,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    path = cfg.MODELS / "pg_an.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    flat = {f"{s}_{k}": v for s, m in metrics.items() for k, v in m.items()}
    save_metrics(flat, cfg.METRICS / "pg_an_metrics.json", "pg_an")
    save_predictions(predictions, cfg.METRICS / "pg_an_predictions.csv", "pg_an")
    save_training_history(history, cfg.METRICS / "pg_an_history.json", "pg_an")
    return {"model": model, "metrics": metrics, "model_path": path, "history": history}


def train_pg_gnn(
    df,
    epochs: int = 100,
    lr: float = 1e-3,
    graph_strategy: str = GraphStrategy.HYBRID,
) -> dict:
    prep, val_prep, test_prep, train_df, val_df, test_df = _prepare_splits(df)
    feat = prep.feature_columns
    group_idx = get_feature_group_indices(feat)
    group_dims, group_idx = _group_dims_from_features(feat, group_idx)

    def run_split(split_prep, split_df):
        X = split_prep.df[feat].values
        y = split_prep.df[cfg.TARGET_COLUMN].values.reshape(-1, 1)
        edge_index = build_edge_index(split_df.reset_index(drop=True), strategy=graph_strategy)
        return X, y, edge_index

    X_train, y_train, edge_train = run_split(prep, train_df)
    X_val, y_val, edge_val = run_split(val_prep, val_df)
    X_test, y_test, edge_test = run_split(test_prep, test_df)

    model = GrapeGraphSAGE(group_dims)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    y_mean, y_std = float(y_train.mean()), max(float(y_train.std()), 1e-6)
    y_train_n = (y_train - y_mean) / y_std
    y_t = torch.tensor(y_train_n, dtype=torch.float32)

    def predict(X, edge):
        model.eval()
        with torch.no_grad():
            pred_n = model(_tensor_groups(X, group_idx), edge).numpy()
        return pred_n * y_std + y_mean

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_rmse": [], "val_rmse": [], "test_rmse": []}
    model.train()
    g_train = _tensor_groups(X_train, group_idx)
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        pred_n = model(g_train, edge_train)
        loss = torch.nn.functional.mse_loss(pred_n, y_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            tr_pred = predict(X_train, edge_train)
            va_pred = predict(X_val, edge_val)
            te_pred = predict(X_test, edge_test)
        model.train()

        history["epoch"].append(epoch)
        history["train_loss"].append(float(loss.detach()))
        history["val_loss"].append(float(np.mean((va_pred - y_val) ** 2)))
        history["train_rmse"].append(compute_metrics(y_train, tr_pred)["rmse"])
        history["val_rmse"].append(compute_metrics(y_val, va_pred)["rmse"])
        history["test_rmse"].append(compute_metrics(y_test, te_pred)["rmse"])

    metrics = {}
    predictions = {}
    for name, X, y, edge in [
        ("train", X_train, y_train, edge_train),
        ("val", X_val, y_val, edge_val),
        ("test", X_test, y_test, edge_test),
    ]:
        pred = predict(X, edge)
        metrics[name] = compute_metrics(y, pred)
        predictions[name] = (y, pred)

    ckpt = {
        "model_state": model.state_dict(),
        "group_dims": group_dims,
        "group_idx": group_idx,
        "features": feat,
        "graph_strategy": graph_strategy,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    path = cfg.MODELS / "pg_gnn.pt"
    torch.save(ckpt, path)
    flat = {f"{s}_{k}": v for s, m in metrics.items() for k, v in m.items()}
    save_metrics(flat, cfg.METRICS / "pg_gnn_metrics.json", "pg_gnn")
    save_predictions(predictions, cfg.METRICS / "pg_gnn_predictions.csv", "pg_gnn")
    save_training_history(history, cfg.METRICS / "pg_gnn_history.json", "pg_gnn")
    return {"model": model, "metrics": metrics, "model_path": path, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Train grape yield models")
    parser.add_argument(
        "--model_type",
        choices=["rf", "pca_rf", "xgboost", "pg_an", "pg_gnn", "pca"],
        default="rf",
    )
    parser.add_argument("--raw_dir", type=str, default=str(cfg.DEFAULT_RAW_DATA_DIR))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--graph_strategy", type=str, default=GraphStrategy.HYBRID.value)
    args = parser.parse_args()

    df = load_merged_vine_year_table(raw_dir=args.raw_dir, save_path=cfg.DATA_PROCESSED / "vine_year_merged.csv")
    print(f"Loaded {len(df)} vine-year rows")

    if args.model_type == "pca":
        out = run_pca_baseline(df)
        print("PCA saved to", out["pca_matrix_path"])
    elif args.model_type == "rf":
        out = train_random_forest(df, use_pca=False)
        print(out["metrics"])
    elif args.model_type == "pca_rf":
        out = train_random_forest(df, use_pca=True)
        print(out["metrics"])
    elif args.model_type == "xgboost":
        if not HAS_XGBOOST:
            raise SystemExit("Install xgboost: pip install xgboost")
        out = train_xgboost(df)
        print(out["metrics"])
    elif args.model_type == "pg_an":
        out = train_pg_an(df, epochs=args.epochs)
        print(out["metrics"])
    elif args.model_type == "pg_gnn":
        if not HAS_PYG:
            raise SystemExit("Install PyTorch Geometric: pip install torch-geometric")
        out = train_pg_gnn(df, epochs=args.epochs, graph_strategy=args.graph_strategy)
        print(out["metrics"])


if __name__ == "__main__":
    main()
