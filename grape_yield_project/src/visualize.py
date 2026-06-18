"""Generate model performance visualizations: metrics, convergence, predicted vs actual."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from .evaluate import compute_metrics

MODEL_LABELS = {
    "rf": "Random Forest",
    "pca_rf": "PCA + RF",
    "xgboost": "XGBoost",
    "pg_an": "PG-AN",
    "pg_gnn": "PG-GNN",
}

SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
METRIC_NAMES = ["rmse", "mae", "r2", "correlation"]


def _load_metrics_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_split_metrics(payload: dict) -> dict[str, dict[str, float]]:
    """Convert flat train_rmse keys into nested {split: {metric: value}}."""
    out: dict[str, dict[str, float]] = {s: {} for s in ("train", "val", "test")}
    for key, val in payload.items():
        if key == "model":
            continue
        for split in out:
            prefix = f"{split}_"
            if key.startswith(prefix):
                metric = key[len(prefix):]
                out[split][metric] = val
    return out


def load_all_model_metrics(metrics_dir: Path | None = None) -> dict[str, dict[str, dict[str, float]]]:
    metrics_dir = metrics_dir or cfg.METRICS
    all_metrics = {}
    for model_key in MODEL_LABELS:
        path = metrics_dir / f"{model_key}_metrics.json"
        payload = _load_metrics_json(path)
        if payload:
            all_metrics[model_key] = _parse_split_metrics(payload)
    return all_metrics


def plot_metrics_comparison(
    all_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """Grouped bar chart: RMSE and R² across models and splits."""
    all_metrics = all_metrics or load_all_model_metrics()
    if not all_metrics:
        raise FileNotFoundError(f"No metrics JSON files found in {cfg.METRICS}")

    models = list(all_metrics.keys())
    splits = ["train", "val", "test"]
    x = np.arange(len(models))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, split in enumerate(splits):
        rmse_vals = [all_metrics[m][split].get("rmse", np.nan) for m in models]
        axes[0].bar(x + i * width, rmse_vals, width, label=split.capitalize(), color=SPLIT_COLORS[split])
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=15, ha="right")
    axes[0].set_ylabel("RMSE (t/d)")
    axes[0].set_title("RMSE by model and split")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    for i, split in enumerate(splits):
        r2_vals = [all_metrics[m][split].get("r2", np.nan) for m in models]
        axes[1].bar(x + i * width, r2_vals, width, label=split.capitalize(), color=SPLIT_COLORS[split])
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=15, ha="right")
    axes[1].set_ylabel("R²")
    axes[1].set_title("R² by model and split")
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Model performance comparison (year-based splits)", fontsize=13, y=1.02)
    fig.tight_layout()
    output_path = Path(output_path or cfg.FIGURES / "model_metrics_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_test_metrics_summary(
    all_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """Focus chart on test-set metrics only."""
    all_metrics = all_metrics or load_all_model_metrics()
    models = list(all_metrics.keys())
    metrics_to_plot = ["rmse", "mae", "r2", "correlation"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for ax, metric in zip(axes, metrics_to_plot):
        vals = [all_metrics[m]["test"].get(metric, np.nan) for m in models]
        bars = ax.bar([MODEL_LABELS.get(m, m) for m in models], vals, color=colors)
        ax.set_title(f"Test {metric.upper()}" if metric != "r2" else "Test R²")
        ax.set_ylabel(metric.upper() if metric != "r2" else "R²")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle("Test set performance (2025)", fontsize=13)
    fig.tight_layout()
    output_path = Path(output_path or cfg.FIGURES / "model_test_metrics.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_predicted_vs_actual(
    predictions_dir: Path | None = None,
    output_path: Path | str | None = None,
) -> Path | None:
    """Scatter plots of predicted vs actual yield per model (test split)."""
    predictions_dir = predictions_dir or cfg.METRICS
    pred_files = sorted(predictions_dir.glob("*_predictions.csv"))
    if not pred_files:
        return None

    n_models = len(pred_files)
    ncols = min(3, n_models)
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, pf in zip(axes, pred_files):
        df = pd.read_csv(pf)
        model_name = pf.stem.replace("_predictions", "")
        test = df[df["split"] == "test"]
        if test.empty:
            test = df
        y_true = test["y_true"].values
        y_pred = test["y_pred"].values
        m = compute_metrics(y_true, y_pred)

        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color=SPLIT_COLORS["test"])
        lims = [0, max(y_true.max(), y_pred.max()) * 1.05]
        ax.plot(lims, lims, "k--", linewidth=1, label="Perfect")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual yield (t/d)")
        ax.set_ylabel("Predicted yield (t/d)")
        ax.set_title(
            f"{MODEL_LABELS.get(model_name, model_name)} — Test\n"
            f"R²={m['r2']:.2f}, RMSE={m['rmse']:.2f}"
        )
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes[len(pred_files):]:
        ax.set_visible(False)

    fig.suptitle("Predicted vs actual yield (test set)", fontsize=13)
    fig.tight_layout()
    output_path = Path(output_path or cfg.FIGURES / "model_predicted_vs_actual.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_convergence(
    history_paths: dict[str, Path] | None = None,
    output_path: Path | str | None = None,
) -> Path | None:
    """Plot training/validation loss and RMSE curves from history JSON files."""
    if history_paths is None:
        history_paths = {}
        for model in ("pg_an", "pg_gnn", "xgboost", "rf", "pca_rf"):
            p = cfg.METRICS / f"{model}_history.json"
            if p.exists():
                history_paths[model] = p

    if not history_paths:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for model_key, path in history_paths.items():
        with open(path, encoding="utf-8") as f:
            hist = json.load(f)
        epochs = hist.get("epoch", list(range(1, len(hist.get("train_loss", [])) + 1)))
        label = MODEL_LABELS.get(model_key, model_key)

        if "train_loss" in hist:
            axes[0].plot(epochs, hist["train_loss"], label=f"{label} train")
        if "val_loss" in hist:
            axes[0].plot(epochs, hist["val_loss"], linestyle="--", label=f"{label} val")

        if "train_rmse" in hist:
            axes[1].plot(epochs, hist["train_rmse"], label=f"{label} train")
        if "val_rmse" in hist:
            axes[1].plot(epochs, hist["val_rmse"], linestyle="--", label=f"{label} val")
        if "test_rmse" in hist and len(hist["test_rmse"]) == len(epochs):
            axes[1].plot(epochs, hist["test_rmse"], linestyle=":", label=f"{label} test")

    axes[0].set_xlabel("Epoch / iteration")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training convergence — loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch / iteration")
    axes[1].set_ylabel("RMSE (t/d)")
    axes[1].set_title("Training convergence — RMSE")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Model training convergence", fontsize=13)
    fig.tight_layout()
    output_path = Path(output_path or cfg.FIGURES / "model_training_convergence.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_rf_learning_curve(
    output_path: Path | str | None = None,
) -> Path | None:
    """Plot RF / PCA-RF learning curves (n_estimators vs RMSE)."""
    curves = []
    for name in ("rf", "pca_rf"):
        path = cfg.METRICS / f"{name}_learning_curve.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                curves.append(json.load(f))
    if not curves:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    for hist in curves:
        label_base = MODEL_LABELS.get(hist.get("model", "rf"), "RF")
        steps = hist.get("n_estimators", [])
        for key, style, split in [("train_rmse", "-", "train"), ("val_rmse", "--", "val"), ("test_rmse", ":", "test")]:
            if key in hist:
                ax.plot(steps, hist[key], style, label=f"{label_base} {split}")
    ax.set_xlabel("Number of trees (n_estimators)")
    ax.set_ylabel("RMSE (t/d)")
    ax.set_title("Random Forest learning curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path = Path(output_path or cfg.FIGURES / "rf_learning_curve.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_split_comparison_heatmap(
    all_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """Heatmap of R² across models and splits."""
    all_metrics = all_metrics or load_all_model_metrics()
    models = list(all_metrics.keys())
    splits = ["train", "val", "test"]
    data = np.array([[all_metrics[m][s].get("r2", np.nan) for s in splits] for m in models])

    fig, ax = plt.subplots(figsize=(7, max(3, len(models) * 0.6 + 1)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=1.0)
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models])
    for i in range(len(models)):
        for j in range(len(splits)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("R² heatmap — train / val / test")
    fig.colorbar(im, ax=ax, label="R²")
    fig.tight_layout()
    output_path = Path(output_path or cfg.FIGURES / "model_r2_heatmap.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_predicted_vs_actual_all_splits(
    predictions_dir: Path | None = None,
    output_path: Path | str | None = None,
) -> Path | None:
    """Faceted predicted vs actual for train/val/test per model."""
    predictions_dir = predictions_dir or cfg.METRICS
    pred_files = sorted(predictions_dir.glob("*_predictions.csv"))
    if not pred_files:
        return None

    splits = ["train", "val", "test"]
    n_models = len(pred_files)
    fig, axes = plt.subplots(n_models, 3, figsize=(14, 4.2 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    for row, pf in enumerate(pred_files):
        df = pd.read_csv(pf)
        model_name = pf.stem.replace("_predictions", "")
        for col, split in enumerate(splits):
            ax = axes[row, col]
            sub = df[df["split"] == split]
            if sub.empty:
                ax.set_visible(False)
                continue
            y_true = sub["y_true"].values
            y_pred = sub["y_pred"].values
            m = compute_metrics(y_true, y_pred)
            ax.scatter(y_true, y_pred, alpha=0.45, s=18, color=SPLIT_COLORS[split])
            lim = max(y_true.max(), y_pred.max(), 1) * 1.05
            ax.plot([0, lim], [0, lim], "k--", linewidth=1)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"{MODEL_LABELS.get(model_name, model_name)} — {split}\nR²={m['r2']:.2f} RMSE={m['rmse']:.2f}")
            ax.grid(alpha=0.3)

    fig.suptitle("Predicted vs actual yield — all splits", fontsize=13)
    fig.tight_layout()
    output_path = Path(output_path or cfg.FIGURES / "model_predicted_vs_actual_all_splits.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_all_figures(metrics_dir: Path | None = None) -> dict[str, Path | None]:
    """Generate all available performance figures."""
    metrics_dir = metrics_dir or cfg.METRICS
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    all_metrics = load_all_model_metrics(metrics_dir)

    outputs = {
        "metrics_comparison": plot_metrics_comparison(all_metrics),
        "test_metrics": plot_test_metrics_summary(all_metrics),
        "r2_heatmap": plot_split_comparison_heatmap(all_metrics),
        "predicted_vs_actual": plot_predicted_vs_actual(metrics_dir),
        "predicted_vs_actual_all_splits": plot_predicted_vs_actual_all_splits(metrics_dir),
        "convergence": plot_convergence(),
        "rf_learning_curve": plot_rf_learning_curve(),
    }
    return outputs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate model performance visualizations")
    parser.add_argument("--metrics_dir", type=str, default=str(cfg.METRICS))
    args = parser.parse_args()
    outputs = generate_all_figures(Path(args.metrics_dir))
    print("Generated figures:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
