"""Generate exploratory yield visualizations and matching result tables.

Figures: visualizations/figures/
Tables:  results/  (CSV, one per chart plus full modeling table)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from wine_data import build_modeling_table, resolve_data_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = resolve_data_dir()
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

# Default focus vineyards; set to None for all CS blocks in POT.
VINEYARDS = ["ENZ", "ELR", "RMM"]

ID_COLS = ["harvest_year", "vineyard", "block"]
DOT_LABEL = "Each dot = one block in one harvest year"


def _annotate_dot_grain(ax: plt.Axes) -> None:
    ax.text(
        0.02,
        0.02,
        DOT_LABEL,
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
        va="bottom",
        ha="left",
        style="italic",
    )


def _save_figure(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_csv(df: pd.DataFrame, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


# --- 00: full modeling table ------------------------------------------------

def export_modeling_table(data: pd.DataFrame) -> Path:
    return _save_csv(data, "00_modeling_table")


# --- 01: yield by year -------------------------------------------------------

def table_yield_by_year(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["harvest_year", "vineyard"], as_index=False)["yield_actual"]
        .agg(
            n="count",
            mean_yield="mean",
            std_yield="std",
            min_yield="min",
            max_yield="max",
            median_yield="median",
        )
        .sort_values(["harvest_year", "vineyard"])
    )


def plot_yield_by_year(data: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=data, x="harvest_year", y="yield_actual", hue="vineyard", ax=ax)
    ax.set_title("Actual yield by harvest year and vineyard")
    ax.set_xlabel("Harvest year")
    ax.set_ylabel("Actual yield")
    ax.text(
        0.02,
        0.98,
        "Each box = block-year yields for that year and vineyard",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
        va="top",
        ha="left",
        style="italic",
    )
    ax.legend(title="Vineyard", bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save_figure(fig, "01_yield_by_year")


# --- 02: yield by vineyard ---------------------------------------------------

def table_yield_by_vineyard(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby("vineyard", as_index=False)["yield_actual"]
        .agg(
            n="count",
            mean_yield="mean",
            std_yield="std",
            min_yield="min",
            max_yield="max",
            median_yield="median",
        )
        .sort_values("mean_yield", ascending=False)
    )


def plot_yield_by_vineyard(summary: pd.DataFrame) -> Path:
    plot_df = summary.sort_values("mean_yield", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["vineyard"], plot_df["mean_yield"], color="steelblue")
    ax.set_title("Mean actual yield by vineyard")
    ax.set_xlabel("Mean actual yield")
    ax.set_ylabel("Vineyard")
    for y, row in enumerate(plot_df.itertuples()):
        ax.text(row.mean_yield, y, f" n={row.n}", va="center", fontsize=8, ha="left")
    return _save_figure(fig, "02_yield_by_vineyard")


# --- 03: predicted vs actual -------------------------------------------------

def table_predicted_vs_actual(data: pd.DataFrame) -> pd.DataFrame | None:
    if "yield_estimation" not in data.columns:
        return None
    cols = [c for c in ID_COLS + ["yield_estimation", "yield_actual"] if c in data.columns]
    return data[cols].dropna(subset=["yield_estimation", "yield_actual"])


def plot_predicted_vs_actual(subset: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(
        data=subset,
        x="yield_estimation",
        y="yield_actual",
        hue="vineyard",
        alpha=0.6,
        ax=ax,
    )
    lim = max(subset["yield_estimation"].max(), subset["yield_actual"].max())
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="Perfect prediction")
    ax.set_title("Predicted vs actual yield (POT baseline)")
    ax.set_xlabel("Predicted yield (estimation)")
    ax.set_ylabel("Actual yield")
    _annotate_dot_grain(ax)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save_figure(fig, "03_predicted_vs_actual")


# --- 04: prediction error ----------------------------------------------------

def table_prediction_error(data: pd.DataFrame) -> pd.DataFrame | None:
    if "yield_estimation" not in data.columns:
        return None
    cols = [c for c in ID_COLS + ["yield_estimation", "yield_actual"] if c in data.columns]
    out = data[cols].dropna(subset=["yield_estimation", "yield_actual"]).copy()
    out["prediction_error"] = out["yield_actual"] - out["yield_estimation"]
    out["abs_prediction_error"] = out["prediction_error"].abs()
    return out


def table_prediction_error_summary(error_table: pd.DataFrame) -> pd.DataFrame:
    errors = error_table["prediction_error"]
    return pd.DataFrame(
        {
            "metric": ["count", "mean", "std", "median", "min", "max", "mae", "rmse"],
            "value": [
                len(errors),
                errors.mean(),
                errors.std(),
                errors.median(),
                errors.min(),
                errors.max(),
                error_table["abs_prediction_error"].mean(),
                np.sqrt((errors**2).mean()),
            ],
        }
    )


def plot_prediction_error(error_table: pd.DataFrame) -> Path:
    errors = error_table["prediction_error"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=30, color="coral", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(
        errors.mean(),
        color="navy",
        linestyle="-",
        linewidth=1,
        label=f"Mean error: {errors.mean():.2f}",
    )
    ax.set_title("Prediction error (actual − predicted)")
    ax.set_xlabel("Error")
    ax.set_ylabel("Count of block-years")
    ax.text(
        0.02,
        0.98,
        "Each block-year = one block in one harvest year",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
        va="top",
        ha="left",
        style="italic",
    )
    ax.legend()
    return _save_figure(fig, "04_prediction_error")


# --- 05: previous year vs actual ---------------------------------------------

def table_prev_year_vs_actual(data: pd.DataFrame) -> pd.DataFrame | None:
    if "yield_prev_year" not in data.columns:
        return None
    cols = [c for c in ID_COLS + ["yield_prev_year", "yield_actual"] if c in data.columns]
    out = data[cols].dropna(subset=["yield_prev_year", "yield_actual"])
    return out if not out.empty else None


def plot_prev_year_vs_actual(subset: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(
        data=subset,
        x="yield_prev_year",
        y="yield_actual",
        hue="vineyard",
        alpha=0.6,
        ax=ax,
    )
    ax.set_title("Previous-year yield vs current actual yield")
    ax.set_xlabel("Previous year actual yield")
    ax.set_ylabel("Current actual yield")
    _annotate_dot_grain(ax)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save_figure(fig, "05_prev_year_vs_actual")


# --- 06: weather vs yield ----------------------------------------------------

def _weather_feature_cols(data: pd.DataFrame) -> list[str]:
    preferred = [
        "weather_gdd_from_mar_max",
        "weather_rain_daily_sum",
        "weather_temp_max_mean",
        "weather_vpd_max_mean",
    ]
    weather_cols = [
        c for c in data.columns if c.startswith("weather_") and c != "weather_station"
    ]
    cols = [c for c in preferred if c in weather_cols]
    return cols if cols else weather_cols[:4]


def table_weather_vs_yield(data: pd.DataFrame) -> pd.DataFrame | None:
    weather_cols = _weather_feature_cols(data)
    if not weather_cols:
        return None
    cols = [c for c in ID_COLS + ["yield_actual"] if c in data.columns] + weather_cols
    return data[cols].dropna(subset=["yield_actual"] + weather_cols, how="any")


def plot_weather_vs_yield(subset: pd.DataFrame) -> Path:
    weather_cols = [c for c in subset.columns if c.startswith("weather_")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for ax, col in zip(axes, weather_cols[:4]):
        plot_df = subset.dropna(subset=[col])
        sns.scatterplot(data=plot_df, x=col, y="yield_actual", alpha=0.5, ax=ax)
        label = col.removeprefix("weather_").replace("_", " ")
        ax.set_xlabel(label)
        ax.set_ylabel("Actual yield")
        ax.set_title(label)
        _annotate_dot_grain(ax)
    for ax in axes[len(weather_cols[:4]) :]:
        ax.set_visible(False)
    fig.suptitle("Weather summaries vs actual yield", y=1.02)
    fig.tight_layout()
    return _save_figure(fig, "06_weather_vs_yield")


# --- 07: cluster vs yield ----------------------------------------------------

def table_cluster_vs_yield(data: pd.DataFrame) -> pd.DataFrame | None:
    cluster_cols = [c for c in ("cluster_count", "cluster_weight") if c in data.columns]
    if not cluster_cols:
        return None
    cols = [c for c in ID_COLS + ["yield_actual"] if c in data.columns] + cluster_cols
    out = data[cols].dropna(subset=["yield_actual"])
    mask = pd.Series(True, index=out.index)
    for col in cluster_cols:
        mask &= out[col].isna() | (out[col] > 0)
    out = out.loc[mask]
    return out if not out.empty else None


def plot_cluster_vs_yield(subset: pd.DataFrame) -> Path:
    cluster_cols = [c for c in ("cluster_count", "cluster_weight") if c in subset.columns]
    fig, axes = plt.subplots(1, len(cluster_cols), figsize=(6 * len(cluster_cols), 5))
    if len(cluster_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cluster_cols):
        plot_df = subset.dropna(subset=[col])
        sns.scatterplot(data=plot_df, x=col, y="yield_actual", alpha=0.5, ax=ax)
        ax.set_title(f"{col.replace('_', ' ').title()} vs yield")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Actual yield")
        _annotate_dot_grain(ax)
    fig.tight_layout()
    return _save_figure(fig, "07_cluster_vs_yield")


# --- 08: correlation with yield ----------------------------------------------

def table_correlation_with_yield(data: pd.DataFrame) -> pd.DataFrame:
    numeric = data.select_dtypes(include=[np.number])
    if "yield_actual" not in numeric.columns:
        corr = numeric.corr(numeric_only=True)
        out = corr.stack().reset_index()
        out.columns = ["feature_a", "feature_b", "correlation"]
        return out.sort_values("correlation", key=abs, ascending=False)

    raw_corr = numeric.corr(numeric_only=True)["yield_actual"].drop("yield_actual", errors="ignore")
    out = (
        raw_corr.dropna()
        .reset_index()
        .rename(columns={"index": "feature", "yield_actual": "correlation_with_yield"})
    )
    out["abs_correlation"] = out["correlation_with_yield"].abs()
    return out.sort_values("abs_correlation", ascending=False)


def plot_correlation_heatmap(corr_table: pd.DataFrame, data: pd.DataFrame) -> Path:
    numeric = data.select_dtypes(include=[np.number])
    if "correlation_with_yield" in corr_table.columns:
        top = corr_table.head(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            top["feature"],
            top["correlation_with_yield"],
            color=[
                "#d73027" if v < 0 else "#4575b4" for v in top["correlation_with_yield"]
            ],
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Top 20 features correlated with actual yield")
        ax.set_xlabel("Pearson correlation with yield")
    else:
        corr = numeric.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, linewidths=0.3)
        ax.set_title("Feature correlation heatmap")
    fig.tight_layout()
    return _save_figure(fig, "08_correlation_heatmap")


# --- main --------------------------------------------------------------------

def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    kwargs: dict = {"include_predicted_yield": True}
    if VINEYARDS:
        kwargs["vineyards"] = VINEYARDS

    data = build_modeling_table(DATA_DIR, verbose=False, **kwargs)
    if data.empty:
        raise ValueError("Modeling table is empty — check data/ and vineyard filters.")

    print(f"Using data from: {DATA_DIR}")

    saved_figures: list[Path] = []
    saved_tables: list[Path] = []

    saved_tables.append(export_modeling_table(data))

    t01 = table_yield_by_year(data)
    saved_tables.append(_save_csv(t01, "01_yield_by_year"))
    saved_figures.append(plot_yield_by_year(data))

    t02 = table_yield_by_vineyard(data)
    saved_tables.append(_save_csv(t02, "02_yield_by_vineyard"))
    saved_figures.append(plot_yield_by_vineyard(t02))

    t03 = table_predicted_vs_actual(data)
    if t03 is not None:
        saved_tables.append(_save_csv(t03, "03_predicted_vs_actual"))
        saved_figures.append(plot_predicted_vs_actual(t03))

    t04 = table_prediction_error(data)
    if t04 is not None:
        saved_tables.append(_save_csv(t04, "04_prediction_error"))
        saved_tables.append(_save_csv(table_prediction_error_summary(t04), "04_prediction_error_summary"))
        saved_figures.append(plot_prediction_error(t04))

    t05 = table_prev_year_vs_actual(data)
    if t05 is not None:
        saved_tables.append(_save_csv(t05, "05_prev_year_vs_actual"))
        saved_figures.append(plot_prev_year_vs_actual(t05))

    t06 = table_weather_vs_yield(data)
    if t06 is not None:
        saved_tables.append(_save_csv(t06, "06_weather_vs_yield"))
        saved_figures.append(plot_weather_vs_yield(t06))

    t07 = table_cluster_vs_yield(data)
    if t07 is not None:
        saved_tables.append(_save_csv(t07, "07_cluster_vs_yield"))
        saved_figures.append(plot_cluster_vs_yield(t07))

    t08 = table_correlation_with_yield(data)
    saved_tables.append(_save_csv(t08, "08_correlation_with_yield"))
    saved_figures.append(plot_correlation_heatmap(t08, data))

    print(f"Loaded {len(data):,} block-year rows")
    if VINEYARDS:
        print(f"Vineyards: {', '.join(VINEYARDS)}")
    print(f"\nSaved {len(saved_figures)} figures to {FIGURES_DIR}/")
    for path in saved_figures:
        print(f"  {path.name}")
    print(f"\nSaved {len(saved_tables)} tables to {RESULTS_DIR}/")
    for path in saved_tables:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
