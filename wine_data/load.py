"""Load and combine Cabernet Sauvignon yield data from Excel workbooks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from wine_data.schema import (
    CLUSTER_RENAME,
    HARVEST_RENAME,
    POT_RENAME,
    VARIETY_CS,
    VINE_ARCHIVE_RENAME,
    WEATHER_RENAME,
)

DEFAULT_DATA_DIR = Path("data")
DEFAULT_SHEET = "DataSheet"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _excel_files(data_dir: Path) -> list[Path]:
    """Return non-test ``.xlsx`` workbooks in ``data_dir``."""
    return sorted(
        f for f in data_dir.glob("*.xlsx") if not f.name.startswith("test_")
    )


def resolve_data_dir(preferred: Path | str | None = None) -> Path:
    """Find a directory that contains yield Excel workbooks.

    Checks, in order:
    1. ``preferred`` (default: ``<project>/data``)
    2. ``<project>/grape_yield_project/data/raw``
    3. ``<project>/data`` again via absolute path
    """
    root = _project_root()
    candidates = [
        Path(preferred) if preferred is not None else root / "data",
        root / "grape_yield_project" / "data" / "raw",
        root / "data",
    ]
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _excel_files(resolved):
            return resolved
    searched = ", ".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(
        f"No .xlsx files found. Place workbooks in data/ or "
        f"grape_yield_project/data/raw/. Searched: {searched}"
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names and strip whitespace."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _detect_dataset_kind(columns: Iterable[str]) -> str:
    cols = {str(c).strip() for c in columns}
    lower = {c.lower() for c in cols}
    if "YIELD ESTIMATION" in cols or "yield_estimation" in lower:
        return "yield_pot"
    if "WEATHER STATION" in cols or "weather_station" in lower:
        return "vine"
    if "קוד תחנה" in cols or "station" in lower:
        return "weather"
    if "CLUSTER COUNT" in cols or "cluster_count" in lower:
        return "cluster"
    if "כרם" in cols:
        return "harvest"
    return "unknown"


def load_excel_files(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    *,
    sheet_name: str | int | None = DEFAULT_SHEET,
    verbose: bool = True,
) -> list[pd.DataFrame]:
    """Load every ``.xlsx`` file in ``data_dir``, tagging rows with ``source_file``."""
    data_dir = Path(data_dir)
    excel_files = _excel_files(data_dir)
    if not excel_files:
        raise FileNotFoundError(f"No .xlsx files found in {data_dir.resolve()}")

    tables: list[pd.DataFrame] = []
    for file in excel_files:
        xls = pd.ExcelFile(file)
        if verbose:
            print(file.name, xls.sheet_names)

        sheet = sheet_name if sheet_name is not None else 0
        if isinstance(sheet, str) and sheet not in xls.sheet_names:
            sheet = xls.sheet_names[0]

        df = pd.read_excel(file, sheet_name=sheet)
        df["source_file"] = file.name
        tables.append(df)

    return tables


def classify_and_load(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    *,
    sheet_name: str | int | None = DEFAULT_SHEET,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load workbooks and split them into typed tables (yield, vine, weather, etc.)."""
    data_dir = Path(data_dir)
    excel_files = _excel_files(data_dir)
    if not excel_files:
        raise FileNotFoundError(f"No .xlsx files found in {data_dir.resolve()}")

    buckets: dict[str, list[pd.DataFrame]] = {
        "yield_pot": [],
        "vine": [],
        "weather": [],
        "cluster": [],
        "harvest": [],
        "unknown": [],
    }

    for file in excel_files:
        xls = pd.ExcelFile(file)
        if verbose:
            print(file.name, xls.sheet_names)

        sheet = sheet_name if sheet_name is not None else 0
        if isinstance(sheet, str) and sheet not in xls.sheet_names:
            sheet = xls.sheet_names[0]

        df = pd.read_excel(file, sheet_name=sheet)
        df["source_file"] = file.name
        kind = _detect_dataset_kind(df.columns)
        buckets[kind].append(df)

    result: dict[str, pd.DataFrame] = {}
    for kind, frames in buckets.items():
        if not frames:
            continue
        result[kind] = pd.concat(frames, ignore_index=True)

    return result


def _prepare_yield_pot(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={k: v for k, v in POT_RENAME.items() if k in df.columns})
    return _normalize_columns(out)


def _prepare_vine_archive(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(
        columns={k: v for k, v in VINE_ARCHIVE_RENAME.items() if k in df.columns}
    )
    out = _normalize_columns(out)
    out = out[out["block"].notna() & (out["block"] != "*")]
    return out.drop_duplicates(subset=["block"], keep="first")


def _prepare_weather(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={k: v for k, v in WEATHER_RENAME.items() if k in df.columns})
    out = _normalize_columns(out)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["station"] = out["station"].astype(str).str.strip()
    return out


def _prepare_cluster(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={k: v for k, v in CLUSTER_RENAME.items() if k in df.columns})
    out = _normalize_columns(out)
    out = out[(out["harvest_year"] > 0) & (out["variety"] == VARIETY_CS)]
    return out


def _prepare_harvest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={k: v for k, v in HARVEST_RENAME.items() if k in df.columns})
    return _normalize_columns(out)


def aggregate_weather_by_year(
    weather: pd.DataFrame,
    *,
    year_col: str = "harvest_year",
) -> pd.DataFrame:
    """Summarize daily weather to one row per station and calendar year."""
    w = weather.copy()
    w[year_col] = w["date"].dt.year

    agg_spec = {
        "temp_max": "mean",
        "temp_min": "mean",
        "vpd_max": "mean",
        "rain_daily": "sum",
        "gdd_from_mar": "max",
        "solar_rad": "mean",
        "cold_hours": "sum",
    }
    present = {col: fn for col, fn in agg_spec.items() if col in w.columns}
    grouped = w.groupby(["station", year_col], as_index=False).agg(present)
    rename = {col: f"weather_{col}_{fn}" for col, fn in present.items()}
    return grouped.rename(columns=rename)


def add_previous_year_yield(yield_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``yield_prev_year`` from lagged actual yield at block level."""
    out = yield_df.sort_values(["block", "harvest_year"]).copy()
    out["yield_prev_year"] = out.groupby("block")["yield_actual"].shift(1)
    return out


def build_modeling_table(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    *,
    variety: str = VARIETY_CS,
    vineyards: Iterable[str] | None = None,
    years: Iterable[int] | None = None,
    include_predicted_yield: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Combine yield, vine, cluster, and weather data into one modeling table.

    Parameters
    ----------
    variety:
        Grape variety code (default ``CS`` for Cabernet Sauvignon).
    vineyards:
        Optional vineyard codes to keep (e.g. ``["ENZ", "ELR", "RMM"]``).
    years:
        Optional harvest years to keep (default 2020–2025 present in data).
    include_predicted_yield:
        If False (default), ``yield_estimation`` is dropped so it is not used
        as a model feature. Keep it for baseline comparison outside the model.
    """
    tables = classify_and_load(data_dir, verbose=verbose)

    if "yield_pot" not in tables:
        raise ValueError("POT yield workbook not found in data directory.")

    yield_df = _prepare_yield_pot(tables["yield_pot"])
    yield_df = yield_df[yield_df["variety"] == variety]
    yield_df = add_previous_year_yield(yield_df)

    if vineyards is not None:
        vineyard_set = {v.strip().upper() for v in vineyards}
        yield_df = yield_df[yield_df["vineyard"].isin(vineyard_set)]

    if years is not None:
        year_set = set(years)
        yield_df = yield_df[yield_df["harvest_year"].isin(year_set)]

    if "vine" in tables:
        vine_df = _prepare_vine_archive(tables["vine"])
        vine_df = vine_df[vine_df["variety"] == variety]
        overlap = set(yield_df.columns) & set(vine_df.columns) - {"block"}
        vine_cols = [c for c in vine_df.columns if c not in overlap | {"source_file"}]
        yield_df = yield_df.merge(vine_df[vine_cols], on="block", how="left")

    if "cluster" in tables:
        cluster_df = _prepare_cluster(tables["cluster"])
        cluster_cols = ["block", "harvest_year", "cluster_count", "cluster_weight"]
        yield_df = yield_df.merge(
            cluster_df[cluster_cols],
            on=["block", "harvest_year"],
            how="left",
        )

    if "weather" in tables:
        weather_df = _prepare_weather(tables["weather"])
        weather_yearly = aggregate_weather_by_year(weather_df)
        if "weather_station" in yield_df.columns:
            weather_yearly = weather_yearly.rename(columns={"station": "weather_station"})
            yield_df = yield_df.merge(
                weather_yearly,
                on=["weather_station", "harvest_year"],
                how="left",
            )

    if not include_predicted_yield and "yield_estimation" in yield_df.columns:
        yield_df = yield_df.drop(columns=["yield_estimation"])

    drop_cols = [c for c in ("source_file",) if c in yield_df.columns]
    if drop_cols:
        yield_df = yield_df.drop(columns=drop_cols)

    return yield_df.reset_index(drop=True)


def split_feature_groups(
    df: pd.DataFrame,
    *,
    include_predicted_yield: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Separate the modeling table into feature groups and the target.

    Returns keys: ``ids``, ``vine``, ``weather``, ``previous_yield``,
    ``predicted_yield`` (optional), ``target``.
    """
    from wine_data.schema import (
        CLUSTER_COLUMNS,
        ID_COLUMNS,
        PREDICTED_YIELD_COLUMNS,
        PREVIOUS_YIELD_COLUMNS,
        TARGET_COLUMN,
        VINE_COLUMNS,
        WEATHER_AGG_COLUMNS,
    )

    ids = [c for c in ID_COLUMNS if c in df.columns]
    vine_cols = [c for c in VINE_COLUMNS + CLUSTER_COLUMNS if c in df.columns and c not in ids]
    weather_cols = [
        c for c in df.columns if c.startswith("weather_") and c != "weather_station"
    ]
    prev_cols = [c for c in PREVIOUS_YIELD_COLUMNS if c in df.columns]
    pred_cols = [c for c in PREDICTED_YIELD_COLUMNS if c in df.columns]
    target_cols = [TARGET_COLUMN] if TARGET_COLUMN in df.columns else []

    groups: dict[str, pd.DataFrame] = {
        "ids": df[ids],
        "vine": df[ids + vine_cols],
        "weather": df[ids + weather_cols],
        "previous_yield": df[ids + prev_cols],
        "target": df[ids + target_cols],
    }

    if include_predicted_yield and pred_cols:
        groups["predicted_yield"] = df[ids + pred_cols]

    return groups
