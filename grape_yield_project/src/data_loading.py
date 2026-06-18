"""Load and merge .xlsx / .csv tabular grape yield data into a vine-year table.

Uses auto-detection of workbook types (POT yield, harvest, cluster, archive,
weather). When ``wine_data`` is installed in the parent project, it can delegate
to ``wine_data.load.build_modeling_table`` and map columns to this project's schema.

Block-level sources: ``vine_id`` = ``{vineyard}_{block}`` when vine-level rows
are absent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config as cfg

# Optional reuse of parent wine_data package
_PARENT = Path(__file__).resolve().parents[2]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

try:
    from wine_data.load import build_modeling_table, classify_and_load
    from wine_data.schema import POT_RENAME as WD_POT_RENAME

    HAS_WINE_DATA = True
except ImportError:
    HAS_WINE_DATA = False


def _resolve_raw_dir(raw_dir: Path | str | None) -> Path:
    candidates = [
        Path(raw_dir) if raw_dir else None,
        cfg.DATA_RAW,
        cfg.DEFAULT_RAW_DATA_DIR,
        _PARENT / "data",
        _PARENT,
    ]
    for c in candidates:
        if c and c.is_dir():
            xlsx = list(c.glob("*.xlsx"))
            csv = list(c.glob("*.csv"))
            if xlsx or csv:
                return c
    return Path(raw_dir or cfg.DATA_RAW)


def _read_table(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        xl = pd.ExcelFile(path)
        sheet = xl.sheet_names[sheet_name] if isinstance(sheet_name, int) else sheet_name
        return pd.read_excel(path, sheet_name=sheet)
    raise ValueError(f"Unsupported file type: {path}")


def _detect_kind(columns: Iterable[str]) -> str:
    cols = {str(c).strip() for c in columns}
    lower = {c.lower() for c in cols}
    if "YIELD ESTIMATION" in cols or "yield_estimation" in lower:
        return "pot"
    if "WEATHER STATION" in cols or "weather_station" in lower:
        return "archive"
    if "קוד תחנה" in cols or "station" in lower:
        return "weather"
    if "CLUSTER COUNT" in cols or "cluster_count" in lower:
        return "cluster"
    if "כרם" in cols:
        return "harvest"
    return "unknown"


def _load_typed_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    if HAS_WINE_DATA:
        wd = classify_and_load(raw_dir, verbose=False)
        mapping = {
            "yield_pot": "pot",
            "vine": "archive",
            "weather": "weather",
            "cluster": "cluster",
            "harvest": "harvest",
        }
        return {mapping.get(k, k): v for k, v in wd.items()}

    buckets: dict[str, list[pd.DataFrame]] = {}
    for path in sorted(raw_dir.glob("*.xlsx")) + sorted(raw_dir.glob("*.csv")):
        df = _read_table(path)
        kind = _detect_kind(df.columns)
        buckets.setdefault(kind, []).append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in buckets.items()}


def _rename_df(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def _normalize_variety(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _filter_variety(df: pd.DataFrame) -> pd.DataFrame:
    if cfg.COL_VARIETY not in df.columns:
        return df
    v = _normalize_variety(df[cfg.COL_VARIETY])
    mask = v.isin(cfg.VARIETY_FILTER) | v.str.contains("Cabernet", case=False, na=False)
    return df.loc[mask].copy()


def _filter_years(df: pd.DataFrame) -> pd.DataFrame:
    if cfg.COL_YEAR not in df.columns:
        return df
    years = pd.to_numeric(df[cfg.COL_YEAR], errors="coerce")
    return df.loc[years.between(2020, 2025)].copy()


def _extract_row_from_block(block: str) -> str | None:
    if not isinstance(block, str):
        return None
    m = re.search(r"(\d{2,4})", block)
    return m.group(1) if m else None


def aggregate_weather_yearly(weather: pd.DataFrame) -> pd.DataFrame:
    w = _rename_df(weather, cfg.WEATHER_RENAME)
    if "date" not in w.columns:
        return pd.DataFrame()
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    w[cfg.COL_YEAR] = w["date"].dt.year
    w = _filter_years(w)
    agg_cols = {k: v for k, v in cfg.WEATHER_AGG.items() if k in w.columns}
    grouped = (
        w.groupby(["station", cfg.COL_YEAR], as_index=False)
        .agg(agg_cols)
        .rename(columns={k: f"weather_{k}_{v}" for k, v in agg_cols.items()})
    )
    return grouped.rename(columns={"station": "weather_station"})


def _estimate_vine_count(row: pd.Series) -> float:
    vpa = row.get("vines_per_quarter_acre")
    size = row.get("size")
    if pd.notna(vpa) and pd.notna(size) and float(vpa) > 0 and float(size) > 0:
        return float(vpa) * float(size) * 4.0
    return float("nan")


def _derive_vigor_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "cluster_count" in out.columns:
        vine_est = out.apply(_estimate_vine_count, axis=1)
        out[cfg.COL_SHOOTS] = out["cluster_count"] / vine_est.replace(0, float("nan"))
    if "cluster_weight" in out.columns:
        out[cfg.COL_CANOPY] = out["cluster_weight"]
    elif "size" in out.columns:
        out[cfg.COL_CANOPY] = out["size"]
    if cfg.COL_SHOOTS in out.columns and cfg.COL_CANOPY in out.columns:
        out["shoots_per_canopy_area"] = out[cfg.COL_SHOOTS] / out[cfg.COL_CANOPY].replace(0, float("nan"))
    return out


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values([cfg.COL_VINE_ID, cfg.COL_YEAR]).copy()
    for col, new_col in [
        (cfg.COL_ACTUAL_YIELD, cfg.COL_PREV_ACTUAL),
        (cfg.COL_PRED_YIELD, cfg.COL_PREV_PRED),
        (cfg.COL_YIELD_ERROR, cfg.COL_PREV_ERROR),
        (cfg.COL_CANOPY, "previous_year_canopy_size"),
    ]:
        if col in out.columns:
            out[new_col] = out.groupby(cfg.COL_VINE_ID)[col].shift(1)
    if cfg.COL_CANOPY in out.columns and "previous_year_canopy_size" in out.columns:
        out["canopy_change_from_previous_year"] = out[cfg.COL_CANOPY] - out["previous_year_canopy_size"]
    return out


def _map_wine_data_table(df: pd.DataFrame) -> pd.DataFrame:
    """Map wine_data column names to grape_yield_project schema."""
    rename = {
        "harvest_year": cfg.COL_YEAR,
        "yield_actual": cfg.COL_ACTUAL_YIELD,
        "yield_estimation": cfg.COL_PRED_YIELD,
        "yield_prev_year": cfg.COL_PREV_ACTUAL,
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if cfg.COL_VINEYARD not in out.columns and "vineyard" in out.columns:
        out[cfg.COL_VINEYARD] = out["vineyard"]
    if cfg.COL_BLOCK not in out.columns and "block" in out.columns:
        out[cfg.COL_BLOCK] = out["block"]
    if cfg.COL_VARIETY not in out.columns and "variety" in out.columns:
        out[cfg.COL_VARIETY] = out["variety"]
    return out


def merge_vine_year_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "pot" not in tables:
        raise ValueError("POT / yield workbook not found. Place .xlsx files in data/raw/.")

    pot = _rename_df(tables["pot"], cfg.POT_RENAME)
    pot[cfg.COL_YEAR] = pd.to_numeric(pot[cfg.COL_YEAR], errors="coerce").astype("Int64")
    pot = _filter_variety(_filter_years(pot))
    base = pot.drop_duplicates(subset=[cfg.COL_YEAR, cfg.COL_VINEYARD, cfg.COL_BLOCK])

    merge_keys = [cfg.COL_YEAR, cfg.COL_VINEYARD, cfg.COL_BLOCK]

    if "cluster" in tables:
        cluster = _rename_df(tables["cluster"], cfg.CLUSTER_RENAME)
        cluster[cfg.COL_YEAR] = pd.to_numeric(cluster[cfg.COL_YEAR], errors="coerce").astype("Int64")
        cluster = cluster[cluster[cfg.COL_YEAR] > 0]
        cluster = _filter_variety(cluster)
        base = base.merge(
            cluster.drop_duplicates(subset=merge_keys),
            on=merge_keys,
            how="left",
            suffixes=("", "_cluster"),
        )

    if "harvest" in tables:
        harvest = _rename_df(tables["harvest"], cfg.HARVEST_RENAME)
        harvest[cfg.COL_YEAR] = pd.to_numeric(harvest[cfg.COL_YEAR], errors="coerce").astype("Int64")
        harvest = _filter_variety(_filter_years(harvest))
        hcols = [c for c in harvest.columns if c not in base.columns or c in merge_keys]
        base = base.merge(harvest[hcols].drop_duplicates(subset=merge_keys), on=merge_keys, how="left")

    if "archive" in tables:
        archive = _rename_df(tables["archive"], cfg.ARCHIVE_RENAME)
        archive = _filter_variety(archive)
        archive = archive[archive[cfg.COL_BLOCK].astype(str) != "*"]
        acols = [c for c in archive.columns if c != cfg.COL_YEAR]
        base = base.merge(
            archive[acols].drop_duplicates(subset=[cfg.COL_BLOCK], keep="last"),
            on=[cfg.COL_VINEYARD, cfg.COL_BLOCK],
            how="left",
        )

    if "weather" in tables:
        weather_yearly = aggregate_weather_yearly(tables["weather"])
        if not weather_yearly.empty and "weather_station" in base.columns:
            base = base.merge(
                weather_yearly,
                on=["weather_station", cfg.COL_YEAR],
                how="left",
            )

    base[cfg.COL_VINE_ID] = base[cfg.COL_VINEYARD].astype(str) + "_" + base[cfg.COL_BLOCK].astype(str)
    base[cfg.COL_ROW] = base[cfg.COL_BLOCK].map(_extract_row_from_block)
    base[cfg.COL_VINE_POSITION] = pd.NA
    base[cfg.COL_YIELD_ERROR] = base[cfg.COL_ACTUAL_YIELD] - base[cfg.COL_PRED_YIELD]

    base = _derive_vigor_features(base)
    base = _add_lag_features(base)

    if cfg.VINEYARD_FILTER:
        base = base[base[cfg.COL_VINEYARD].isin(cfg.VINEYARD_FILTER)]

    return base.reset_index(drop=True)


def load_merged_vine_year_table(
    raw_dir: Path | str | None = None,
    save_path: Path | str | None = None,
    use_wine_data: bool = True,
) -> pd.DataFrame:
    raw_dir = _resolve_raw_dir(raw_dir)

    if use_wine_data and HAS_WINE_DATA:
        try:
            wd_df = build_modeling_table(
                raw_dir,
                variety="CS",
                vineyards=cfg.VINEYARD_FILTER,
                include_predicted_yield=True,
                verbose=False,
            )
            merged = _map_wine_data_table(wd_df)
            merged[cfg.COL_VINE_ID] = merged[cfg.COL_VINEYARD].astype(str) + "_" + merged[cfg.COL_BLOCK].astype(str)
            merged[cfg.COL_ROW] = merged[cfg.COL_BLOCK].map(_extract_row_from_block)
            merged[cfg.COL_VINE_POSITION] = pd.NA
            if cfg.COL_PRED_YIELD in merged.columns and cfg.COL_ACTUAL_YIELD in merged.columns:
                merged[cfg.COL_YIELD_ERROR] = merged[cfg.COL_ACTUAL_YIELD] - merged[cfg.COL_PRED_YIELD]
            merged = _derive_vigor_features(merged)
            merged = _add_lag_features(merged)
        except (FileNotFoundError, ValueError):
            tables = _load_typed_tables(raw_dir)
            merged = merge_vine_year_tables(tables)
    else:
        tables = _load_typed_tables(raw_dir)
        merged = merge_vine_year_tables(tables)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix == ".parquet":
            merged.to_parquet(save_path, index=False)
        else:
            merged.to_csv(save_path, index=False)
    return merged


def load_csv_files(
    csv_paths: Iterable[Path | str],
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        df = _read_table(Path(p))
        if column_map:
            df = df.rename(columns=column_map)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = _filter_variety(_filter_years(out))
    return out


if __name__ == "__main__":
    df = load_merged_vine_year_table(save_path=cfg.DATA_PROCESSED / "vine_year_merged.csv")
    print(f"Merged table: {df.shape}")
    print(df.head())
