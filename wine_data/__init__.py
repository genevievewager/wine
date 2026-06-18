"""Excel data-loading pipeline for CS grape yield modeling."""

from wine_data.load import (
    add_previous_year_yield,
    aggregate_weather_by_year,
    build_modeling_table,
    classify_and_load,
    load_excel_files,
    resolve_data_dir,
    split_feature_groups,
)
from wine_data.schema import TARGET_COLUMN, VARIETY_CS

__all__ = [
    "TARGET_COLUMN",
    "VARIETY_CS",
    "add_previous_year_yield",
    "aggregate_weather_by_year",
    "build_modeling_table",
    "classify_and_load",
    "load_excel_files",
    "resolve_data_dir",
    "split_feature_groups",
]
