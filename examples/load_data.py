"""Example: load CS yield data from Excel workbooks in ``data/``."""

from pathlib import Path

import pandas as pd

from wine_data import build_modeling_table, resolve_data_dir, split_feature_groups

# Resolve data directory (data/ or grape_yield_project/data/raw).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_dir = resolve_data_dir()
print(f"Using data from: {data_dir}")

# Low-level: inspect and concatenate all workbooks
excel_files = sorted(f for f in data_dir.glob("*.xlsx") if not f.name.startswith("test_"))
if not excel_files:
    raise FileNotFoundError(
        f"No .xlsx files in {data_dir}. "
        f"Run from project root or place workbooks in data/."
    )
all_tables: list[pd.DataFrame] = []

for file in excel_files:
    xls = pd.ExcelFile(file)
    print(file.name, xls.sheet_names)
    df = pd.read_excel(file)
    df["source_file"] = file.name
    all_tables.append(df)

raw = pd.concat(all_tables, ignore_index=True)
print(f"\nRaw concatenated rows: {len(raw):,}")

# Modeling table: block-year grain with separated feature groups
# Restrict to three vineyards when desired, e.g. ENZ, ELR, RMM
data = build_modeling_table(
    data_dir,
    vineyards=["ENZ", "ELR", "RMM"],
    include_predicted_yield=True,
)

groups = split_feature_groups(data, include_predicted_yield=True)
target = groups["target"]["yield_actual"]

print(f"\nModeling rows: {len(data):,}")
print(f"Target (actual yield) mean: {target.mean():.2f}")
print(f"Vine features: {list(groups['vine'].columns)}")
print(f"Weather features: {list(groups['weather'].columns)}")
