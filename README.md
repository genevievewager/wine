# Wine Yield Prediction

Cabernet Sauvignon block-level yield forecasting from vineyard Excel workbooks (weather, vigor, historical yield). Includes EDA, multiple ML models, and accuracy reporting.

## Project layout

```
wine/
  wine_data/          # Data loading package (Excel → modeling table)
  src/                # Training, preprocessing, models, plots
  data/raw/           # Source .xlsx workbooks
  data/processed/     # Merged vine-year table, PCA artifacts
  outputs/
    eda/              # Exploratory figures + CSV tables
    models/           # Checkpoints, metrics, model figures
  notebooks/          # Interactive exploration
  tests/              # Pytest suite
  reference/pg-an/  # Original AAAI 2023 reference code (TensorFlow)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[ml,dev]"
# Optional: pip install -e ".[gnn,boost]"
```

Place vineyard `.xlsx` files in `data/raw/`.

## Workflow

### 1. Merge data

```bash
python -m src.data_loading   # → data/processed/vine_year_merged.csv
```

### 2. Train models

```bash
python -m src.train --model_type pca --raw_dir data/raw
python -m src.train --model_type rf --raw_dir data/raw
python -m src.train --model_type pca_rf --raw_dir data/raw
python -m src.train --model_type pg_an --raw_dir data/raw --epochs 100
python -m src.train --model_type pg_gnn --raw_dir data/raw --epochs 100 --graph_strategy hybrid
python -m src.train --model_type xgboost --raw_dir data/raw   # requires [boost]
```

### 3. Visualize

```bash
python -m src.plots --all          # EDA + model figures
python -m src.plots --eda          # outputs/eda/
python -m src.plots --models       # outputs/models/figures/
python -m src.model_accuracy_report
```

### 4. Example data API

```bash
python examples/load_data.py
```

## Current model results (test 2025)

| Model | Test R² | Test RMSE | Test correlation |
|-------|---------|-----------|------------------|
| PCA + RF | 0.63 | 3.66 | 0.80 |
| PG-AN | 0.51 | 4.22 | 0.80 |
| Random Forest | 0.33 | 4.95 | 0.59 |
| PG-GNN | 0.25 | 5.25 | 0.61 |

Metrics: `outputs/models/metrics/*_metrics.json`  
EDA tables: `outputs/eda/tables/`

## Tests

```bash
pytest
```
