Generate visualizations of the current state of prediction error, and relation between weather/cluster and yield. 

```bash
python visualizations/generate_plots.py
```

In the grape_yield_project train a model to predict yield based on previous year data.
```bash
# load/merge data 
 python -m src.data_loading # → data/processed/vine_year_merged.csv

# Generate PCA baseline
 python -m src.train --model_type pca --raw_dir data/raw

# PCA is informative: PC1 ≈ climate, PC2 ≈ vigor/yield history; together they explain ~33% of variance and show a visible yield gradient on PC2.

# Generate Random Forest baseline
# Original features
 python -m src.train --model_type rf --raw_dir data/raw

# PCA-reduced features
 python -m src.train --model_type pca_rf --raw_dir data/raw

# Grape PG-AN
 python -m src.train --model_type pg_an --raw_dir data/raw --epochs 100

# Grape PG-GNN (GraphSAGE)
 python -m src.train --model_type pg_gnn --raw_dir data/raw --epochs 100 \
  --graph_strategy hybrid
```
Current Model Results
Random Forest (original features)
Strong on training (R² 0.97, RMSE 1.6) but weaker on 2025 (R² 0.33, RMSE 4.9). Classic temporal overfitting: the forest memorizes block patterns from 2020–23 that don’t fully transfer to 2024–25. Test correlation (0.59) is moderate — ranking is OK, absolute errors are larger.

PCA + Random Forest — best test performer
Test R² 0.63 and RMSE 3.66 — clearly better than raw RF on 2025. PCA compression (weather + vigor into ~10 components) appears to reduce overfitting and generalize better across years. Val also improves (R² 0.72 vs 0.66)

PG-AN
Test R² 0.34 — similar to raw RF on 2025, but much worse on train (R² 0.71). High test correlation (0.79) with mediocre R² suggests correct direction/trend but biased or scaled predictions. ***With only 100 epochs and no hyperparameter tuning, it underperforms the tree baselines on this tabular setup.***

```bash
# To run real data
 cd grape_yield_project
 source ../.venv/bin/activate
 python -m src.train --model_type rf --raw_dir data/raw

# To visualize model results

# Retrain (now saves predictions + history automatically)
python -m src.train --model_type rf --raw_dir data/raw
python -m src.train --model_type pca_rf --raw_dir data/raw
python -m src.train --model_type pg_an --raw_dir data/raw --epochs 100

# Generate all plots
python -m src.visualize
```