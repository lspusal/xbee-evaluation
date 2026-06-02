# X-BEE: Explainable Bayesian Economic Ensemble for ICT Nowcasting

Code and data to reproduce the empirical results of the X-BEE framework for
nowcasting quarterly revenue of the Spanish Information and Communication
Technology (ICT) sector.

X-BEE combines five regularized linear learners with an AIC-selected ARIMA
one-step forecast (added as a shared feature) and aggregates them with a
**Bayesian-shrinkage simplex stacking** rule: non-negative weights that sum to
one, estimated on nested out-of-fold predictions and shrunk toward an
inverse-variance (precision) prior. Because the base learners are linear, the
fitted ensemble is an explicit linear model, which gives exact SHAP attributions.

## Data

All inputs are public and included under `data/` (2019Q1–2024Q2):

| Source | Variable | Frequency |
|---|---|---|
| CNMC | Telecommunications revenue (target), mobile lines | Quarterly |
| Google Trends | Technology-related search interest | Monthly → quarterly |
| INE | ICT-sector turnover, employment, R&D | Annual → quarterly |
| OECD STAN | ICT value added, ICT employment | Annual → quarterly |

After lag construction the usable sample is 19 quarterly observations, evaluated
with one-step-ahead expanding-window cross-validation (minimum training size 10,
nine out-of-sample folds covering 2022Q2–2024Q2).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`lightgbm`, `xgboost` and `lime` are optional; the benchmark and explainability
scripts degrade gracefully if they are not installed.

## Usage

```bash
# Full leaderboard (X-BEE vs all competing methods)
python scripts/run_benchmark.py --data data --seeds 5

# Skip the deep-learning baselines (much faster, CPU-friendly)
python scripts/run_benchmark.py --data data --no-deep

# SHAP / permutation / LIME importances and their cross-method agreement
python scripts/run_explainability.py --data data

# Diebold-Mariano tests, Theil's U, bootstrap interval, ablations
python scripts/run_robustness.py --data data
```

Results are written as JSON under `results/`.

### Minimal example

```python
from xbee import XBEE, load_dataset, build_features, expanding_window_folds
from xbee.evaluation import evaluate_xbee

df = load_dataset("data")
X, y, names, dates = build_features(df)
folds = expanding_window_folds(len(y))
preds, actuals, summary = evaluate_xbee(X.to_numpy(), y, folds)
print(summary["rmse"], summary["r2"], summary["mape"])
```

## Repository layout

```
xbee/
  data.py          loading, feature engineering, folds, ARIMA helper
  model.py         X-BEE ensemble and Bayesian-shrinkage aggregation
  evaluation.py    expanding-window CV, metrics, competitors, econometric baselines
  dl.py            deep-learning baselines (PyTorch)
  explain.py       SHAP / permutation / LIME and cross-method consistency
  diagnostics.py   Diebold-Mariano, Theil's U, bootstrap, ablation, sensitivity
scripts/           command-line entry points
data/              public input datasets
```

## Notes on methodology

- All per-fold operations (feature selection, ARIMA order, ensemble weights) are
  computed strictly within the training window; the ARIMA forecast feature is
  shared with every multivariate competitor for a fair comparison.
- Feature engineering uses lagged values for any revenue- or line-derived
  quantity to avoid contemporaneous-target leakage.

## License

MIT (see `LICENSE`).
