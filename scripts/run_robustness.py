"""Significance and robustness diagnostics: Diebold-Mariano tests against key
baselines, Theil's U2, directional accuracy, a block-bootstrap RMSE interval,
data-source ablation, and training-length sensitivity.

    python scripts/run_robustness.py --data data
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import BayesianRidge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xbee.data import build_features, expanding_window_folds, load_dataset
from xbee.diagnostics import (block_bootstrap_rmse_ci, data_source_ablation, diebold_mariano,
                             directional_accuracy, theils_u2, training_length_sensitivity)
from xbee.evaluation import evaluate_arima, evaluate_estimator, evaluate_random_walk, evaluate_xbee


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results/robustness.json")
    args = parser.parse_args()

    df = load_dataset(args.data)
    X_frame, y, feature_names, _ = build_features(df)
    X = X_frame.to_numpy()
    folds = expanding_window_folds(len(y))

    xbee_pred, actuals, _ = evaluate_xbee(X, y, folds)
    naive = np.array([float(y[tr][-1]) for tr, _ in folds])
    err_xbee = xbee_pred - actuals

    br_pred, _ = evaluate_estimator(lambda: BayesianRidge(alpha_1=1e-6, alpha_2=1e-6), X, y, folds)
    arima_pred, _ = evaluate_arima(y, folds, (1, 1, 1))
    rw_pred, _ = evaluate_random_walk(y, folds)

    dm = {
        "vs Bayesian Ridge": diebold_mariano(err_xbee, br_pred - actuals),
        "vs ARIMA(1,1,1)": diebold_mariano(err_xbee, arima_pred - actuals),
        "vs Random Walk": diebold_mariano(err_xbee, rw_pred - actuals),
    }
    lo, hi = block_bootstrap_rmse_ci(actuals, xbee_pred)
    report = {
        "diebold_mariano": {k: {"statistic": s, "p_value": p} for k, (s, p) in dm.items()},
        "theils_u2": theils_u2(actuals, xbee_pred, naive),
        "directional_accuracy": directional_accuracy(actuals, xbee_pred, naive),
        "rmse_ci95": [lo, hi],
        "data_source_ablation": data_source_ablation(X, y, folds, feature_names),
        "training_length_sensitivity": training_length_sensitivity(X, y),
    }

    print("Diebold-Mariano (negative statistic favours X-BEE):")
    for name, (stat, p) in dm.items():
        print(f"  {name:20s} statistic={stat:+.3f}  p={p:.3f}")
    print(f"Theil U2={report['theils_u2']:.3f}  directional accuracy={report['directional_accuracy']:.3f}")
    print(f"RMSE 95% CI=[{lo:.1f}, {hi:.1f}]")
    print("Data-source ablation (RMSE):", report["data_source_ablation"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
