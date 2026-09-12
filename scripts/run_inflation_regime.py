"""Inflationary-regime robustness analysis (second-round revision).

Isolates the 2022--2023 inflation surge / ECB tightening episode as a distinct
structural regime within the post-pandemic test window (2022Q2--2024Q2):

  Regime A - "inflation surge & ECB tightening" : 2022Q2-2023Q3
             Euro-area HICP inflation averaged ~7.7% y/y (peak 10.6% in
             Oct-2022); the ECB raised the deposit facility rate from -0.50%
             to 4.00% in ten consecutive hikes (Jul-2022 to Sep-2023).
  Regime B - "disinflation & rate plateau"      : 2023Q4-2024Q2
             Euro-area HICP fell below 3% y/y; the ECB held rates at the peak.

Outputs (results/inflation_regime.json):
  (i)  X-BEE predictive accuracy per regime (RMSE, MAE, MAPE, directional
       accuracy) under the standard expanding-window protocol;
  (ii) out-of-sample SHAP attribution stability across regimes: exact linear
       SHAP of each test-quarter nowcast (fixed feature set, per-fold fitted
       effective model), averaged within regime, plus the Spearman rank
       correlation between the two regime importance profiles.

Usage:
    python scripts/run_inflation_regime.py --data data [--aggregation bayes_shrink]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xbee import load_dataset, build_features, expanding_window_folds
from xbee.data import select_features, arima_forecast_feature
from xbee.evaluation import metrics
from xbee.model import XBEE, base_learners, _nested_oof, _combination_weights

REGIME_A = "inflation_surge_ecb_tightening"  # 2022Q2-2023Q3
REGIME_B = "disinflation_rate_plateau"       # 2023Q4-2024Q2


def regime_of(quarter: str) -> str:
    """Exogenous regime label from documented Eurostat/ECB chronology."""
    return REGIME_A if quarter <= "2023Q3" else REGIME_B


def regime_metrics(records: list[dict]) -> dict:
    actuals = np.array([r["actual"] for r in records])
    preds = np.array([r["pred"] for r in records])
    directional = [
        np.sign(r["actual"] - r["prev_actual"]) == np.sign(r["pred"] - r["prev_actual"])
        for r in records
    ]
    return {
        **metrics(actuals, preds),
        "n_quarters": len(records),
        "quarters": [r["quarter"] for r in records],
        "mae": float(np.mean(np.abs(actuals - preds))),
        "directional_accuracy": float(np.mean(directional)),
    }


def fold_shap_attribution(X, y, train_idx, test_idx, sel, aggregation, tau):
    """Exact linear SHAP of one out-of-sample nowcast on a FIXED feature set,
    so attribution profiles are comparable across folds and regimes."""
    fitted, forecast, _ = arima_forecast_feature(y[train_idx])
    x_tr = np.hstack([X[train_idx][:, sel], fitted.reshape(-1, 1)])
    x_te = np.hstack([X[test_idx][:, sel], np.full((len(test_idx), 1), forecast)])
    scaler = RobustScaler()
    xs_tr = scaler.fit_transform(x_tr)
    xs_te = scaler.transform(x_te)

    learners = base_learners()
    coefs, intercepts = {}, {}
    for name, model in learners.items():
        model.fit(xs_tr, y[train_idx].astype(float))
        coefs[name] = model.coef_.copy()
        intercepts[name] = float(model.intercept_)
    oof_pred, oof_true = _nested_oof(xs_tr, y[train_idx].astype(float))
    weights = _combination_weights(oof_pred, oof_true, aggregation, tau)
    names = list(learners)
    eff_coef = np.sum([weights[i] * coefs[names[i]] for i in range(len(names))], axis=0)
    return eff_coef * (xs_te[0] - xs_tr.mean(axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--aggregation", default="bayes_shrink",
                        choices=["bayes_shrink", "invvar", "nnls", "mean"])
    parser.add_argument("--out", default="results/inflation_regime.json")
    args = parser.parse_args()

    df = load_dataset(args.data)
    feat, y, feature_names, dates = build_features(df)
    X = feat.to_numpy()
    folds = expanding_window_folds(len(y))
    quarters = [f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in dates]
    print(f"n={len(y)} folds={len(folds)} test={quarters[10]}..{quarters[-1]}")

    records = {REGIME_A: [], REGIME_B: []}
    shap_rows = {REGIME_A: [], REGIME_B: []}
    sel = select_features(X, y, 8)
    selected_names = [feature_names[s] for s in sel] + ["arima_forecast"]

    for train_idx, test_idx in folds:
        quarter = quarters[test_idx[0]]
        regime = regime_of(quarter)
        model = XBEE(aggregation=args.aggregation).fit(X[train_idx], y[train_idx])
        pred = float(model.predict(X[test_idx])[0])
        records[regime].append({
            "quarter": quarter,
            "actual": float(y[test_idx[0]]),
            "pred": pred,
            "prev_actual": float(y[test_idx[0] - 1]),
        })
        phi = fold_shap_attribution(X, y, train_idx, test_idx, sel,
                                    args.aggregation, model.tau)
        shap_rows[regime].append(np.abs(phi))
        print(f"  {quarter}  {regime:32s} actual={y[test_idx[0]]:9.1f} pred={pred:9.1f}")

    profile_a = np.mean(shap_rows[REGIME_A], axis=0)
    profile_b = np.mean(shap_rows[REGIME_B], axis=0)
    rho, pvalue = spearmanr(profile_a, profile_b)
    pct_a = profile_a / profile_a.sum() * 100
    pct_b = profile_b / profile_b.sum() * 100

    results = {
        "metadata": {
            "purpose": "Inflationary-regime robustness (second-round revision)",
            "aggregation": args.aggregation,
            "test_period": f"{quarters[10]}..{quarters[-1]}",
            "regime_definitions": {
                REGIME_A: "2022Q2-2023Q3: HICP avg ~7.7% y/y (peak 10.6% Oct-2022); "
                          "ECB deposit facility rate -0.50% -> 4.00%",
                REGIME_B: "2023Q4-2024Q2: HICP < 3% y/y; deposit facility rate at 4.00%",
            },
        },
        "predictive_stability": {
            REGIME_A: regime_metrics(records[REGIME_A]),
            REGIME_B: regime_metrics(records[REGIME_B]),
            "full_test": regime_metrics(records[REGIME_A] + records[REGIME_B]),
        },
        "attribution_stability": {
            "feature_names": selected_names,
            "regime_A_mean_abs_shap_pct": dict(zip(selected_names, pct_a.tolist())),
            "regime_B_mean_abs_shap_pct": dict(zip(selected_names, pct_b.tolist())),
            "spearman_rank_corr_between_regimes": float(rho),
            "spearman_pvalue": float(pvalue),
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)

    ma, mb = (results["predictive_stability"][REGIME_A],
              results["predictive_stability"][REGIME_B])
    print(f"\nRegime A (2022Q2-2023Q3): RMSE={ma['rmse']:.2f} MAPE={ma['mape']:.3f}% "
          f"dir.acc={ma['directional_accuracy']:.2f}")
    print(f"Regime B (2023Q4-2024Q2): RMSE={mb['rmse']:.2f} MAPE={mb['mape']:.3f}% "
          f"dir.acc={mb['directional_accuracy']:.2f}")
    print(f"SHAP rank corr across regimes: rho={rho:.3f} (p={pvalue:.3f})")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
