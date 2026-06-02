#!/usr/bin/env python3
"""
Aggregator weight-stability analysis for the X-BEE revision.

Reviewer follow-up: NNLS stacking attains a marginally lower RMSE than the
proposed Bayesian-shrinkage combination (Table 5), so "why not just use NNLS?"
This script answers that question quantitatively by recovering the per-fold
combination weights over the five base learners for each aggregator and
measuring how concentrated and how stable those weights are.

For each aggregator we report:
  * concentration   : mean/max of the largest single weight, and the effective
                      number of active learners 1/sum(w^2) (inverse HHI;
                      5 = perfectly even, 1 = all mass on one learner)
  * sparsity        : mean number of effectively-zero learners per fold (w < 0.01)
  * fold-to-fold     : mean across learners of the std of each learner's weight
    instability       across folds, and the mean pairwise L1 distance between the
                      per-fold weight vectors (how much the allocation jumps)

It uses the *canonical* X-BEE package shipped in GitHub3 unchanged, so the
reported RMSE values reproduce the manuscript leaderboard (X-BEE bayes_shrink
RMSE = 64.61 EUR M; NNLS = 58.10).

Usage (from this directory, with the canonical data shipped in GitHub3):
    python aggregator_weight_stability.py --data ../GitHub3/data \
        --xbee ../GitHub3
"""
import argparse
import json
import os
import sys
import numpy as np

AGGREGATORS = ["invvar", "nnls", "bayes_shrink"]


def weight_stability(weights):
    """weights: list of per-fold weight lists (len = n_base_learners) or None."""
    W = np.array([w for w in weights if w is not None], dtype=float)  # (folds, K)
    effective = 1.0 / np.sum(W ** 2, axis=1)            # active learners per fold
    near_zero = np.sum(W < 0.01, axis=1)                # ~zero learners per fold
    per_learner_std = W.std(axis=0)                     # drift of each learner
    n = len(W)
    pairwise_l1 = [np.abs(W[i] - W[j]).sum() for i in range(n) for j in range(i + 1, n)]
    return {
        "n_folds": int(n),
        "max_weight_mean": round(float(W.max(axis=1).mean()), 4),
        "max_weight_max": round(float(W.max(axis=1).max()), 4),
        "effective_n_learners_mean": round(float(effective.mean()), 4),
        "effective_n_learners_min": round(float(effective.min()), 4),
        "near_zero_learners_mean": round(float(near_zero.mean()), 4),
        "weight_std_across_folds_mean": round(float(per_learner_std.mean()), 4),
        "mean_pairwise_L1_distance": round(float(np.mean(pairwise_l1)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../GitHub3/data", help="canonical data dir")
    ap.add_argument("--xbee", default="../GitHub3", help="dir containing the xbee package")
    ap.add_argument("--out", default="aggregator_weight_stability_results.json")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.xbee))
    from xbee.data import build_features, expanding_window_folds, load_dataset
    from xbee.evaluation import evaluate_xbee

    df = load_dataset(args.data)
    feat, y, names, index = build_features(df)
    X = feat.values
    folds = expanding_window_folds(len(y))
    print(f"n={len(y)}  features={X.shape[1]}  folds={len(folds)}\n")

    out = {
        "description": "Per-fold combination-weight concentration and stability "
                       "across base learners for each aggregation rule.",
        "base_learners": ["bayesian_ridge", "huber", "ridge", "elastic_net", "lasso"],
        "aggregators": {},
    }
    for agg in AGGREGATORS:
        preds, actuals, res = evaluate_xbee(X, y, folds, aggregation=agg)
        stab = weight_stability(res["weights"])
        out["aggregators"][agg] = {
            "rmse": round(res["rmse"], 4),
            "r2": round(res["r2"], 4),
            "mape": round(res["mape"], 4),
            "stability": stab,
            "per_fold_weights": [
                [round(v, 4) for v in w] for w in res["weights"] if w is not None
            ],
        }
        print(f"== {agg} ==  RMSE={res['rmse']:.2f}  R2={res['r2']:.4f}  MAPE={res['mape']:.3f}%")
        print(f"   largest single weight (mean / max) : {stab['max_weight_mean']:.3f} / {stab['max_weight_max']:.3f}")
        print(f"   effective # active learners (mean) : {stab['effective_n_learners_mean']:.2f}"
              f"  (min {stab['effective_n_learners_min']:.2f} of 5)")
        print(f"   near-zero learners per fold (mean) : {stab['near_zero_learners_mean']:.2f} of 5")
        print(f"   weight std across folds (mean)     : {stab['weight_std_across_folds_mean']:.3f}")
        print(f"   mean pairwise L1 weight distance   : {stab['mean_pairwise_L1_distance']:.3f}\n")

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
