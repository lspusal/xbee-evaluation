"""Compute global SHAP / permutation / LIME importances, their cross-method
agreement, and cross-fold importance stability for the fitted X-BEE ensemble.

    python scripts/run_explainability.py --data data
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xbee.data import build_features, expanding_window_folds, load_dataset
from xbee.explain import cross_fold_stability, feature_importance_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results/explainability.json")
    args = parser.parse_args()

    df = load_dataset(args.data)
    X_frame, y, feature_names, _ = build_features(df)
    X = X_frame.to_numpy()
    folds = expanding_window_folds(len(y))

    report = feature_importance_report(X, y, feature_names)
    rho, n_pairs = cross_fold_stability(X, y, folds)
    report["cross_fold_stability"] = {"mean_pairwise_spearman": rho, "n_pairs": n_pairs}

    print("Global SHAP share (%):")
    for feat, share in sorted(report["shap_share_pct"].items(), key=lambda kv: -kv[1]):
        print(f"  {feat:32s}{share:6.1f}")
    print("\nCross-method consistency (Spearman):", report["consistency_spearman"])
    print(f"Cross-fold importance stability: rho={rho:.3f} ({n_pairs} pairs)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
