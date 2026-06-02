"""Reproduce the full leaderboard: X-BEE against all competing methods under one
expanding-window protocol. Writes results/leaderboard.json and prints a ranking.

    python scripts/run_benchmark.py --data data --seeds 5
"""
import argparse
import json
import os
import sys

from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, RandomForestRegressor, StackingRegressor,
                              VotingRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xbee.data import build_features, expanding_window_folds, load_dataset
from xbee.evaluation import (evaluate_arima, evaluate_estimator, evaluate_random_walk,
                            evaluate_sarima, evaluate_xbee)

try:
    import xgboost as xgb
    import lightgbm as lgb
    HAS_BOOST = True
except ImportError:
    HAS_BOOST = False


def competitors():
    models = {
        "Bayesian Ridge": (lambda: BayesianRidge(alpha_1=1e-6, alpha_2=1e-6), "Traditional ML"),
        "Huber Regressor": (lambda: HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000), "Traditional ML"),
        "Ridge": (lambda: Ridge(alpha=1.0), "Traditional ML"),
        "Elastic Net": (lambda: ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=5000), "Traditional ML"),
        "Lasso": (lambda: Lasso(alpha=0.01, max_iter=5000), "Traditional ML"),
        "Bridge Equation": (lambda: LinearRegression(), "Econometric"),
        "Random Forest": (lambda: RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=2, random_state=42), "Ensemble Methods"),
        "Gradient Boosting": (lambda: GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42), "Ensemble Methods"),
        "Extra Trees": (lambda: ExtraTreesRegressor(n_estimators=300, max_depth=4, random_state=42), "Ensemble Methods"),
        "AdaBoost": (lambda: AdaBoostRegressor(n_estimators=80, learning_rate=0.1, random_state=42), "Ensemble Methods"),
        "Bagging Regressor": (lambda: BaggingRegressor(n_estimators=80, random_state=42), "Ensemble Methods"),
        "Decision Tree": (lambda: DecisionTreeRegressor(max_depth=4, random_state=42), "Traditional ML"),
        "SVR": (lambda: SVR(kernel="rbf", C=100, epsilon=0.1), "Traditional ML"),
        "KNN": (lambda: KNeighborsRegressor(n_neighbors=3), "Traditional ML"),
        "Gaussian Process": (lambda: GaussianProcessRegressor(alpha=1.0, normalize_y=True, random_state=42), "Traditional ML"),
        "Voting Regressor": (lambda: VotingRegressor([("br", BayesianRidge()), ("r", Ridge(1.0)), ("l", Lasso(0.01)), ("en", ElasticNet(0.05)), ("h", HuberRegressor(max_iter=2000))]), "Ensemble Methods"),
        "Stacking Regressor": (lambda: StackingRegressor(estimators=[("br", BayesianRidge()), ("r", Ridge(1.0)), ("l", Lasso(0.01))], final_estimator=BayesianRidge()), "Ensemble Methods"),
    }
    if HAS_BOOST:
        models["XGBoost"] = (lambda: xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42, verbosity=0), "Ensemble Methods")
        models["LightGBM"] = (lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42, verbose=-1), "Ensemble Methods")
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-deep", action="store_true", help="skip deep-learning baselines")
    parser.add_argument("--out", default="results/leaderboard.json")
    args = parser.parse_args()

    df = load_dataset(args.data)
    X_frame, y, feature_names, dates = build_features(df)
    X = X_frame.to_numpy()
    folds = expanding_window_folds(len(y))
    quarters = [f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in dates]
    print(f"n={len(y)} features={X.shape[1]} folds={len(folds)} test={quarters[10]}..{quarters[-1]}")

    board = {}
    _, _, xbee = evaluate_xbee(X, y, folds)
    board["X-BEE"] = {"rmse": xbee["rmse"], "r2": xbee["r2"], "mape": xbee["mape"], "category": "Ensemble (Ours)"}

    for name, (factory, category) in competitors().items():
        try:
            _, m = evaluate_estimator(factory, X, y, folds)
            board[name] = {**m, "category": category}
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: skipped ({exc})")

    for name, order in (("ARIMA(2,1,1)", (2, 1, 1)), ("ARIMA(1,1,1)", (1, 1, 1)), ("ARIMA(1,1,0)", (1, 1, 0))):
        _, m = evaluate_arima(y, folds, order)
        board[name] = {**m, "category": "Econometric"}
    _, m = evaluate_sarima(y, folds)
    board["SARIMA"] = {**m, "category": "Econometric"}
    _, m = evaluate_random_walk(y, folds)
    board["Random Walk"] = {**m, "category": "Econometric"}

    if not args.no_deep:
        from xbee.dl import evaluate_deep_models
        for name, m in evaluate_deep_models(X, y, folds, seeds=args.seeds).items():
            board[name] = {**m, "category": "Deep Learning"}

    ranked = sorted(board.items(), key=lambda kv: kv[1]["rmse"])
    print(f"\n{'Rank':>4}  {'Method':24s}{'RMSE':>9}{'R2':>9}{'MAPE':>8}")
    for i, (name, m) in enumerate(ranked, 1):
        mark = "  <-- proposed" if name == "X-BEE" else ""
        print(f"{i:>4}  {name:24s}{m['rmse']:9.2f}{m['r2']:9.4f}{m['mape']:8.3f}{mark}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"n_methods": len(board), "ranking": [{"rank": i, "method": n, **m} for i, (n, m) in enumerate(ranked, 1)]}, fh, indent=2)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
