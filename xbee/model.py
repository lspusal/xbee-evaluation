"""The X-BEE ensemble: ARIMA-augmented regularized linear learners combined by
Bayesian-shrinkage simplex stacking.

The five base learners are linear, so the fitted ensemble is itself an explicit
linear model, which yields exact SHAP attributions (see ``explain.py``). The
combination weights are non-negative, sum to one, are estimated on nested
out-of-fold predictions, and are shrunk toward an inverse-variance (precision)
prior. This is the MAP estimate of the mixing weights under a Dirichlet-type
prior centred on the precision-weighted combination -- a finite-sample,
stable counterpart of Bayesian model averaging.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, nnls
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, Ridge
from sklearn.preprocessing import RobustScaler

from .data import arima_forecast_feature, select_features

DEFAULT_TAU = 1e4


def base_learners():
    """Five complementary regularized linear models."""
    return {
        "bayesian_ridge": BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6),
        "huber": HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000),
        "ridge": Ridge(alpha=1.0),
        "elastic_net": ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=5000),
        "lasso": Lasso(alpha=0.01, max_iter=5000),
    }


def _nested_oof(x_scaled: np.ndarray, y: np.ndarray):
    """Out-of-fold base-learner predictions via an inner expanding window."""
    n = len(y)
    inner_start = max(5, n - 6)
    preds, targets = [], []
    for i in range(inner_start, n):
        row = []
        for model in base_learners().values():
            model.fit(x_scaled[:i], y[:i])
            row.append(float(model.predict(x_scaled[i : i + 1])[0]))
        preds.append(row)
        targets.append(float(y[i]))
    return np.asarray(preds), np.asarray(targets)


def shrinkage_weights(oof_pred: np.ndarray, oof_true: np.ndarray, tau: float = DEFAULT_TAU):
    """Non-negative simplex weights minimising OOF error with shrinkage toward
    the inverse-variance prior."""
    k = oof_pred.shape[1]
    mse = np.mean((oof_pred - oof_true[:, None]) ** 2, axis=0)
    prior = 1.0 / (mse + 1e-9)
    prior /= prior.sum()

    def objective(w):
        return np.mean((oof_pred @ w - oof_true) ** 2) + tau * np.sum((w - prior) ** 2)

    res = minimize(
        objective,
        prior,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = res.x if res.success else prior
    w = np.clip(w, 0.0, None)
    return w / w.sum() if w.sum() > 0 else prior


def _combination_weights(oof_pred, oof_true, scheme, tau):
    names = oof_pred.shape[1]
    if oof_pred.shape[0] < 3:
        return np.ones(names) / names
    if scheme == "bayes_shrink":
        return shrinkage_weights(oof_pred, oof_true, tau)
    if scheme == "invvar":
        w = 1.0 / (np.mean((oof_pred - oof_true[:, None]) ** 2, axis=0) + 1e-9)
        return w / w.sum()
    if scheme == "nnls":
        w, _ = nnls(oof_pred, oof_true)
        return w / w.sum() if w.sum() > 1e-9 else np.ones(names) / names
    return np.ones(names) / names  # mean


class XBEE:
    """ARIMA-augmented Bayesian-shrinkage ensemble.

    Parameters
    ----------
    aggregation : {"bayes_shrink", "invvar", "nnls", "mean", "median"}
        Combination rule. ``bayes_shrink`` is the proposed method.
    n_features : int
        Number of mutual-information-selected predictors per fold.
    tau : float
        Shrinkage strength toward the inverse-variance prior.
    """

    def __init__(self, aggregation: str = "bayes_shrink", n_features: int = 8, tau: float = DEFAULT_TAU):
        self.aggregation = aggregation
        self.n_features = n_features
        self.tau = tau

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.selected_ = select_features(X, y, self.n_features)
        fitted, self.arima_forecast_, self.arima_order_ = arima_forecast_feature(y)
        x_aug = np.hstack([X[:, self.selected_], fitted.reshape(-1, 1)])
        self.scaler_ = RobustScaler()
        x_scaled = self.scaler_.fit_transform(x_aug)

        self.models_ = base_learners()
        for model in self.models_.values():
            model.fit(x_scaled, y.astype(float))

        if self.aggregation == "median":
            self.weights_ = None
        else:
            oof_pred, oof_true = _nested_oof(x_scaled, y.astype(float))
            self.weights_ = _combination_weights(oof_pred, oof_true, self.aggregation, self.tau)
        return self

    def _augment(self, X: np.ndarray) -> np.ndarray:
        col = np.full((X.shape[0], 1), self.arima_forecast_)
        return self.scaler_.transform(np.hstack([X[:, self.selected_], col]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_scaled = self._augment(X)
        member = np.column_stack([m.predict(x_scaled) for m in self.models_.values()])
        self.member_predictions_ = member
        if self.weights_ is None:
            return np.median(member, axis=1)
        return member @ self.weights_

    def prediction_uncertainty(self) -> float:
        """Disagreement (std) across base learners for the last prediction."""
        return float(np.std(self.member_predictions_[0]))

    def effective_linear_model(self):
        """Coefficients and intercept of the equivalent linear model, used for
        exact SHAP attribution. Only defined for weighted (non-median) schemes."""
        if self.weights_ is None:
            raise ValueError("effective_linear_model requires a weighted aggregation")
        names = list(self.models_)
        coef = np.sum([self.weights_[i] * self.models_[names[i]].coef_ for i in range(len(names))], axis=0)
        intercept = float(np.sum([self.weights_[i] * self.models_[names[i]].intercept_ for i in range(len(names))]))
        return coef, intercept
