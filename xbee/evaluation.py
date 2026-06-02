"""Expanding-window evaluation under a single, leakage-free protocol.

Every multivariate competitor receives the same feature set, including the
shared ARIMA one-step forecast, so that any advantage of X-BEE reflects its
combination architecture rather than privileged information.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .data import arima_forecast_feature, select_features
from .model import XBEE


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(1.0 - ss_res / ss_tot),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def augmented_fold(X, y, train_idx, test_idx, k=8):
    """Build train/test matrices with MI selection and the shared ARIMA feature."""
    sel = select_features(X[train_idx], y[train_idx], k)
    fitted, forecast, _ = arima_forecast_feature(y[train_idx])
    x_tr = np.hstack([X[train_idx][:, sel], fitted.reshape(-1, 1)])
    x_te = np.hstack([X[test_idx][:, sel], np.full((len(test_idx), 1), forecast)])
    return x_tr, x_te


def evaluate_xbee(X, y, folds, aggregation="bayes_shrink"):
    preds, actuals, weights, uncertainty = [], [], [], []
    for train_idx, test_idx in folds:
        model = XBEE(aggregation=aggregation).fit(X[train_idx], y[train_idx])
        preds.append(float(model.predict(X[test_idx])[0]))
        actuals.append(float(y[test_idx][0]))
        weights.append(None if model.weights_ is None else model.weights_.tolist())
        uncertainty.append(model.prediction_uncertainty())
    preds, actuals = np.asarray(preds), np.asarray(actuals)
    return preds, actuals, {**metrics(actuals, preds), "weights": weights,
                            "uncertainty": float(np.mean(uncertainty))}


def evaluate_estimator(factory, X, y, folds, k=8):
    """Evaluate any scikit-learn-style regressor on the shared augmented features."""
    preds, actuals = [], []
    for train_idx, test_idx in folds:
        x_tr, x_te = augmented_fold(X, y, train_idx, test_idx, k)
        scaler = RobustScaler()
        model = factory()
        model.fit(scaler.fit_transform(x_tr), y[train_idx])
        preds.append(float(np.atleast_1d(model.predict(scaler.transform(x_te)))[0]))
        actuals.append(float(y[test_idx][0]))
    return np.asarray(preds), metrics(actuals, preds)


def evaluate_arima(y, folds, order=(1, 1, 1)):
    preds, actuals = [], []
    for train_idx, test_idx in folds:
        try:
            preds.append(float(np.asarray(ARIMA(y[train_idx], order=order).fit().forecast(1)).reshape(-1)[0]))
        except Exception:
            preds.append(float(y[train_idx][-1]))
        actuals.append(float(y[test_idx][0]))
    return np.asarray(preds), metrics(actuals, preds)


def evaluate_sarima(y, folds):
    preds, actuals = [], []
    for train_idx, test_idx in folds:
        try:
            fit = SARIMAX(y[train_idx], order=(1, 1, 1), seasonal_order=(1, 0, 1, 4)).fit(
                disp=False, maxiter=200
            )
            preds.append(float(np.asarray(fit.forecast(1)).reshape(-1)[0]))
        except Exception:
            preds.append(float(y[train_idx][-1]))
        actuals.append(float(y[test_idx][0]))
    return np.asarray(preds), metrics(actuals, preds)


def evaluate_random_walk(y, folds):
    preds = [float(y[tr][-1]) for tr, _ in folds]
    actuals = [float(y[te][0]) for _, te in folds]
    return np.asarray(preds), metrics(actuals, preds)
