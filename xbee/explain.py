"""Explainability for the fitted X-BEE ensemble.

Because the ensemble is an explicit linear model, SHAP values are exact
(no sampling). We additionally compute permutation importance and aggregated
LIME coefficients and report their rank agreement as a robustness check.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .data import arima_forecast_feature, select_features
from .model import XBEE


def _effective_model_on_full_sample(X, y, n_features=8):
    model = XBEE(aggregation="bayes_shrink", n_features=n_features).fit(X, y)
    sel = model.selected_
    fitted, _, _ = arima_forecast_feature(y)
    x_aug = np.hstack([X[:, sel], fitted.reshape(-1, 1)])
    x_scaled = model.scaler_.transform(x_aug)
    coef, intercept = model.effective_linear_model()
    names = [f"feature_{i}" for i in sel] + ["arima_forecast"]
    return x_scaled, coef, intercept, names, model


def shap_values(x_scaled, coef):
    """Exact SHAP for a linear model: phi_ij = coef_j * (x_ij - mean_j)."""
    return coef[None, :] * (x_scaled - x_scaled.mean(axis=0)[None, :])


def permutation_importance(x_scaled, y, coef, intercept, repeats=50, seed=42):
    rng = np.random.default_rng(seed)
    base = _r2(y, x_scaled @ coef + intercept)
    importance = np.zeros(x_scaled.shape[1])
    for j in range(x_scaled.shape[1]):
        drops = []
        for _ in range(repeats):
            permuted = x_scaled.copy()
            rng.shuffle(permuted[:, j])
            drops.append(base - _r2(y, permuted @ coef + intercept))
        importance[j] = np.mean(drops)
    return importance


def lime_importance(x_scaled, coef, intercept, names, seed=42):
    """Mean absolute LIME coefficient per feature. Returns None if lime is absent."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        return None
    predict = lambda a: a @ coef + intercept
    explainer = LimeTabularExplainer(
        x_scaled, feature_names=names, mode="regression",
        discretize_continuous=False, random_state=seed,
    )
    agg = np.zeros(len(names))
    for i in range(x_scaled.shape[0]):
        contributions = dict(explainer.explain_instance(x_scaled[i], predict, num_features=len(names)).as_map()[1])
        for idx, weight in contributions.items():
            agg[idx] += abs(weight)
    return agg / x_scaled.shape[0]


def cross_method_consistency(shap_imp, perm_imp, lime_imp):
    out = {
        "shap_vs_permutation": float(spearmanr(shap_imp, perm_imp).statistic),
    }
    if lime_imp is not None:
        out["shap_vs_lime"] = float(spearmanr(shap_imp, lime_imp).statistic)
        out["permutation_vs_lime"] = float(spearmanr(perm_imp, lime_imp).statistic)
    return out


def feature_importance_report(X, y, feature_names, n_features=8):
    """Global SHAP / permutation / LIME importances and their rank agreement."""
    x_scaled, coef, intercept, sel_names, model = _effective_model_on_full_sample(X, y, n_features)
    readable = [feature_names[i] for i in model.selected_] + ["arima_forecast"]
    shap_imp = np.abs(shap_values(x_scaled, coef)).mean(axis=0)
    perm_imp = permutation_importance(x_scaled, y, coef, intercept)
    lime_imp = lime_importance(x_scaled, coef, intercept, readable)
    return {
        "feature_names": readable,
        "base_weights": dict(zip(model.models_, model.weights_.tolist())),
        "shap_share_pct": dict(zip(readable, (shap_imp / shap_imp.sum() * 100).tolist())),
        "permutation": dict(zip(readable, perm_imp.tolist())),
        "lime": None if lime_imp is None else dict(zip(readable, lime_imp.tolist())),
        "consistency_spearman": cross_method_consistency(shap_imp, perm_imp, lime_imp),
    }


def cross_fold_stability(X, y, folds, n_features=8):
    """Mean pairwise Spearman correlation of global importances across folds."""
    per_fold = []
    for train_idx, _ in folds:
        x_scaled, coef, _, _, _ = _effective_model_on_full_sample(X[train_idx], y[train_idx], n_features)
        per_fold.append(np.abs(shap_values(x_scaled, coef)).mean(axis=0))
    rhos = [
        spearmanr(per_fold[i], per_fold[j]).statistic
        for i in range(len(per_fold))
        for j in range(i + 1, len(per_fold))
    ]
    return float(np.nanmean(rhos)), len(rhos)


def _r2(y_true, y_pred):
    return 1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
