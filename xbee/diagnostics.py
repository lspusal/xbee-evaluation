"""Statistical significance and robustness diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.stats import t as t_dist

from .evaluation import evaluate_xbee


def diebold_mariano(errors_a, errors_b, horizon=1):
    """DM test on the squared-error loss differential (model A vs B) with the
    Harvey-Leybourne-Newbold small-sample correction. Negative statistic favours A."""
    d = np.asarray(errors_a) ** 2 - np.asarray(errors_b) ** 2
    n = len(d)
    d_bar = d.mean()
    lag = max(1, int(np.floor(n ** (1 / 3))))
    var = np.mean((d - d_bar) ** 2)
    for k in range(1, min(lag, n - 1) + 1):
        cov = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        var += 2 * (1 - k / (lag + 1)) * cov
    var /= n
    if var <= 0:
        return float("nan"), float("nan")
    stat = d_bar / np.sqrt(var)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat *= correction
    p_value = 2 * (1 - t_dist.cdf(abs(stat), df=n - 1))
    return float(stat), float(p_value)


def theils_u2(y_true, y_pred, naive_pred):
    num = np.sqrt(np.mean((np.asarray(y_pred) - y_true) ** 2))
    den = np.sqrt(np.mean((np.asarray(naive_pred) - y_true) ** 2))
    return float(num / den)


def directional_accuracy(y_true, y_pred, previous):
    return float(np.mean(np.sign(y_true - previous) == np.sign(np.asarray(y_pred) - previous)))


def block_bootstrap_rmse_ci(y_true, y_pred, n_boot=3000, block=3, seed=42):
    rng = np.random.default_rng(seed)
    err = np.asarray(y_pred) - y_true
    n = len(err)
    n_blocks = int(np.ceil(n / block))
    rmses = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        rmses.append(np.sqrt(np.mean(err[idx] ** 2)))
    return float(np.percentile(rmses, 2.5)), float(np.percentile(rmses, 97.5))


def data_source_ablation(X, y, folds, feature_names):
    """RMSE when each data source is removed from the predictor set."""
    groups = {
        "CNMC autoregressive": [
            i for i, f in enumerate(feature_names)
            if any(s in f for s in ("revenue_lag", "lines_lag", "delta_revenue", "rpl_lag", "lines_qoq", "revenue_qoq"))
        ],
        "Google Trends": [i for i, f in enumerate(feature_names) if f.startswith("trends_")],
        "INE": [i for i, f in enumerate(feature_names) if f.startswith("ine_")],
        "OECD": [i for i, f in enumerate(feature_names) if f.startswith("oecd_")],
    }
    _, _, full = evaluate_xbee(X, y, folds)
    out = {"full": full["rmse"]}
    for name, idx in groups.items():
        keep = [i for i in range(X.shape[1]) if i not in idx]
        _, _, res = evaluate_xbee(X[:, keep], y, folds)
        out[name] = res["rmse"]
    return out


def training_length_sensitivity(X, y, min_train_values=(8, 10, 12, 14)):
    from .data import expanding_window_folds

    out = {}
    for mt in min_train_values:
        folds = expanding_window_folds(len(y), min_train=mt)
        _, _, res = evaluate_xbee(X, y, folds)
        out[mt] = {"rmse": res["rmse"], "n_folds": len(folds)}
    return out
