"""Data loading and feature engineering for the Spanish ICT nowcasting study.

The target is quarterly CNMC telecommunications revenue. Predictors combine
quarterly subscriber data, monthly Google Trends aggregated to quarters, and
annual INE/OECD indicators expanded to quarterly frequency. All engineered
predictors use only information dated strictly before the forecast quarter.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.arima.model import ARIMA

TARGET = "ingresos_millones_eur"
TREND_TERMS = ("tecnología", "software", "telecomunicaciones", "informática")
ARIMA_GRID = ((1, 1, 0), (1, 1, 1), (2, 1, 1), (2, 1, 0), (0, 1, 1))


def load_dataset(data_dir: str) -> pd.DataFrame:
    """Merge the four public sources into a single quarterly frame."""
    cnmc = pd.read_csv(os.path.join(data_dir, "cnmc_telecomunicaciones_trimestral.csv"))
    cnmc["year"] = cnmc["periodo"].str[:4].astype(int)
    cnmc["quarter"] = cnmc["periodo"].str[-1].astype(int)
    cnmc["date"] = pd.to_datetime(
        cnmc["year"].astype(str) + "-" + (cnmc["quarter"] * 3).astype(str).str.zfill(2) + "-01"
    ) + pd.offsets.MonthEnd(0)
    cnmc = cnmc.set_index("date").sort_index()

    trends = pd.read_csv(os.path.join(data_dir, "google_trends_spain_ict.csv"))
    trends["date"] = pd.to_datetime(trends["date"])
    trends = trends.set_index("date").sort_index()
    trends_q = pd.DataFrame()
    for term in TREND_TERMS:
        if term in trends.columns:
            trends_q[f"trends_{term}_mean"] = trends[term].resample("QE").mean()
            trends_q[f"trends_{term}_std"] = trends[term].resample("QE").std()

    ine = pd.read_csv(os.path.join(data_dir, "ine_sector_ict_spain.csv"))
    ine["date"] = pd.to_datetime(ine["año"].astype(str) + "-12-31")
    ine_q = ine.set_index("date").sort_index().resample("QE").ffill()

    oecd = pd.read_csv(os.path.join(data_dir, "oecd_stan_spain_ict.csv"))
    oecd["date"] = pd.to_datetime(oecd["year"].astype(str) + "-12-31")
    oecd_q = (
        oecd.set_index("date").sort_index().drop(columns=["year", "country"]).resample("QE").ffill()
    )

    df = (
        cnmc.join(trends_q, how="left")
        .join(ine_q, how="left", rsuffix="_ine")
        .join(oecd_q, how="left", rsuffix="_oecd")
    )
    return df.ffill().bfill()


def build_features(df: pd.DataFrame):
    """Return (feature_frame, target_array, feature_names, dates).

    Revenue- and line-derived features are built from lagged values only, so no
    contemporaneous target information enters the predictors.
    """
    feat = pd.DataFrame(index=df.index)
    for lag in (1, 2, 3):
        feat[f"revenue_lag{lag}"] = df[TARGET].shift(lag)
    feat["lines_lag1"] = df["lineas_moviles_millones"].shift(1)
    feat["delta_revenue_lag1"] = df[TARGET].diff().shift(1)
    feat["delta_revenue_lag2"] = df[TARGET].diff().shift(2)
    feat["rpl_lag1"] = df[TARGET].shift(1) / df["lineas_moviles_millones"].shift(1)
    feat["lines_qoq"] = df["lineas_moviles_millones"].pct_change(1) * 100
    feat["revenue_qoq_lag1"] = df[TARGET].shift(1).pct_change(1) * 100

    for col in df.columns:
        if col.startswith("trends_"):
            feat[col] = df[col]
    for col in df.columns:
        if col.startswith("trends_") and col.endswith("_mean"):
            feat[f"{col}_diff"] = df[col].diff()

    for col in ("cifra_negocios_millones_eur", "personal_ocupado", "gasto_id_millones_eur"):
        if col in df.columns:
            feat[f"ine_{col}"] = df[col]
    for col in ("ict_value_added_millions_usd", "ict_employment_thousands"):
        if col in df.columns:
            feat[f"oecd_{col}"] = df[col]

    feat["Q1"] = (df.index.quarter == 1).astype(int)
    feat["Q2"] = (df.index.quarter == 2).astype(int)
    feat["Q3"] = (df.index.quarter == 3).astype(int)
    feat["Q4"] = (df.index.quarter == 4).astype(int)
    feat["time_trend"] = np.arange(len(df))

    feat = feat[feat.notna().all(axis=1)]
    y = df.loc[feat.index, TARGET].to_numpy()
    keep = feat.std()[feat.std() > 1e-10].index.tolist()
    feat = feat[keep]
    return feat, y, list(feat.columns), feat.index


def expanding_window_folds(n: int, min_train: int = 10):
    """One-step-ahead expanding window: train on [0, i), test on {i}."""
    return [(np.arange(i), np.array([i])) for i in range(min_train, n)]


def select_features(x_train: np.ndarray, y_train: np.ndarray, k: int = 8) -> np.ndarray:
    """Top-k feature indices by mutual information, fit on the training fold only."""
    mi = mutual_info_regression(x_train, y_train, random_state=42, n_neighbors=3)
    return np.argsort(mi)[-k:][::-1]


def arima_forecast_feature(y_train: np.ndarray):
    """Fit an AIC-selected ARIMA on the training target.

    Returns the in-sample fitted values (aligned to the training window) and the
    one-step-ahead forecast, used as an additional predictor for the ensemble.
    """
    best_order, best_aic, fits = (1, 1, 1), np.inf, {}
    for order in ARIMA_GRID:
        try:
            fit = ARIMA(y_train, order=order).fit()
            fits[order] = fit
            if fit.aic < best_aic:
                best_aic, best_order = fit.aic, order
        except Exception:
            continue
    fit = fits.get(best_order) or ARIMA(y_train, order=(1, 1, 1)).fit()
    fitted = np.asarray(fit.fittedvalues, dtype=float)
    if len(fitted) < len(y_train):
        fitted = np.concatenate([np.full(len(y_train) - len(fitted), fitted[0]), fitted])
    forecast = float(np.asarray(fit.forecast(steps=1)).reshape(-1)[0])
    return fitted, forecast, best_order
