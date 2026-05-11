"""
evaluation.py
=============

Evaluation utilities for the forecasting models defined in `modelling.py`.

Contents
--------
- rmse(y_true, y_pred)        -> Root Mean Squared Error.
- mae(y_true, y_pred)         -> Mean Absolute Error.
- mape(y_true, y_pred)        -> Mean Absolute Percentage Error
                                 (informational only; the target can take
                                 small values, so MAPE is not used as the
                                 primary metric).
- compute_metrics(y_true, y_pred)
                              -> Dictionary with RMSE, MAE, MAPE.
- compare_models(results, y_true)
                              -> Tidy DataFrame comparing several models,
                                 sorted by RMSE ascending.
- plot_actual_vs_predicted(y_true, predictions, title, start, end)
                              -> Overlay of actual values and one or more
                                 model forecasts (optionally zoomed on a
                                 time window).
- plot_residuals(y_true, y_pred, title)
                              -> Residual plot + histogram for a single
                                 model.

Role in the project
-------------------
After the models in `modelling.py` have been trained and have produced
their forecasts on the test set, this module turns those forecasts into
the quantitative comparison (RMSE, MAE) and the qualitative diagnostic
plots (Actual vs Predicted, residuals) that go into Section 3 of the
technical report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _align(y_true: pd.Series, y_pred: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Align two series on their common index and return clean numpy arrays."""
    common = y_true.index.intersection(y_pred.index)
    if len(common) == 0:
        raise ValueError("y_true and y_pred have no overlapping index.")
    a = y_true.loc[common].to_numpy(dtype=float)
    b = y_pred.loc[common].to_numpy(dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask]


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root Mean Squared Error."""
    a, b = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Error."""
    a, b = _align(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def mape(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-3) -> float:
    """
    Mean Absolute Percentage Error (in percent).

    Not used as the primary metric because the target can be close to zero
    in some hours (low household consumption); `eps` is added to the
    denominator only to avoid division by zero.
    """
    a, b = _align(y_true, y_pred)
    denom = np.where(np.abs(a) < eps, eps, np.abs(a))
    return float(100.0 * np.mean(np.abs((a - b) / denom)))


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Return RMSE, MAE and MAPE in a single dictionary."""
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
    }


def compare_models(
    results: dict[str, pd.Series],
    y_true: pd.Series,
) -> pd.DataFrame:
    """
    Build a comparison table for a collection of model predictions.

    Parameters
    ----------
    results : dict
        Maps model name (str) to its predicted pandas.Series.
    y_true : pandas.Series
        Ground-truth series (test set).

    Returns
    -------
    pandas.DataFrame
        Rows are model names, columns are RMSE, MAE, MAPE_%. Sorted by
        RMSE ascending (best model on top).
    """
    rows = {name: compute_metrics(y_true, pred) for name, pred in results.items()}
    table = pd.DataFrame(rows).T
    return table.sort_values("RMSE")
