"""
modelling.py
============

Forecasting models for the Enel Global ICT project on electric load
forecasting. Three families of models are provided, each with a consistent
`fit` + `forecast_rolling` interface:

1. NaiveSeasonalForecaster  -> trivial baseline (same hour, previous day).
2. SARIMAForecaster         -> classical statistical model (SARIMA).
3. LSTMForecaster           -> deep-learning recurrent model (PyTorch).

Evaluation protocol
-------------------
All three models are compared on the same rolling one-step-ahead protocol:
at each test timestamp t, the model observes the actual values up to t-1
and predicts the value at t. This is the standard fair-comparison set-up
between statistical and recurrent models on hourly load data.

For SARIMA the rolling update is implemented efficiently via
`SARIMAXResults.append(..., refit=False)`, which updates the Kalman filter
state without re-estimating the parameters at every step.

Role in the project
-------------------
This module is invoked after preprocessing. Its outputs (numerical
forecasts) are then fed into `evaluation.py` to compute RMSE / MAE and to
produce the Actual-vs-Predicted and residual plots reported in Section 3
of the technical report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Naive seasonal baseline
# ---------------------------------------------------------------------------

class NaiveSeasonalForecaster:
    """
    Seasonal naive baseline: the forecast for time t is the observed value
    at time t - period. With period=24 on hourly data, this means
    "the load at hour H of day D+1 equals the load at hour H of day D".

    This is the standard, intentionally trivial baseline against which
    SARIMA and LSTM are compared. A serious model must beat it.

    Parameters
    ----------
    period : int
        Seasonal period in number of observations (24 for daily seasonality
        on hourly data).
    """

    def __init__(self, period: int = 24):
        self.period = period
        self.history_: pd.Series | None = None

    def fit(self, y_train: pd.Series) -> "NaiveSeasonalForecaster":
        """Store the training series. Nothing is learned."""
        self.history_ = y_train.copy()
        return self

    def forecast_rolling(self, y_test: pd.Series) -> pd.Series:
        """
        Rolling one-step-ahead forecast on `y_test`.

        For each timestamp t in the test set, the prediction is the actual
        observed value at t - period. The "actual" history is the union of
        train and test, so as t walks through the test set the lookup
        progressively uses test values themselves (one-step-ahead protocol).

        Parameters
        ----------
        y_test : pandas.Series
            Test series, hourly indexed, immediately following the training
            series.

        Returns
        -------
        pandas.Series
            Forecast indexed exactly as `y_test`.
        """
        if self.history_ is None:
            raise RuntimeError("Call .fit() before .forecast_rolling().")

        # Concatenate train + test, then shift by `period` so that each
        # position t holds the value observed `period` steps earlier.
        full = pd.concat([self.history_, y_test])
        shifted = full.shift(self.period)
        preds = shifted.loc[y_test.index]
        preds.name = "naive_forecast"
        return preds


