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


# ---------------------------------------------------------------------------
# 2. SARIMA
# ---------------------------------------------------------------------------

class SARIMAForecaster:
    """
    SARIMA(p, d, q)(P, D, Q, s) wrapper around `statsmodels.SARIMAX`.

    The default seasonal period s=24 captures the daily cycle that the EDA
    identified as the strongest seasonality of the household load series.
    The default order (2,0,1)(1,1,1,24) follows the ACF/PACF analysis on
    the training set (see Step 4.3 of the notebook): two AR lags, one MA
    lag, one seasonal AR + MA lag, and one seasonal differentiation to
    handle the daily periodicity.

    `enforce_stationarity` and `enforce_invertibility` are set to False by
    default: on long, noisy real-world series, enforcing them often
    prevents the optimiser from converging without a meaningful gain in
    fit quality. This is the standard set-up reported in the applied
    literature on energy load forecasting.

    Parameters
    ----------
    order : tuple of int
        Non-seasonal (p, d, q) ARIMA order.
    seasonal_order : tuple of int
        Seasonal (P, D, Q, s) order. s should match the dominant seasonal
        period (24 for daily seasonality on hourly data).
    enforce_stationarity, enforce_invertibility : bool
        Forwarded to SARIMAX. Default False (see above).
    """

    def __init__(
        self,
        order: tuple = (2, 0, 1),
        seasonal_order: tuple = (1, 1, 1, 24),
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.results_ = None

    def fit(self, y_train: pd.Series) -> "SARIMAForecaster":
        """
        Fit the SARIMAX model on `y_train`.

        Note: SARIMA fitting on long hourly series is computationally
        expensive. For this project we restrict the training window to the
        last year of the train set (see the notebook). LSTM is instead
        trained on the full train set, since recurrent models benefit
        from more data.
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model = SARIMAX(
            y_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        # `disp=False` suppresses the verbose optimiser output.
        self.results_ = model.fit(disp=False)
        return self

    def forecast_static(self, steps: int) -> pd.Series:
        """
        Static multi-step forecast: predict `steps` hours ahead starting
        immediately after the training series, without observing any new
        data along the way.

        Used for the diagnostic "one-week forecast" plot in the report.
        """
        if self.results_ is None:
            raise RuntimeError("Call .fit() before .forecast_static().")
        forecast = self.results_.get_forecast(steps=steps)
        mean = forecast.predicted_mean
        mean.name = "sarima_forecast_static"
        return mean

    def forecast_rolling(self, y_test: pd.Series) -> pd.Series:
        """
        Rolling one-step-ahead forecast on `y_test`.

        At each step t the model:
        1. predicts the value at t using its current state;
        2. appends the *actual* observed value y_test[t] to the state
           (without re-estimating the parameters: `refit=False`).

        This is the same fair-comparison protocol used for the Naive
        baseline and the LSTM model.

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
        if self.results_ is None:
            raise RuntimeError("Call .fit() before .forecast_rolling().")

        results = self.results_
        preds = np.zeros(len(y_test), dtype=float)

        for i in range(len(y_test)):
            # 1. Predict one step ahead from the current state.
            preds[i] = float(results.forecast(steps=1).iloc[0])
            # 2. Update the state with the actual observed value (no refit).
            results = results.append(y_test.iloc[i : i + 1], refit=False)

        return pd.Series(preds, index=y_test.index, name="sarima_forecast")

    def summary(self):
        """Return the statsmodels summary table (used in the report)."""
        if self.results_ is None:
            raise RuntimeError("Call .fit() before .summary().")
        return self.results_.summary()


