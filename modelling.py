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

    def forecast_rolling_strided(
        self,
        y_test: pd.Series,
        stride: int = 24,
    ) -> pd.Series:
        """
        Strided rolling forecast: predict `stride` steps ahead at a time,
        then update the state with the corresponding `stride` actual values
        (no parameter re-estimation). With stride=24 on hourly data this is
        the standard "day-ahead" forecasting protocol used in operational
        electric load forecasting.

        Parameters
        ----------
        y_test : pandas.Series
            Test series, hourly indexed, immediately following the training
            series.
        stride : int
            Forecasting horizon in number of observations between successive
            state updates. Default 24 (= one day on hourly data).

        Returns
        -------
        pandas.Series
            Forecast indexed exactly as `y_test`. The last block may be
            shorter than `stride` if `len(y_test)` is not a multiple of it.
        """
        if self.results_ is None:
            raise RuntimeError("Call .fit() before .forecast_rolling_strided().")

        results = self.results_
        n = len(y_test)
        preds = np.zeros(n, dtype=float)

        i = 0
        while i < n:
            block_len = min(stride, n - i)
            # 1. Predict `block_len` steps ahead from the current state.
            block_pred = results.forecast(steps=block_len)
            preds[i : i + block_len] = block_pred.to_numpy()
            # 2. Update the state with the actual block (no refit).
            results = results.append(y_test.iloc[i : i + block_len], refit=False)
            i += block_len

        return pd.Series(preds, index=y_test.index, name="sarima_forecast")
        

    def summary(self):
        """Return the statsmodels summary table (used in the report)."""
        if self.results_ is None:
            raise RuntimeError("Call .fit() before .summary().")
        return self.results_.summary()


# ---------------------------------------------------------------------------
# 3. LSTM
# ---------------------------------------------------------------------------

def _make_sliding_windows(
    series: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) sliding-window pairs for one-step-ahead training.

    Given a 1-D array of length T, produces:
    - X of shape (T - lookback, lookback, 1): each row is `lookback`
      consecutive past observations.
    - y of shape (T - lookback,): the next observation immediately after
      the window.

    Used internally by `LSTMForecaster.fit`.

    Parameters
    ----------
    series : numpy.ndarray (1-D)
        Time series (already standardised).
    lookback : int
        Number of past observations used as input.
    """
    series = np.asarray(series, dtype=np.float32).reshape(-1)
    n = len(series)
    n_samples = n - lookback
    if n_samples <= 0:
        raise ValueError(
            f"Series too short ({n}) for the requested lookback={lookback}."
        )
    X = np.zeros((n_samples, lookback, 1), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        X[i, :, 0] = series[i : i + lookback]
        y[i] = series[i + lookback]
    return X, y


class LSTMForecaster:
    """
    Single-layer LSTM for one-step-ahead univariate hourly forecasting.

    Architecture
    ------------
    Input  : (batch, lookback, 1)
    LSTM   : hidden_size units, num_layers stacked
    Linear : hidden_size -> 1
    Output : (batch,)

    The series is standardised internally using the training-set mean and
    standard deviation (the same statistics are then applied to the test
    set, so no information leaks from test to train).

    Forecasting protocol
    --------------------
    `forecast_rolling(y_test)` performs rolling one-step-ahead prediction:
    at each test timestamp t the LSTM observes the *actual* last
    `lookback` hours (taken from train+test history) and predicts the
    value at t. Identical comparison protocol to the Naive baseline and
    SARIMA.

    Parameters
    ----------
    lookback : int
        Number of past hours used as input. Default 48 (= 2 days).
    hidden_size : int
        Number of LSTM hidden units. Default 64.
    num_layers : int
        Number of stacked LSTM layers. Default 1.
    dropout : float
        Dropout probability between LSTM layers (active only if
        num_layers > 1).
    learning_rate : float
        Adam learning rate. Default 1e-3.
    n_epochs : int
        Number of training epochs. Default 20.
    batch_size : int
        Mini-batch size. Default 128.
    device : str or None
        'cuda' or 'cpu'. If None, auto-detected.
    """

    def __init__(
        self,
        lookback: int = 48,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        n_epochs: int = 20,
        batch_size: int = 128,
        device: str | None = None,
    ):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        self.model_ = None
        self.scaler_mean_: float | None = None
        self.scaler_std_: float | None = None
        self.history_: pd.Series | None = None
        self.loss_curve_: list = []

    # -- internal helpers ---------------------------------------------------

    def _build_model(self):
        """Construct the PyTorch LSTM module."""
        import torch.nn as nn

        class _LSTMNet(nn.Module):
            def __init__(self, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x):
                # x: (batch, lookback, 1)
                out, _ = self.lstm(x)
                last = out[:, -1, :]              # take the last hidden state
                return self.head(last).squeeze(-1)

        return _LSTMNet(self.hidden_size, self.num_layers, self.dropout)

    def _resolve_device(self):
        import torch
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _standardise(self, series: np.ndarray) -> np.ndarray:
        return (series - self.scaler_mean_) / self.scaler_std_

    def _inverse_standardise(self, series: np.ndarray) -> np.ndarray:
        return series * self.scaler_std_ + self.scaler_mean_

    # -- public API ---------------------------------------------------------

    def fit(self, y_train: pd.Series) -> "LSTMForecaster":
        """
        Train the LSTM on the standardised training series.

        Standardisation statistics (mean, std) are estimated on the
        training set only and stored for later use on the test set.
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self.history_ = y_train.copy()

        values = y_train.to_numpy(dtype=np.float32)
        self.scaler_mean_ = float(values.mean())
        self.scaler_std_ = float(values.std() + 1e-8)
        scaled = self._standardise(values)

        X, y = _make_sliding_windows(scaled, lookback=self.lookback)

        device = self._resolve_device()
        X_t = torch.from_numpy(X).to(device)
        y_t = torch.from_numpy(y).to(device)
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model_ = self._build_model().to(device)
        optimiser = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()

        self.loss_curve_ = []
        self.model_.train()
        for _ in range(self.n_epochs):
            epoch_loss = 0.0
            n_seen = 0
            for xb, yb in loader:
                optimiser.zero_grad()
                pred = self.model_(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * xb.size(0)
                n_seen += xb.size(0)
            self.loss_curve_.append(epoch_loss / n_seen)
        return self

    def forecast_rolling(self, y_test: pd.Series) -> pd.Series:
        """
        Rolling one-step-ahead forecast on `y_test`.

        At step t the model receives the actual last `lookback` hourly
        values (drawn from the train+test history up to t-1) and predicts
        the value at t.

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
        import torch

        if self.model_ is None:
            raise RuntimeError("Call .fit() before .forecast_rolling().")

        device = self._resolve_device()
        self.model_.eval()

        # Concatenate train and test, standardise with the train statistics.
        full = np.concatenate([
            self.history_.to_numpy(dtype=np.float32),
            y_test.to_numpy(dtype=np.float32),
        ])
        full_scaled = self._standardise(full)
        n_train = len(self.history_)
        n_test = len(y_test)

        preds_scaled = np.zeros(n_test, dtype=np.float32)
        with torch.no_grad():
            for i in range(n_test):
                start = n_train + i - self.lookback
                window = full_scaled[start : n_train + i].reshape(1, self.lookback, 1)
                x = torch.from_numpy(window).to(device)
                preds_scaled[i] = self.model_(x).cpu().item()

        preds = self._inverse_standardise(preds_scaled)
        return pd.Series(preds, index=y_test.index, name="lstm_forecast")


