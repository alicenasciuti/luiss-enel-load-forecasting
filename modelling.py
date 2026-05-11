from __future__ import annotations

import numpy as np
import pandas as pd


class NaiveSeasonalForecaster:

    def __init__(self, period: int = 24):
        self.period = period
        self.history_: pd.Series | None = None

    def fit(self, y_train: pd.Series) -> "NaiveSeasonalForecaster":
        self.history_ = y_train.copy()
        return self

    def forecast_rolling(self, y_test: pd.Series) -> pd.Series:
        if self.history_ is None:
            raise RuntimeError("Call .fit() before .forecast_rolling().")

        full = pd.concat([self.history_, y_test])
        shifted = full.shift(self.period)
        preds = shifted.loc[y_test.index]
        preds.name = "naive_forecast"
        return preds

    def forecast_rolling_strided(
        self,
        y_test: pd.Series,
        stride: int = 24,
    ) -> pd.Series:
        if self.history_ is None:
            raise RuntimeError("Call .fit() before .forecast_rolling_strided().")

        full = pd.concat([self.history_, y_test])
        shifted = full.shift(self.period)
        preds = shifted.loc[y_test.index]
        preds.name = "naive_forecast"
        return preds


class SARIMAForecaster:

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
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model = SARIMAX(
            y_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        self.results_ = model.fit(disp=False)
        return self

    def forecast_static(self, steps: int) -> pd.Series:
        if self.results_ is None:
            raise RuntimeError("Call .fit() before .forecast_static().")
        forecast = self.results_.get_forecast(steps=steps)
        mean = forecast.predicted_mean
        mean.name = "sarima_forecast_static"
        return mean

    def forecast_rolling(self, y_test: pd.Series) -> pd.Series:
        if self.results_ is None:
            raise RuntimeError("Call .fit() before .forecast_rolling().")

        results = self.results_
        preds = np.zeros(len(y_test), dtype=float)

        for i in range(len(y_test)):
            preds[i] = float(results.forecast(steps=1).iloc[0])
            results = results.append(y_test.iloc[i : i + 1], refit=False)

        return pd.Series(preds, index=y_test.index, name="sarima_forecast")

    def forecast_rolling_strided(
        self,
        y_test: pd.Series,
        stride: int = 24,
    ) -> pd.Series:
        if self.results_ is None:
            raise RuntimeError("Call .fit() before .forecast_rolling_strided().")

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        n = len(y_test)
        preds = np.zeros(n, dtype=float)

        full_series = pd.concat([self.results_.data.orig_endog, y_test])
        params = self.results_.params

        i = 0
        while i < n:
            block_len = min(stride, n - i)
            end_idx = len(full_series) - (n - i)
            current_data = full_series.iloc[:end_idx]

            tmp_model = SARIMAX(
                current_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
            )
            tmp_results = tmp_model.filter(params)
            block_pred = tmp_results.forecast(steps=block_len)
            preds[i : i + block_len] = block_pred.to_numpy()
            i += block_len

        return pd.Series(preds, index=y_test.index, name="sarima_forecast")

    def summary(self):
        if self.results_ is None:
            raise RuntimeError("Call .fit() before .summary().")
        return self.results_.summary()


def _make_sliding_windows(
    series: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
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

    def _build_model(self):
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
                out, _ = self.lstm(x)
                last = out[:, -1, :]
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

    def fit(self, y_train: pd.Series) -> "LSTMForecaster":
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
        import torch

        if self.model_ is None:
            raise RuntimeError("Call .fit() before .forecast_rolling().")

        device = self._resolve_device()
        self.model_.eval()

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

    def forecast_rolling_strided(
        self,
        y_test: pd.Series,
        stride: int = 24,
    ) -> pd.Series:
        import torch

        if self.model_ is None:
            raise RuntimeError("Call .fit() before .forecast_rolling_strided().")

        device = self._resolve_device()
        self.model_.eval()

        full = np.concatenate([
            self.history_.to_numpy(dtype=np.float32),
            y_test.to_numpy(dtype=np.float32),
        ])
        full_scaled = self._standardise(full)
        n_train = len(self.history_)
        n_test = len(y_test)

        preds_scaled = np.zeros(n_test, dtype=np.float32)
        with torch.no_grad():
            i = 0
            while i < n_test:
                block_len = min(stride, n_test - i)
                window = full_scaled[n_train + i - self.lookback : n_train + i].copy()
                for k in range(block_len):
                    x = torch.from_numpy(window.reshape(1, self.lookback, 1)).to(device)
                    p = self.model_(x).cpu().item()
                    preds_scaled[i + k] = p
                    window = np.concatenate([window[1:], [p]])
                i += block_len

        preds = self._inverse_standardise(preds_scaled)
        return pd.Series(preds, index=y_test.index, name="lstm_forecast")
