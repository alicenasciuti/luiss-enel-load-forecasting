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


