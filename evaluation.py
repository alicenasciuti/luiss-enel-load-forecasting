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

