"""
File Title : Model Evaluation
File Name  : evaluation.py

Description:
Contains the evaluation and performance analysis utilities used
to assess forecasting model outputs against real observed values.
This file is expected to implement metric calculation functions,
prediction comparison systems, residual analysis utilities,
forecast validation logic, and visualization tools for analysing
model accuracy, error distributions, and predictive behaviour
across multiple forecasting approaches.

Role in Project:
Provides the evaluation and validation layer of the project
architecture by transforming model predictions into quantitative
performance metrics and diagnostic visual analyses. This module
interacts with the modelling components to compare forecasting
results, support model selection decisions, validate predictive
quality, and generate the analytical outputs required for the
final reporting and performance assessment stages of the project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _align(y_true: pd.Series, y_pred: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    
    common = y_true.index.intersection(y_pred.index)
    if len(common) == 0:
        raise ValueError("y_true and y_pred have no overlapping index.")
    a = y_true.loc[common].to_numpy(dtype=float)
    b = y_pred.loc[common].to_numpy(dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask]

def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    
    a, b = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    
    a, b = _align(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))

def mape(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-3) -> float:
    
    a, b = _align(y_true, y_pred)
    denom = np.where(np.abs(a) < eps, eps, np.abs(a))
    return float(100.0 * np.mean(np.abs((a - b) / denom)))

def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
    }

def compare_models(
    results: dict[str, pd.Series],
    y_true: pd.Series,
) -> pd.DataFrame:
    
    rows = {name: compute_metrics(y_true, pred) for name, pred in results.items()}
    table = pd.DataFrame(rows).T
    return table.sort_values("RMSE")

def plot_actual_vs_predicted(
    y_true: pd.Series,
    predictions: dict[str, pd.Series],
    title: str = "Actual vs Predicted",
    start=None,
    end=None,
):
    
    fig, ax = plt.subplots(figsize=(14, 4))

    y_plot = y_true.loc[start:end] if (start or end) else y_true
    ax.plot(y_plot.index, y_plot.values,
            label="Actual", color="black", linewidth=1.0)

    palette = ["#3a7bd5", "#e76f51", "#2a9d8f", "#9b5de5", "#f4a261"]
    for i, (name, pred) in enumerate(predictions.items()):
        p_plot = pred.loc[start:end] if (start or end) else pred
        ax.plot(
            p_plot.index, p_plot.values,
            label=name,
            color=palette[i % len(palette)],
            linewidth=0.9,
            alpha=0.85,
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Global active power (kW)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax

def plot_residuals(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "Residuals",
):
    
    common = y_true.index.intersection(y_pred.index)
    a = y_true.loc[common].to_numpy(dtype=float)
    b = y_pred.loc[common].to_numpy(dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    residuals = pd.Series((a - b)[mask], index=common[mask])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(residuals.index, residuals.values,
                 color="#2a9d8f", linewidth=0.6)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title(f"{title} — residuals over time")
    axes[0].set_ylabel("Residual (kW)")
    axes[0].grid(alpha=0.3)

    axes[1].hist(residuals.values, bins=50,
                 color="#3a7bd5", edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title(f"{title} — residuals histogram")
    axes[1].set_xlabel("Residual (kW)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig, axes
