"""
File Title : Pipeline Orchestrator
File Name  : main.py

Description:
End-to-end orchestration script that runs the complete forecasting
pipeline from raw data ingestion to final results. This file is
expected to coordinate dataset acquisition, preprocessing,
chronological train/test splitting, fitting of the baseline,
statistical (SARIMA) and deep learning (LSTM) forecasters,
rolling day-ahead forecasting, computation of evaluation metrics,
generation of comparison plots, and persistence of all final
artefacts (metrics tables and figures) to disk.

Role in Project:
Acts as the single executable entry point of the project pipeline,
satisfying the reproducibility requirement that the source code
must be runnable in its entirety. This module wires together
data_loader, preprocessing, modelling and evaluation components
to produce, in one shot, the numerical and visual results that
back the technical report and the executive summary.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import ensure_dataset, load_raw_data
from preprocessing import run_preprocessing_pipeline
from modelling import (
    NaiveSeasonalForecaster,
    SARIMAForecaster,
    LSTMForecaster,
)
from evaluation import compute_metrics, compare_models, plot_actual_vs_predicted, plot_residuals
from utils import set_global_seed



HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CACHE_DIR = HERE / "cache"
OUTPUTS_DIR = HERE / "outputs"

RAW_FILE = DATA_DIR / "household_power_consumption.txt"
TARGET_COL = "Global_active_power"
TEST_SIZE = 0.2
SEED = 42


SARIMA_ORDER = (2, 0, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 24)
SARIMA_TRAIN_WINDOW_HOURS = 24 * 365  


LSTM_LOOKBACK = 48
LSTM_HIDDEN = 64
LSTM_LAYERS = 1
LSTM_EPOCHS = 20
LSTM_BATCH = 128
LSTM_LR = 1e-3

STRIDE = 24  



def _load_cached_predictions(path: Path, name: str) -> pd.Series | None:
    """Return cached predictions as a Series if `path` exists, else None."""
    if not path.exists():
        return None
    print(f"[main] Loading cached predictions: {path.name}")
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
    s.name = name
    return s


def _save_predictions(series: pd.Series, path: Path) -> None:
    """Persist a prediction Series to CSV (single column with header)."""
    series.to_csv(path, header=True)
    print(f"[main] Saved predictions to: {path.name}")



def run_naive(y_train: pd.Series, y_test: pd.Series, use_cache: bool) -> pd.Series:
    cache = CACHE_DIR / "y_pred_naive.csv"
    if use_cache:
        cached = _load_cached_predictions(cache, "naive_forecast")
        if cached is not None:
            return cached
    print("[main] Fitting Naive seasonal baseline (period=24)...")
    model = NaiveSeasonalForecaster(period=24).fit(y_train)
    preds = model.forecast_rolling_strided(y_test, stride=STRIDE)
    _save_predictions(preds, cache)
    return preds


def run_sarima(y_train: pd.Series, y_test: pd.Series, use_cache: bool) -> pd.Series:
    cache = CACHE_DIR / "y_pred_sarima.csv"
    if use_cache:
        cached = _load_cached_predictions(cache, "sarima_forecast")
        if cached is not None:
            return cached
    y_train_window = y_train.iloc[-SARIMA_TRAIN_WINDOW_HOURS:]
    print(
        f"[main] Fitting SARIMA{SARIMA_ORDER}x{SARIMA_SEASONAL_ORDER} "
        f"on {len(y_train_window):,} hours (~2-5 min)..."
    )
    model = SARIMAForecaster(
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
    ).fit(y_train_window)
    print("[main] Running SARIMA rolling forecast (~5-15 min)...")
    preds = model.forecast_rolling_strided(y_test, stride=STRIDE)
    _save_predictions(preds, cache)
    return preds


def run_lstm(y_train: pd.Series, y_test: pd.Series, use_cache: bool) -> pd.Series:
    cache = CACHE_DIR / "y_pred_lstm.csv"
    if use_cache:
        cached = _load_cached_predictions(cache, "lstm_forecast")
        if cached is not None:
            return cached
    print(
        f"[main] Training LSTM (lookback={LSTM_LOOKBACK}, hidden={LSTM_HIDDEN}, "
        f"epochs={LSTM_EPOCHS}) on {len(y_train):,} hours (~3-8 min on CPU)..."
    )
    model = LSTMForecaster(
        lookback=LSTM_LOOKBACK,
        hidden_size=LSTM_HIDDEN,
        num_layers=LSTM_LAYERS,
        learning_rate=LSTM_LR,
        n_epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
    ).fit(y_train)
    print("[main] Running LSTM rolling forecast (~1-2 min)...")
    preds = model.forecast_rolling_strided(y_test, stride=STRIDE)
    _save_predictions(preds, cache)
    return preds


def save_figures(y_test: pd.Series, predictions: dict[str, pd.Series]) -> None:
    """Save the comparative actual-vs-predicted plot and per-model residual plots."""
    
    zoom_start = y_test.index.min()
    zoom_end = zoom_start + pd.Timedelta(days=14)
    fig, _ = plot_actual_vs_predicted(
        y_test,
        predictions,
        title="Actual vs Predicted — first 14 days of test set",
        start=zoom_start,
        end=zoom_end,
    )
    out = OUTPUTS_DIR / "actual_vs_predicted.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[main] Saved figure: {out.name}")

    for name, preds in predictions.items():
        fig, _ = plot_residuals(y_test, preds, title=name)
        out = OUTPUTS_DIR / f"residuals_{name.lower()}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[main] Saved figure: {out.name}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full forecasting pipeline.")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached predictions and re-train all models from scratch.",
    )
    args = parser.parse_args()
    use_cache = not args.no_cache

    set_global_seed(SEED)
    for d in (DATA_DIR, CACHE_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

   
    print("\n=== [1/5] Data acquisition ===")
    ensure_dataset(RAW_FILE)
    df_raw = load_raw_data(RAW_FILE)
    print(f"[main] Raw shape: {df_raw.shape}  |  Range: {df_raw.index.min()} → {df_raw.index.max()}")

    
    print("\n=== [2/5] Preprocessing & train/test split ===")
    df_train, df_test = run_preprocessing_pipeline(
        df_raw,
        target_col=TARGET_COL,
        test_size=TEST_SIZE,
    )
    y_train = df_train[TARGET_COL]
    y_test = df_test[TARGET_COL]
    print(f"[main] Train: {len(y_train):,} hours  |  Test: {len(y_test):,} hours")

    
    print("\n=== [3/5] Models ===")
    y_pred_naive = run_naive(y_train, y_test, use_cache)
    y_pred_sarima = run_sarima(y_train, y_test, use_cache)
    y_pred_lstm = run_lstm(y_train, y_test, use_cache)

    predictions = {
        "Naive": y_pred_naive,
        "SARIMA": y_pred_sarima,
        "LSTM": y_pred_lstm,
    }

    
    print("\n=== [4/5] Evaluation ===")
    tab
