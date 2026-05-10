"""
eda.py
======

Exploratory Data Analysis (EDA) for the household power consumption dataset.

Contents
--------
- describe_dataset(df): basic shape, time range and descriptive statistics.
- analyse_missing_values(df): per-column count and percentage of NaN values,
  plus longest contiguous run of missing values for each column.
- analyse_temporal_continuity(df, freq='1min'): check whether the timestamp
  index is continuous at the expected frequency.
- locate_missing_blocks(df, col, min_gap_minutes=60): list the time windows
  in which `col` has consecutive NaN values longer than `min_gap_minutes`.
- plot_full_series(df, col, freq='D', ax=None): plot the time series of `col`
  resampled at `freq` (default daily mean) over the whole period.
- plot_zoom(df, col, start, end, ax=None): plot `col` between `start` and
  `end` (string dates).
- plot_seasonal_boxplots(df, col): three boxplots side by side, by hour of
  day, day of week and month.

Role in the project
-------------------
This module contains the functions used during the exploratory phase of the
project. It does not modify the input DataFrame. The output of these functions
(numerical summaries and matplotlib figures) is what drives the design choices
documented in the technical report (target variable, sampling frequency,
model hyperparameters).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --- Quality checks --------------------------------------------------------

def describe_dataset(df: pd.DataFrame) -> dict:
    """
    Return a dictionary with high-level information about the dataset.
    """
    info = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "start": df.index.min(),
        "end": df.index.max(),
        "duration_days": (df.index.max() - df.index.min()).days,
        "dtypes": df.dtypes.to_dict(),
        "describe": df.describe(),
    }
    return info


def analyse_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-column missing-value statistics: count, percentage and
    longest contiguous run of NaN.
    """
    out = pd.DataFrame(index=df.columns)
    out["missing_count"] = df.isna().sum()
    out["missing_pct"] = 100.0 * out["missing_count"] / len(df)

    longest_gap = {}
    for col in df.columns:
        is_na = df[col].isna().to_numpy()
        if not is_na.any():
            longest_gap[col] = 0
            continue
        max_run = 0
        run = 0
        for v in is_na:
            if v:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        longest_gap[col] = max_run

    out["longest_gap_minutes"] = pd.Series(longest_gap)
    return out


def analyse_temporal_continuity(df: pd.DataFrame, freq: str = "1min") -> dict:
    """
    Check whether the DataFrame index is continuous at the given frequency.
    """
    full_range = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    expected_n = len(full_range)
    actual_n = len(df.index.unique())
    n_missing = expected_n - actual_n

    deltas = df.index.to_series().diff().dropna()
    expected_step = pd.Timedelta(freq)
    longest_gap = deltas.max() if len(deltas) else pd.Timedelta(0)

    return {
        "expected_n": expected_n,
        "actual_n": actual_n,
        "n_missing_timestamps": n_missing,
        "pct_missing_timestamps": 100.0 * n_missing / expected_n,
        "expected_step": expected_step,
        "longest_continuous_gap": longest_gap,
    }


def locate_missing_blocks(
    df: pd.DataFrame,
    col: str,
    min_gap_minutes: int = 60,
) -> pd.DataFrame:
    """
    Locate contiguous blocks of NaN longer than `min_gap_minutes` in `col`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame indexed by timestamp.
    col : str
        Column name to inspect.
    min_gap_minutes : int
        Minimum length (in minutes) of NaN runs to be reported.

    Returns
    -------
    pandas.DataFrame
        Columns: ['start', 'end', 'duration_minutes'], one row per block.
    """
    is_na = df[col].isna().to_numpy()
    timestamps = df.index.to_numpy()

    blocks = []
    in_run = False
    run_start = None

    for i, v in enumerate(is_na):
        if v and not in_run:
            in_run = True
            run_start = i
        elif not v and in_run:
            in_run = False
            duration = i - run_start
            if duration >= min_gap_minutes:
                blocks.append({
                    "start": pd.Timestamp(timestamps[run_start]),
                    "end": pd.Timestamp(timestamps[i - 1]),
                    "duration_minutes": duration,
                })
    # Catch a run that ends at the very last row.
    if in_run:
        duration = len(is_na) - run_start
        if duration >= min_gap_minutes:
            blocks.append({
                "start": pd.Timestamp(timestamps[run_start]),
                "end": pd.Timestamp(timestamps[-1]),
                "duration_minutes": duration,
            })

    return pd.DataFrame(blocks)


# --- Time-series visualisations --------------------------------------------

def plot_full_series(
    df: pd.DataFrame,
    col: str,
    freq: str = "D",
    ax=None,
):
    """
    Plot the full time series of `col` resampled to `freq` (default daily mean).
    Useful to visualise the global behaviour of the series.
    """
    series = df[col].resample(freq).mean()
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure
    series.plot(ax=ax, color="#3a7bd5", linewidth=0.8)
    ax.set_title(f"{col} — resampled at '{freq}' (mean)")
    ax.set_ylabel(col)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_zoom(
    df: pd.DataFrame,
    col: str,
    start: str,
    end: str,
    ax=None,
):
    """
    Plot `col` between `start` and `end` (string dates parseable by pandas).
    Used to inspect daily/weekly patterns at minute resolution.
    """
    series = df.loc[start:end, col]
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure
    series.plot(ax=ax, color="#e76f51", linewidth=0.7)
    ax.set_title(f"{col} — zoom {start} to {end}")
    ax.set_ylabel(col)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_seasonal_boxplots(df: pd.DataFrame, col: str):
    """
    Three boxplots side by side: distribution of `col` by hour of day,
    day of week and month. Reveals daily, weekly and yearly seasonalities.
    """
    s = df[col].dropna()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Hour of day.
    by_hour = [s[s.index.hour == h].values for h in range(24)]
    axes[0].boxplot(by_hour, showfliers=False)
    axes[0].set_title(f"{col} by hour of day")
    axes[0].set_xlabel("Hour")
    axes[0].set_xticks(range(1, 25, 3))
    axes[0].set_xticklabels(range(0, 24, 3))

    # Day of week.
    by_dow = [s[s.index.dayofweek == d].values for d in range(7)]
    axes[1].boxplot(by_dow, showfliers=False)
    axes[1].set_title(f"{col} by day of week")
    axes[1].set_xlabel("Day (Mon=0)")

    # Month.
    by_month = [s[s.index.month == m].values for m in range(1, 13)]
    axes[2].boxplot(by_month, showfliers=False)
    axes[2].set_title(f"{col} by month")
    axes[2].set_xlabel("Month")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, axes
