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
  index is continuous at the expected frequency, and report the number,
  total duration and longest streak of missing timestamps.

Role in the project
-------------------
This module contains the functions used during the exploratory phase of the
project. It does not modify the input DataFrame. The output of these functions
(numerical summaries, figures saved into figures/eda/) is what drives the
design choices documented in the technical report (target variable, sampling
frequency, model hyperparameters).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def describe_dataset(df: pd.DataFrame) -> dict:
    """
    Return a dictionary with high-level information about the dataset:
    shape, time range covered, dtypes and `describe()` summary.

    Parameters
    ----------
    df : pandas.DataFrame
        The raw dataset as returned by `data_loader.load_raw_data()`.

    Returns
    -------
    dict
        Keys: 'n_rows', 'n_cols', 'start', 'end', 'duration_days',
        'dtypes', 'describe'.
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
    Compute per-column missing-value statistics:
    - count of NaN
    - percentage of NaN over the whole DataFrame
    - longest contiguous run of NaN

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Indexed by column name, columns: ['missing_count', 'missing_pct',
        'longest_gap_minutes'].
    """
    out = pd.DataFrame(index=df.columns)
    out["missing_count"] = df.isna().sum()
    out["missing_pct"] = 100.0 * out["missing_count"] / len(df)

    # Longest contiguous run of NaN per column.
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
    Returns the number of missing timestamps, their total duration and the
    longest contiguous gap.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a DatetimeIndex.
    freq : str
        Expected sampling frequency (default '1min').

    Returns
    -------
    dict
        Keys: 'expected_n', 'actual_n', 'n_missing_timestamps',
        'pct_missing_timestamps', 'expected_step', 'longest_continuous_gap'.
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
