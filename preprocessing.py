"""
preprocessing.py
================

Preprocessing of the household power consumption dataset.

Contents
--------
- resample_to_hourly(df, agg='mean'): downsamples the minute-level series
  to hourly frequency using the chosen aggregation (default: mean).
- handle_missing(series, short_gap_hours=6): fills NaN values in a series.
  Short gaps (<= short_gap_hours) are filled by linear interpolation; longer
  gaps are filled by the value of the same hour on the previous (or next)
  day, repeated until no NaN remains.
- add_time_features(df): adds calendar features (hour, dayofweek, month,
  is_weekend) extracted from the DatetimeIndex.
- train_test_split_chronological(df, test_size=0.2): splits the DataFrame
  in two contiguous chunks, the last `test_size` fraction going to the test
  set. No shuffling.
- run_preprocessing_pipeline(df_raw, target_col='Global_active_power'):
  end-to-end function that chains the four steps above and returns
  (df_train, df_test).

Role in the project
-------------------
This module transforms the raw minute-level dataset returned by
`data_loader.load_raw_data()` into the hourly, missing-free,
feature-augmented DataFrame on which the modelling will be performed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --- Resampling ------------------------------------------------------------

def resample_to_hourly(df: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    """
    Downsample a minute-level DataFrame to hourly frequency.

    Parameters
    ----------
    df : pandas.DataFrame
        Minute-level DataFrame with a DatetimeIndex.
    agg : {'mean', 'sum'}
        Aggregation function to use. Default is 'mean' (suitable for power
        in kW). Use 'sum' for energy quantities (Wh) such as Sub_metering_*.

    Returns
    -------
    pandas.DataFrame
        Hourly-resampled DataFrame. NaN are produced when an entire hour
        has no observations.
    """
    if agg == "mean":
        return df.resample("h").mean()
    elif agg == "sum":
        # `min_count=1` returns NaN when the hour is fully empty (instead of 0).
        return df.resample("h").sum(min_count=1)
    else:
        raise ValueError(f"Unsupported aggregation: {agg!r}")


# --- Missing-value imputation ----------------------------------------------

def handle_missing(series: pd.Series, short_gap_hours: int = 6) -> pd.Series:
    """
    Fill NaN values in an hourly time series.

    Strategy
    --------
    1. Short gaps (length <= `short_gap_hours`): linear interpolation.
    2. Longer gaps: each NaN is filled with the value at the same hour on
       the previous day. If still NaN (i.e. the previous day was also NaN),
       fall back to the next day. The procedure is repeated until no NaN
       remains.

    Parameters
    ----------
    series : pandas.Series
        Hourly series with a DatetimeIndex.
    short_gap_hours : int
        Threshold (in hours) separating short from long gaps. Default 6.

    Returns
    -------
    pandas.Series
        Series without NaN.
    """
    s = series.copy()

    # Identify the length of each NaN run (in number of hours).
    is_na = s.isna()
    if not is_na.any():
        return s

    # Group consecutive NaN runs.
    run_id = (is_na != is_na.shift()).cumsum()
    run_lengths = is_na.groupby(run_id).transform("sum")

    # Step 1 - linear interpolation only inside short runs.
    short_mask = is_na & (run_lengths <= short_gap_hours)
    s_interp = s.interpolate(method="linear", limit_direction="both")
    s = s.where(~short_mask, s_interp)

    # Step 2 - long runs: fill with same-hour-previous-day, then next-day, loop.
    max_iterations = 30  # safety bound (covers up to 30 days of missing).
    for _ in range(max_iterations):
        if not s.isna().any():
            break
        prev_day = s.shift(24)
        s = s.fillna(prev_day)
        if s.isna().any():
            next_day = s.shift(-24)
            s = s.fillna(next_day)

    return s


# --- Calendar features -----------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar features extracted from the DatetimeIndex.

    Adds the following columns:
    - 'hour'        : hour of the day (0-23)
    - 'dayofweek'   : day of the week (0=Mon, 6=Sun)
    - 'month'       : month of the year (1-12)
    - 'is_weekend'  : 1 if Saturday or Sunday, 0 otherwise

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a DatetimeIndex.

    Returns
    -------
    pandas.DataFrame
        Copy of `df` with the new columns appended.
    """
    out = df.copy()
    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month
    out["is_weekend"] = (out.index.dayofweek >= 5).astype(int)
    return out


# --- Train/test split ------------------------------------------------------

def train_test_split_chronological(
    df: pd.DataFrame,
    test_size: float = 0.2,
):
    """
    Split a DataFrame into train/test in a chronological way (no shuffle).

    Parameters
    ----------
    df : pandas.DataFrame
        Sorted DataFrame with a DatetimeIndex.
    test_size : float
        Fraction of the dataset to put in the test set. Default 0.2 (20%).

    Returns
    -------
    tuple of pandas.DataFrame
        (train_df, test_df).
    """
    if not 0 < test_size < 1:
        raise ValueError("`test_size` must be in (0, 1).")
    n = len(df)
    cutoff = int(n * (1 - test_size))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


# --- End-to-end pipeline ---------------------------------------------------

def run_preprocessing_pipeline(
    df_raw: pd.DataFrame,
    target_col: str = "Global_active_power",
    test_size: float = 0.2,
):
    """
    End-to-end preprocessing: resample, impute, add features and split.

    Parameters
    ----------
    df_raw : pandas.DataFrame
        Raw minute-level DataFrame from `data_loader.load_raw_data()`.
    target_col : str
        Name of the target column. Default 'Global_active_power'.
    test_size : float
        Fraction of the dataset for the test split. Default 0.2.

    Returns
    -------
    tuple of pandas.DataFrame
        (df_train, df_test). Both contain the imputed target, the other
        original columns (also resampled) and the four calendar features.
    """
    # 1. Resample to hourly (mean).
    df_h = resample_to_hourly(df_raw, agg="mean")

    # 2. Impute the target column.
    df_h[target_col] = handle_missing(df_h[target_col])

    # 3. Impute the other columns as well (so we can use them later if needed).
    for col in df_h.columns:
        if col != target_col and df_h[col].isna().any():
            df_h[col] = handle_missing(df_h[col])

    # 4. Add calendar features.
    df_h = add_time_features(df_h)

    # 5. Chronological train/test split.
    df_train, df_test = train_test_split_chronological(df_h, test_size=test_size)
    return df_train, df_test
