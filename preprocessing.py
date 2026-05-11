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
    
    if agg == "mean":
        return df.resample("h").mean()
    elif agg == "sum":
        # `min_count=1` returns NaN when the hour is fully empty (instead of 0).
        return df.resample("h").sum(min_count=1)
    else:
        raise ValueError(f"Unsupported aggregation: {agg!r}")



def handle_missing(series: pd.Series, short_gap_hours: int = 6) -> pd.Series:
    
    s = series.copy()

    
    is_na = s.isna()
    if not is_na.any():
        return s

    
    run_id = (is_na != is_na.shift()).cumsum()
    run_lengths = is_na.groupby(run_id).transform("sum")

    
    short_mask = is_na & (run_lengths <= short_gap_hours)
    s_interp = s.interpolate(method="linear", limit_direction="both")
    s = s.where(~short_mask, s_interp)

    
    max_iterations = 30  
    for _ in range(max_iterations):
        if not s.isna().any():
            break
        prev_day = s.shift(24)
        s = s.fillna(prev_day)
        if s.isna().any():
            next_day = s.shift(-24)
            s = s.fillna(next_day)

    return s




def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    
    out = df.copy()
    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month
    out["is_weekend"] = (out.index.dayofweek >= 5).astype(int)
    return out




def train_test_split_chronological(
    df: pd.DataFrame,
    test_size: float = 0.2,
):
    
    if not 0 < test_size < 1:
        raise ValueError("`test_size` must be in (0, 1).")
    n = len(df)
    cutoff = int(n * (1 - test_size))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()




def run_preprocessing_pipeline(
    df_raw: pd.DataFrame,
    target_col: str = "Global_active_power",
    test_size: float = 0.2,
):
    
    
    df_h = resample_to_hourly(df_raw, agg="mean")

    
    df_h[target_col] = handle_missing(df_h[target_col])

    
    for col in df_h.columns:
        if col != target_col and df_h[col].isna().any():
            df_h[col] = handle_missing(df_h[col])

    
    df_h = add_time_features(df_h)

    
    df_train, df_test = train_test_split_chronological(df_h, test_size=test_size)
    return df_train, df_test
