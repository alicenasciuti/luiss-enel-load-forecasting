"""
File Title : Data Preprocessing
File Name  : preprocessing.py

Description:
Contains the data preprocessing and transformation logic required
to prepare the household power consumption dataset for analytical
and forecasting tasks. This file is expected to implement data
cleaning procedures, missing value handling systems, temporal
resampling utilities, feature engineering functions, chronological
dataset splitting mechanisms, and preprocessing pipeline operations
that convert raw time series data into structured and model-ready
datasets.

Role in Project:
Provides the data preparation and transformation layer of the
project architecture by bridging the gap between raw dataset
ingestion and downstream analytical or modelling components.
This module interacts with the data loading system to standardize,
clean, enrich, and partition the dataset before it is consumed
by exploratory analysis, forecasting models, and evaluation
workflows throughout the project pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
