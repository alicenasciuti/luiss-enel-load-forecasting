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
- decompose_series(series, period, model='additive'): seasonal decomposition
  (trend + seasonal + residual) using statsmodels.
- adf_test(series, name=''): Augmented Dickey-Fuller stationarity test;
  prints results and returns the dictionary.
- kpss_test(series, name=''): KPSS stationarity test (complement to ADF);
  prints results and returns the dictionary.
- plot_acf_pacf(series, lags, title=''): autocorrelation and partial
  autocorrelation plots side by side.
- plot_correlation_heatmap(df, cols=None): Pearson correlation heatmap among
  numeric columns.

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
    if in_run:
        duration = len(is_na) - run_start
        if duration >= min_gap_minutes:
            blocks.append({
                "start": pd.Timestamp(timestamps[run_start]),
                "end": pd.Timestamp(timestamps[-1]),
                "duration_minutes": duration,
            })

    return pd.DataFrame(blocks)




def plot_full_series(df, col, freq="D", ax=None):
    
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


def plot_zoom(df, col, start, end, ax=None):
    
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


def plot_seasonal_boxplots(df, col):
    
    s = df[col].dropna()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    by_hour = [s[s.index.hour == h].values for h in range(24)]
    axes[0].boxplot(by_hour, showfliers=False)
    axes[0].set_title(f"{col} by hour of day")
    axes[0].set_xlabel("Hour")
    axes[0].set_xticks(range(1, 25, 3))
    axes[0].set_xticklabels(range(0, 24, 3))

    by_dow = [s[s.index.dayofweek == d].values for d in range(7)]
    axes[1].boxplot(by_dow, showfliers=False)
    axes[1].set_title(f"{col} by day of week")
    axes[1].set_xlabel("Day (Mon=0)")

    by_month = [s[s.index.month == m].values for m in range(1, 13)]
    axes[2].boxplot(by_month, showfliers=False)
    axes[2].set_title(f"{col} by month")
    axes[2].set_xlabel("Month")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, axes




def decompose_series(series: pd.Series, period: int, model: str = "additive"):
    
    from statsmodels.tsa.seasonal import seasonal_decompose

    result = seasonal_decompose(series, model=model, period=period)
    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(
        f"Seasonal decomposition (period={period}, model={model})",
        y=1.02,
    )
    fig.tight_layout()
    return fig, result


def adf_test(series: pd.Series, name: str = "") -> dict:
    
    from statsmodels.tsa.stattools import adfuller

    s = series.dropna()
    result = adfuller(s, autolag="AIC")

    out = {
        "adf_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }

    if name:
        print(f"--- ADF test: {name} ---")
    print(f"ADF Statistic:  {out['adf_statistic']:.4f}")
    print(f"p-value:        {out['p_value']:.6f}")
    print(f"# lags used:    {out['n_lags']}")
    print(f"# observations: {out['n_obs']}")
    print("Critical values:")
    for level, val in out["critical_values"].items():
        print(f"  {level}: {val:.4f}")
    verdict = "STATIONARY ✅" if out["is_stationary"] else "NON-STATIONARY ❌"
    print(f"Verdict (alpha=0.05): {verdict}")
    print()

    return out

def kpss_test(series: pd.Series, name: str = "") -> dict:
    
    from statsmodels.tsa.stattools import kpss

    s = series.dropna()
    stat, p_value, n_lags, crit = kpss(s, regression="c", nlags="auto")

    out = {
        "kpss_statistic": stat,
        "p_value": p_value,
        "n_lags": n_lags,
        "critical_values": crit,
        "is_stationary": p_value >= 0.05,
    }

    if name:
        print(f"--- KPSS test: {name} ---")
    print(f"KPSS Statistic: {out['kpss_statistic']:.4f}")
    print(f"p-value:        {out['p_value']:.6f}")
    print(f"# lags used:    {out['n_lags']}")
    print("Critical values:")
    for level, val in out["critical_values"].items():
        print(f"  {level}: {val:.4f}")
    verdict = "STATIONARY ✅" if out["is_stationary"] else "NON-STATIONARY ❌"
    print(f"Verdict (alpha=0.05): {verdict}")
    print()

    return out


def plot_acf_pacf(series: pd.Series, lags: int = 48, title: str = ""):
    
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF{' — ' + title if title else ''}")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"PACF{' — ' + title if title else ''}")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, axes


def plot_correlation_heatmap(df: pd.DataFrame, cols=None):
    
    if cols is not None:
        sub = df[cols]
    else:
        sub = df.select_dtypes(include=[np.number])

    corr = sub.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center",
                    color="black" if abs(corr.iloc[i, j]) < 0.5 else "white",
                    fontsize=9)

    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    return fig, ax
