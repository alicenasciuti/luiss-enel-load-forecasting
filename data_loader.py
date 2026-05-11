"""
data_loader.py
==============

Loading of the raw `Individual Household Electric Power Consumption` dataset
(UCI Machine Learning Repository) used in the Enel Global ICT project.

Contents
--------
- RAW_FILENAME: expected filename of the raw dataset.
- DATE_COL, TIME_COL, NUMERIC_COLS: column names present in the raw file.
- load_raw_data(path): reads `household_power_consumption.txt`, parses
  Date+Time into a single DatetimeIndex, converts '?' tokens into NaN and
  casts the seven measurement columns to float64. Returns a pandas
  DataFrame indexed by timestamp.

Role in the project
-------------------
This is the very first step of the pipeline. Every other module
(preprocessing, eda, modelling, evaluation) starts from the DataFrame
returned by `load_raw_data()`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_FILENAME = "household_power_consumption.txt"

DATE_COL = "Date"
TIME_COL = "Time"
NUMERIC_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def load_raw_data(path) -> pd.DataFrame:
    
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}.")

    
    df = pd.read_csv(
        raw_path,
        sep=";",
        na_values=["?"],
        low_memory=False,
        dtype={col: "string" for col in NUMERIC_COLS},
    )

   
    timestamp = pd.to_datetime(
        df[DATE_COL] + " " + df[TIME_COL],
        format="%d/%m/%Y %H:%M:%S",
    )

    
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

    df = df.drop(columns=[DATE_COL, TIME_COL])
    df.index = timestamp
    df.index.name = "timestamp"
    df = df.sort_index()

    return df
