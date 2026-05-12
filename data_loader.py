"""
File Title : Data Loader
File Name  : data_loader.py

Description:
Contains the utilities and constants responsible for loading the raw
household electric power consumption dataset, parsing timestamps,
handling missing values, and converting measurement columns into the
appropriate numerical formats.

Role in Project:
Acts as the main data ingestion layer of the project by providing the
initial structured DataFrame used by all downstream modules, including
preprocessing, exploratory analysis, modelling, and evaluation.
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
