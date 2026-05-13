"""
File Title : Data Loader
File Name  : data_loader.py

Description:
Contains the data ingestion and loading logic responsible for
reading the raw household power consumption dataset, validating
dataset availability, parsing timestamp information, handling
missing values, and converting raw measurements into structured
numerical formats. This file is expected to implement constants,
utility functions, and data-loading procedures that transform
raw source files into standardized pandas DataFrame objects
ready for downstream processing and analysis.

Role in Project:
Acts as the entry point of the project data pipeline by providing
the foundational dataset structure consumed by preprocessing,
exploratory analysis, feature engineering, modelling, and
evaluation components. This module ensures that all subsequent
systems operate on a clean, consistent, and correctly formatted
data representation across the entire project architecture.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

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

UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)

def ensure_dataset(path) -> Path:
    
    raw_path = Path(path)

    if raw_path.exists():
        return raw_path

    raw_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[data_loader] Raw dataset not found at {raw_path}.")
    print(f"[data_loader] Downloading from UCI: {UCI_DATASET_URL}")
    with urlopen(UCI_DATASET_URL) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        member = next(
            (name for name in zf.namelist() if name.endswith(RAW_FILENAME)),
            None,
        )
        if member is None:
            raise RuntimeError(
                f"{RAW_FILENAME} not found inside the UCI archive."
            )
        with zf.open(member) as src, open(raw_path, "wb") as dst:
            dst.write(src.read())

    print(f"[data_loader] Dataset saved to {raw_path}.")
    return raw_path


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
