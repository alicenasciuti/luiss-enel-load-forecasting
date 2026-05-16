

# LUISS x Enel Global ICT — Electric Load Forecasting

End-to-end pipeline for short-term household electric load forecasting on the UCI Individual Household Electric Power Consumption dataset. Three models are compared under the same day-ahead rolling protocol: a seasonal Naive baseline, SARIMA, and an LSTM (implemented in PyTorch).

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Setup — step by step](#2-setup--step-by-step)
3. [How to reproduce the results](#3-how-to-reproduce-the-results)
4. [Expected runtime](#4-expected-runtime)
5. [Project structure](#5-project-structure)

---

## 1. Requirements

- **Python 3.12.4** (any Python 3.12.x should work, but 3.12.4 is the tested version)
- An active internet connection on the first run (the dataset is downloaded automatically from the UCI ML Repository)

Dependencies pinned in `requirements.txt`:

- `numpy==2.0.2`
- `pandas==2.2.2`
- `matplotlib==3.10.0`
- `statsmodels==0.14.6`
- `torch==2.10.0`


---

## 2. Setup — step by step

### Step 2.1 — Open a terminal in the project folder

- **macOS:** open Finder, navigate to the folder that contains the `src/` directory, right-click on `src/` and choose *Services → New Terminal at Folder*. 
- **Windows:** open File Explorer, navigate inside the `src/` folder, click the address bar at the top, type `cmd`, and press Enter.
- **Linux:** right-click inside the `src/` folder and choose *Open in Terminal*.

Verify you are in the correct folder by listing its contents:

```bash
ls          # macOS / Linux
dir         # Windows
```

You should see `main.py`, `requirements.txt`, `README.md`, and the other Python files.

### Step 2.2 — Create a virtual environment

A virtual environment is an isolated workspace that keeps this project's libraries separate from the rest of your system.

```bash
python -m venv venv
```

This creates a new folder called `venv/` inside `src/`. Wait a few seconds for it to complete.

### Step 2.3 — Activate the virtual environment

- **macOS / Linux:**
```bash
  source venv/bin/activate
```
- **Windows (Command Prompt):**
```bash
  .\venv\Scripts\activate
```
- **Windows (PowerShell):**
```bash
  .\venv\Scripts\Activate.ps1
```

After activation, your terminal prompt should start with `(venv)`. For example:

```
(venv) user@computer src %
```

If you do **not** see `(venv)` at the beginning of the prompt, the environment is not active and the next steps will fail. Repeat Step 2.3.

### Step 2.4 — Install the dependencies

With the virtual environment active, install all required libraries in one command:

```bash
pip install -r requirements.txt
```

This downloads and installs the five libraries listed in `requirements.txt`. 


---

## 3. How to reproduce the results

A single command runs the full pipeline end-to-end:

```bash
python main.py
```

This will, in order:

1. **Acquire the dataset** — downloads `household_power_consumption.txt` from UCI if it is not already in `data/`.
2. **Preprocess** — hourly resampling, missing-value handling, chronological 80/20 train/test split.
3. **Fit and forecast** the three models (Naive, SARIMA, LSTM) under a day-ahead rolling protocol (stride = 24 hours).
4. **Compute metrics** (RMSE, MAE, MAPE) and save the comparison table to `outputs/metrics_comparison.csv`.
5. **Save figures** (Actual vs Predicted and per-model residual plots) to `outputs/`.

When it finishes, the results are saved inside the `outputs/` folder:

- `metrics_comparison.csv` — the table comparing the three models on RMSE, MAE, and MAPE.
- `actual_vs_predicted.png` — predicted vs actual load curves for all models.
- `residuals_naive.png`, `residuals_sarima.png`, `residuals_lstm.png` — residual plots for each model.

### Forcing a fresh run

After the first run, predictions are cached in the `cache/` folder so that re-runs take only seconds. To force a full re-training and ignore the cache, run:

```bash
python main.py --no-cache
```

---

## 4. Expected runtime
| Step                              | Time           |
|-----------------------------------|----------------|
| **Total (first run)**             | **~20–35 min** |
| **Total (cached run)**            | **~30 s**      |


Runtimes may vary on different hardware.


---

## 5. Project structure

Before running the pipeline:

```
src/
├── main.py              # Orchestrator. Single entry point.
├── data_loader.py       # Dataset download and parsing.
├── preprocessing.py     # Resampling, missing-value handling, train/test split.
├── eda.py               # Exploratory analysis utilities.
├── modelling.py         # Naive, SARIMA, LSTM forecasters.
├── evaluation.py        # Metrics and comparison plots.
├── utils.py             # Global random seed.
├── requirements.txt     # Pinned dependencies.
└── README.md            # This file.
```

After execution, three additional folders are created automatically:

```
src/
├── data/                # Raw dataset downloaded from UCI.
├── cache/               # Cached predictions (one CSV per model).
└── outputs/             # Final deliverables: metrics table and figures.
```

---

