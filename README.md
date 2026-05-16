# LUISS x Enel Global ICT — Electric Load Forecasting

End-to-end pipeline for short-term household electric load forecasting on the UCI Individual Household Electric Power Consumption dataset. Three models are compared under the same day-ahead rolling protocol: a seasonal Naive baseline, SARIMA, and an LSTM (implemented in PyTorch).

## 1. Requirements

- **Python 3.12.4**
- Dependencies pinned in `requirements.txt`:
  - `numpy==2.0.2`
  - `pandas==2.2.2`
  - `matplotlib==3.10.0`
  - `statsmodels==0.14.6`
  - `torch==2.10.0`

Install everything in a clean virtual environment:

```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
# .\venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Internet access is required at the first run (the dataset is downloaded automatically from the UCI ML Repository).

## 2. How to reproduce the results

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

To force a full re-training and ignore any cached predictions, run:

```bash
python main.py --no-cache
```

### Expected runtime (CPU)

| Step                              | Time           |
|-----------------------------------|----------------|
| Data download (first run only)    | ~30 s          |
| Preprocessing                     | ~10 s          |
| Naive baseline                    | < 1 s          |
| SARIMA fit + rolling forecast     | 5–20 min       |
| LSTM training + forecast          | 4–10 min       |
| Metrics and figures               | < 30 s         |
| **Total (first run)**             | **~25–35 min** |
| **Total (cached run)**            | **~30 s**      |

Runtimes were measured on an Apple Silicon laptop and may vary on different hardware.

## 3. Project structure

```
src/
├── main.py              # Orchestrator. Single entry point.
├── data_loader.py       # Dataset download and parsing.
├── preprocessing.py     # Resampling, missing-value handling, train/test split.
├── eda.py               # Exploratory analysis utilities.
├── modelling.py         # Naive, SARIMA, LSTM forecasters.
├── evaluation.py        # Metrics and comparison plots.
├── utils.py             # Global random seed.
├── requirements.txt
└── README.md
```

After execution, three additional folders are created:

```
src/
├── data/                # Raw dataset downloaded from UCI.
├── cache/               # Cached predictions (one CSV per model).
└── outputs/             # Final deliverables: metrics table and figures.
```

---
---


# Versione a prova di coglione 


# LUISS x Enel Global ICT — Electric Load Forecasting

End-to-end pipeline for short-term household electric load forecasting on the UCI Individual Household Electric Power Consumption dataset. Three models are compared under the same day-ahead rolling protocol: a seasonal Naive baseline, SARIMA, and an LSTM (implemented in PyTorch).

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Setup — step by step](#2-setup--step-by-step)
3. [How to reproduce the results](#3-how-to-reproduce-the-results)
4. [Expected runtime](#4-expected-runtime)
5. [Project structure](#5-project-structure)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Requirements

- **Python 3.12.4** (any Python 3.12.x should work, but 3.12.4 is the tested version)
- An active internet connection on the first run (the dataset is downloaded automatically from the UCI ML Repository)
- Approximately **500 MB of free disk space** (for the virtual environment, the dataset, and the outputs)

Dependencies pinned in `requirements.txt`:

- `numpy==2.0.2`
- `pandas==2.2.2`
- `matplotlib==3.10.0`
- `statsmodels==0.14.6`
- `torch==2.10.0`

### Don't have Python installed?

Check first by opening a terminal and running:

```bash
python --version
```

If you see `Python 3.12.x`, you are good to go. If you see a different version or an error, install Python 3.12 from [python.org/downloads](https://www.python.org/downloads/) and restart your terminal.

> **macOS users:** Python 3 may be invoked as `python3` instead of `python`. If `python --version` does not work, try `python3 --version`. In that case, use `python3` and `pip3` throughout the rest of this guide.

---

## 2. Setup — step by step

### Step 2.1 — Open a terminal in the project folder

- **macOS:** open Finder, navigate to the folder that contains the `src/` directory, right-click on `src/` and choose *Services → New Terminal at Folder*. (If you don't see this option, enable it in *System Settings → Keyboard → Keyboard Shortcuts → Services*.)
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

This downloads and installs the five libraries listed in `requirements.txt`. Expect this to take **3 to 7 minutes**, mostly because PyTorch is a large package (around 200 MB).

When finished, you will see a line like:
```
Successfully installed matplotlib-3.10.0 numpy-2.0.2 pandas-2.2.2 ...
```

### Step 2.5 — Verify the installation

Run a quick sanity check:

```bash
python -c "import numpy, pandas, matplotlib, statsmodels, torch; print('OK')"
```

If you see `OK`, the setup is complete. If you see `ModuleNotFoundError`, return to Step 2.4.

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

### What you should see

As the pipeline runs, the terminal prints progress messages like:

```
=== [1/5] Data acquisition ===
=== [2/5] Preprocessing & train/test split ===
=== [3/5] Models ===
=== [4/5] Evaluation ===
=== [5/5] Figures ===

[main] Pipeline completed successfully.
```

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
| Data download (first run only)    | ~30 s          |
| Preprocessing                     | ~10 s          |
| Naive baseline                    | < 1 s          |
| SARIMA fit + rolling forecast     | 5–20 min       |
| LSTM training + forecast          | 4–10 min       |
| Metrics and figures               | < 30 s         |
| **Total (first run)**             | **~25–35 min** |
| **Total (cached run)**            | **~30 s**      |

Runtimes were measured on an Apple Silicon laptop and may vary on different hardware. On older machines or low-power CPUs, SARIMA may take longer.

> **Tip:** during the run, do not close the terminal and prevent your computer from going to sleep. On macOS, you can run `caffeinate -i` in a second terminal to keep the system awake.

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

## 6. Troubleshooting

**`command not found: python`**
On some macOS and Linux systems, Python 3 is invoked as `python3`. Try replacing `python` with `python3` and `pip` with `pip3` in all commands.

**`pip install` fails with `Could not find a version that satisfies the requirement torch==2.10.0+cpu`**
This means the `requirements.txt` was edited for a different operating system. On macOS, the correct line is simply `torch==2.10.0` (without the `+cpu` suffix).

**`ModuleNotFoundError: No module named 'xxx'` when running `main.py`**
The virtual environment is probably not active. Check that your prompt starts with `(venv)`. If not, re-run the activation command from Step 2.3.

**The dataset download fails**
Check your internet connection and that `https://archive.ics.uci.edu` is reachable. Corporate networks or VPNs sometimes block this domain. If needed, you can download `household_power_consumption.txt` manually from the [UCI repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) and place it inside a `data/` folder under `src/`.

**The script seems frozen during SARIMA fitting**
SARIMA fitting is computationally heavy and can take several minutes without printing output. As long as your CPU is busy (check Activity Monitor on macOS, Task Manager on Windows), the script is working — be patient.

**I want to start completely from scratch**
From inside the `src/` folder, with the virtual environment deactivated:

```bash
deactivate
rm -rf venv data cache outputs       # macOS / Linux
# rmdir /s venv data cache outputs   # Windows
```

Then repeat the setup from Step 2.2.
