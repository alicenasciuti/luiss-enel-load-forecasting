# LUISS x Enel Global ICT — Electric Load Forecasting

End-to-end pipeline for short-term household electric load forecasting on the
UCI *Individual Household Electric Power Consumption* dataset. Three models are
compared under the same day-ahead rolling protocol: a seasonal Naive baseline,
SARIMA, and an LSTM.

---

## 1. Requirements

- Python **3.12.4**
- Dependencies pinned in `requirements.txt`

Install everything in a clean virtual environment:

```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
# .\venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Internet access is required at the first run (the dataset is downloaded
automatically from the UCI ML Repository).

---

## 2. How to reproduce the results

A single command runs the full pipeline end-to-end:

```bash
python main.py
```

This will, in order:

1. **Acquire the dataset** — downloads `household_power_consumption.txt` from
   UCI if it is not already in `data/`.
2. **Preprocess** — hourly resampling, missing-value handling, chronological
   80/20 train/test split.
3. **Fit and forecast** the three models (Naive, SARIMA, LSTM) under a
   day-ahead rolling protocol (stride = 24 hours).
4. **Compute metrics** (RMSE, MAE, MAPE) and save the comparison table to
   `outputs/metrics_comparison.csv`.
5. **Save figures** (Actual vs Predicted and per-model residual plots) to
   `outputs/`.

To force a full re-training and ignore any cached predictions, run:

```bash
python main.py --no-cache
```

### Expected runtime (CPU)

| Step | Time |
|------|------|
| Data download (first run only) | ~30 s |
| Preprocessing | ~10 s |
| Naive baseline | < 1 s |
| SARIMA fit + rolling forecast | 5-20 min |
| LSTM training + forecast | 4-10 min |
| Metrics and figures | < 30 s |
| **Total (first run)** | **~25-35 min** |
| **Total (cached run)** | **~30 s** |

---

## 3. Project structure

```
src/
├── main.py              Orchestrator. Single entry point.
├── data_loader.py       Dataset download and parsing.
├── preprocessing.py     Resampling, missing-value handling, train/test split.
├── eda.py               Exploratory analysis utilities.
├── modelling.py         Naive, SARIMA, LSTM forecasters.
├── evaluation.py        Metrics and comparison plots.
├── utils.py             Global random seed.
├── requirements.txt
└── README.MD
```

After execution, two additional folders are created:

```
src/
├── cache/               Cached predictions (CSV, one file per model).
└── outputs/             Final deliverables: metrics table and figures.
```

