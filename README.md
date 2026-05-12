# luiss-enel-load-forecasting

Time Series Forecasting for Electric Load — LUISS AI Techniques x Enel Global ICT Project Work.

This repository contains the source code used to load, preprocess, explore, model and evaluate the `Individual Household Electric Power Consumption` dataset from the UCI Machine Learning Repository.

The goal of the project is to forecast household electric load using time series forecasting techniques and compare different models under the same evaluation protocol.

---

## Repository Structure

The project is organised into separate Python modules, each one with a specific role in the pipeline:

- `data_loader.py`  
  Loads the raw dataset and converts the date and time columns into a timestamp index.

- `preprocessing.py`  
  Cleans the data, handles missing values, resamples the time series and prepares the train/test split.

- `eda.py`  
  Performs exploratory data analysis, visualisations and statistical checks on the dataset.

- `modelling.py`  
  Implements the forecasting models used in the project.

- `evaluation.py`  
  Computes evaluation metrics and compares model performances.

- `utils.py`  
  Contains helper and reproducibility functions shared across the project.
  
  ---

## Dataset

The project uses the `Individual Household Electric Power Consumption` dataset from the UCI Machine Learning Repository.

Expected raw dataset filename:

```text
household_power_consumption.txt
```

The raw dataset is loaded through the function:

```python
load_raw_data(path)
```

where `path` is the location of the raw `.txt` file.

Before running `01_eda.ipynb`, make sure that the dataset is available in the path specified inside the notebook.

---

## Requirements
---

The project was developed using Python 3.12.13.

Main libraries used:

```text
numpy
pandas
matplotlib
scikit-learn
statsmodels
xgboost
lightgbm
seaborn
```

The complete environment configuration is available in [requirements.txt](requirements.txt).

---

## How to Run the Code

The recommended way to reproduce the project is to run the notebook:

```python
01_eda.ipynb
```

The notebook performs the following steps:

1. Mount Google Drive (when running on Google Colab).
2. Clone the GitHub repository.
3. Add the repository folder to the Python path.
4. Import the project modules.
5. Load the raw dataset using `load_raw_data(path)`.
6. Run preprocessing and exploratory data analysis.
7. Train the forecasting models.
8. Evaluate the models and generate the final plots and metrics.

Example setup used in Google Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
rm -rf /content/repo
git clone https://github.com/alicenasciuti/luiss-enel-load-forecasting.git /content/repo
```

```python
import sys
sys.path.insert(0, '/content/repo')
```

## Reproducibility

To guarantee reproducibility of the experiments:

- The dataset is loaded from a fixed path.

- The preprocessing pipeline is deterministic.

- The same train/test split is reused across experiments.

- Evaluation metrics are computed consistently for all models.

- All figures and numerical results are generated directly from the notebook execution.

Running the notebook from top to bottom reproduces the complete workflow and results.
