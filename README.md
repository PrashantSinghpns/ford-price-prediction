# Ford Used-Car Price Prediction

[![Python checks](https://github.com/PrashantSinghpns/ford-price-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/PrashantSinghpns/ford-price-prediction/actions/workflows/ci.yml)

Estimate used-car prices from vehicle specifications with a reproducible tabular regression workflow.

## Technical approach

1. Validate the CSV schema and normalize category whitespace and missing values.
2. Reserve an 80/20 holdout with `random_state=42`.
3. Compare Dummy mean, Ridge regression, and Random Forest using training-only cross-validation.
4. Select the winner by mean CV RMSE; refit it on the training partition.
5. Evaluate the selected model once on the test partition, then export the complete preprocessing/model pipeline.

Median imputation and scaling apply to numeric columns. Most-frequent imputation and one-hot encoding apply to categorical columns. `Pipeline` and `ColumnTransformer` keep learned transformations inside each CV fold. Reports include CV means and standard deviations, row counts, the input SHA-256, feature names, and package versions.

## Input contract

| Field group | Columns |
|---|---|
| Target | price |
| Numeric features | year, mileage, tax, mpg, engineSize |
| Categorical features | model, transmission, fuelType |

Ford records require positive prices and years between 1990 and 2027, matching the existing dataset rule; this fixed range needs review for other populations. Missing numeric inputs are imputed, and unknown categories are ignored by the encoder. Random row splits do not measure future-market performance.

The source CSV is not bundled. Use a permitted copy matching [the data contract](data/README.md). Duplicate entities or repeated observations require a group-aware split before using results outside this benchmark.

## Run locally

Use Python 3.11 or 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pytest
python train.py --data data/ford.csv --output-dir artifacts
python -m pytest -q
```

On Windows PowerShell, replace activation with `.\.venv\Scripts\Activate.ps1`. An external CSV path also works.

Batch inference expects the same feature columns without requiring the target:

```bash
python predict.py --model artifacts/model.joblib --data data/new_records.csv --output artifacts/predictions.csv
```

Only load model artifacts from a trusted source: joblib uses executable serialization. Reuse the package versions recorded by the training report when loading a fitted pipeline.

## Outputs and evaluation

| Output | Purpose |
|---|---|
| `artifacts/model.joblib` | Selected fitted preprocessing and model pipeline |
| `artifacts/metrics.json` | CV comparisons, final holdout metrics, data fingerprint, environment versions |
| `artifacts/predictions.csv` | Batch input records with a prediction column |

Final evaluation includes **MAE, RMSE, and R2**. The raw dataset is unavailable in this checkout, so revised real-data scores are not claimed. Prior reported figures are retained in [the historical record](docs/HISTORICAL_RESULTS.md), with their limitations.

## Repository map

| File | Responsibility |
|---|---|
| `train.py` | Input contract, preprocessing, candidate models, training CLI |
| `evaluation.py` | Training-fold selection, final evaluation, export, provenance |
| `predict.py` | Batch inference with the saved pipeline |
| `tests/` | Data contract and evaluation regression checks |
| `.github/workflows/ci.yml` | Python 3.11/3.12 checks |

This is a portfolio ML benchmark. It is not a validated production decision service. Further work should use verified data provenance, temporal/group validation where relevant, calibration or error analysis, and performance monitoring.
