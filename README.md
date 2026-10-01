# Ford Used-Car Price Prediction

A leakage-safe regression workflow that estimates used Ford car prices from vehicle attributes. It compares a mean-price baseline, regularised linear regression, and a random forest using one fixed held-out test set.

## Engineering decisions

- `price` is removed before preprocessing, preventing target leakage.
- Numeric values keep their decimal precision; no full-table integer cast is used.
- One-hot encoding handles vehicle model, transmission, and fuel type without falsely treating them as ordered values.
- Imputation, scaling, encoding, and modelling live in one scikit-learn pipeline, so they fit only on training rows.
- MAE, RMSE, and R² are saved in `artifacts/metrics.json`; the smallest test RMSE selects the exported model.

## Results

On the supplied 17,966-row CSV, one invalid future-year record was removed. With an 80/20 split (`random_state=42`), the selected Random Forest achieved **MAE £837.36**, **RMSE £1,218.05**, and **R² 0.9336** on 3,593 held-out rows. These are one-split estimates, not a guarantee of real market pricing performance.

## Run

```powershell
python -m venv .venv  # Create an isolated environment.
.\.venv\Scripts\python -m pip install -r requirements.txt  # Install the project dependencies.
.\.venv\Scripts\python train.py --data "C:\Users\lenovo\Downloads\ford.csv"  # Train and evaluate the models.
```

## Data

See [`data/README.md`](data/README.md). Do not commit a dataset until its licence is confirmed.
