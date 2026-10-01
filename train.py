"""Train and evaluate a leakage-safe Ford used-car price regression model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "price"  # Keep the prediction target separate from every feature transformation.
NUMERIC_FEATURES = ["year", "mileage", "tax", "mpg", "engineSize"]  # Continuous vehicle attributes.
CATEGORICAL_FEATURES = ["model", "transmission", "fuelType"]  # Nominal categories require one-hot encoding.
REQUIRED_COLUMNS = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES, TARGET]  # Validate the input schema early.


def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Load a CSV and apply only documented, defensible quality checks."""
    frame = pd.read_csv(path)  # Read the user-supplied dataset without modifying the source file.
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))  # Find schema problems before training.
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")  # Fail with an actionable message.

    frame = frame[REQUIRED_COLUMNS].copy()  # Retain only declared features and the target.
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("string").str.strip()  # Remove accidental leading/trailing label spaces.

    frame = frame.dropna(subset=[TARGET])  # A row without a price cannot be used for supervised training.
    frame = frame.loc[frame[TARGET] > 0].copy()  # Prices must be positive for this regression task.
    frame = frame.loc[frame["year"].between(1990, 2027)].copy()  # Remove the clearly implausible future year in this dataset.
    return frame


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing that is fitted only on training rows through a pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  # Learn numeric replacements from training data only.
            ("scaler", StandardScaler()),  # Scale only continuous values; never scale encoded categories.
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # Handle a missing category without dropping a row.
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),  # Avoid inventing an ordinal relationship between models.
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),  # Apply numeric steps to numeric columns.
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),  # Apply category steps to text columns.
        ]
    )


def regression_metrics(y_true: pd.Series, predictions) -> dict[str, float]:
    """Return business-readable regression metrics in pounds."""
    return {
        "mae": round(float(mean_absolute_error(y_true, predictions)), 2),  # Typical absolute pricing error.
        "rmse": round(float(mean_squared_error(y_true, predictions) ** 0.5), 2),  # Penalise large pricing mistakes.
        "r2": round(float(r2_score(y_true, predictions)), 4),  # Proportion of held-out price variance explained.
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)  # Provide a clear command-line interface.
    parser.add_argument("--data", required=True, help="Path to the Ford CSV file")  # Keep raw data outside the repository.
    parser.add_argument("--output-dir", default="artifacts", help="Directory for the model and metrics")  # Store reproducible outputs together.
    args = parser.parse_args()

    frame = load_and_clean(args.data)  # Clean before splitting, without seeing the test target during preprocessing.
    X = frame.drop(columns=TARGET)  # Explicitly exclude price to prevent target leakage.
    y = frame[TARGET]  # Keep the target in a separate series.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # Preserve a fixed held-out test set for honest final evaluation.
    )

    candidates = {
        "dummy_mean": DummyRegressor(strategy="mean"),  # Establish the minimum useful benchmark.
        "ridge": Ridge(alpha=10.0),  # Use a regularised linear model as a transparent baseline.
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1  # Capture non-linear price effects reproducibly.
        ),
    }
    results: dict[str, dict[str, float]] = {}  # Collect metrics before selecting the best model.
    fitted_models = {}  # Retain fitted pipelines for later selection.
    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("model", estimator)]  # Fit preprocessing and model together.
        )
        pipeline.fit(X_train, y_train)  # Learn transformations from training rows only.
        results[name] = regression_metrics(y_test, pipeline.predict(X_test))  # Evaluate once on untouched test data.
        fitted_models[name] = pipeline  # Keep the trained candidate for the winning-model export.

    best_name = min(results, key=lambda name: results[name]["rmse"])  # Select by held-out pricing error, not training score.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create a dedicated artifact location.
    joblib.dump(fitted_models[best_name], output_dir / "model.joblib")  # Save preprocessing and model as one deployable object.
    report = {
        "dataset_rows": len(frame),  # Record the exact cleaned training population.
        "train_rows": len(X_train),  # Record split sizes for reproducibility.
        "test_rows": len(X_test),
        "best_model": best_name,
        "test_metrics": results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")  # Save machine-readable evidence.
    print(json.dumps(report, indent=2))  # Show the final evaluation in the terminal.


if __name__ == "__main__":
    main()  # Run training only when this file is executed directly.
