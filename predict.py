"""Batch predictions from a trusted exported preprocessing/model pipeline."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from train import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"))
    args = parser.parse_args()
    frame = pd.read_csv(args.data)
    required = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        parser.error(f"Missing input columns: {missing}")
    features = frame[required].copy()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="raise")
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].map(lambda v: v.strip() if isinstance(v, str) else v)
        features[column] = features[column].replace("", float("nan")).astype(object)
        features[column] = features[column].where(features[column].notna(), float("nan"))
    # joblib/pickle is executable serialization: load only artifacts you trust.
    pipeline = joblib.load(args.model)
    result = frame.copy()
    result["prediction"] = pipeline.predict(features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")

if __name__ == "__main__":
    main()
