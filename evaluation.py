"""Training-only model selection followed by one final holdout evaluation."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline


def evaluate_and_export(frame, target, preprocessor_factory, candidates, data_path,
                        output_dir, *, classification=False, seed=42):
    """Select candidates using CV only; serialize the winning fitted pipeline."""
    X, y = frame.drop(columns=target), frame[target]
    if len(frame) < 20:
        raise ValueError("At least 20 valid rows are required for a meaningful split.")
    if classification and (y.nunique() != 2 or y.value_counts().min() < 5):
        raise ValueError("Binary classification requires both classes with at least 5 rows each.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if classification else None)
    folds = min(5, int(y_train.value_counts().min())) if classification else 5
    validation = (StratifiedKFold(folds, shuffle=True, random_state=seed)
                  if classification else KFold(folds, shuffle=True, random_state=seed))
    scoring = ({"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
               if classification else
               {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"})
    results = {}
    for name, model in candidates.items():
        pipeline = Pipeline([("preprocessor", preprocessor_factory()), ("model", model)])
        scores = cross_validate(pipeline, X_train, y_train, cv=validation,
                                scoring=scoring, error_score="raise")
        summary = {}
        for metric in scoring:
            values = scores["test_" + metric]
            if metric in {"mae", "rmse"}:
                values = -values
            summary[metric] = {"mean": float(values.mean()), "std": float(values.std())}
        results[name] = summary
    metric = "roc_auc" if classification else "rmse"
    selector = max if classification else min
    best_name = selector(results, key=lambda name: results[name][metric]["mean"])
    winner = Pipeline([("preprocessor", preprocessor_factory()), ("model", candidates[best_name])])
    winner.fit(X_train, y_train)
    predictions = winner.predict(X_test)
    if classification:
        positive_index = list(winner.classes_).index(1)
        probabilities = winner.predict_proba(X_test)[:, positive_index]
        test = {"accuracy": float(accuracy_score(y_test, predictions)),
                "f1_survived": float(f1_score(y_test, predictions, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
                "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
                "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0)}
    else:
        test = {"mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
                "r2": float(r2_score(y_test, predictions))}
    report = {"schema_version": 1, "selected_model": best_name,
              "selection_metric": metric, "cv_folds": folds, "cv_results": results,
              "selected_model_test_metrics": test, "dataset_rows": len(frame),
              "train_rows": len(X_train), "test_rows": len(X_test),
              "random_state": seed, "test_size": 0.2,
              "features": list(X.columns), "target": target,
              "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
              "versions": {name: importlib.metadata.version(name)
                           for name in ["numpy", "pandas", "scikit-learn", "joblib"]}}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
