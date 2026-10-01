import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from evaluation import evaluate_and_export

def test_selection_report_and_saved_pipeline_use_training_cv(tmp_path):
    features = np.linspace(0, 10, 60)
    frame = pd.DataFrame({"x": features, "target": 3 * features + 2})
    data_path = tmp_path / "source.csv"
    frame.to_csv(data_path, index=False)
    report = evaluate_and_export(frame, "target", StandardScaler,
                                 {"dummy": DummyRegressor(), "ridge": Ridge(alpha=0.001)},
                                 data_path, tmp_path / "result")
    assert report["selected_model"] == "ridge"
    assert report["selection_metric"] == "rmse"
    assert set(report["cv_results"]) == {"dummy", "ridge"}
    assert report["train_rows"] + report["test_rows"] == len(frame)
    assert len(report["dataset_sha256"]) == 64
    assert "test_metrics" not in report["cv_results"]["dummy"]
    saved = joblib.load(tmp_path / "result/model.joblib")
    np.testing.assert_allclose(saved.predict(frame[["x"]].iloc[:3]), [2, 2 + 30/59, 2 + 60/59], atol=0.01)
    assert json.loads((tmp_path / "result/metrics.json").read_text())["selected_model"] == "ridge"
