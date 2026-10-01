import pandas as pd
import numpy as np
from train import normalize_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
import train

def test_missing_category_and_unseen_value_reach_fitted_pipeline():
    frame = pd.DataFrame({name: [1., 2., 3., 4.] for name in NUMERIC_FEATURES})
    for name in CATEGORICAL_FEATURES:
        frame[name] = pd.Series([" a ", pd.NA, "b", "a"], dtype="string")
    cleaned = normalize_features(frame)
    factory = getattr(train, "make_preprocessor", None) or train.build_preprocessor
    pipeline = Pipeline([("preprocessor", factory()), ("model", DummyRegressor())])
    pipeline.fit(cleaned, [1., 2., 3., 4.])
    future = cleaned.iloc[:1].copy()
    for name in CATEGORICAL_FEATURES:
        future[name] = "unseen"
    assert np.isfinite(pipeline.predict(future)).all()
    assert cleaned.iloc[0][CATEGORICAL_FEATURES[0]] == "a"
