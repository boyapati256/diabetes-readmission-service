import pandas as pd
from src.features.pipeline import build_feature_pipeline


def test_pipeline_numeric_transform():
    df = pd.DataFrame({"x1": [1.0, 2.0, None], "x2": [4.0, None, 6.0]})
    pipe = build_feature_pipeline(["x1", "x2"])
    out = pipe.fit_transform(df)
    assert out.shape[0] == 3
