import pandas as pd
from src.data.preprocess import preprocess


def test_preprocess_fills_numeric_na():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None], "c": ["x", "y", "z"]})
    out = preprocess(df)
    assert out["a"].isna().sum() == 0
    assert out["b"].isna().sum() == 0
