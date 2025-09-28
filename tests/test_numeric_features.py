import pandas as pd

from spam_classifier.numeric_features import add_numeric_features, NUMERIC_FEATURE_COLUMNS


def test_add_numeric_features_columns():
    df = pd.DataFrame({"description": ["hello world 123", "#vintage #denim"]})
    out = add_numeric_features(df, "description")
    for col in NUMERIC_FEATURE_COLUMNS:
        assert col in out.columns


