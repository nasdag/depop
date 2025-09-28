import os
import pandas as pd
import pytest

from spam_classifier.data_io import load_train_test, TEXT_COLUMN, TARGET_COLUMN


def _write_tsv(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, sep="\t", index=False)


def test_load_train_test_roundtrip(tmp_path):
    train_path = os.path.join(tmp_path, "train.tsv")
    test_path = os.path.join(tmp_path, "test.tsv")
    train_df = pd.DataFrame({TEXT_COLUMN: ["a b", "c d 1"], TARGET_COLUMN: [0, 1]})
    test_df = pd.DataFrame({TEXT_COLUMN: ["x y", "z 2"], TARGET_COLUMN: [0, 1]})
    _write_tsv(train_path, train_df)
    _write_tsv(test_path, test_df)

    splits = load_train_test(train_path, test_path)
    assert list(splits.X_train.columns) == [TEXT_COLUMN]
    assert splits.y_train.tolist() == [0, 1]
    assert list(splits.X_test.columns) == [TEXT_COLUMN]
    assert splits.y_test.tolist() == [0, 1]


def test_load_train_test_missing_column(tmp_path):
    train_path = os.path.join(tmp_path, "train.tsv")
    bad_train = pd.DataFrame({"some": [1]})
    _write_tsv(train_path, bad_train)
    with pytest.raises(ValueError):
        load_train_test(train_path)


