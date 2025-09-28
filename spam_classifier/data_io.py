from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import pandas as pd


TARGET_COLUMN = "label"
TEXT_COLUMN = "description"

"""Data loading utilities for the spam classifier.

This module provides:
- Typed container `DatasetSplits` for holding train/test splits
- Robust TSV readers with basic schema validation
"""


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame | None
    y_test: pd.Series | None


def read_tsv(file_path: str) -> pd.DataFrame:
    """Read a tab-separated file into a DataFrame.

    Raises FileNotFoundError if the path does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path, sep="\t")
    return df


def load_train_test(train_path: str, test_path: str | None = None) -> DatasetSplits:
    """Load train (and optional test) TSVs and return a typed split container.

    Ensures required columns exist and casts the target to int.
    """
    train_df = read_tsv(train_path)

    required_columns = {TEXT_COLUMN, TARGET_COLUMN}
    missing_cols = required_columns - set(train_df.columns)
    if missing_cols:
        raise ValueError(
            f"Train file missing required columns: {sorted(missing_cols)}"
        )

    if test_path:
        test_df = read_tsv(test_path)
        missing_cols_test = required_columns - set(test_df.columns)
        if missing_cols_test:
            raise ValueError(
                f"Test file missing required columns: {sorted(missing_cols_test)}"
            )
        X_train = train_df[[TEXT_COLUMN]].copy()
        y_train = train_df[TARGET_COLUMN].astype(int).copy()
        X_test = test_df[[TEXT_COLUMN]].copy()
        y_test = test_df[TARGET_COLUMN].astype(int).copy()
        return DatasetSplits(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    X_train = train_df[[TEXT_COLUMN]].copy()
    y_train = train_df[TARGET_COLUMN].astype(int).copy()
    return DatasetSplits(X_train=X_train, y_train=y_train, X_test=None, y_test=None)


