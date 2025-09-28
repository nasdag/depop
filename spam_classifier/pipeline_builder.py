from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    XGBClassifier = None  # type: ignore
    _HAS_XGB = False

from .data_io import TEXT_COLUMN
from .numeric_features import NUMERIC_FEATURE_COLUMNS, add_numeric_features
from .text_cleaning import clean_text


# --- Top-level transformers to ensure picklability ---
def _transform_clean_text(series):
    import numpy as _np
    return _np.array([clean_text(t) for t in series], dtype=object)


def _build_numeric_df(df):
    return add_numeric_features(df, TEXT_COLUMN)


def _select_numeric_columns(df):
    return df[NUMERIC_FEATURE_COLUMNS]


def _extract_text_column(X: pd.DataFrame) -> np.ndarray:
    return X[TEXT_COLUMN].values


def build_feature_transformer(max_features: int = 50000) -> ColumnTransformer:
    """Create a ColumnTransformer combining TF-IDF text and numeric features."""
    clean_transformer = FunctionTransformer(_transform_clean_text, validate=False)

    text_pipeline = Pipeline(
        steps=[
            ("clean", clean_transformer),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    max_features=max_features,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("build_numeric", FunctionTransformer(_build_numeric_df, validate=False)),
            ("select_numeric", FunctionTransformer(_select_numeric_columns, validate=False)),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("text", text_pipeline, TEXT_COLUMN),
            ("numeric", numeric_pipeline, [TEXT_COLUMN]),
        ]
    )
    return transformer


def build_estimator(model: str = "lr", random_state: int = 42):
    """Return the estimator by name with sensible defaults."""
    if model == "lr":
        return LogisticRegression(max_iter=300, n_jobs=1, random_state=random_state)
    if model == "xgb" and _HAS_XGB:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=400,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.8,
            max_depth=6,
            random_state=random_state,
            n_jobs=2,
        )
    raise ValueError("Unknown model or missing dependency: use 'lr' or 'xgb'")


def build_pipeline(model: str = "lr", random_state: int = 42, max_features: int = 50000) -> Pipeline:
    """Build the full pipeline: features + model."""
    transformer = build_feature_transformer(max_features=max_features)
    estimator = build_estimator(model=model, random_state=random_state)
    return Pipeline(steps=[("features", transformer), ("model", estimator)])


def default_param_distributions(model: str = "lr") -> Dict[str, Any]:
    """Hyperparameter search space per model type."""
    if model == "lr":
        return {
            "model__C": np.logspace(-2, 2, 20),
            "model__penalty": ["l2"],
            "model__class_weight": [None, "balanced"],
        }
    if model == "xgb":
        return {
            "model__n_estimators": [200, 300, 400, 600],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [4, 6, 8],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_alpha": [0.0, 0.1, 0.5],
            "model__reg_lambda": [0.5, 1.0, 2.0],
        }
    raise ValueError("Unknown model")


def default_scorer():
    """Default scorer (F1) for imbalanced binary classification."""
    return make_scorer(f1_score)


