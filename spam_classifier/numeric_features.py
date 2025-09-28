from __future__ import annotations

import pandas as pd

"""Numeric feature engineering built from cleaned text.

Features include stopword proportion, average word length, number ratio, and
hashtag counts. Text is cleaned idempotently.
"""

from .text_cleaning import (
    clean_text,
    proportion_of_stopwords,
    average_word_length,
    proportion_of_numbers,
    count_hash_tags,
)


NUMERIC_FEATURE_COLUMNS = [
    "prop_stopwords",
    "avg_word_len",
    "prop_numbers",
    "num_hashtags",
]


def add_numeric_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    out = df.copy()
    # Assume text is already cleaned upstream; safe to clean again idempotently
    out[text_col] = out[text_col].astype(str).map(clean_text)
    out["prop_stopwords"] = out[text_col].map(proportion_of_stopwords)
    out["avg_word_len"] = out[text_col].map(average_word_length)
    out["prop_numbers"] = out[text_col].map(proportion_of_numbers)
    out["num_hashtags"] = out[text_col].map(count_hash_tags)
    return out


