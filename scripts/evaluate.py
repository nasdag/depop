from __future__ import annotations

import argparse
import json
import os
import logging
try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    joblib = None  # type: ignore
    _HAS_JOBLIB = False
import pickle
import pandas as pd

from spam_classifier.data_io import load_train_test
from spam_classifier.metrics import compute_metrics, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved spam classifier on test set")
    parser.add_argument("--model_path", required=True, help="Path to saved .joblib model")
    parser.add_argument("--test", required=True, help="Path to test_set.tsv")
    parser.add_argument("--out_dir", default="artifacts", help="Directory to save metrics")
    parser.add_argument("--misclassified_csv", default=None, help="Optional path to export misclassified rows as CSV")
    parser.add_argument("--log_level", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    os.makedirs(args.out_dir, exist_ok=True)
    if _HAS_JOBLIB:
        clf = joblib.load(args.model_path)  # type: ignore[assignment]
    else:
        with open(args.model_path, "rb") as f:
            clf = pickle.load(f)

    data = load_train_test(train_path=args.test, test_path=args.test)
    # Using test both as X_train and X_test to avoid unused split; we only need X_test/y_test

    y_pred = clf.predict(data.X_test)
    try:
        y_proba = clf.predict_proba(data.X_test)[:, 1]
    except Exception:
        y_proba = None
    metrics = compute_metrics(data.y_test, y_pred, y_proba=y_proba)
    save_json(metrics, os.path.join(args.out_dir, "test_metrics.json"))

    # Export misclassified examples if requested
    if args.misclassified_csv:
        df = data.X_test.copy()
        df["y_true"] = data.y_test.values
        df["y_pred"] = y_pred
        if y_proba is not None:
            df["y_proba"] = y_proba
        mis = df[df["y_true"] != df["y_pred"]]
        mis.to_csv(args.misclassified_csv, index=False)
        logging.info("Saved misclassified rows to %s (n=%d)", args.misclassified_csv, len(mis))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


