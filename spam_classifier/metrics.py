from __future__ import annotations

import json
import os
from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba=None) -> Dict[str, Any]:
    """Compute core binary classification metrics and return as a dict."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    out = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
            out["roc_auc"] = float(auc)
        except Exception:
            pass
    return out


def save_json(metrics: Dict[str, Any], out_path: str) -> None:
    """Persist a metrics dictionary as pretty-printed JSON."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


