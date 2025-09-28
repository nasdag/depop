from __future__ import annotations

import argparse
import json
import os
import time
import logging
from typing import Any, Dict
try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    joblib = None  # type: ignore
    _HAS_JOBLIB = False
import pickle

from spam_classifier.data_io import load_train_test
from spam_classifier.metrics import compute_metrics, save_json
from spam_classifier.pipeline_builder import (
    build_pipeline,
    default_param_distributions,
    default_scorer,
)

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train keyword spam classifier")
    parser.add_argument("--train", required=True, help="Path to train_set.tsv")
    parser.add_argument("--test", required=False, help="Path to test_set.tsv")
    parser.add_argument("--model", default="lr", choices=["lr", "xgb"], help="Estimator type")
    parser.add_argument("--max_features", type=int, default=50000, help="Max TF-IDF features")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_iter", type=int, default=25, help="Hyperparameter search iterations")
    parser.add_argument("--cv", type=int, default=5, help="Cross validation folds")
    parser.add_argument("--out_dir", default="artifacts", help="Directory to save model and metrics")
    parser.add_argument("--config", type=str, help="Optional JSON config with params to override")
    parser.add_argument("--log_level", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    return parser.parse_args()


def _apply_config_overrides(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)


def _save_feature_importance(best_estimator, out_dir: str) -> None:
    try:
        import numpy as np
        os.makedirs(out_dir, exist_ok=True)
        model = best_estimator.named_steps.get("model")
        features = best_estimator.named_steps.get("features")
        if model is None or features is None:
            return
        text_tfidf = features.named_transformers_["text"].named_steps["tfidf"]
        text_names = np.array(text_tfidf.get_feature_names_out())
        numeric_names = np.array(["prop_stopwords", "avg_word_len", "prop_numbers", "num_hashtags"])
        if hasattr(model, "coef_"):
            coefs = model.coef_.ravel()
            if coefs.shape[0] == (len(text_names) + len(numeric_names)):
                names = np.concatenate([text_names, numeric_names])
            else:
                names = np.arange(coefs.shape[0]).astype(str)
            top_idx = np.argsort(np.abs(coefs))[::-1][:100]
            imp = [{"feature": str(names[i]), "weight": float(coefs[i])} for i in top_idx]
        elif hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            names = np.concatenate([text_names, numeric_names])
            top_idx = np.argsort(fi)[::-1][:100]
            imp = [{"feature": str(names[i]), "importance": float(fi[i])} for i in top_idx]
        else:
            return
        with open(os.path.join(out_dir, "feature_importance_top100.json"), "w") as f:
            json.dump(imp, f, indent=2)
    except Exception:
        pass


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    os.makedirs(args.out_dir, exist_ok=True)

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = json.load(f)
        _apply_config_overrides(args, cfg)

    logging.info("Loading data ...")
    data = load_train_test(args.train, args.test)

    logging.info("Building pipeline ...")
    pipeline = build_pipeline(model=args.model, random_state=args.seed, max_features=args.max_features)
    param_distributions = default_param_distributions(model=args.model)
    scorer = default_scorer()

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        scoring=scorer,
        cv=cv,
        verbose=1,
        random_state=args.seed,
        n_jobs=2,
        refit=True,
    )

    logging.info("Fitting search ...")
    t0 = time.time()
    search.fit(data.X_train, data.y_train)
    train_seconds = time.time() - t0
    logging.info("Search fit complete in %.2fs", train_seconds)

    model_path = os.path.join(args.out_dir, f"model_{args.model}.joblib")
    if _HAS_JOBLIB:
        joblib.dump(search.best_estimator_, model_path)  # type: ignore[arg-type]
    else:
        with open(model_path, "wb") as f:
            pickle.dump(search.best_estimator_, f)

    results = {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "train_seconds": round(train_seconds, 2),
    }
    save_json(results, os.path.join(args.out_dir, "cv_results.json"))

    _save_feature_importance(search.best_estimator_, args.out_dir)

    if data.X_test is not None and data.y_test is not None:
        y_pred = search.predict(data.X_test)
        try:
            y_proba = search.predict_proba(data.X_test)[:, 1]
        except Exception:
            y_proba = None
        metrics = compute_metrics(data.y_test, y_pred, y_proba=y_proba)
        save_json(metrics, os.path.join(args.out_dir, "test_metrics.json"))

    print(json.dumps({"model_path": model_path, **results}, indent=2))


if __name__ == "__main__":
    main()


