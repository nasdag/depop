# Keyword Spam Classifier (Submission)

This is a production-leaning implementation that re-implements the notebook logic as a clean, reproducible Python CLI pipeline, with clear preprocessing, feature engineering, model training, and evaluation.

## TL;DR
- Problem: classify keyword spamming in item descriptions and demote spam in ranking.
- Deliverable: reproducible CLI pipeline (`scripts/train.py`, `scripts/evaluate.py`) + metrics/artifacts.
- Highlights: strict train/test separation, safe text pipeline (TF‑IDF + numeric features), CV with F1, optional spaCy.
- Results: strong baseline (F1 ≈ 0.95 on test in our runs); details below.

### Reviewer quick start
```bash
# 1) Environment
pyenv local 3.11.9
python -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt

# 2) Train (saves model + CV results under artifacts/)
export PYTHONPATH=$(pwd):$PYTHONPATH
python scripts/train.py --train data/train_set.tsv --test data/test_set.tsv --model lr --out_dir artifacts --log_level INFO

# 3) Evaluate and export misclassified examples
python scripts/evaluate.py --model_path artifacts/model_lr.joblib --test data/test_set.tsv --out_dir artifacts --misclassified_csv artifacts/misclassified.csv
```

## Goals
- Train a classifier to detect keyword spamming in item descriptions.
- Address issues from the original notebook: structure, features, tuning, metrics.

## Project layout
- `spam_classifier/`
  - `data_io.py`: TSV loaders and splits.
  - `text_cleaning.py`: text normalization, lemmatization (optional), simple stats.
  - `numeric_features.py`: engineered numeric features.
  - `pipeline_builder.py`: TF-IDF + numeric features, model, hyperparams.
  - `metrics.py`: evaluation and JSON export.
- `scripts/`
  - `train.py`: train with CV + randomized search, persist best model.
  - `evaluate.py`: evaluate saved model on test set.
- `notebooks/` (optional): exploration and visualization.

## Environment
We standardized on Python 3.11 and a single virtualenv `.venv311` (created with `pyenv` + `python -m venv`).

```bash
# Ensure Python 3.11 is available
pyenv install -s 3.11.9
pyenv local 3.11.9

# Create and activate env
python -m venv .venv311
source .venv311/bin/activate

# Install requirements
pip install -r requirements.txt

# Optional: enable spaCy lemmatization
python -m spacy download en_core_web_sm
```

Notes:
- The pipeline runs WITHOUT spaCy (lemmatization and stopword features gracefully disable).
- We removed the old `.venv` and now only use `.venv311`.

## Quickstart
Train with CV and evaluate on the provided splits:

```bash
python scripts/train.py --train data/train_set.tsv --test data/test_set.tsv --model lr --out_dir artifacts
```

Evaluate a saved model on the test set:

```bash
echo "Using saved model:" artifacts/model_lr.joblib
python scripts/evaluate.py --model_path artifacts/model_lr.joblib --test data/test_set.tsv --out_dir artifacts
```

Outputs in `artifacts/`:
- `model_*.joblib`: best estimator.
- `cv_results.json`: best CV params/score (F1).
- `test_metrics.json`: accuracy, precision, recall, F1, ROC-AUC (if available).

## Results (our runs)
- Python 3.13 without spaCy (TF‑IDF + LR):
  - Test metrics: accuracy 0.948, precision 0.925, recall 0.981, F1 0.952, ROC‑AUC 0.990
- Python 3.11 with spaCy enabled (small CV search):
  - CV best F1 ≈ 0.876 (quick 3x3 search)
- Python 3.11 with spaCy enabled (larger CV search):
  - CV best F1 ≈ 0.934

Notes:
- Differences reflect search size/seed; larger searches generally improve F1.
- spaCy enables lemmatization/stopwords; TF‑IDF remains the main signal either way.

### Latest run snapshot
```json
{
  "best_f1_cv": 0.9341784349511897,
  "train_seconds": 2152.09,
  "best_params": {"penalty": "l2", "class_weight": "balanced", "C": 23.3572}
}
```
Notes:
- Repeated sklearn linear_model runtime warnings during CV are benign and stem from extreme C values; they don’t affect final metrics.

## Testing, logging, and config
- Run tests:
  ```bash
  source .venv311/bin/activate
  export PYTHONPATH=$(pwd):$PYTHONPATH
  pytest -q
  ```
- Training supports logging and JSON config overrides:
  ```bash
  python scripts/train.py --train data/train_set.tsv --test data/test_set.tsv --model lr --out_dir artifacts --log_level INFO --config configs/train.json
  ```
  Example `configs/train.json`:
  ```json
  {"n_iter": 50, "cv": 5, "max_features": 75000, "model": "lr"}
  ```
- Diagnostics saved:
  - `cv_results.json` includes training time.
  - `feature_importance_top100.json` lists top TF‑IDF/numeric signals when available.
  - `evaluate.py` can export misclassified rows via `--misclassified_csv`.

### Export misclassified examples
```bash
python scripts/evaluate.py \
  --model_path artifacts/model_lr.joblib \
  --test data/test_set.tsv \
  --misclassified_csv artifacts/misclassified.csv
```

## What we’d do next
- Threshold/operating point tuning and probability calibration.
- Add brand/NER features with robust capitalization and entity filtering.
- Add confusion matrix and PR/ROC curve exports.
- Compare LR vs XGBoost/LightGBM with dimensionality reduction (e.g., SVD) or learned embeddings.
- Add config-driven experiment tracking and model registry hooks.

## Design choices
- TF-IDF with unigrams+bigrams, sublinear TF, feature cap for robustness.
- Lightweight numeric features: stopword proportion, average word length, numbers ratio, hashtag count.
- Two models available: Logistic Regression (default), XGBoost (optional).
- Hyperparameter tuning via randomized search and Stratified K-Fold; scored on F1.
- Defensive spaCy usage so the pipeline works even if model isn’t installed.

## Next steps (if we had more time)
- Add brand/NER features reliably (case restoration and entity filtering).
- Calibrate probabilities; add threshold tuning for ranking demotion use-cases.
- Add robust text pipelines (URLs, emoji/hashtag handling, language detection).
- Train/serve split, model registry, and experiment tracking.
- Error analysis notebook and slice metrics.

## Notebooks
The repo intentionally has no notebooks checked in for the submission (kept minimal and reproducible via CLI). If desired, add your own under `notebooks/` for visualization. The CLI remains the source of truth.

## What we fixed from the initial notebook
- Data leakage risk: training and test were concatenated before feature engineering. We now keep strict separation and fit transforms inside a Pipeline.
- Fragile NLP flow: spaCy assumed everywhere and NER used incorrectly. We made spaCy optional; lemmatization/stopwords disable safely; removed broken NER (to reintroduce properly later).
- Dynamic/unsafe patterns: removed `eval`-based feature application and any storage of spaCy `Doc` objects in dataframes; replaced with explicit, testable functions.
- Unstructured workflow: moved from an all-in-one notebook to a package + CLI (`scripts/train.py`, `scripts/evaluate.py`) with persisted models and metrics.
- Evaluation gaps: added cross-validated randomized search (F1 scorer), plus test metrics (accuracy, precision, recall, F1, ROC-AUC) saved to JSON.
- Serialization issues: replaced lambdas in transformers with top-level functions so the pipeline is picklable and persistable.
