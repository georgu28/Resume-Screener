"""
Train and evaluate the resume category classifier.

Pipeline: TF-IDF (word 1-2 grams) -> calibrated LinearSVC, over the
Resume-Atlas dataset (12k+ distinct labeled resumes, 43 categories).

The script:
  1. loads the dataset (downloading it first if missing),
  2. deduplicates and splits it (stratified hold-out),
  3. reports honest held-out metrics (top-1 / top-3 accuracy, macro-F1),
  4. refits on the full dataset and saves the model to models/resume_clf.joblib.

Usage:
    python train.py
"""

import json
import os
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from scripts.download_data import download

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "resume_atlas.csv")
MODEL_PATH = os.path.join(HERE, "models", "resume_clf.joblib")
METRICS_PATH = os.path.join(HERE, "models", "metrics.json")
RANDOM_STATE = 2024


def build_pipeline() -> Pipeline:
    """TF-IDF + calibrated LinearSVC. Calibration gives usable predict_proba."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=30000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        # LinearSVC is a strong, fast text classifier but has no probabilities;
        # CalibratedClassifierCV wraps it to expose calibrated predict_proba.
        ("clf", CalibratedClassifierCV(LinearSVC(), cv=3)),
    ])


def top_k_accuracy(model: Pipeline, X, y_true, k: int = 3) -> float:
    """Fraction of samples whose true label is among the model's top-k guesses."""
    proba = model.predict_proba(X)
    classes = model.classes_
    topk = classes[np.argsort(proba, axis=1)[:, -k:]]
    y_true = np.asarray(y_true)
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))


def main() -> None:
    download(dest=DATA_PATH)  # no-op if already present

    df = pd.read_csv(DATA_PATH).dropna().drop_duplicates(subset=["Text"])
    X, y = df["Text"], df["Category"]
    print(f"Loaded {len(df)} distinct resumes across {y.nunique()} categories")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("Training (held-out evaluation)...")
    t0 = time.time()
    model = build_pipeline().fit(X_train, y_train)

    y_pred = model.predict(X_test)
    top1 = accuracy_score(y_test, y_pred)
    top3 = top_k_accuracy(model, X_test, y_test, k=3)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n  Held-out top-1 accuracy : {top1:.3f}")
    print(f"  Held-out top-3 accuracy : {top3:.3f}")
    print(f"  Macro F1                : {macro_f1:.3f}")
    print(f"  Trained in {time.time() - t0:.1f}s\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Refit on the full dataset for the shipped model (more data = better).
    print("Refitting on full dataset for the production model...")
    model = build_pipeline().fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metrics = {
        "n_resumes": int(len(df)),
        "n_categories": int(y.nunique()),
        "holdout_top1_accuracy": round(top1, 4),
        "holdout_top3_accuracy": round(top3, 4),
        "holdout_macro_f1": round(macro_f1, 4),
        "trained_at": time.strftime("%Y-%m-%d"),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}: {metrics}")


if __name__ == "__main__":
    main()
