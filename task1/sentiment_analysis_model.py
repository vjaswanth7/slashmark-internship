"""
Sentiment Analysis Model
3-class sentiment classifier: positive / negative / neutral

Features:
- Text cleaning: regex cleanup, lowercase, stopword removal, lemmatization if available
- TF-IDF features
- Logistic Regression and Linear SVM comparison
- Accuracy, precision, recall, F1 evaluation
- Confusion matrix
- Model saving/loading with joblib

Usage:
    python sentiment_analysis_model.py --data demo_sentiment_data.csv --text "The food was great"
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

# Optional NLTK support
try:
    import nltk
    from nltk.stem import WordNetLemmatizer, SnowballStemmer
    from nltk.corpus import stopwords as nltk_stopwords
    NLTK_AVAILABLE = True
except Exception:
    nltk = None
    WordNetLemmatizer = None
    SnowballStemmer = None
    nltk_stopwords = None
    NLTK_AVAILABLE = False

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


LABELS = ["negative", "neutral", "positive"]


def _try_get_nltk_stopwords() -> set:
    if not NLTK_AVAILABLE:
        return set(ENGLISH_STOP_WORDS)
    try:
        return set(nltk_stopwords.words("english"))
    except Exception:
        return set(ENGLISH_STOP_WORDS)


STOP_WORDS = _try_get_nltk_stopwords()


class TextPreprocessor:
    """Clean and normalize text for sentiment analysis."""

    def __init__(self) -> None:
        self.lemmatizer = None
        self.stemmer = None

        if NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
                # wordnet corpus may be missing; we'll fall back gracefully
                _ = self.lemmatizer.lemmatize("cars")
            except Exception:
                self.lemmatizer = None
            try:
                self.stemmer = SnowballStemmer("english")
            except Exception:
                self.stemmer = None

    def clean(self, text: str) -> str:
        text = str(text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = text.lower()
        tokens = text.split()

        cleaned_tokens: List[str] = []
        for token in tokens:
            if token in STOP_WORDS or len(token) < 2:
                continue
            if self.lemmatizer is not None:
                try:
                    token = self.lemmatizer.lemmatize(token)
                except Exception:
                    pass
            elif self.stemmer is not None:
                token = self.stemmer.stem(token)
            cleaned_tokens.append(token)

        return " ".join(cleaned_tokens)


@dataclass
class SentimentModelResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float


def load_data(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if path.suffix.lower() in [".tsv", ".txt"]:
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    expected_cols = {"text", "label"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Dataset must contain columns {sorted(expected_cols)}. Found: {list(df.columns)}"
        )

    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)

    if len(df) < 30:
        raise ValueError(
            f"Need more training samples for a useful model. Found only {len(df)} rows."
        )

    return df


def build_pipeline(model_name: str) -> Pipeline:
    vectorizer = TfidfVectorizer(
        preprocessor=TextPreprocessor().clean,
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1,
        max_df=0.95,
    )

    if model_name == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
    elif model_name == "svm":
        clf = LinearSVC(class_weight="balanced", random_state=42)
    else:
        raise ValueError("model_name must be either 'logreg' or 'svm'")

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def evaluate_model(y_true, y_pred, model_name: str) -> SentimentModelResult:
    labels = LABELS
    return SentimentModelResult(
        model_name=model_name,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        recall=recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        f1=f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
    )


def train_and_select_best(df: pd.DataFrame):
    X = df["text"].values
    y = df["label"].values

    # Stratify ensures each class is represented in train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logreg": build_pipeline("logreg"),
        "svm": build_pipeline("svm"),
    }

    results: List[SentimentModelResult] = []
    trained = {}

    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        result = evaluate_model(y_test, preds, name)
        results.append(result)
        trained[name] = pipe

    best = max(results, key=lambda r: (r.f1, r.accuracy))
    best_model = trained[best.model_name]

    best_preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_preds, labels=LABELS)
    report = classification_report(y_test, best_preds, labels=LABELS, zero_division=0)

    return best_model, results, cm, report, (X_test, y_test)


def save_model(model, output_path: str) -> None:
    joblib.dump(model, output_path)


def load_model(model_path: str):
    return joblib.load(model_path)


def predict_sentiment(model, text: str) -> str:
    return str(model.predict([text])[0])


def batch_predict(model, texts: Iterable[str]) -> List[str]:
    return list(model.predict(list(texts)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 3-class sentiment analysis model.")
    parser.add_argument("--data", type=str, default="demo_sentiment_data.csv", help="Path to CSV/TSV with columns text,label")
    parser.add_argument("--model-out", type=str, default="sentiment_model.joblib", help="Where to save the trained model")
    parser.add_argument("--text", type=str, default=None, help="Optional single text for prediction after training")
    args = parser.parse_args()

    df = load_data(args.data)
    model, results, cm, report, _ = train_and_select_best(df)

    print("\n=== Model Comparison ===")
    for r in results:
        print(
            f"{r.model_name:>7} | Accuracy: {r.accuracy:.3f} | Precision: {r.precision:.3f} | "
            f"Recall: {r.recall:.3f} | F1: {r.f1:.3f}"
        )

    print(f"\nBest model: {max(results, key=lambda r: (r.f1, r.accuracy)).model_name}")
    print("\n=== Classification Report ===")
    print(report)
    print("\n=== Confusion Matrix ===")
    print(cm)

    save_model(model, args.model_out)
    print(f"\nSaved model to: {args.model_out}")

    if args.text:
        pred = predict_sentiment(model, args.text)
        print(f"\nText: {args.text}\nPrediction: {pred}")


if __name__ == "__main__":
    main()
