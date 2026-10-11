"""Train and use a TF-IDF game-review classifier.

Each review gets a binary sentiment prediction and zero or more matching
reasons. CSV input columns: comment, sentiment, reason. Multiple reasons are
separated with |.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import FeatureUnion


NEGATIVE_REASONS = (
    "optimization",
    "FPS drops",
    "bad community",
    "hard game",
    "connectivity issues",
    "price",
    "bugs/glitches",
    "poor controls",
    "poor soundtrack/audio",
    "low replayability",
)
POSITIVE_REASONS = (
    "fun to play",
    "multiplayer accessible",
    "mod compatible",
    "graphically pleasing",
    "developer connection",
    "depth",
    "gameplay",
    "story driven",
    "responsive controls",
    "good soundtrack/audio",
    "high replayability",
)
REASONS = tuple(dict.fromkeys(NEGATIVE_REASONS + POSITIVE_REASONS))
REASON_LOOKUP = {reason.lower(): reason for reason in REASONS}
MODEL_VERSION = 4


def normalize_reason(value: str) -> str:
    """Normalize capitalization, underscores, and whitespace in reason labels."""
    return " ".join(value.strip().lower().replace("_", " ").split())


def parse_sentiment(value: str, row_number: int) -> int:
    normalized = value.strip().lower()
    if normalized in {"1", "positive", "pos"}:
        return 1
    if normalized in {"0", "negative", "neg"}:
        return 0
    raise ValueError(
        f"Row {row_number}: sentiment must be 0/negative or 1/positive; got {value!r}."
    )


def read_dataset(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Read and validate labeled CSV rows."""
    texts: list[str] = []
    sentiments: list[int] = []
    reason_rows: list[list[int]] = []
    reason_is_annotated: list[bool] = []

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError(f"{path} is empty or has no CSV header.")

        columns = {name.strip().lower(): name for name in reader.fieldnames if name}
        if "comment" not in columns or "sentiment" not in columns:
            raise ValueError("CSV must have 'comment' and 'sentiment' columns.")
        reason_column = columns.get("reason")
        if not reason_column:
            raise ValueError("CSV must have a 'reason' column.")

        for row_number, row in enumerate(reader, start=2):
            comment = (row.get(columns["comment"]) or "").strip()
            if not comment:
                raise ValueError(f"Row {row_number}: comment cannot be empty.")

            sentiment = parse_sentiment(row.get(columns["sentiment"]) or "", row_number)
            raw_reasons = (row.get(reason_column) or "").strip()
            selected: set[str] = set()
            annotation_complete = raw_reasons.lower() not in {"", "null"}
            if raw_reasons.lower() not in {"", "null", "none", "[]"}:
                for raw_reason in raw_reasons.split("|"):
                    normalized_reason = normalize_reason(raw_reason)
                    if normalized_reason not in REASON_LOOKUP:
                        raise ValueError(
                            f"Row {row_number}: unknown reason {raw_reason!r}. "
                            f"Choose from: {', '.join(REASONS)}."
                        )
                    reason = REASON_LOOKUP[normalized_reason]
                    selected.add(reason)

            allowed_reasons = POSITIVE_REASONS if sentiment == 1 else NEGATIVE_REASONS
            mismatched_reasons = selected.difference(allowed_reasons)
            if mismatched_reasons:
                sentiment_name = "positive" if sentiment == 1 else "negative"
                raise ValueError(
                    f"Row {row_number}: {sentiment_name} sentiment cannot use reason(s): "
                    f"{', '.join(sorted(mismatched_reasons))}."
                )

            texts.append(comment)
            sentiments.append(sentiment)
            reason_rows.append([int(reason in selected) for reason in REASONS])
            reason_is_annotated.append(annotation_complete)

    if not texts:
        raise ValueError(f"{path} has a header but no labeled rows.")
    return (
        texts,
        np.asarray(sentiments, dtype=int),
        np.asarray(reason_rows, dtype=int),
        np.asarray(reason_is_annotated, dtype=bool),
    )


def make_vectorizer() -> FeatureUnion:
    """Combine word and character TF-IDF features for robust text matching."""
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=300_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def train(data_path: Path, model_path: Path) -> None:
    texts, sentiment_targets, reason_targets, reason_is_annotated = read_dataset(data_path)
    if len(set(sentiment_targets.tolist())) < 2:
        raise ValueError("Training data must contain both positive and negative examples.")

    vectorizer = make_vectorizer()
    features = vectorizer.fit_transform(texts)
    sentiment_model = LogisticRegression(class_weight="balanced", max_iter=1200, C=3.0)
    sentiment_model.fit(features, sentiment_targets)

    reason_models: dict[str, Any] = {}
    annotated_targets = reason_targets[reason_is_annotated]
    for index, reason in enumerate(REASONS):
        targets = annotated_targets[:, index]
        if len(targets) == 0:
            continue
        if len(set(targets.tolist())) == 1:
            # Keep training usable while being explicit that a reason with no
            # positive examples cannot be learned from this dataset.
            model = DummyClassifier(strategy="constant", constant=int(targets[0]))
        else:
            model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(features[reason_is_annotated], targets)
        reason_models[reason] = model

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": MODEL_VERSION,
            "reasons": REASONS,
            "vectorizer": vectorizer,
            "sentiment_model": sentiment_model,
            "reason_models": reason_models,
            "reason_annotation_count": int(reason_is_annotated.sum()),
        },
        model_path,
    )

    print(f"Trained on {len(texts)} labeled comments.")
    print(f"Saved model to {model_path}")
    if not reason_is_annotated.any():
        print("Warning: no reasons have been annotated; reason predictions will be empty.")
    for index, reason in enumerate(REASONS):
        if not reason_is_annotated.any():
            break
        positives = int(annotated_targets[:, index].sum())
        if positives == 0:
            print(f"Warning: no training examples labeled '{reason}'; it will always predict absent.")
        elif positives == len(annotated_targets):
            print(f"Warning: every training example is labeled '{reason}'; it will always predict present.")


def load_model(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run the train command first.")
    model = joblib.load(path)
    if model.get("version") != MODEL_VERSION:
        raise ValueError("Unsupported model version. Retrain with the current classifier.py.")
    return model


def predict_rows(model: dict[str, Any], texts: list[str], threshold: float) -> list[dict[str, Any]]:
    features = model["vectorizer"].transform(texts)
    sentiment_model = model["sentiment_model"]
    sentiment_probabilities = sentiment_model.predict_proba(features)[:, 1]

    predictions: list[dict[str, Any]] = []
    for row_index, text in enumerate(texts):
        positive_probability = float(sentiment_probabilities[row_index])
        sentiment = int(positive_probability >= 0.5)
        allowed_reasons = POSITIVE_REASONS if sentiment == 1 else NEGATIVE_REASONS
        reason_scores: dict[str, float] = {}
        for reason in allowed_reasons:
            reason_model = model["reason_models"].get(reason)
            if reason_model is None:
                continue
            probabilities = reason_model.predict_proba(features[row_index])[0]
            positive_class = np.flatnonzero(reason_model.classes_ == 1)
            # A constant classifier has only one probability column.
            reason_scores[reason] = float(probabilities[positive_class[0]]) if len(positive_class) else 0.0
        predictions.append(
            {
                "comment": text,
                "sentiment": sentiment,
                "sentiment_label": "positive" if sentiment else "negative",
                "sentiment_confidence": round(max(positive_probability, 1 - positive_probability), 4),
                "positive_probability": round(positive_probability, 4),
                "reason": [reason for reason, score in reason_scores.items() if score >= threshold],
                "reason_scores": {reason: round(score, 4) for reason, score in reason_scores.items()},
            }
        )
    return predictions


def evaluate(data_path: Path, model_path: Path, threshold: float) -> None:
    texts, sentiment_targets, reason_targets, reason_is_annotated = read_dataset(data_path)
    model = load_model(model_path)
    predictions = predict_rows(model, texts, threshold)
    sentiment_predictions = np.asarray([item["sentiment"] for item in predictions])
    reason_predictions = np.asarray(
        [[int(reason in item["reason"]) for reason in model["reasons"]] for item in predictions]
    )

    print(f"Evaluated on {len(texts)} held-out comments")
    print(f"Sentiment accuracy: {accuracy_score(sentiment_targets, sentiment_predictions):.3f}")
    print(f"Sentiment macro F1: {f1_score(sentiment_targets, sentiment_predictions, average='macro', zero_division=0):.3f}")
    if reason_is_annotated.any():
        known_targets = reason_targets[reason_is_annotated]
        known_predictions = reason_predictions[reason_is_annotated]
        print(f"Reason micro F1: {f1_score(known_targets, known_predictions, average='micro', zero_division=0):.3f}")
        print(f"Reason macro F1: {f1_score(known_targets, known_predictions, average='macro', zero_division=0):.3f}")
        print("\nPer-reason F1:")
        for index, reason in enumerate(model["reasons"]):
            score = f1_score(known_targets[:, index], known_predictions[:, index], zero_division=0)
            print(f"  {reason}: {score:.3f}")
    else:
        print("Reason metrics skipped: the held-out CSV has no annotated reasons.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Game-review sentiment and reason classification")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train classifiers from a labeled CSV")
    train_parser.add_argument("--data", type=Path, required=True, help="CSV with comment, sentiment, and reason columns")
    train_parser.add_argument("--model", type=Path, default=Path("models/review_classifier.joblib"))

    predict_parser = commands.add_parser("predict", help="classify one comment")
    predict_parser.add_argument("--text", required=True, help="comment to classify")
    predict_parser.add_argument("--model", type=Path, default=Path("models/review_classifier.joblib"))
    predict_parser.add_argument("--threshold", type=float, default=0.5, help="reason probability threshold (default: 0.5)")

    evaluate_parser = commands.add_parser("evaluate", help="evaluate against a separate labeled CSV")
    evaluate_parser.add_argument("--data", type=Path, required=True, help="held-out labeled CSV")
    evaluate_parser.add_argument("--model", type=Path, default=Path("models/review_classifier.joblib"))
    evaluate_parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    try:
        if args.command == "train":
            train(args.data, args.model)
        elif args.command == "predict":
            if not 0 <= args.threshold <= 1:
                raise ValueError("threshold must be between 0 and 1")
            model = load_model(args.model)
            result = predict_rows(model, [args.text], args.threshold)[0]
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            if not 0 <= args.threshold <= 1:
                raise ValueError("threshold must be between 0 and 1")
            evaluate(args.data, args.model, args.threshold)
    except (FileNotFoundError, ValueError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
