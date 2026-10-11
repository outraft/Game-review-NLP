"""Compare majority, word-only, and word+character sentiment models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from classifier import make_vectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "splits" / "train.csv"
DEFAULT_VALIDATION = PROJECT_ROOT / "data" / "splits" / "validation.csv"


def read_split(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as data_file:
        reader = csv.DictReader(data_file)
        if not reader.fieldnames or not {"comment", "sentiment", "app_id"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain comment, sentiment, and app_id columns.")
        rows = list(reader)

    texts = [row["comment"] for row in rows]
    labels = np.asarray([int(row["sentiment"]) for row in rows], dtype=int)
    groups = np.asarray([row["app_id"] for row in rows], dtype=object)
    return texts, labels, groups


def word_model() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)),
        ]
    )


def word_char_model() -> Pipeline:
    return Pipeline(
        [
            ("features", make_vectorizer()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1200, C=3.0)),
        ]
    )


def report_scores(name: str, truth: np.ndarray, predictions: np.ndarray) -> None:
    print(
        f"{name}: accuracy={accuracy_score(truth, predictions):.4f} "
        f"macro-F1={f1_score(truth, predictions, average='macro'):.4f} "
        f"weighted-F1={f1_score(truth, predictions, average='weighted'):.4f}"
    )


def benchmark(train_path: Path, validation_path: Path) -> None:
    train_texts, train_labels, train_games = read_split(train_path)
    valid_texts, valid_labels, _ = read_split(validation_path)
    n_splits = min(3, len(np.unique(train_games)))
    if n_splits < 2:
        raise ValueError("Grouped cross-validation requires at least two games in the training split.")
    cv = GroupKFold(n_splits=n_splits)

    print(f"Training rows: {len(train_texts)}; validation rows: {len(valid_texts)}")
    print(f"Training titles: {len(np.unique(train_games))}; grouped CV folds: {n_splits}")
    baseline_oof = cross_val_predict(
        word_model(), train_texts, train_labels, cv=cv, groups=train_games, n_jobs=1
    )
    baseline_cv_f1 = f1_score(train_labels, baseline_oof, average="macro")
    print(f"Word-only grouped-CV macro-F1: {baseline_cv_f1:.4f}")

    search = GridSearchCV(
        word_char_model(),
        param_grid={"classifier__C": [0.3, 1.0, 3.0]},
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(train_texts, train_labels, groups=train_games)
    print(
        f"Tuned word+character grouped-CV macro-F1: {search.best_score_:.4f} "
        f"(C={search.best_params_['classifier__C']})"
    )

    majority = DummyClassifier(strategy="most_frequent").fit(train_texts, train_labels)
    baseline = word_model().fit(train_texts, train_labels)
    tuned = search.best_estimator_
    majority_predictions = majority.predict(valid_texts)
    baseline_predictions = baseline.predict(valid_texts)
    tuned_predictions = tuned.predict(valid_texts)

    print("\nHeld-out review-level validation (used only after CV model selection):")
    report_scores("Majority baseline", valid_labels, majority_predictions)
    report_scores("Word-only baseline", valid_labels, baseline_predictions)
    report_scores("Tuned word+character model", valid_labels, tuned_predictions)
    print("\nTuned model classification report:")
    print(
        classification_report(
            valid_labels,
            tuned_predictions,
            labels=[0, 1],
            target_names=["negative", "positive"],
            digits=4,
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=true [negative, positive], columns=predicted):")
    print(confusion_matrix(valid_labels, tuned_predictions, labels=[0, 1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    args = parser.parse_args()
    benchmark(args.train, args.validation)


if __name__ == "__main__":
    main()
