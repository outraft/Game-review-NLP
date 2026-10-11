"""Evaluate the pretrained zero-shot DeBERTa NLI model on sentiment labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from annotate_reasons import DEFAULT_MODEL, DEFAULT_REVISION, build_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION = PROJECT_ROOT / "data" / "splits" / "validation.csv"
POSITIVE_CANDIDATE = "positive and recommending the game"
NEGATIVE_CANDIDATE = "negative and not recommending the game"
HYPOTHESIS_TEMPLATE = "The overall sentiment of this game review is {}."


def read_validation(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as validation_file:
        reader = csv.DictReader(validation_file)
        if not reader.fieldnames or not {"comment", "sentiment"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain comment and sentiment columns.")
        rows = list(reader)
    texts = [row["comment"] for row in rows]
    labels = np.asarray([int(row["sentiment"]) for row in rows], dtype=int)
    if not set(labels.tolist()).issubset({0, 1}):
        raise ValueError("Sentiment labels must be 0 or 1.")
    return texts, labels


def benchmark(
    validation_path: Path,
    model_name: str,
    revision: str,
    batch_size: int,
    device: str,
) -> None:
    texts, true_labels = read_validation(validation_path)
    classifier, device_name = build_pipeline(model_name, revision, device)
    predictions: list[int] = []
    confidences: list[float] = []

    candidates = [POSITIVE_CANDIDATE, NEGATIVE_CANDIDATE]
    for start in range(0, len(texts), batch_size):
        results = classifier(
            texts[start : start + batch_size],
            candidate_labels=candidates,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=False,
            batch_size=batch_size,
            truncation=True,
        )
        if isinstance(results, dict):
            results = [results]
        for result in results:
            label_scores = dict(zip(result["labels"], result["scores"]))
            positive_score = float(label_scores[POSITIVE_CANDIDATE])
            negative_score = float(label_scores[NEGATIVE_CANDIDATE])
            predictions.append(int(positive_score > negative_score))
            confidences.append(max(positive_score, negative_score))

    predicted_labels = np.asarray(predictions, dtype=int)
    print(f"Validation rows: {len(texts)}")
    print(f"Model: {model_name}@{revision}")
    print(f"Device: {device_name}; prediction = highest-scoring hypothesis")
    print(f"Accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")
    print(f"Macro-F1: {f1_score(true_labels, predicted_labels, average='macro'):.4f}")
    print(f"Weighted-F1: {f1_score(true_labels, predicted_labels, average='weighted'):.4f}")
    mean_score = float(np.mean(confidences)) if confidences else float("nan")
    print(f"Mean winning-label score (not calibrated): {mean_score:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=[0, 1],
            target_names=["negative", "positive"],
            digits=4,
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=true [negative, positive], columns=predicted):")
    print(confusion_matrix(true_labels, predicted_labels, labels=[0, 1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")
    benchmark(
        args.validation,
        args.model,
        args.revision,
        args.batch_size,
        args.device,
    )


if __name__ == "__main__":
    main()
