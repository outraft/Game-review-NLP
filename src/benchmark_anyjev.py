"""Benchmark a local AnyJev/Tacit typed-decision model on sentiment."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION = PROJECT_ROOT / "data" / "splits" / "validation.csv"
DEFAULT_MODEL = "morriszjm/Tacit-1.7B"
DEFAULT_REVISION = "1cb46fff0ffeefc4965455185074be5c62adccf7"
QUESTION = "What overall sentiment does the reviewer express about the game?"
OPTIONS = ("positive", "negative")


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


def benchmark(validation_path: Path, model_name: str, revision: str, batch_size: int) -> None:
    try:
        from huggingface_hub import snapshot_download
        import torch
        from anyjev import Tacit
    except ImportError as error:
        raise RuntimeError(
            "AnyJev dependencies are missing. Install them with: "
            "python -m pip install -r requirements-jev.txt"
        ) from error

    texts, true_labels = read_validation(validation_path)
    model_path = snapshot_download(repo_id=model_name, revision=revision)
    model = Tacit.from_pretrained(
        model_path,
        dtype="bfloat16",
        device_map="auto",
    )

    predicted_labels: list[int] = []
    winning_scores: list[float] = []
    routes: dict[str, int] = {}
    for start in range(0, len(texts), batch_size):
        items = [
            {"state": text, "question": QUESTION, "options": list(OPTIONS)}
            for text in texts[start : start + batch_size]
        ]
        decisions = model.decide_batch(items)
        for decision in decisions:
            answer = decision["answer"].strip().lower()
            if answer not in OPTIONS:
                raise ValueError(f"AnyJev returned unexpected answer {answer!r}.")
            predicted_labels.append(int(answer == "positive"))
            winning_scores.append(float(decision["probs"][answer]))
            route = str(decision.get("route", "unknown"))
            routes[route] = routes.get(route, 0) + 1

    predictions = np.asarray(predicted_labels, dtype=int)
    print(f"Validation rows: {len(texts)}")
    print(f"Method: AnyJev Tacit typed choice; model: {model_name}@{revision}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("Input: review text; options: positive / negative")
    print(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
    print(f"Macro-F1: {f1_score(true_labels, predictions, average='macro'):.4f}")
    print(f"Weighted-F1: {f1_score(true_labels, predictions, average='weighted'):.4f}")
    print(f"Mean winning-option score (not calibrated): {np.mean(winning_scores):.4f}")
    print(f"Decision routes: {routes}")
    print("\nClassification report:")
    print(
        classification_report(
            true_labels,
            predictions,
            labels=[0, 1],
            target_names=["negative", "positive"],
            digits=4,
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=true [negative, positive], columns=predicted):")
    print(confusion_matrix(true_labels, predictions, labels=[0, 1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")
    try:
        benchmark(args.validation, args.model, args.revision, args.batch_size)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
