"""Assign provisional game-review reasons with a local zero-shot NLI model.

These outputs are machine-generated pseudo-labels, not verified ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from classifier import NEGATIVE_REASONS, POSITIVE_REASONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "steam_reviews_cleaned.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "annotated" / "steam_reviews_reason_predictions.csv"
DEFAULT_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
DEFAULT_REVISION = "8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8"
HYPOTHESIS_TEMPLATE = "The review's reason is {}."

# Use short natural-language descriptions as NLI candidates, but store the
# project's canonical labels in the output CSV.
NEGATIVE_CANDIDATES = {
    "optimization": "poor game optimization or performance problems",
    "FPS drops": "frequent frame-rate or FPS drops",
    "bad community": "a toxic or unpleasant player community",
    "hard game": "the game being too difficult",
    "connectivity issues": "connectivity or server problems",
    "price": "the game being too expensive or poor value for money",
    "bugs/glitches": "bugs, glitches, or crashes",
    "poor controls": "poor, unresponsive, or awkward controls",
    "poor soundtrack/audio": "a poor soundtrack, audio, or sound design",
    "low replayability": "low replayability or little reason to replay",
}
POSITIVE_CANDIDATES = {
    "fun to play": "the game being fun to play",
    "multiplayer accessible": "accessible multiplayer that is easy to join",
    "mod compatible": "support for mods and mod compatibility",
    "graphically pleasing": "pleasing graphics or visual design",
    "developer connection": "good communication between developers and players",
    "depth": "depth or meaningful game systems",
    "gameplay": "enjoyable gameplay",
    "story driven": "a compelling story or story-driven design",
    "responsive controls": "responsive and satisfying controls",
    "good soundtrack/audio": "a good soundtrack, audio, or sound design",
    "high replayability": "high replayability and reasons to play again",
}


def identity(row: dict[str, str]) -> str:
    app_id = (row.get("app_id") or "").strip()
    review_id = (row.get("steam_review_id") or "").strip()
    if app_id and review_id:
        return f"{app_id}:{review_id}"
    digest = hashlib.sha256((row.get("comment") or "").casefold().encode("utf-8")).hexdigest()
    return f"text:{digest}"


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            raise ValueError(f"{path} is empty or missing a header.")
        required = {"comment", "sentiment", "reason"}
        if not required.issubset({name.strip().lower() for name in reader.fieldnames}):
            raise ValueError("Input CSV must include comment, sentiment, and reason columns.")
        return reader.fieldnames, list(reader)


def load_processed_ids(path: Path, model_name: str, threshold: float, overwrite: bool) -> set[str]:
    if overwrite or not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as output_file:
        reader = csv.DictReader(output_file)
        if not reader.fieldnames:
            return set()
        rows = list(reader)
    existing_models = {
        row.get("reason_model", "")
        for row in rows
        if row.get("reason_source") == "zero_shot_auto" and row.get("reason_model")
    }
    if existing_models and existing_models != {model_name}:
        raise ValueError(
            f"Output already contains pseudo-labels from {sorted(existing_models)}. "
            "Use --overwrite to regenerate them with a different model."
        )
    existing_thresholds = {
        row.get("reason_threshold", "")
        for row in rows
        if row.get("reason_source") == "zero_shot_auto" and row.get("reason_threshold")
    }
    threshold_value = f"{threshold:.8g}"
    if existing_thresholds and existing_thresholds != {threshold_value}:
        raise ValueError(
            f"Output contains pseudo-labels with thresholds {sorted(existing_thresholds)}. "
            "Use --overwrite to apply a different threshold."
        )
    return {identity(row) for row in rows}


def build_pipeline(model_name: str, revision: str, device_option: str) -> tuple[Any, str]:
    try:
        import torch
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError(
            "Zero-shot dependencies are missing. Install them with: "
            "python -m pip install -r requirements-zero-shot.txt"
        ) from error

    if device_option == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA GPU.")
        device: int = 0
        device_name = torch.cuda.get_device_name(0)
    elif device_option == "cpu":
        device = -1
        device_name = "CPU"
    elif torch.cuda.is_available():
        device = 0
        device_name = torch.cuda.get_device_name(0)
    else:
        device = -1
        device_name = "CPU"

    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        revision=revision,
        device=device,
    )
    return classifier, device_name


def write_summary(rows: list[dict[str, str]], output_path: Path, model_name: str) -> None:
    reason_counts: dict[tuple[str, str], list[float]] = defaultdict(list)
    auto_labeled = 0
    total = 0
    for row in rows:
        total += 1
        if row.get("reason_source") != "zero_shot_auto" or row.get("reason_model") != model_name:
            continue
        auto_labeled += 1
        scores = json.loads(row.get("reason_scores") or "{}")
        selected = set(filter(None, (row.get("reason") or "").split("|"))) - {"none"}
        sentiment = "positive" if row.get("sentiment") == "1" else "negative"
        for reason in selected:
            if reason in scores:
                reason_counts[(sentiment, reason)].append(float(scores[reason]))

    negative = set(NEGATIVE_REASONS)
    positive = set(POSITIVE_REASONS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(
            ["sentiment", "reason", "review_count", "percent_of_all_reviews", "mean_score", "model"]
        )
        for sentiment, reasons in (("negative", negative), ("positive", positive)):
            for reason in sorted(reasons):
                scores = reason_counts.get((sentiment, reason), [])
                writer.writerow(
                    [
                        sentiment,
                        reason,
                        len(scores),
                        round(100 * len(scores) / total, 2) if total else 0,
                        round(sum(scores) / len(scores), 4) if scores else "",
                        model_name,
                    ]
                )
    print(f"Auto-labeled reviews: {auto_labeled}/{total}")
    print(f"Reason summary: {output_path}")


def summary_path_for(output_path: Path) -> Path:
    if output_path.name == DEFAULT_OUTPUT.name:
        return output_path.parent / "reason_summary.csv"
    return output_path.with_name(f"{output_path.stem}_summary.csv")


def annotate(
    input_path: Path,
    output_path: Path,
    model_name: str,
    revision: str,
    threshold: float,
    batch_size: int,
    device_option: str,
    limit: int | None,
    overwrite: bool,
    inference_batch_size: int,
) -> None:
    fields, all_rows = load_rows(input_path)
    rows = all_rows[:limit] if limit is not None else all_rows
    if limit is not None and limit < 1:
        raise ValueError("--limit must be a positive integer")

    output_fields = list(fields)
    for field in ("reason_scores", "reason_source", "reason_model", "reason_threshold"):
        if field not in output_fields:
            output_fields.append(field)

    model_ref = f"{model_name}@{revision}"
    threshold_value = f"{threshold:.8g}"
    already_done = load_processed_ids(output_path, model_ref, threshold, overwrite)
    pending_by_sentiment: dict[int, list[dict[str, str]]] = {0: [], 1: []}
    passthrough: list[dict[str, str]] = []
    for row in rows:
        row_id = identity(row)
        if row_id in already_done:
            continue
        try:
            sentiment = int(row["sentiment"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Review {row_id}: sentiment must be 0 or 1.") from error
        if sentiment not in (0, 1):
            raise ValueError(f"Review {row_id}: sentiment must be 0 or 1.")
        if (row.get("reason") or "").strip():
            copied = {**row, "reason_source": row.get("reason_source") or "provided"}
            passthrough.append(copied)
        else:
            pending_by_sentiment[sentiment].append(row)

    if not passthrough and not any(pending_by_sentiment.values()) and output_path.exists() and not overwrite:
        print("Nothing to annotate; all rows already exist in the output.")
        with output_path.open("r", encoding="utf-8-sig", newline="") as output_file:
            annotated_rows = list(csv.DictReader(output_file))
        write_summary(annotated_rows, summary_path_for(output_path), model_ref)
        return

    model, device_name = build_pipeline(model_name, revision, device_option)
    print(f"Loaded {model_ref} on {device_name}; threshold={threshold:.2f}; batch size={batch_size}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = overwrite or not output_path.exists()
    mode = "w" if write_header else "a"
    completed = 0

    with output_path.open(mode, encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=output_fields, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()
            output_file.flush()

        for row in passthrough:
            writer.writerow({field: row.get(field, "") for field in output_fields})
            completed += 1

        for sentiment, candidates in (
            (0, NEGATIVE_CANDIDATES),
            (1, POSITIVE_CANDIDATES),
        ):
            pending = pending_by_sentiment[sentiment]
            candidate_descriptions = list(candidates.values())
            description_to_reason = {description: reason for reason, description in candidates.items()}
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                texts = [row.get("comment", "") for row in batch]
                results = model(
                    texts,
                    candidate_labels=candidate_descriptions,
                    hypothesis_template=HYPOTHESIS_TEMPLATE,
                    multi_label=True,
                    batch_size=inference_batch_size,
                    truncation=True,
                )
                if isinstance(results, dict):
                    results = [results]

                for row, result in zip(batch, results):
                    score_by_description = dict(zip(result["labels"], result["scores"]))
                    scores = {
                        description_to_reason[description]: round(float(score), 5)
                        for description, score in score_by_description.items()
                    }
                    selected = [
                        reason
                        for reason in candidates
                        if scores.get(reason, 0.0) >= threshold
                    ]
                    output_row = {
                        **row,
                        "reason": "|".join(selected) if selected else "none",
                        "reason_scores": json.dumps(scores, ensure_ascii=False, sort_keys=True),
                        "reason_source": "zero_shot_auto",
                        "reason_model": model_ref,
                        "reason_threshold": threshold_value,
                    }
                    writer.writerow({field: output_row.get(field, "") for field in output_fields})
                    completed += 1

                # Flush each completed batch so an interrupted run can resume.
                output_file.flush()
                print(f"Processed {completed} new reviews in this run.", end="\r", flush=True)

    print(f"\nWrote/resumed pseudo-labels at {output_path}")
    with output_path.open("r", encoding="utf-8-sig", newline="") as output_file:
        annotated_rows = list(csv.DictReader(output_file))
    write_summary(annotated_rows, summary_path_for(output_path), model_ref)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="immutable Hugging Face revision")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=64, help="reviews per checkpoint chunk")
    parser.add_argument("--inference-batch-size", type=int, default=16, help="NLI pairs per model batch")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, help="process only the first N rows (for a smoke test)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output rather than resume it")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")
    if args.inference_batch_size < 1:
        parser.error("--inference-batch-size must be a positive integer")

    try:
        annotate(
            args.input,
            args.output,
            args.model,
            args.revision,
            args.threshold,
            args.batch_size,
            args.device,
            args.limit,
            args.overwrite,
            args.inference_batch_size,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
