"""Create reproducible, stratified train/validation CSV splits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "steam_reviews_cleaned.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"


def split_data(input_path: Path, output_dir: Path, validation_size: float, seed: int) -> None:
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"{input_path} is empty or missing a header.")
        fields = reader.fieldnames
        columns = {name.strip().lower(): name for name in fields if name}
        if "sentiment" not in columns:
            raise ValueError("Input CSV must include a 'sentiment' column.")
        rows = list(reader)

    labels = [row[columns["sentiment"]].strip() for row in rows]
    if len(set(labels)) < 2:
        raise ValueError("Need examples from both sentiment classes to make a stratified split.")

    training, validation = train_test_split(
        rows,
        test_size=validation_size,
        random_state=seed,
        stratify=labels,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, split_rows in (("train.csv", training), ("validation.csv", validation)):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fields, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(split_rows)

    print(f"Training reviews: {len(training)}")
    print(f"Validation reviews: {len(validation)}")
    print(f"Total reviews: {len(training) + len(validation)}")
    for label in sorted(set(labels)):
        train_count = sum(row[columns["sentiment"]] == label for row in training)
        validation_count = sum(row[columns["sentiment"]] == label for row in validation)
        print(f"Sentiment {label}: train={train_count}, validation={validation_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/validation data splits.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.validation_size < 1:
        parser.error("--validation-size must be between 0 and 1")
    split_data(args.input, args.output_dir, args.validation_size, args.seed)


if __name__ == "__main__":
    main()
