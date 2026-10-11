"""Validate and deduplicate the normalized Steam review CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "steam_reviews.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "steam_reviews_cleaned.csv"
DEFAULT_CONFLICTS = PROJECT_ROOT / "data" / "audit" / "conflicting_reviews.csv"


def normalize_comment(comment: str) -> str:
    return " ".join(comment.casefold().split())


def parse_sentiment(value: str, row_number: int) -> str:
    normalized = value.strip().lower()
    if normalized in {"1", "positive", "pos", "true"}:
        return "1"
    if normalized in {"0", "negative", "neg", "false"}:
        return "0"
    raise ValueError(f"Row {row_number}: invalid sentiment {value!r}; expected 0/1.")


def clean_reviews(input_path: Path, output_path: Path, conflict_path: Path) -> tuple[int, int]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"{input_path} is empty or missing a header.")
        fields = reader.fieldnames
        columns = {name.strip().lower(): name for name in fields if name}
        if "comment" not in columns or "sentiment" not in columns:
            raise ValueError("Input CSV must include 'comment' and 'sentiment' columns.")

        raw_rows = list(reader)

    # Remove all copies of an identical text if it was assigned conflicting
    # sentiments; retaining one would make the target label ambiguous.
    labels_by_comment: dict[str, set[str]] = defaultdict(set)
    counts_by_comment: dict[str, int] = defaultdict(int)
    for row_number, row in enumerate(raw_rows, start=2):
        text = (row.get(columns["comment"]) or "").strip()
        if not text:
            continue
        label = parse_sentiment(row.get(columns["sentiment"]) or "", row_number)
        key = normalize_comment(text)
        labels_by_comment[key].add(label)
        counts_by_comment[key] += 1

    conflicts = {text for text, labels in labels_by_comment.items() if len(labels) > 1}
    conflict_path.parent.mkdir(parents=True, exist_ok=True)
    with conflict_path.open("w", encoding="utf-8", newline="") as audit_file:
        writer = csv.writer(audit_file)
        writer.writerow(["normalized_comment", "sentiment_labels", "row_count"])
        for text in sorted(conflicts):
            labels = "|".join(sorted(labels_by_comment[text]))
            writer.writerow([text, labels, counts_by_comment[text]])

    kept_rows: list[dict[str, str]] = []
    seen_review_ids: set[tuple[str, str]] = set()
    seen_comments: set[str] = set()
    removed_empty = removed_conflict = removed_duplicate_id = removed_duplicate_text = 0

    for row_number, row in enumerate(raw_rows, start=2):
        text = (row.get(columns["comment"]) or "").strip()
        if not text:
            removed_empty += 1
            continue
        label = parse_sentiment(row.get(columns["sentiment"]) or "", row_number)
        normalized = normalize_comment(text)
        if normalized in conflicts:
            removed_conflict += 1
            continue

        # A Steam recommendation ID is unique per game. Legacy rows without
        # IDs fall back to text deduplication below.
        app_id = (row.get(columns.get("app_id", "")) or "").strip()
        review_id = (row.get(columns.get("steam_review_id", "")) or "").strip()
        identity = (app_id, review_id)
        if review_id and identity in seen_review_ids:
            removed_duplicate_id += 1
            continue
        if normalized in seen_comments:
            removed_duplicate_text += 1
            continue

        row[columns["comment"]] = text
        row[columns["sentiment"]] = label
        kept_rows.append(row)
        seen_comments.add(normalized)
        if review_id:
            seen_review_ids.add(identity)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as cleaned_file:
        writer = csv.DictWriter(cleaned_file, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"Input rows: {len(raw_rows)}")
    print(f"Removed empty comments: {removed_empty}")
    print(f"Removed conflicting-text rows: {removed_conflict}")
    print(f"Removed duplicate Steam IDs: {removed_duplicate_id}")
    print(f"Removed duplicate comments: {removed_duplicate_text}")
    print(f"Clean rows: {len(kept_rows)}")
    print(f"Conflict audit: {conflict_path}")
    print(f"Cleaned CSV: {output_path}")
    return len(raw_rows), len(kept_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and validate Steam review CSV data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--conflicts", type=Path, default=DEFAULT_CONFLICTS)
    args = parser.parse_args()
    clean_reviews(args.input, args.output, args.conflicts)


if __name__ == "__main__":
    main()
