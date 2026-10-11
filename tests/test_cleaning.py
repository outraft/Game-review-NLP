import csv
import tempfile
import unittest
from pathlib import Path

from src.cleaning import clean_reviews


class CleaningTests(unittest.TestCase):
    def test_removes_empty_duplicates_and_conflicting_text(self) -> None:
        rows = [
            {"steam_review_id": "1", "app_id": "10", "comment": "Same text", "sentiment": "1"},
            {"steam_review_id": "2", "app_id": "10", "comment": " SAME   TEXT ", "sentiment": "1"},
            {"steam_review_id": "3", "app_id": "10", "comment": "Conflict", "sentiment": "0"},
            {"steam_review_id": "4", "app_id": "10", "comment": " conflict ", "sentiment": "1"},
            {"steam_review_id": "5", "app_id": "10", "comment": "  ", "sentiment": "0"},
            {"steam_review_id": "6", "app_id": "10", "comment": "Keep me", "sentiment": "positive"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.csv"
            cleaned = root / "cleaned.csv"
            conflicts = root / "conflicts.csv"
            with source.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            input_count, kept_count = clean_reviews(source, cleaned, conflicts)
            with cleaned.open("r", encoding="utf-8", newline="") as csv_file:
                cleaned_rows = list(csv.DictReader(csv_file))
            with conflicts.open("r", encoding="utf-8", newline="") as csv_file:
                conflict_rows = list(csv.DictReader(csv_file))

        self.assertEqual(input_count, 6)
        self.assertEqual(kept_count, 2)
        self.assertEqual({row["comment"] for row in cleaned_rows}, {"Same text", "Keep me"})
        self.assertEqual(cleaned_rows[1]["sentiment"], "1")
        self.assertEqual(len(conflict_rows), 1)
        self.assertEqual(conflict_rows[0]["normalized_comment"], "conflict")


if __name__ == "__main__":
    unittest.main()
