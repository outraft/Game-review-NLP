import csv
import tempfile
import unittest
from pathlib import Path

from src.classifier import NEGATIVE_REASONS, POSITIVE_REASONS, REASONS, read_dataset


class ReadDatasetTests(unittest.TestCase):
    def test_blank_reasons_are_unknown_and_none_is_an_annotation(self) -> None:
        rows = [
            {"comment": "Positive review text", "sentiment": "1", "reason": ""},
            {"comment": "Negative review text", "sentiment": "0", "reason": "none"},
            {"comment": "A reasoned negative review", "sentiment": "0", "reason": "FPS drops|bugs/glitches"},
            {"comment": "A reasoned positive review", "sentiment": "positive", "reason": "story driven"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            with path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["comment", "sentiment", "reason"])
                writer.writeheader()
                writer.writerows(rows)

            texts, sentiments, reason_targets, reason_is_annotated = read_dataset(path)

        self.assertEqual(len(texts), 4)
        self.assertEqual(sentiments.tolist(), [1, 0, 0, 1])
        self.assertEqual(reason_is_annotated.tolist(), [False, True, True, True])
        self.assertEqual(int(reason_targets[1].sum()), 0)  # explicit "none"
        self.assertEqual(reason_targets[2, REASONS.index("FPS drops")], 1)
        self.assertEqual(reason_targets[2, REASONS.index("bugs/glitches")], 1)
        self.assertEqual(reason_targets[3, REASONS.index("story driven")], 1)
        self.assertIn("FPS drops", NEGATIVE_REASONS)
        self.assertIn("fun to play", POSITIVE_REASONS)

    def test_rejects_reason_from_wrong_sentiment_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            with path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["comment", "sentiment", "reason"])
                writer.writeheader()
                writer.writerow({"comment": "Bad review", "sentiment": "0", "reason": "fun to play"})

            with self.assertRaisesRegex(ValueError, "cannot use reason"):
                read_dataset(path)


if __name__ == "__main__":
    unittest.main()
