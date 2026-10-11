### Dataset Distribution

| Category | Number of Reviews |
|---|---:|
| Original dataset | 3,000 |
| Cleaned dataset | 2,563 |
| Positive reviews | 1,146 |
| Negative reviews | 1,417 |
| Training set (80%) | 2,049 |
| Validation set (20%) | 514 |

### Files

- `src/cleaning.py` — Cleans reviews, checks conflicting labels, and corrects selected sentiment labels.
- `src/split_data.py` — Splits the cleaned dataset into training and validation sets.
- `steam_reviews_cleaned.csv` — Final cleaned dataset.
- `conflicting_reviews.csv` — Reviews identified with conflicting labels.
- `train.csv` — Training dataset.
- `validation.csv` — Validation dataset.

### Run the Scripts

```bash
python src/cleaning.py
python src/split_data.py
```

Some short, multilingual, or ambiguous reviews may still require manual review. The dataset has not been fully verified for sentiment-label accuracy.
