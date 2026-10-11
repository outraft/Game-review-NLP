<<<<<<< HEAD
### Dataset Distribution
=======
## Dataset Distribution

The original dataset contained 3,000 Steam game reviews. After data cleaning and label checks, 2,563 reviews remained. The dataset was then divided into training and validation sets.
>>>>>>> a0e7d1b (Improve README dataset documentation)

| Category | Number of Reviews |
|---|---:|
| Original dataset | 3,000 |
| Cleaned dataset | 2,563 |
| Positive reviews | 1,146 |
| Negative reviews | 1,417 |
| Training set (80%) | 2,049 |
| Validation set (20%) | 514 |

## Project Files

- `src/cleaning.py` — Removes duplicate and selected meaningless reviews, checks conflicting sentiment labels, and corrects three selected labels.
- `src/split_data.py` — Divides the cleaned dataset into training and validation sets.
- `steam_reviews_cleaned.csv` — Contains the cleaned reviews.
- `conflicting_reviews.csv` — Lists review texts identified with conflicting sentiment labels.
- `train.csv` — Contains the training data.
- `validation.csv` — Contains the validation data.

## How to Run

Run the following commands from the project directory:

```bash
python src/cleaning.py
python src/split_data.py
```

<<<<<<< HEAD
Some short, multilingual, or ambiguous reviews may still require manual review. The dataset has not been fully verified for sentiment-label accuracy.
=======
## Notes

Some short, multilingual, or ambiguous reviews may still require manual inspection. Only selected sentiment labels have been corrected, so the entire dataset has not yet been manually verified for label accuracy.
>>>>>>> a0e7d1b (Improve README dataset documentation)
