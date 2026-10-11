## Data Cleaning and Preparation

The Steam review dataset was cleaned and prepared for sentiment classification.

### Data Cleaning

The original dataset contained 3,000 reviews. The cleaning process included:

- Removing empty and duplicate reviews.
- Removing numeric-only and punctuation-only comments.
- Filtering selected meaningless or random-text reviews.
- Identifying reviews with conflicting sentiment labels.
- Removing reviews with conflicting labels.

After cleaning, the final dataset contains **2,580 reviews**.

### Dataset Distribution

| Category | Number of Reviews |
|---|---:|
| Original dataset | 3,000 |
| Cleaned dataset | 2,580 |
| Positive reviews | 1,155 |
| Negative reviews | 1,425 |
| Training set (80%) | 2,064 |
| Validation set (20%) | 516 |

### Files

- `src/cleaning.py` — Cleans the dataset and checks conflicting labels.
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

The dataset includes positive and negative sentiment labels. Some reviews are multilingual, and reason-category annotations have not yet been added.
