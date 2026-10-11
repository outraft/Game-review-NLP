## Data Cleaning and Preparation

Before model training, we cleaned and prepared the Steam reviews dataset using Python.

### Data Cleaning

We started with 3,000 raw reviews.

- Removed empty and duplicate comments.
- Removed comments containing only numbers or punctuation.
- Removed selected meaningless comments.
- Checked positive and negative sentiment labels.
- Identified and separated reviews with conflicting labels.

After cleaning, 2,598 reviews remained.

### Label Distribution

| Sentiment | Reviews |
|---|---:|
| Positive (1) | 1,168 |
| Negative (0) | 1,430 |
| Total | 2,598 |

### Training and Validation Split

We divided the cleaned dataset into training and validation sets using an approximately 80/20 split.

We separated positive and negative reviews before splitting to keep a similar sentiment distribution in both datasets.

We used `random.seed(42)` to make the results reproducible.

| Dataset | Reviews |
|---|---:|
| Training | 2,078 |
| Validation | 520 |

### Data Preparation Files

- `src/cleaning.py` - Cleans reviews and checks conflicting labels.
- `src/split_data.py` - Splits the cleaned data into training and validation sets.
- `steam_reviews_raw.csv` - Original dataset.
- `steam_reviews_cleaned.csv` - Cleaned dataset.
- `conflicting_reviews.csv` - Reviews with conflicting labels.
- `train.csv` - Training dataset.
- `validation.csv` - Validation dataset.

### Run Data Preparation

```bash
python src/cleaning.py
python src/split_data.py
```

The prepared data contains sentiment labels. The model's required reason labels and column format should be checked before training.
