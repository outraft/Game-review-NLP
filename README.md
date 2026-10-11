# Game-review NLP classifier

This project classifies a game review from its text. At prediction time, the input is **only the comment**. The classifier predicts a sentiment and one or more reasons from the matching sentiment's category list.

## Reason categories

**Negative reviews (`0`)**

- optimization
- FPS drops
- bad community
- hard game
- connectivity issues
- price
- bugs/glitches
- poor controls
- poor soundtrack/audio
- low replayability

**Positive reviews (`1`)**

- fun to play
- multiplayer accessible
- mod compatible
- graphically pleasing
- developer connection
- depth
- gameplay
- story driven
- responsive controls
- good soundtrack/audio
- high replayability

A review may receive multiple reasons, but every selected reason comes only from the list matching its predicted sentiment. No other categories are included.

## How the model works

The baseline uses TF-IDF text features with logistic-regression classifiers. It predicts sentiment (`0` = negative, `1` = positive), then selects zero or more reasons from that sentiment's list. The CLI returns JSON fields for the comment, numeric sentiment, sentiment label, and selected `reason` labels. Reasons are category labels, not a generated free-text explanation.

## Install

Requires Python 3.10 or later.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Training data

The repository includes cleaned Steam review data and separate training and validation CSV files. The current prepared files use these columns:

```csv
comment,label,positive or negative,votes_up,weighted_vote_score
```

Here, `label` contains the sentiment label (`0` = negative, `1` = positive). **These prepared files are not yet in the classifier's required training format.**

The classifier requires a UTF-8 CSV with these columns:

```csv
comment,sentiment,reason
```

`sentiment` accepts `0`/`negative` or `1`/`positive`. Put one or more reason labels in `reason`, separated by `|`. Leave it blank or use `none` if no listed reason applies. Reasons must match the list above and must be compatible with the row's sentiment.

Before using the prepared CSV files with the classifier, the sentiment column must be mapped to `sentiment`, and the `reason` column must be prepared or left blank where appropriate. The team should check this format before training.

## Train, predict, and evaluate

Train the model using a CSV in the required `comment,sentiment,reason` format:

```bash
python src/classifier.py train --data path/to/training.csv
```

The model is saved to `models/review_classifier.joblib` by default. Predict from a comment only:

```bash
python src/classifier.py predict --text "Your game review text goes here."
```

The optional `--threshold` argument changes the cutoff for selecting reasons (default `0.5`). Evaluate with a separate held-out CSV in the same format:

```bash
python src/classifier.py evaluate --data path/to/held_out.csv
```

Keep training and held-out comments separate. The model needs examples from both sentiment classes and examples for each reason you want it to learn.

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
