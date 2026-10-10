# Game-review NLP classifier

This project classifies a game review from its text. At prediction time, the
input is **only the comment**. The classifier predicts a sentiment and one or
more reasons from the matching sentiment's category list.

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

A review may receive multiple reasons, but every selected reason comes only
from the list matching its predicted sentiment. No other categories are
included.

## How the model works

The baseline uses TF-IDF text features with logistic-regression classifiers.
It predicts sentiment (`0` = negative, `1` = positive), then selects zero or
more reasons from that sentiment's list. The CLI returns JSON fields for the
comment, numeric sentiment, sentiment label, and selected `reason` labels.
Reasons are category labels, not a generated free-text explanation.

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

Training requires labeled examples so the model can learn from comments and
their sentiment/reason labels. No data is included in this repository. When
your training data is ready, provide a UTF-8 CSV with these columns:

```csv
comment,sentiment,reason
```

`sentiment` accepts `0`/`negative` or `1`/`positive`. Put one or more reason
labels in `reason`, separated by `|`. Leave it blank or use `none` if no listed
reason applies. Reasons must match the list above and must be compatible with
the row's sentiment.

## Train, predict, and evaluate

Train the model:

```bash
python src/classifier.py train --data path/to/training.csv
```

The model is saved to `models/review_classifier.joblib` by default. Predict
from a comment only:

```bash
python src/classifier.py predict --text "Your game review text goes here."
```

The optional `--threshold` argument changes the cutoff for selecting reasons
(default `0.5`). Evaluate with a separate held-out CSV in the same format:

```bash
python src/classifier.py evaluate --data path/to/held_out.csv
```

Keep training and held-out comments separate. The model needs examples from
both sentiment classes and examples for each reason you want it to learn.
