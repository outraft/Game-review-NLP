# Review NLP classifier

A small, explainable baseline for classifying review comments into:

1. **Sentiment:** `0` (negative) or `1` (positive).
2. **Aspects:** any combination of `hygiene`, `taste`, `customer service`,
   `location`, `transportation options`, and `wait times`.

Aspect classification is multi-label: a comment may have zero, one, or several
aspects. The model uses TF-IDF text features with logistic-regression
classifiers. It does not include or generate a dataset.

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

## Training data (for when your labeled data is ready)

Training requires example comments paired with their correct sentiment and
reasoning labels, so the model can learn the relationship. The training command
will read a UTF-8 CSV with these columns:

```csv
comment,sentiment,reasoning
"A labeled training comment",1,"taste|wait times"
```

`sentiment` accepts `0`/`negative` or `1`/`positive`. `reasoning` is a
`|`-separated set of labels from the six categories above; leave it blank (or
use `none`) when no category applies. `aspects` is also accepted as the third
column name. No training data is included in this repo.

## Prediction input and output

At prediction time, provide **only the comment**—not its sentiment or reasoning:

```bash
python src/classifier.py predict --text "The meal was delicious, but service was slow."
```

The output contains the predicted sentiment and `reasoning`, a list containing
zero or more selected categories. These are category labels, not a generated
free-text explanation.

## Train and evaluate

Once you have labeled training data:

```bash
python src/classifier.py train --data path/to/training.csv
```

The model is saved to `models/review_classifier.joblib` by default. You can
choose another path with `--model`. Prediction prints JSON with the sentiment,
confidence, selected reasoning categories, and per-category scores. Adjust the
category cutoff with `--threshold 0.4` if needed.

Evaluate using a separate, held-out CSV with the same format:

```bash
python src/classifier.py evaluate --data path/to/held_out.csv
```

Keep training and held-out comments separate. The held-out set is for measuring
performance, not for training. A useful model needs labeled examples for both
sentiments and positive examples of each aspect; this baseline will warn when
an aspect has no positive training examples.
