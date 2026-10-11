Game-review NLP classifier
This project classifies a game review from its text. At prediction time, the input is only the comment. The classifier predicts a sentiment and one or more reasons from the matching sentiment's category list.

Reason categories
Negative reviews (0)

optimization
FPS drops
bad community
hard game
connectivity issues
price
bugs/glitches
poor controls
poor soundtrack/audio
low replayability
Positive reviews (1)

fun to play
multiplayer accessible
mod compatible
graphically pleasing
developer connection
depth
gameplay
story driven
responsive controls
good soundtrack/audio
high replayability
A review may receive multiple reasons, but every selected reason comes only from the list matching its predicted sentiment. No other categories are included.

How the model works
The baseline uses TF-IDF text features with logistic-regression classifiers. It predicts sentiment (0 = negative, 1 = positive), then selects zero or more reasons from that sentiment's list. The CLI returns JSON fields for the comment, numeric sentiment, sentiment label, and selected reason labels. Reasons are category labels, not a generated free-text explanation.

Install
Requires Python 3.10 or later.

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
Training data
Training requires labeled examples so the model can learn from comments and their sentiment/reason labels. No data is included in this repository. When your training data is ready, provide a UTF-8 CSV with these columns:

comment,sentiment,reason
sentiment accepts 0/negative or 1/positive. Put one or more reason labels in reason, separated by |. Leave it blank or use none if no listed reason applies. Reasons must match the list above and must be compatible with the row's sentiment.

Train, predict, and evaluate
Train the model:

python src/classifier.py train --data path/to/training.csv
The model is saved to models/review_classifier.joblib by default. Predict from a comment only:

python src/classifier.py predict --text "Your game review text goes here."
The optional --threshold argument changes the cutoff for selecting reasons (default 0.5). Evaluate with a separate held-out CSV in the same format:

python src/classifier.py evaluate --data path/to/held_out.csv
Keep training and held-out comments separate. The model needs examples from both sentiment classes and examples for each reason you want it to learn.

About
No description, website, or topics provided.
Resources
Readme
GPL-3.0 license
Activity
Stars
0 stars
Watchers
0 watching
Forks
0 forks
Releases
No releases published
Create a new release
Packages
No packages published
Publish your first package
Contributors
1
 (1)
@outraft
outraftErsin
Languages
Python
100%
Suggested workflows
Based on your tech stack

Python Package using Anaconda logo
Python Package using Anaconda
Create and test a Python package on multiple Python versions using Anaconda for package management.
By GitHub Actions
SLSA Generic generator logo
SLSA Generic generator
Generate SLSA3 provenance for your existing release workflows
By Open Source Security Foundation (OpenSSF)
Publish Python Package logo
Publish Python Package
Publish a Python Package to PyPI on release.
By GitHub Actions
More workflows
Footer
© 2026 GitH

## Data Cleaning and Preparation

The Steam review dataset was cleaned and prepared for sentiment classification.

### Data Cleaning

The original dataset contained 3,000 reviews. The cleaning process included:

- Removing empty and duplicate reviews.
- Removing numeric-only and punctuation-only comments.
- Filtering selected meaningless or random-text reviews.
- Identifying and removing reviews with conflicting sentiment labels.
- Manually correcting three selected sentiment labels.

After cleaning, the final dataset contains **2,563 reviews**.

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