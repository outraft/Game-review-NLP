# Game-review NLP

This project analyzes English-language Steam game reviews. Its input at
prediction time is **the comment text only**. It predicts:

1. **Sentiment:** `1` = positive, `0` = negative.
2. **Reason label(s):** zero or more labels from the sentiment-specific lists
   below.

Steam provides the positive/negative recommendation label, but it does not
provide our project-specific reason labels. Those reasons are generated with
local zero-shot NLP and are marked as provisional predictions, not ground truth.
The task is therefore framed as **multi-label review reason/aspect
classification conditioned on sentiment**, rather than sentiment analysis
alone; sentiment is also reported and used to restrict which reason labels can
be selected.

## Project layout

```text
src/
├── steam_scraper.py       # collect reviews from Steam's public App Reviews endpoint
├── cleaning.py            # validate, deduplicate, and audit review data
├── split_data.py          # create reproducible stratified splits
├── benchmark.py           # compare sentiment models and report metrics
├── benchmark_zero_shot.py # evaluate pretrained DeBERTa zero-shot sentiment
├── benchmark_anyjev.py   # evaluate a local AnyJev/Tacit typed-decision model
├── annotate_reasons.py    # infer reason labels with a local zero-shot NLI model
└── classifier.py          # train, evaluate, and predict sentiment/reasons

contributions/
├── CONTRIBUTORS.md        # team contribution index
├── ERSİN_EGE_TEZGULER_2409011008_outraft.md # individual contribution record
└── OMAR_ALTUHAIF_2309116431_omaraltuhaif.md # individual contribution record

tests/
├── test_classifier.py     # label parsing and reason-annotation checks
└── test_cleaning.py       # preprocessing behavior checks

root files:
├── requirements.txt          # scraper and core sentiment pipeline
├── requirements-zero-shot.txt # local DeBERTa reason model dependencies
├── requirements-jev.txt      # optional AnyJev/Tacit benchmark dependencies
└── LICENSE                   # GPL-3.0 for original code; see data license notice

data/
├── raw/                   # scraped Steam export
├── processed/             # cleaned review CSV
├── splits/                # sentiment train/validation CSVs
├── annotated/             # zero-shot reason predictions, summary, and splits
├── audit/                 # conflicting review text removed during cleaning
└── DATA_LICENSE.md        # data source and third-party rights notice

models/                    # locally trained model files (git-ignored)
```

The redundant legacy V1 CSV snapshots are excluded to keep the repository
focused on the current reproducible pipeline. The trained model file is not
committed; recreate it with the train commands below.

## Current dataset

The current scrape contains **12,000 rows**: 1,000 positive and 1,000 negative
review requests for each of six games:

| Game | Steam App ID |
| --- | ---: |
| Cyberpunk 2077 | 1091500 |
| Counter-Strike 2 | 730 |
| Grand Theft Auto V | 271590 |
| The Witcher 3: Wild Hunt | 292030 |
| Baldur's Gate 3 | 1086940 |
| ELDEN RING | 1245620 |

The API filters by review type, but a few returned Steam labels did not match
the requested type, so the raw labels are 5,997 negative and 6,003 positive.
Cleaning removed 628 rows whose identical text had conflicting sentiments and
941 repeated comment texts. The processed file has **10,431 reviews**: 5,699
negative and 4,732 positive. Empty comments removed: 0.

| Cleaned game | Total | Negative | Positive |
| --- | ---: | ---: | ---: |
| Cyberpunk 2077 | 1,784 | 973 | 811 |
| Counter-Strike 2 | 1,625 | 899 | 726 |
| Grand Theft Auto V | 1,639 | 926 | 713 |
| The Witcher 3: Wild Hunt | 1,802 | 993 | 809 |
| Baldur's Gate 3 | 1,822 | 959 | 863 |
| ELDEN RING | 1,759 | 949 | 810 |

The current stratified 80/20 split has **8,344 training** and **2,087
validation** reviews. It keeps all six game titles represented in both sets.
The train split contains 4,559 negative and 3,785 positive reviews; validation
contains 1,140 negative and 947 positive reviews. The additional game-grouped
cross-validation result below tests performance on titles held out from training.

## CSV schema and score fields

Raw and processed CSVs share this schema:

| Column | Meaning |
| --- | --- |
| `steam_review_id` | Steam review identifier |
| `app_id`, `game_title` | Game metadata |
| `comment` | Review text; the NLP input |
| `sentiment` | `1` positive / `0` negative, based on Steam's recommendation |
| `reason` | Reason label(s), separated by `|` after annotation |
| `created_at` | Review creation timestamp in UTC |
| `votes_up` | Steam helpfulness vote count |
| `weighted_vote_score` | Steam's weighted helpfulness score |

For `reason`, a **blank** value means not annotated. `none` means the
annotation process completed but no reason passed its cutoff. `reason_source`
distinguishes a `zero_shot_auto` prediction from a provided or human-verified
label. The annotated output also includes `reason_scores`, `reason_model`, and
`reason_threshold`.

`votes_up` and `weighted_vote_score` are retained as metadata, but are **not
used as model weights**. They measure helpfulness/community response, not
confidence that the sentiment label is correct. Weighting by them could bias
the model toward popular reviews.

## Code and data licensing

The root `LICENSE` provides GPL-3.0 terms for this project's original code.
Steam review text is third-party user-generated content and is **not** covered
by that code license or relicensed by this project. See
[`data/DATA_LICENSE.md`](data/DATA_LICENSE.md) for sources, terms to check, and
the public-redistribution warning. Public access to Steam's endpoint does not
itself grant permission to republish a bulk review dump. Confirm redistribution
rights with the instructor/Valve before publishing the CSVs. The pretrained
zero-shot model has its own MIT license, linked in the model section below.

## Sentiment model and results

The sentiment classifier uses:

- Word TF-IDF unigrams and bigrams to represent terms and short phrases.
- Character-within-word TF-IDF n-grams of lengths 3–5 (up to 300,000
  features) to help with informal spelling and word variants.
- Class-balanced logistic regression to account for the cleaned set's class
  imbalance.

Both feature extractors lowercase and strip accents, use sublinear term
frequency, and ignore n-grams seen fewer than twice.

The word+character model's regularization parameter was selected from
`C = 0.3, 1, 3` using grouped cross-validation by Steam App ID on the training
split. The selected value was `C = 3`.

Macro-F1 is reported alongside accuracy because the cleaned classes are
moderately imbalanced (5,699 negative versus 4,732 positive); macro-F1 gives
both sentiment classes equal weight. Accuracy is retained for an intuitive
overall-correctness measure.

`python src/benchmark.py` compares the tuned model with the current word-only
logistic-regression baseline and a majority-class baseline. Results on this
dataset are:

| Model | Accuracy | Macro-F1 | Evaluation |
| --- | ---: | ---: | --- |
| Majority-class baseline | 0.5462 | 0.3533 | 2,087-review validation set |
| Word-only TF-IDF logistic regression | 0.8448 | 0.8437 | 2,087-review validation set |
| **Tuned word + character TF-IDF** | **0.8639** | **0.8632** | 2,087-review validation set |
| DeBERTa-v3-base zero-shot NLI (new method) | 0.8385 | 0.8314 | 2,087-review validation set |
| AnyJev Tacit-1.7B typed choice (new method) | 0.8601 | 0.8591 | 2,087-review validation set |
| Word-only model | — | 0.8004 | 3-fold App-ID-grouped CV on training data |
| **Tuned word + character model** | — | **0.8296** | 3-fold App-ID-grouped CV on training data |

On the review-level validation set, the tuned model improves macro-F1 by **1.95
percentage points** over the word-only baseline. In grouped cross-validation,
which holds out game titles in each fold, it improves macro-F1 by **2.92
points** over that baseline. The tuned model's validation confusion matrix
(rows = actual negative/positive, columns = predicted negative/positive) is:

```text
[[976, 164],
 [120, 827]]
```

The per-class validation scores were:

| Sentiment | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Negative | 0.8905 | 0.8561 | 0.8730 |
| Positive | 0.8345 | 0.8733 | 0.8535 |

Reproduce the benchmark:

```bash
python src/benchmark.py
```

The DeBERTa row is a new, pretrained zero-shot method, not a baseline. It uses
the same pinned DeBERTa-v3-base checkpoint described below, without fine-tuning
on these reviews. The model predicts the higher-scoring of a positive-review
and a negative-review hypothesis. Compare it with:

```bash
python src/benchmark_zero_shot.py --device auto
```

This zero-shot benchmark runs in the optional PyTorch environment. In model
terms, **DeBERTa-v3-base** is the architecture/size; `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`
is the specific pretrained checkpoint. Its zero-shot scores are not calibrated
probabilities. The TF-IDF models are the current in-domain baselines; if the
new method is changed, benchmark it against the same in-domain baselines on the
same held-out data.

The AnyJev row is the optional JEV-alternative method described in the course
email. It uses AnyJev's `Tacit-1.7B` typed-choice decision model without
fine-tuning on the review set. The choices are exactly `positive` and
`negative`; the validation labels still come from Steam recommendations. Its
winning-option scores are decision distributions, not calibrated probabilities.
The base architecture is **Qwen3-1.7B**; `morriszjm/Tacit-1.7B` is the specific
self-distilled checkpoint and `AnyJev` is the typed-decision library/API.

To reproduce the AnyJev benchmark, activate the Python 3.12 PyTorch/CUDA
environment, install the optional package, and run:

```bash
python -m pip install -r requirements-jev.txt
python src/benchmark_anyjev.py --batch-size 8
```

The pinned `morriszjm/Tacit-1.7B@1cb46fff0ffeefc4965455185074be5c62adccf7`
checkpoint is about 3.5 GB. The benchmark downloads it on first run and uses
the GPU when Transformers' automatic device mapping can use one. The project
pins AnyJev package version 0.3.0. AnyJev is a named alternative from the
professor's email; this project does not claim to implement TypeSafe's original
Jev API. The AnyJev package and Tacit checkpoint have separate Apache-2.0
licenses.

The review-level validation set includes the same game titles as training, so
it measures performance on new reviews of known titles, not exclusively on
unseen games. The grouped result addresses unseen-title generalization more
directly. Steam's recommendation is a noisy sentiment target: jokes, sarcasm,
mixed opinions, and changed recommendations may disagree with the text. These
are project results, not a guarantee of performance on every game or review
site.

## Reason categories

Reasons are restricted to the list corresponding to the review's sentiment.
A comment may receive multiple labels.

**Negative (`0`):**

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

**Positive (`1`):**

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

## Automatic reason analysis (zero-shot NLI)

The project uses a local **zero-shot natural-language-inference model** to
suggest reason labels without manually labeling all 10,431 reviews. For each
comment, it compares natural-language hypotheses for the possible reasons and
scores whether the text supports them. It considers only the reason list that
matches the review's existing Steam sentiment, then selects every label above
the cutoff. No review text is sent to a hosted inference API.

During batch annotation, the known Steam recommendation selects the positive
or negative candidate list. During single-comment prediction, the sentiment
model's predicted polarity selects that list instead. Within the selected list,
reasons are multi-label: more than one may be returned.

The model is
[`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0),
MIT-licensed, pinned to revision
`8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8`. The first run downloads the model
weights (roughly 1 GB of cached files). The model scores are **not calibrated
probabilities**, and its predictions are **pseudo-labels**, not human-verified
ground truth. A small, representative hand-reviewed sample is needed before
presenting reason counts as factual findings. Do not report agreement with
these pseudo-labels as reason-classification accuracy.

Reason annotation results at the current `0.7` cutoff:

| Result | Reviews |
| --- | ---: |
| Reviews with one or more selected reasons | 5,805 |
| Reviews with no reason above the cutoff (`none`) | 4,626 |
| Total selected reason assignments | 10,771 |

Because this is multi-label classification, one review may contribute multiple
assignments. Most frequent provisional labels:

| Sentiment | Reason | Count |
| --- | --- | ---: |
| Positive | gameplay | 1,974 |
| Negative | bugs/glitches | 1,326 |
| Negative | low replayability | 1,244 |
| Negative | poor controls | 1,224 |
| Negative | optimization | 1,052 |
| Positive | fun to play | 915 |
| Negative | bad community | 610 |
| Positive | story driven | 547 |

All category counts and mean selected scores are in
`data/annotated/reason_summary.csv`. Rare categories have very few predictions
and should not be interpreted strongly without validation.

### Install zero-shot dependencies

The core sentiment pipeline does not require PyTorch. For local zero-shot
reason labeling, use Python 3.12 (tested) and install the optional dependencies.
In Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-zero-shot.txt
```

On macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-zero-shot.txt
```

The default PyTorch install can run on CPU. For an NVIDIA GPU, install a
CUDA-enabled PyTorch build compatible with its driver. The development machine
used a GeForce RTX 4060 Laptop GPU; its CUDA 13.0 install command was:

```bash
python -m pip install --force-reinstall "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130
```

The annotator defaults to `--device auto`, which uses a CUDA GPU if PyTorch can
access one and otherwise uses CPU. CPU inference over the full dataset will be
slower.

### Run reason annotation

First run a small smoke test, then process the full cleaned dataset:

```bash
python src/annotate_reasons.py --limit 20 --output data/annotated/smoke_test.csv
python src/annotate_reasons.py
```

The full output files are:

- `data/annotated/steam_reviews_reason_predictions.csv`: original review
  columns plus `reason`, per-category `reason_scores`, `reason_source`,
  `reason_model`, and `reason_threshold`.
- `data/annotated/reason_summary.csv`: counts and mean scores by sentiment and
  reason.

The cutoff defaults to `0.7`; adjust it with `--threshold 0.6` after reviewing
sample predictions. `--batch-size` controls how many reviews are checkpointed
at a time; interrupted runs can resume from the same output. Changing the model
revision or threshold requires `--overwrite`. `--limit` is for smoke tests, not
for creating the final analytics. With the smoke-test command above, its files
are `data/annotated/smoke_test.csv` and
`data/annotated/smoke_test_summary.csv`; the full run writes to the separate
default output and summary paths.

## Train and use the classifier

### Sentiment-only model

The regular `data/splits/` files have blank reasons. Training on them trains
sentiment only; it intentionally does not treat unannotated reasons as negative
examples.

```bash
python -m pip install -r requirements.txt
python src/classifier.py train --data data/splits/train.csv --model models/sentiment_only.joblib
python src/classifier.py evaluate --data data/splits/validation.csv --model models/sentiment_only.joblib
python src/classifier.py predict --text "Your game review text here" --model models/sentiment_only.joblib
```

### Student model for sentiment and reasons

After generating zero-shot pseudo-labels, make dedicated train/validation
splits, then train the TF-IDF model to imitate them:

```bash
python src/split_data.py --input data/annotated/steam_reviews_reason_predictions.csv --output-dir data/annotated/splits
python src/classifier.py train --data data/annotated/splits/train.csv
python src/classifier.py evaluate --data data/annotated/splits/validation.csv
python src/classifier.py predict --text "Your game review text here"
```

This student model is smaller and faster to run than the zero-shot teacher.
Its reason F1 against a held-out set of **pseudo-labels** only measures how
well it imitates that teacher. It does not establish real-world reason accuracy.
In the current run it achieved micro-F1 `0.569` and macro-F1 `0.406` against
those pseudo-labels. Do not present those values as reason accuracy. Use a
human-reviewed reason sample for an independent estimate.

Prediction takes only a comment and prints JSON containing `sentiment` (0/1),
`sentiment_label`, confidence values, selected `reason` labels, and their
scores. The model is saved to `models/review_classifier.joblib` by default.
Model files are git-ignored; train the model after cloning or pulling the
project.

## Rebuild the dataset

The scraper uses Steam's public App Reviews endpoint, requests English recent
reviews with all purchase types, pauses 1.25 seconds between requests, and
retries HTTP 429, server, and network errors with backoff. Its default target is
1,000 reviews per sentiment for each of the six games (12,000 rows). It writes
a partial result but exits with an error if fewer than 10,000 are collected.

```bash
python -m pip install -r requirements.txt
python src/steam_scraper.py
python src/cleaning.py
python src/split_data.py
```

Common scraper options:

```bash
python src/steam_scraper.py --per-class 1200 --minimum-total 10000 --delay 1.5
python src/steam_scraper.py --apps 1091500 730 271590 --output data/raw/selected_games.csv
```

Use Steam App IDs for `--apps`. `--per-class` is the requested quota for each
sentiment **per game**; the quota can be lower if Steam has fewer matching
reviews. `--minimum-total` sets the required total before the command exits
successfully. Cleaning and split paths can be changed with `--input`, `--output`,
`--conflicts`, and `--output-dir`; see `--help` on each script.

Outputs:

1. `data/raw/steam_reviews.csv` — scraped export.
2. `data/processed/steam_reviews_cleaned.csv` — normalized, deduplicated
   reviews; sentiment labels are not manually changed.
3. `data/audit/conflicting_reviews.csv` — identical comment text found with
   conflicting sentiment labels.
4. `data/splits/train.csv` and `data/splits/validation.csv` — reproducible,
   stratified 80/20 sentiment split (seed 42).

The splitter preserves columns and stratifies on sentiment. It is a
review-level split, not a game-level split; use `benchmark.py` for grouped
cross-validation by game title.

## Automated checks

The lightweight unit tests exercise reason-label parsing, the difference
between blank (unknown) and `none` (annotated with no reason), sentiment/reason
compatibility, and cleaning behavior for empty, duplicate, and conflicting
comments. They use temporary fixtures and do not edit the project datasets.

```bash
python -m unittest discover -s tests -v
```

## References

- Steamworks, [User Reviews API](https://partner.steamgames.com/doc/store/getreviews).
- Laurer et al., [Building Efficient Universal Classifiers with Natural
  Language Inference](https://arxiv.org/abs/2312.17543), 2023. See also the
  [zero-shot model card](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0)
  for model details and license.
- Nokia Applied Research, [AnyJev](https://github.com/nokia-applied-research/AnyJev),
  its [technical report](https://arxiv.org/abs/2610.00831), and the
  [Tacit-1.7B model card](https://huggingface.co/morriszjm/Tacit-1.7B).
