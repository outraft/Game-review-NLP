# Contribution Record — ERSİN EGE TEZGÜLER

- **Student number:** 2409011008
- **Project handle:** `outraft`

## Main contribution and scope

My primary contribution to this project was the **NLP design, implementation,
evaluation, and integration**. I developed the comment-classification workflow
so a review comment can be analyzed for sentiment and likely game-review
reasons.

The scraper, review data, and data-preparation scripts already existed in the
project. I **patched and integrated** those components to support the NLP
workflow; I did not create the data collection and preparation pipeline from
scratch. The existing raw data and preparation scripts were patched and
integrated; redundant legacy snapshots were excluded from the final layout.

## Project propositions

- Shift the project focus from restaurant reviews to **video-game reviews**.
- At prediction time, provide **only the comment**. The NLP should predict a
  sentiment (`1` positive, `0` negative) and zero or more reason categories.
- Keep reason categories separate by sentiment. The negative categories are
  optimization, FPS drops, bad community, hard game, connectivity issues,
  price, bugs/glitches, poor controls, poor soundtrack/audio, and low
  replayability. Positive categories are fun to play, multiplayer accessible,
  mod compatible, graphically pleasing, developer connection, depth, gameplay,
  story driven, responsive controls, good soundtrack/audio, and high
  replayability.
- Retain Steam vote scores as metadata, but do not use them as model weights:
  helpfulness votes do not establish sentiment-label confidence.
- Avoid manually annotating every review for reasons. Instead, generate
  provisional reason labels with a local zero-shot NLI model and clearly mark
  them as unverified pseudo-labels.

## NLP implementation and evaluation

- Built a binary sentiment classifier using word and character TF-IDF features
  with class-balanced logistic regression.
- Compared it with a majority-class baseline and a word-only TF-IDF baseline;
  tuned regularization using grouped cross-validation by game title.
- Added a zero-shot DeBERTa-v3-base NLI sentiment comparison against the same
  Steam recommendation labels, without fine-tuning on the review dataset.
- Added an AnyJev/Tacit-1.7B typed-choice benchmark as the optional JEV
  alternative named in the course instructions; this does not claim to use
  TypeSafe's original Jev API.
- Integrated local zero-shot NLI reason inference using
  `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, with per-category scores,
  sentiment-specific candidate lists, and saved model provenance.
- Added a smaller TF-IDF classifier that can learn to imitate the generated
  reason labels. Its reason F1 against pseudo-labels is **not** an independent
  measure of reason accuracy.
- Integrated the scraper, cleaning, splitting, benchmark, reason annotation,
  and classifier stages, and documented commands and limitations in the
  README.
- Added unit tests for cleaning behavior and reason-label parsing/validation;
  the test suite runs with `python -m unittest discover -s tests -v`.
- Documented the Steam source and third-party data-rights limitation in
  `data/DATA_LICENSE.md`; the project code license is not asserted over Steam
  review text.

## Data integration and preparation patches

The existing Steam scraper was extended rather than replaced. Its defaults
collect 12,000 review rows from six games, include review/game metadata, retain
the score fields as metadata, and enforce a 10,000-row minimum. Current CSVs
were organized into raw, processed, split, and audit locations.
The existing cleaning and splitting scripts were patched to use a consistent
schema and reproducible splits.

The processed dataset contains **10,431 reviews** after removing repeated text
and conflicting sentiment rows. Steam provides recommendation sentiment but
not the project's reason labels; the zero-shot NLP step therefore supplies
provisional reasons. A small human-reviewed sample is still needed before
reason counts can be presented as verified findings.

## Reported sentiment results

On the current 2,087-review validation set, the tuned word+character model
achieved **0.8639 accuracy** and **0.8632 macro-F1**. The word-only baseline
achieved **0.8448 accuracy** and **0.8437 macro-F1**; the majority baseline
achieved **0.5462 accuracy** and **0.3533 macro-F1**. Three-fold game-grouped
cross-validation macro-F1 was **0.8296** for the tuned model versus **0.8004**
for the word-only baseline.

On the same validation set, the DeBERTa zero-shot model scored 0.8385 accuracy
and 0.8314 macro-F1. AnyJev/Tacit-1.7B scored 0.8601 accuracy and 0.8591
macro-F1. These pretrained alternatives were evaluated without fine-tuning on
the review set.

These are preliminary project results on Steam's noisy recommendation labels.
The review-level validation split contains the same game titles as training;
the grouped cross-validation result is the more direct check on unseen titles.
