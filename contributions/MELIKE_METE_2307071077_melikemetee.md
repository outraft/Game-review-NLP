# Contribution Record — Melike Mete

- **Student number:** 2307071077
- **GitHub username:** melikemetee

## Main contribution and scope

My main role was data preparation and cleaning. I worked on the initial cleaning script to prepare game reviews for the NLP pipeline.

## Data cleaning

- Removed spaces at the beginning and end of review text.
- Compared comments in lowercase to detect repeated reviews.
- Removed empty reviews and duplicate comments.
- Removed number-only reviews, punctuation-only reviews, and selected meaningless text.
- Checked whether the same comment had different sentiment labels.
- Saved conflicting comments in a separate CSV file and excluded their rows from the cleaned dataset.
- Added manual label corrections for three selected comments.
- Updated the positive or negative field to match the sentiment label.
- Saved the cleaned reviews in a CSV file.

## Data checks

- Counted positive and negative reviews.
- Printed the number of removed reviews and corrected labels.
- Checked the final number of reviews after cleaning.

## Scope and limitations

The cleaning script uses simple rules and a manually prepared list of meaningless comments. It cannot detect every meaningless review automatically. My contribution focused on data preparation rather than NLP model development.