## Data Quality and Limitations

The Steam review dataset was cleaned and prepared for sentiment classification. The cleaning process included removing duplicate reviews, filtering meaningless comments, identifying conflicting sentiment labels, and correcting three selected labels.

Additional manual review and label verification may further improve data quality, especially for short, multilingual, or ambiguous reviews.

### Final Dataset Summary

| Category | Number of Reviews |
|---|---:|
| Original dataset | 3,000 |
| Cleaned dataset | 2,554 |
| Positive reviews | 1,140 |
| Negative reviews | 1,414 |
| Training set (80%) | 2,043 |
| Validation set (20%) | 511 |

The dataset was split into 80% training data and 20% validation data. We also checked that the same reviews do not appear in both datasets.