# Contribution Record — Doruk Ünsal

- **Student number:** 2309011046
- **Project handle:** `Hadessd4`

## Main contribution and scope

My main role in this repository centered on the design, execution, evaluation, and integration of the NLP pipeline. I engineered a comment-classification system capable of parsing review text to determine both the author's sentiment and the specific factors driving their feedback.

While the data collection scraper, raw review dataset, and initial preprocessing scripts were pre-existing assets, I adapted and unified these legacy components to feed directly into the new NLP architecture. Rather than building the data pipeline from the ground up, I patched the existing framework and purged redundant historical snapshots to create a streamlined, integrated layout..

## Project propositions
-Pivoted the analytical domain from restaurant feedback to video-game reviews.
-Configured the inference engine to process only the raw comment text. From this single input, the NLP pipeline extracts a binary sentiment classification (1 for positive, 0 for negative) alongside any applicable reason tags.
-Segmented the underlying reason categories strictly by sentiment polarity:
-Negative drivers: poor optimization, frame rate (FPS) drops, toxic community, excessive difficulty, networking/connectivity problems, high price, bugs/glitches, clunky controls, weak audio/soundtrack, and lack of replay value.
-Positive drivers: general fun factor, strong multiplayer, mod support, impressive graphics, active developer engagement, mechanical depth, engaging gameplay, strong narrative/story, tight controls, excellent audio/soundtrack, and high replayability.
-Preserved Steam's "helpfulness" upvotes strictly as informational metadata. These were deliberately excluded from model weighting, as a review's helpfulness rating does not inherently reflect the confidence level of its sentiment.
-Bypassed exhaustive manual tagging by generating provisional reason labels via a local zero-shot NLI model. These are explicitly logged and treated as unverified machine-generated "pseudo-labels."

## NLP implementation and evaluation

-Developed a binary sentiment model utilizing a class-balanced logistic regression approach, fueled by both word- and character-level TF-IDF features.
-Benchmarked this classifier against a standard majority-class baseline and a purely word-level TF-IDF baseline. Regularization was optimized using grouped cross-validation, segmented by game title to prevent data leakage.
-Introduced a zero-shot sentiment baseline using a DeBERTa-v3-base NLI model. This was evaluated directly against native Steam recommendation tags without any task-specific fine-tuning on the dataset.
-Incorporated the AnyJev/Tacit-1.7B typed-choice evaluation to satisfy the optional JEV alternative requested in the course guidelines (expressly noting this operates independently of TypeSafe's proprietary Jev API).
-Deployed local zero-shot NLI for reason classification using the MoritzLaurer/deberta-v3-base-zeroshot-v2.0 model. This implementation handles distinct category scoring, applies sentiment-specific candidate filtering, and tracks model provenance.
-Trained a lightweight, supplementary TF-IDF classifier designed to mimic the NLI-generated reason labels. Its resulting F1 score strictly measures its ability to replicate the pseudo-labels, rather than representing objective, ground-truth accuracy.
-Successfully unified all pipeline stages—scraping, preprocessing, data splitting, benchmarking, zero-shot annotation, and final classification—while thoroughly documenting operational commands and system limitations in the README.
-Authored a robust unit testing suite covering text sanitation and reason-label validation, fully executable via python -m unittest discover -s tests -v.
-Addressed data provenance and legal scope by documenting Steam's third-party rights in data/DATA_LICENSE.md, explicitly clarifying that the repository's source code license does not extend to the scraped Steam review text.
