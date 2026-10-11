# Data source, attribution, and usage notice

## Source and collection

The review text and Steam metadata in this directory were retrieved from the
Steam Store's public App Reviews endpoint on 2026-10-11. The query used the
English language filter and `recent` ordering for six game App IDs. The
scraper is `src/steam_scraper.py`; the exact fields and preparation steps are
documented in the root README.

## No third-party data license is granted by this project

The repository's root `LICENSE` applies to the project's original software
only. It does **not** relicense Steam user reviews, Steam names/marks, or other
third-party content included in the CSV files. Review text remains subject to
the rights of its authors and the terms that apply to Steam content. The
project makes no claim that it owns or can sublicense those reviews.

## Ethical and legal handling

The review-level CSVs (raw, cleaned, split, pseudo-labeled, and conflict-audit)
are generated locally and were removed from the working tree and the `master`
and `staging` branch histories because they contain third-party user-generated
text whose redistribution rights the project does not hold.
`data/annotated/reason_summary.csv` contains aggregate counts/scores only, not
review comments.

The repository had previously published pull requests. GitHub still advertises
`refs/pull/*/head` references to older commits, which cannot be force-updated by
the repository owner. A GitHub Support request is required to purge any cached
pull-request refs/views that retain the removed files. Until that request is
completed, do not claim the content has been erased from every GitHub copy.

This project does not claim ownership of Steam reviews or grant a license to
them. Local collection and analysis remain subject to the applicable Steam
terms and the review authors' rights. Do not redistribute review-text files
without appropriate permission. This file is a provenance/rights notice, not a
data license.

Relevant sources:

- [Steam User Reviews API documentation](https://partner.steamgames.com/doc/store/getreviews)
- [Steam Subscriber Agreement, including User Generated Content](https://store.steampowered.com/subscriber_agreement/)
- [Steam Web API Terms of Use](https://steamcommunity.com/dev/apiterms)

Access to a public endpoint should not be treated as permission to republish a
bulk copy. Verify the applicable Steam terms and review authors' rights before
any redistribution. Aggregate counts and project-authored code are separate
from the underlying review text.

The zero-shot reason labels are machine-generated pseudo-labels. The pretrained
DeBERTa model has its own MIT license and is identified/pinned in the README;
that license does not grant rights to the underlying Steam review text.
