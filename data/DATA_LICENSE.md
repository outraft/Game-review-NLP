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

Relevant sources:

- [Steam User Reviews API documentation](https://partner.steamgames.com/doc/store/getreviews)
- [Steam Subscriber Agreement, including User Generated Content](https://store.steampowered.com/subscriber_agreement/)
- [Steam Web API Terms of Use](https://steamcommunity.com/dev/apiterms)

Access to a public endpoint should not be treated as permission to republish a
bulk copy. Before making a repository containing the review-text CSVs public,
the team must verify that redistribution is permitted by the applicable Steam
terms and the review authors' rights, and follow any instructor or
institutional requirements. If permission cannot be confirmed, remove the
review-text CSVs from the public release and provide the scraper and
reproducible preparation steps instead. Aggregate counts and project-authored
code are separate from the underlying review text.

The zero-shot reason labels are machine-generated pseudo-labels. The pretrained
DeBERTa model has its own MIT license and is identified/pinned in the README;
that license does not grant rights to the underlying Steam review text.
