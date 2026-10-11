#!/usr/bin/env python3
"""
steam_scraper.py
================

Fetch raw English user reviews for selected Steam games via the public
Steam App Reviews endpoint and export them as a CSV ready for the NLP
data-cleaning pipeline.

Endpoint:  https://store.steampowered.com/appreviews/<APP_ID>?json=1
Docs:      https://partner.steamgames.com/doc/store/getreviews

Output columns (`data/raw/steam_reviews.csv`):
    steam_review_id, app_id, game_title, comment, sentiment, reason,
    created_at, votes_up, weighted_vote_score

`reason` is intentionally blank: Steam does not supply the project's reason
categories. The score fields are retained as metadata, not training weights.

Positive and negative reviews are fetched separately (review_type=positive /
review_type=negative) so each game contributes an equal number of each,
giving a class-balanced dataset.

Usage:
    python src/steam_scraper.py                    # 12,000 review target
    python src/steam_scraper.py --per-class 500 --apps 1091500 730 271590
    python src/steam_scraper.py --output data/raw/custom.csv --delay 1.5
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

BASE_URL = "https://store.steampowered.com/appreviews/{app_id}"

# Six popular games provide variety and a 12,000-review default target.
DEFAULT_APPS: Dict[int, str] = {
    1091500: "Cyberpunk 2077",
    730: "Counter-Strike 2",
    271590: "Grand Theft Auto V",
    292030: "The Witcher 3: Wild Hunt",
    1086940: "Baldur's Gate 3",
    1245620: "ELDEN RING",
}

DEFAULT_PER_CLASS = 1000       # 1000 positive + 1000 negative per game
DEFAULT_MINIMUM_TOTAL = 10000
PAGE_SIZE = 100                # Steam's maximum num_per_page
DEFAULT_DELAY = 1.25           # seconds between requests (rate limiting)
MAX_RETRIES = 5                # retries on 429 / 5xx / network errors
REQUEST_TIMEOUT = 30           # seconds
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "steam_reviews.csv"

CSV_COLUMNS = [
    "steam_review_id",
    "app_id",
    "game_title",
    "comment",
    "sentiment",
    "reason",
    "created_at",
    "votes_up",
    "weighted_vote_score",
]

log = logging.getLogger("steam_scraper")


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #

def build_session() -> requests.Session:
    """Create a requests session with a descriptive User-Agent."""
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "game-review-nlp/1.0 (educational research project)"}
    )
    return session


def fetch_page(
    session: requests.Session,
    app_id: int,
    cursor: str,
    review_type: str,
    delay: float,
) -> dict:
    """
    Fetch one page of reviews.

    Retries with exponential backoff on HTTP 429, 5xx responses and network
    errors. Honours the Retry-After header when Steam sends one.
    """
    params = {
        "json": 1,
        "language": "english",
        "filter": "recent",          # 'recent' paginates reliably with cursor
        "review_type": review_type,  # 'positive' or 'negative'
        "purchase_type": "all",
        "num_per_page": PAGE_SIZE,
        "cursor": cursor,            # requests URL-encodes this for us
    }
    url = BASE_URL.format(app_id=app_id)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                wait = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else delay * (2 ** attempt) + random.uniform(0, 1)
                )
                log.warning(
                    "HTTP %s for app %s (attempt %d/%d) - sleeping %.1fs",
                    resp.status_code, app_id, attempt, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            if data.get("success") != 1:
                raise RuntimeError(f"Steam returned success={data.get('success')}")
            return data

        except (requests.RequestException, ValueError) as exc:
            wait = delay * (2 ** attempt) + random.uniform(0, 1)
            log.warning(
                "Request error for app %s (attempt %d/%d): %s - sleeping %.1fs",
                app_id, attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)

    raise RuntimeError(f"Giving up on app {app_id} after {MAX_RETRIES} attempts")


# --------------------------------------------------------------------------- #
# Pagination
# --------------------------------------------------------------------------- #

def iter_reviews(
    session: requests.Session,
    app_id: int,
    review_type: str,
    limit: int,
    delay: float,
) -> Iterator[dict]:
    """
    Yield up to `limit` unique, non-empty raw review objects for one app and
    one class,
    following Steam's cursor-based pagination.

    Stops when the limit is reached, a page comes back empty, or the cursor
    stops advancing (Steam repeats the last cursor at the end of the list).
    """
    cursor = "*"
    seen_ids: set = set()
    seen_cursors: set = set()

    while len(seen_ids) < limit:
        data = fetch_page(session, app_id, cursor, review_type, delay)
        reviews = data.get("reviews", [])
        next_cursor = data.get("cursor")

        if not reviews:
            log.info("  app %s [%s]: no more reviews", app_id, review_type)
            break

        for review in reviews:
            rid = review.get("recommendationid")
            # Skip duplicates and empty-text reviews *before* counting them,
            # so the quota is filled with usable reviews only.
            if rid in seen_ids or not (review.get("review") or "").strip():
                continue
            seen_ids.add(rid)
            yield review
            if len(seen_ids) >= limit:
                break

        if not next_cursor or next_cursor in seen_cursors:
            log.info("  app %s [%s]: cursor exhausted", app_id, review_type)
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor

        log.info("  app %s [%s]: %d/%d", app_id, review_type, len(seen_ids), limit)
        time.sleep(delay)  # rate limiting between pages


# --------------------------------------------------------------------------- #
# Transformation
# --------------------------------------------------------------------------- #

def to_row(review: dict, app_id: int) -> Optional[dict]:
    """
    Map a raw Steam review object to the output CSV schema.
    Returns None for reviews with empty text (useless for NLP).
    """
    text = (review.get("review") or "").strip()
    if not text:
        return None

    voted_up = bool(review.get("voted_up"))
    try:
        score = float(review.get("weighted_vote_score", 0) or 0)
    except (TypeError, ValueError):
        score = 0.0

    try:
        created_at = datetime.fromtimestamp(
            int(review.get("timestamp_created", 0) or 0), tz=timezone.utc
        ).isoformat()
    except (TypeError, ValueError, OSError):
        created_at = ""

    return {
        "steam_review_id": str(review.get("recommendationid", "")),
        "app_id": app_id,
        "game_title": DEFAULT_APPS.get(app_id, str(app_id)),
        "comment": text,
        "sentiment": 1 if voted_up else 0,
        # Steam supplies sentiment, but not our project-specific reason labels.
        # Empty means awaiting reason annotation; "none" will mean reviewed and no match.
        "reason": "",
        "created_at": created_at,
        "votes_up": int(review.get("votes_up", 0) or 0),
        "weighted_vote_score": score,
    }


def scrape_app(
    session: requests.Session,
    app_id: int,
    per_class: int,
    delay: float,
) -> List[dict]:
    """Collect a balanced set of positive and negative rows for one game."""
    name = DEFAULT_APPS.get(app_id, str(app_id))
    log.info("Scraping %s (%s)", name, app_id)

    rows_by_class: Dict[str, List[dict]] = {}
    seen_review_ids: set[str] = set()
    for review_type in ("positive", "negative"):
        rows = []
        for review in iter_reviews(session, app_id, review_type, per_class, delay):
            review_id = str(review.get("recommendationid", ""))
            if review_id and review_id in seen_review_ids:
                log.warning("%s: skipping duplicate review ID %s", name, review_id)
                continue
            row = to_row(review, app_id)
            if row is not None:
                rows.append(row)
                if review_id:
                    seen_review_ids.add(review_id)
        rows_by_class[review_type] = rows
        time.sleep(delay)

    # Trim to the smaller class so the game stays exactly balanced.
    n = min(len(rows_by_class["positive"]), len(rows_by_class["negative"]))
    if n < per_class:
        log.warning("%s: only %d reviews per class available", name, n)
    balanced = rows_by_class["positive"][:n] + rows_by_class["negative"][:n]
    log.info("%s: %d rows (%d pos / %d neg)", name, len(balanced), n, n)
    return balanced


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_csv(rows: List[dict], path: Path) -> None:
    """Write rows to CSV (UTF-8, all fields quoted so multiline text is safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape English Steam reviews to CSV.")
    p.add_argument("--apps", type=int, nargs="+", default=list(DEFAULT_APPS),
                   help="Steam app IDs (default: %(default)s)")
    p.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS,
                   help="Reviews per class (pos/neg) per game (default: %(default)s)")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                   help="Seconds to sleep between requests (default: %(default)s)")
    p.add_argument("--minimum-total", type=int, default=DEFAULT_MINIMUM_TOTAL,
                   help="Exit nonzero if fewer reviews are collected (default: %(default)s)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output CSV path (default: %(default)s)")
    p.add_argument("--seed", type=int, default=42,
                   help="Shuffle seed for the final dataset (default: %(default)s)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.per_class < 1 or args.minimum_total < 1 or args.delay < 0:
        log.error("--per-class and --minimum-total must be positive; --delay cannot be negative")
        return 2
    session = build_session()

    all_rows: List[dict] = []
    for app_id in args.apps:
        try:
            all_rows.extend(scrape_app(session, app_id, args.per_class, args.delay))
        except RuntimeError as exc:
            log.error("Skipping app %s: %s", app_id, exc)

    if not all_rows:
        log.error("No reviews collected - nothing written.")
        return 1

    # Shuffle so classes and games are interleaved for downstream splits.
    random.Random(args.seed).shuffle(all_rows)
    write_csv(all_rows, args.output)

    pos = sum(int(r["sentiment"]) for r in all_rows)
    log.info("Done: %d total | %d positive | %d negative",
             len(all_rows), pos, len(all_rows) - pos)
    if len(all_rows) < args.minimum_total:
        log.error("Collected %d reviews; minimum requested was %d",
                  len(all_rows), args.minimum_total)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
