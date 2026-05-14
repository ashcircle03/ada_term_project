# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic research crawler for fruitsfamily.com (Korean vintage fashion marketplace). The crawler collects listing/seller/review data into a SQLite DB, which is then analyzed to validate three hypotheses about seller "signature" patterns and pricing.

## Setup

```bash
pip install requests beautifulsoup4 lxml
# For analysis only:
pip install pandas numpy scipy scikit-learn matplotlib xgboost pyarrow hdbscan scikit-posthocs
```

**Before first run:** Edit `src/config.py`:
1. Fill in `USER_AGENT` with a contact email
2. Replace `ANONYMIZATION_SALT` with a secret string (used for SHA256 username hashing)

## Commands

```bash
# Crawler (run from project root)
python -m src.main init                    # Initialize DB schema
python -m src.main seed                    # Discover listing IDs from category/brand pages
python -m src.main listings --limit 20     # Crawl listing detail pages (start small to verify parsers)
python -m src.main sellers --limit 50      # Crawl seller detail pages
python -m src.main reviews --limit 50      # Crawl seller review pages
python -m src.main stats                   # Show crawl progress counts

# Analysis (requires data in DB first)
python -m analysis.data_loader             # Check data availability / preview
python -m analysis.features               # Preview engineered features
python -m analysis.h1_clustering          # Run clustering (must run before H2/H3)
python -m analysis.h2_anova               # Price premium by cluster (needs H1 output)
python -m analysis.h3_prediction          # Price prediction comparison (needs H1 output)
python -m analysis.run_all                # Full pipeline
```

To reset and re-crawl from scratch: delete `data/fruitsfamily.db`, then `python -m src.main init`.

Logs go to `logs/crawler.log` (also mirrored to stdout).

## Architecture

### Crawler (`src/`)

**4-stage pipeline** — each stage is independently resumable via `crawl_state` table:

1. **`seed_categories`** — hits category search pages + brand pages to discover `product_id`s, inserts placeholder rows with `seller_id='_pending_'`
2. **`crawl_listings`** — fetches product detail pages for `_pending_` rows, fills in all fields via `upsert_listing`
3. **`crawl_sellers`** — for sellers discovered in listings but not yet fetched, gets seller metadata + additional listing cards
4. **`crawl_reviews`** — only for sellers with `total_sales >= min_sales`, fetches review pages

**Key design invariants:**
- `Fetcher` handles all HTTP: rate limiting (2–4s random delay), retry (3x with exponential backoff), 4xx = immediate abort, and raw HTML saved to `data/raw_html/` by URL hash for re-parsing without re-fetching
- `parsers.py` uses structural heuristics (URL patterns, relative position to `<h1>`, text proximity) — NOT CSS class names, which are auto-generated and unstable in the React SSR output. Parsers never raise exceptions; missing fields return `None`.
- `db.upsert_listing` uses `COALESCE` to never overwrite real values with `NULL`, and specifically handles the `_pending_` → real `seller_id` transition
- `crawl_state` table keys like `"category:MEN:26:done"` allow skipping already-processed seeds on resume

### Analysis (`analysis/`)

**Data flow:** `fruitsfamily.db` → `data_loader` (DataFrames + parquet cache) → `features` (engineered variables: `signature_text`, `consistency`, `matched_pairs`) → H1 clustering → H2/H3 statistical tests → `results/*.json` + `results/figures/*.png`

**Three hypotheses:**
- **H1** (`h1_clustering.py`): Seller signature clusters exist — TF-IDF on listing text + K-means
- **H2** (`h2_anova.py`): Signature clusters correlate with pricing — Kruskal-Wallis + matched-pair analysis
- **H3** (`h3_prediction.py`): Signature features improve price prediction — XGBoost model A vs B, paired t-test on RMSE

H2 and H3 depend on `seller_clusters.parquet` output from H1.

**Minimum data for analysis:** 50+ sellers with 3+ listings (H1), 500+ listings (H2), 1,000+ listings (H3).

### DB Schema

Five tables: `seller`, `listing`, `review`, `crawl_state`, `fetch_failure`. Full column definitions in `src/db.py` `SCHEMA` constant. `seller.username_hash` is a salted SHA256 of the raw username — the salt is only in `config.ANONYMIZATION_SALT`.
