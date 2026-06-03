# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic research crawler targeting [fruitsfamily.com](https://fruitsfamily.com), a Korean vintage C2C fashion platform. Data is used to empirically test hypotheses about seller style "signatures" and their effect on pricing. The project collects listings, seller metadata, and reviews into a SQLite DB, then exports parquet caches for analysis.

## Environment Setup

```bash
# Python 3.11 via .conda (already created at ./.conda)
conda activate ./.conda
pip install -r requirements.txt

# If using Jupyter:
python -m ipykernel install --user --name ada --display-name "ada (Python 3.11)"
```

On macOS, XGBoost requires OpenMP: `brew install libomp` before `pip install`.

## Common Commands

```bash
# Check crawl progress
python -m src.main stats

# Recommended crawl workflow (run each step, verify before next)
python -m src.main init                        # create/migrate the SQLite DB
python -m src.main seed                        # discover listing IDs from categories + brands
python -m src.main listings --limit 100        # small batch first to validate parser
python -m src.main listings --limit 5000       # full run
python -m src.main sellers --limit 500
python -m src.main reviews --limit 200 --min-sales 5
python -m src.main wishlists --limit 1100      # seller public wishlists (1 page = ~40 items each)

# Fill Apollo fields (condition/like_count/view_count) via HTTP for listings that lack them
python -m src.main fill --limit 30000

# Parallel fill across N workers (WAL mode lets writers run concurrently).
# Launch one process per shard; rows are partitioned by rowid % n_shards.
python -m src.main fill --n-shards 4 --shard-id 0   # worker 0
python -m src.main fill --n-shards 4 --shard-id 1   # worker 1  ... etc

# Backfill Apollo fields from cached HTML (no HTTP) after a parser change
python -m src.reparse --limit 5000

# Backfill view_count from cached seller pages (no HTTP, works offline)
python -m src.backfill_view_count

# Seed new product_ids discovered in wishlists into listing table (no HTTP).
# Run before launching a large remote crawl (Oracle Cloud etc.) to expand the universe.
python -m src.seed_from_wishlist
```

`python -m src.main full` runs `seed → listings → sellers` end-to-end, but the staged workflow above is preferred so the parser can be validated on a small batch first.

## Architecture

### Crawl Pipeline (stages)

```
seed_categories → crawl_listings → crawl_sellers → crawl_reviews → crawl_wishlists
```

Each stage is independently resumable — it queries the DB for unprocessed records and picks up where it left off. No in-memory state; every record is committed immediately after fetch. Resumability checkpoints live in the `crawl_state` key-value table.

**Stage 1 — Seed** (`crawler.py:seed_categories`): Scrapes category search pages + brand pages to collect `product_id` seeds. Inserts rows with `seller_id='_pending_'` as placeholders. `crawl_state` stores per-source done flags (e.g. `category:MEN:26:done = "1"`) so re-running `seed` skips processed sources.

**Stage 2 — Listings** (`crawler.py:crawl_listings`): Fetches each `_pending_` product detail page. Parses price, brand, size, sold status, and seller ID. Upserts over the placeholder row.

**Stage 3 — Sellers** (`crawler.py:crawl_sellers`): Fetches seller profile pages for any seller ID found in listings but not yet detailed in the `seller` table. Also ingests additional listing cards found on seller pages (without Apollo fields — those need `fill`).

**Stage 4 — Reviews** (`crawler.py:crawl_reviews`): Fetches review pages for sellers with `total_sales >= min_sales`.

**Stage 5 — Wishlists** (`crawler.py:crawl_wishlists`): Fetches `/seller/{sid}/x/like` for each seller. Extracts the public wishlist (max 40 items per first page). Wished products are usually from *other* sellers — stored as `(owner_seller_id, product_id, rank)` triples in the `wishlist` table. Uses `crawl_state` key `wishlist:{seller_id}:done` for resumability; empty wishlists are still marked done.

**Fill** (`crawler.py:fill_listing_details`): Not a sequential stage — a backfill pass that re-fetches product pages missing Apollo fields. Supports sharding via `--shard-id`/`--n-shards` (partitions by `rowid % n_shards`) so multiple workers can run in parallel against the WAL-mode DB.

### Parser Architecture (`parsers.py`)

Fruitsfamily is a React SPA with SSR. Each page has two data sources parsed in priority order:

1. **`__APOLLO_STATE__`** JSON (script tag `id="__APOLLO_STATE__"`) — structured GraphQL cache; most reliable. Contains `condition`, `like_count`, `view_count`, `createdAt`, etc.
2. **Text heuristics** (BeautifulSoup) — fallback when Apollo state is missing a field.

Key subtlety: Apollo state uses **numeric IDs** (`ProductNotMine:8867516`) as keys, but the URL uses **shortcodes** (`/product/5a27w/`). `_apollo_product()` resolves this via `ROOT_QUERY.seeProductResponse.__ref`. Seller IDs are always taken from the HTML `/seller/` link (not Apollo) to stay consistent with the shortcode-based DB schema.

### Database (`db.py`)

SQLite at `data/fruitsfamily.db`, opened in **WAL journal mode** (`db.py` `_migrate()`) so parallel `fill` workers can write concurrently. Tables: `seller`, `listing`, `review`, `wishlist` (seller's public liked items, with rank), `crawl_state` (checkpoint key-value store), `fetch_failure`.

**Apollo numeric ↔ shortcode mapping**: Apollo state uses numeric product IDs (`ProductNotMine:8867516`), but the DB stores URL shortcodes (`5a27w`). For *list contexts* (seller pages, wishlist pages), the order of `ROOT_QUERY.searchProducts(...)` / `seeUserLikes(...)` matches the order of `<a href="/product/{shortcode}">` cards in the HTML — pair them by index, skipping when lengths disagree. This is how `backfill_view_count.py` and `parse_wishlist_page` resolve shortcodes.

**`view_count` quirk**: Product page Apollo SSR includes `view_count` inconsistently (~20% of pages). Seller page Apollo SSR includes `view_count` for **every** product in the seller's listing. Therefore `view_count` is reliably backfilled via cached seller HTML (`src/backfill_view_count.py`), not via product page reparse.

`upsert_listing` has explicit COALESCE logic so NULL values from card-level scrapes never overwrite real data from detailed product pages. The `condition` column (NEW/GOOD_CONDITION/LIGHTLY_WORN/WORN), `like_count`, `view_count`, `created_at`, and `gender` are Apollo-only fields added via migration in `_migrate()`.

### Data Flow for Analysis

Raw HTML is saved to `data/raw_html/` (keyed by SHA256 of URL). If the parser is updated, run `python -m src.reparse` to re-extract fields from cached HTML without hitting the network.

Parquet caches at `data/cache/` are produced separately for analysis notebooks (listings, sellers, reviews, seller_clusters). Both `data/raw_html/` and `data/cache/` are gitignored; only `data/fruitsfamily.db` is committed (to preserve crawl state).

## Key Configuration (`config.py`)

All crawl limits, URL patterns, seed categories/brands, and rate-limit parameters are centralized here. Change `SEARCH_SORT`, `SEED_CATEGORIES`, or `SEED_BRANDS` to adjust scope.

- `SEARCH_SORT = "RECENT"` — deliberately *not* `POPULAR`: popular sort floods every category with the same hot brands (Ignota etc.), collapsing brand diversity and breaking seller-signature clustering with brand bias.
- `ANONYMIZATION_SALT` should be rotated before a real run — seller usernames are SHA256-hashed with this salt before DB storage.
- Rate limiting: 1–2 second random delay between requests (`REQUEST_DELAY_MIN`/`MAX`), exponential backoff on 5xx, immediate give-up on 4xx. The `USER_AGENT` is an explicit academic-research identifier (not a browser disguise) so the site operator can choose to allow or block it.

## Research Context

Setting hypotheses to conduct EDA and derive valuable insights