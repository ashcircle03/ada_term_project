# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.11 research crawler and analysis project for FruitsFamily data.
Core crawler code lives in `src/`: `main.py` exposes the CLI, `crawler.py` owns crawl stages, `parsers.py` extracts SSR/Apollo data, `db.py` manages SQLite, and `config.py` centralizes paths, crawl limits, URLs, and rate limits. Analysis helpers live in `analysis/`, while paired notebook scripts and notebooks live in `notebooks/` (`00_eda.py` / `00_eda.ipynb`, etc.). Runtime data is under `data/`, logs under `logs/`, generated outputs under `results/`, and reference PDFs/docs under `sample/`.

## Build, Test, and Development Commands

Set up the local environment:

```bash
conda activate ./.conda
pip install -r requirements.txt
python -m ipykernel install --user --name ada --display-name "ada (Python 3.11)"
```

Common crawler commands:

```bash
python -m src.main init
python -m src.main stats
python -m src.main listings --limit 100
python -m src.main sellers --limit 50
python -m src.reparse --limit 5000
python -m analysis.build_features
python -m analysis.nbmake notebooks/00_eda.py
```

Prefer staged crawl runs over `python -m src.main full` so parser behavior can be validated on small batches first.

## Coding Style & Naming Conventions

Use 4-space indentation, standard-library imports before third-party imports, then local imports. Keep module names lowercase with underscores. Use explicit function names such as `crawl_wishlists`, `build_listing_features`, or `parse_wishlist_page`. Keep configuration constants in `src/config.py` as `UPPER_SNAKE_CASE`. Existing comments and docstrings include Korean project notes; preserve that style when editing nearby code.

## Testing Guidelines

There is no formal test suite in this repository. Validate crawler changes with small limits (`listings --limit 20`, `sellers --limit 10`) and verify `python -m src.main stats`. For parser or schema changes, run `python -m src.reparse --limit 100` against cached HTML before network crawls. For analysis changes, rerun `python -m analysis.build_features` and the relevant notebook/script pair.

## Commit & Pull Request Guidelines

Git history uses concise Conventional Commit-style prefixes, for example `fix:`, `perf:`, `docs:`, and `analysis:`. Keep messages specific: `fix: preserve seller_id during reparse`. Pull requests should describe the data or behavior changed, include commands run for validation, and attach updated figures or notebook outputs when analysis results change.

## Security & Configuration Tips

Do not commit secrets, private keys, cloud hostnames, or fresh raw crawl artifacts. Keep crawler identity and rate limits explicit in `src/config.py`; do not disguise the bot as an ordinary browser. Rotate `ANONYMIZATION_SALT` before real anonymized runs.
