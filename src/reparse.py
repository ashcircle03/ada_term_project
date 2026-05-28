"""저장된 raw HTML을 재파싱해 DB를 업데이트하는 마이그레이션.

새 파서(Apollo __APOLLO_STATE__ 우선)로 condition/like_count/view_count/
created_at/gender 등을 채운다. HTTP 요청 없음.

사용법:
    python -m src.reparse [--limit N]
"""
import hashlib
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _url_to_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def run(limit: int = 0) -> None:
    from . import config, db
    from .parsers import parse_product_page

    raw_dir: Path = config.RAW_HTML_DIR

    with db.get_conn() as conn:
        db._migrate(conn)

        rows = conn.execute(
            "SELECT product_id, seller_id FROM listing WHERE seller_id != '_pending_'"
        ).fetchall()

    products = [(r[0], r[1]) for r in rows]
    if limit:
        products = products[:limit]

    total = len(products)
    updated = skipped = errors = 0

    with db.get_conn() as conn:
        for i, (pid, existing_seller_id) in enumerate(products, 1):
            url = f"{config.BASE_URL}/product/{pid}/x"
            path = raw_dir / f"{_url_to_hash(url)}.html"

            if not path.exists():
                skipped += 1
                continue

            html = path.read_text(encoding="utf-8", errors="replace")
            try:
                parsed = parse_product_page(html, url)
            except Exception as e:
                logger.warning(f"파싱 오류 {pid}: {e}")
                errors += 1
                continue

            if not parsed:
                skipped += 1
                continue

            # 파서가 셀러 링크를 못 찾으면 DB의 기존 seller_id 보존 (reparse는 기존 행 갱신)
            if not parsed.get("seller_id"):
                parsed["seller_id"] = existing_seller_id

            try:
                db.upsert_listing(conn, parsed)
                updated += 1
            except Exception as e:
                logger.warning(f"DB 오류 {pid}: {e}")
                errors += 1

            if i % 500 == 0:
                conn.commit()
                print(f"  {i}/{total}  updated={updated} skipped={skipped} errors={errors}")

        conn.commit()

    print(f"\n완료: total={total}  updated={updated}  skipped={skipped}  errors={errors}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    limit = 0
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        limit = int(sys.argv[idx + 1])
    run(limit=limit)
