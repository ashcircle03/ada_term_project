"""저장된 raw HTML을 재파싱해 DB를 업데이트하는 마이그레이션.

새 파서(Apollo __APOLLO_STATE__ 우선)로 condition/like_count/view_count/
created_at/gender/discount_pct/price_original 등을 채운다. HTTP 요청 없음.

파싱(BeautifulSoup)이 CPU 병목이라 멀티프로세스로 병렬화한다.
워커는 HTML을 읽고 파싱만 하며 DB는 건드리지 않는다 — 쓰기는 메인 프로세스
단일 writer가 전담해 SQLite WAL 쓰기 경합을 피한다.

사용법:
    python -m src.reparse [--limit N] [--workers K]
"""
import hashlib
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)


def _url_to_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _parse_one(task):
    """워커 함수: (product_id, seller_id) → ('ok', parsed) | ('skip'|'error', pid).

    DB를 절대 건드리지 않는다 (메인 프로세스만 씀).
    """
    from . import config
    from .parsers import parse_product_page

    pid, seller_id = task
    url = f"{config.BASE_URL}/product/{pid}/x"
    path = config.RAW_HTML_DIR / f"{_url_to_hash(url)}.html"
    if not path.exists():
        return ("skip", pid)
    try:
        html = path.read_text(encoding="utf-8", errors="replace")
        parsed = parse_product_page(html, url)
    except Exception:
        return ("error", pid)
    if not parsed:
        return ("skip", pid)
    # 파서가 셀러 링크를 못 찾으면 DB의 기존 seller_id 보존 (reparse는 기존 행 갱신)
    if not parsed.get("seller_id"):
        parsed["seller_id"] = seller_id
    return ("ok", parsed)


def run(limit: int = 0, workers: int = None) -> None:
    from . import config, db

    if workers is None:
        workers = os.cpu_count() or 4

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
    print(f"시작: total={total:,} workers={workers}")

    def consume(i, res):
        nonlocal updated, skipped, errors
        tag = res[0]
        if tag == "ok":
            try:
                db.upsert_listing(conn, res[1])
                updated += 1
            except Exception as e:
                logger.warning(f"DB 오류: {e}")
                errors += 1
        elif tag == "skip":
            skipped += 1
        else:
            errors += 1
        if i % 2000 == 0:
            conn.commit()
            print(f"  {i:,}/{total:,}  updated={updated:,} skipped={skipped:,} errors={errors:,}", flush=True)

    with db.get_conn() as conn:
        if workers <= 1:
            # 직렬 폴백
            for i, (pid, seller_id) in enumerate(products, 1):
                consume(i, _parse_one((pid, seller_id)))
        else:
            # 워커는 DB를 안 쓰므로 conn 핸들이 fork돼도 무해
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for i, res in enumerate(ex.map(_parse_one, products, chunksize=200), 1):
                    consume(i, res)
        conn.commit()

    print(f"\n완료: total={total:,}  updated={updated:,}  skipped={skipped:,}  errors={errors:,}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    def _arg(flag, default):
        if flag in sys.argv:
            return int(sys.argv[sys.argv.index(flag) + 1])
        return default

    run(limit=_arg("--limit", 0), workers=_arg("--workers", None))
