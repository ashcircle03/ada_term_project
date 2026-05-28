"""discount_pct + price_original 만 빠르게 재채운다.

reparse와 달리 BeautifulSoup 풀 파싱을 생략하고 __APOLLO_STATE__ JSON을
직접 슬라이싱→파싱한다(행당 ~10배 빠름). discount_pct/price_original 두
컬럼만 UPDATE하므로 다른 필드(is_sold/condition 등)엔 영향이 없다.
HTTP 요청 없음. 멀티프로세스 병렬.

사용법:
    python -m src.refill_discount [--limit N] [--workers K]
"""
import hashlib
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

_OPEN = '<script id="__APOLLO_STATE__"'
# price_original은 할인율에서 유도되는 값뿐이라 보존할 다른 출처가 없음 → 직접 덮어씀.
_UPDATE_SQL = "UPDATE listing SET discount_pct = ?, price_original = ? WHERE product_id = ?"


def _extract_apollo_fast(html: str) -> dict | None:
    """__APOLLO_STATE__ JSON만 문자열 슬라이싱으로 추출.

    스크립트 내용엔 raw '<'가 없음을 확인했으므로 </script> 경계가 안전.
    실패 시 None.
    """
    i = html.find(_OPEN)
    if i < 0:
        return None
    start = html.find(">", i)
    if start < 0:
        return None
    end = html.find("</script>", start)
    if end < 0:
        return None
    try:
        return json.loads(html[start + 1:end])
    except Exception:
        return None


def _compute(pid: str):
    """워커: product_id → ('ok', pid, discount_pct, price_original) | ('skip'|'error', pid).

    DB를 건드리지 않는다.
    """
    from . import config
    from .parsers import _apollo_product

    url = f"{config.BASE_URL}/product/{pid}/x"
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    path = config.RAW_HTML_DIR / f"{h}.html"
    if not path.exists():
        return ("skip", pid)
    try:
        html = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ("error", pid)

    # file은 있지만 Apollo에 product가 없는 페이지(주로 Sold/삭제) → 할인 알 수 없음.
    # 깨끗한 기본값 (0, NULL)로 정리한다 (full 파서의 텍스트 휴리스틱 garbage 방지).
    apollo = _extract_apollo_fast(html)
    prod = _apollo_product(apollo, pid) if apollo else None
    if not prod or prod.get("price") is None:
        return ("ok", pid, 0, None)
    try:
        price = int(prod["price"])
        rate = int(prod.get("discount_rate") or 0)
    except (TypeError, ValueError):
        return ("ok", pid, 0, None)

    orig = round(price / (1 - rate / 100)) if 0 < rate < 100 else None
    return ("ok", pid, rate, orig)


def run(limit: int = 0, workers: int = None) -> None:
    from . import config, db

    if workers is None:
        workers = os.cpu_count() or 4

    with db.get_conn() as conn:
        db._migrate(conn)
        rows = conn.execute(
            "SELECT product_id FROM listing WHERE seller_id != '_pending_'"
        ).fetchall()

    pids = [r[0] for r in rows]
    if limit:
        pids = pids[:limit]

    total = len(pids)
    updated = skipped = errors = 0
    batch = []
    print(f"시작: total={total:,} workers={workers}")

    with db.get_conn() as conn:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, res in enumerate(ex.map(_compute, pids, chunksize=500), 1):
                if res[0] == "ok":
                    _, pid, rate, orig = res
                    batch.append((rate, orig, pid))
                    updated += 1
                elif res[0] == "skip":
                    skipped += 1
                else:
                    errors += 1

                if len(batch) >= 1000:
                    conn.executemany(_UPDATE_SQL, batch)
                    conn.commit()
                    batch.clear()

                if i % 5000 == 0:
                    print(f"  {i:,}/{total:,}  updated={updated:,} skipped={skipped:,} errors={errors:,}", flush=True)

        if batch:
            conn.executemany(_UPDATE_SQL, batch)
        conn.commit()

    print(f"\n완료: total={total:,}  updated={updated:,}  skipped={skipped:,}  errors={errors:,}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    def _arg(flag, default):
        if flag in sys.argv:
            return int(sys.argv[sys.argv.index(flag) + 1])
        return default

    run(limit=_arg("--limit", 0), workers=_arg("--workers", None))
