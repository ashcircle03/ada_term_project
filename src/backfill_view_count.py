"""캐시된 셀러 페이지 HTML에서 view_count를 추출해 listing 테이블에 백필.

원인:
  - parse_product_page는 상품 페이지 Apollo SSR에서 view_count를 잡으려 하지만,
    상품 페이지 Apollo state에 view_count가 일관되게 포함되지 않음 (~20%만 있음).
  - 반면 셀러 페이지 Apollo state의 ProductNotMine 항목에는 view_count가 항상 있음.
  - parse_seller_page는 Apollo를 안 보고 HTML 텍스트 휴리스틱만 사용해서 놓쳤음.

매핑:
  - 셀러 페이지 Apollo의 ROOT_QUERY.searchProducts(...) 는 ProductNotMine __ref 리스트.
  - 같은 페이지 HTML의 /product/{shortcode} 카드 순서가 1:1 대응.
  - 길이 동일할 때만 안전하게 매핑 (불일치 셀러는 skip).

HTTP 요청 없음 — data/raw_html/만 읽음.
"""
import hashlib
import logging
import sys
from pathlib import Path
from bs4 import BeautifulSoup

from . import config, db
from .parsers import _extract_apollo, PRODUCT_URL_RE


logger = logging.getLogger(__name__)


def _seller_html_path(seller_id: str) -> Path:
    """crawl_sellers가 저장한 셀러 페이지 파일 경로."""
    url = f"{config.BASE_URL}/seller/{seller_id}/x"
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return config.RAW_HTML_DIR / f"{url_hash}.html"


def _extract_view_counts(html: str) -> dict[str, int]:
    """셀러 페이지 HTML → {shortcode: view_count} 매핑.

    매핑 실패(길이 불일치 등) 시 빈 dict 반환.
    """
    soup = BeautifulSoup(html, "lxml")
    apollo = _extract_apollo(soup)
    if not apollo:
        return {}

    # 1. searchProducts 리스트 (Apollo 순서)
    refs: list[str] = []
    for k, v in apollo.get("ROOT_QUERY", {}).items():
        if k.startswith("searchProducts") and isinstance(v, list):
            refs = [item.get("__ref") for item in v if isinstance(item, dict) and item.get("__ref")]
            break
    if not refs:
        return {}

    # 2. HTML 카드의 shortcode 순서 (dedup, parse_seller_page와 동일 로직)
    shortcodes: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=PRODUCT_URL_RE):
        m = PRODUCT_URL_RE.search(a.get("href", ""))
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            shortcodes.append(m.group(1))

    # 3. 길이 안 맞으면 안전을 위해 포기
    if len(refs) != len(shortcodes):
        return {}

    out: dict[str, int] = {}
    for ref, sc in zip(refs, shortcodes):
        prod = apollo.get(ref, {})
        vc = prod.get("view_count")
        if isinstance(vc, int):
            out[sc] = vc
    return out


def run() -> None:
    with db.get_conn() as conn:
        seller_ids = [r[0] for r in conn.execute("SELECT seller_id FROM seller").fetchall()]

    total_sellers = len(seller_ids)
    no_cache = 0
    map_failed = 0
    pairs: dict[str, int] = {}

    for sid in seller_ids:
        path = _seller_html_path(sid)
        if not path.exists():
            no_cache += 1
            continue

        html = path.read_text(encoding="utf-8", errors="replace")
        mapping = _extract_view_counts(html)
        if not mapping:
            map_failed += 1
            continue

        # 같은 shortcode가 여러 셀러 페이지에 나오면 마지막 값 사용
        # (값이 거의 같을 것이고 큰 의미 없음)
        pairs.update(mapping)

    print(f"셀러 수: {total_sellers}")
    print(f"  캐시 없음: {no_cache}")
    print(f"  매핑 실패 (길이 불일치): {map_failed}")
    print(f"수집된 (shortcode → view_count) 쌍: {len(pairs):,}")

    if not pairs:
        print("백필할 데이터 없음.")
        return

    # DB 업데이트 — view_count != new_value 인 행만 갱신
    updated = 0
    not_in_db = 0
    with db.get_conn() as conn:
        for sc, new_vc in pairs.items():
            row = conn.execute(
                "SELECT view_count FROM listing WHERE product_id = ?", (sc,)
            ).fetchone()
            if row is None:
                not_in_db += 1
                continue
            current = row["view_count"] or 0
            if current != new_vc:
                conn.execute(
                    "UPDATE listing SET view_count = ? WHERE product_id = ?",
                    (new_vc, sc),
                )
                updated += 1
        conn.commit()

    print(f"DB에 없는 shortcode: {not_in_db}")
    print(f"실제 UPDATE된 listing: {updated:,}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
