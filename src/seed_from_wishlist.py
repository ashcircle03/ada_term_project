"""위시리스트에서 발견된 신규 product_id를 listing 테이블에 시드.

wishlist 테이블에 있지만 listing 테이블에는 없는 product_id를 추출해
seller_id='_pending_' 상태로 INSERT OR IGNORE 한다. 이후 표준 파이프라인
(crawl_listings → crawl_sellers)이 자동으로 픽업한다.

HTTP 요청 없음. DB 쓰기만 수행.

사용법 (Oracle Cloud 등 별도 환경에서):
    1) python -m src.seed_from_wishlist             # _pending_ 시드 주입
    2) python -m src.main listings --limit 25000    # 매물 상세 크롤 (~18h)
    3) python -m src.main sellers --limit 2000      # 신규 셀러 메타 크롤
    4) python -m src.backfill_view_count            # 셀러 페이지 캐시 기반 view_count 백필
"""
import logging
from . import db


logger = logging.getLogger(__name__)


def run() -> None:
    with db.get_conn() as conn:
        # 1) 신규 product_id 추출 (wishlist에 있고 listing에 없는 것)
        rows = conn.execute(
            """
            SELECT DISTINCT w.product_id
            FROM wishlist w
            LEFT JOIN listing l ON l.product_id = w.product_id
            WHERE l.product_id IS NULL
            """
        ).fetchall()
        new_ids = [r["product_id"] for r in rows]

        n_wishlist = conn.execute("SELECT COUNT(*) FROM wishlist").fetchone()[0]
        n_listing = conn.execute("SELECT COUNT(*) FROM listing").fetchone()[0]
        print(f"현재 wishlist 행: {n_wishlist:,}")
        print(f"현재 listing 행:  {n_listing:,}")
        print(f"시드 대상 신규 product_id: {len(new_ids):,}")

        if not new_ids:
            print("시드할 매물 없음.")
            return

        # 2) _pending_ 행으로 일괄 삽입 — 표준 seed와 동일한 placeholder
        inserted = 0
        for pid in new_ids:
            cur = conn.execute(
                """INSERT OR IGNORE INTO listing
                   (product_id, seller_id, crawled_at)
                   VALUES (?, ?, ?)""",
                (pid, "_pending_", db.now()),
            )
            inserted += cur.rowcount
        conn.commit()

        print(f"INSERT OR IGNORE 결과: {inserted:,}건 실제 삽입")

        # 3) 안내
        pending = conn.execute(
            "SELECT COUNT(*) FROM listing WHERE seller_id = '_pending_'"
        ).fetchone()[0]
        print(f"\n현재 _pending_ 매물 총 수: {pending:,}")
        print(f"다음 명령으로 본격 크롤 가능:")
        print(f"  python -m src.main listings --limit {pending + 100}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
