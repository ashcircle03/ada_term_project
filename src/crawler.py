"""크롤링 오케스트레이션.

4단계로 분리:
  1) seed_categories  — 카테고리 페이지에서 매물 ID 시드 수집
  2) crawl_listings   — 각 매물 상세 페이지 → DB
  3) crawl_sellers    — 매물에서 발견된 셀러들의 상세 페이지 → DB
  4) crawl_reviews    — 활성 셀러의 리뷰 → DB

각 단계는 독립 실행 가능. 중단되면 같은 명령으로 재개.
"""
import logging
from . import config, db, parsers
from .fetcher import Fetcher


logger = logging.getLogger(__name__)


class Crawler:
    def __init__(self, fetcher: Fetcher = None):
        self.fetcher = fetcher or Fetcher()

    # ============================================================
    # 1단계: 시드 → 매물 ID 발견
    # ============================================================
    def seed_categories(self, conn):
        """카테고리 검색 + 브랜드 페이지 둘 다 돌며 매물 ID 수집.

        다양성을 위해 두 채널 병행:
        - 카테고리: 의류 주요 서브카테고리, 최신순 정렬
        - 브랜드: 후르츠 인기 브랜드 30개 페이지

        둘 다 INSERT OR IGNORE라 중복 매물은 자동 dedup.
        """
        seeded_cat = self._seed_from_categories(conn)
        seeded_brand = self._seed_from_brands(conn)
        logger.info(
            f"[SEED 완료] 카테고리 {seeded_cat}건 + 브랜드 {seeded_brand}건 = {seeded_cat + seeded_brand}건 (중복 제외 전)"
        )

    def _seed_from_categories(self, conn) -> int:
        seeded = 0
        for gender, sub_id, label in config.SEED_CATEGORIES:
            state_key = f"category:{gender}:{sub_id}:done"
            if db.get_state(conn, state_key) == "1":
                logger.info(f"[SKIP] 카테고리 이미 처리됨: {label}")
                continue

            url = config.URL_PATTERNS["search_category"].format(
                gender=gender, sub_id=sub_id, sort=config.SEARCH_SORT
            )
            logger.info(f"[SEED-CATEGORY] {label}: {url}")
            html = self.fetcher.get(url)
            if not html:
                db.log_failure(conn, url, err="seed page fetch failed")
                continue

            ids = parsers.parse_search_page(html)
            logger.info(f"  → {len(ids)}개 매물 ID 발견")

            for pid in ids:
                conn.execute(
                    """INSERT OR IGNORE INTO listing
                       (product_id, seller_id, subcategory_id, crawled_at)
                       VALUES (?, ?, ?, ?)""",
                    (pid, "_pending_", sub_id, db.now()),
                )
            db.set_state(conn, state_key, "1")
            seeded += len(ids)
        return seeded

    def _seed_from_brands(self, conn) -> int:
        """브랜드 페이지에서 매물 ID 수집. parse_search_page를 그대로 재사용 가능
        (브랜드 페이지도 /product/ 링크 패턴이 같음)."""
        from urllib.parse import quote
        seeded = 0
        for brand in config.SEED_BRANDS:
            state_key = f"brand:{brand}:done"
            if db.get_state(conn, state_key) == "1":
                logger.info(f"[SKIP] 브랜드 이미 처리됨: {brand}")
                continue

            url = config.URL_PATTERNS["brand"].format(brand=quote(brand))
            logger.info(f"[SEED-BRAND] {brand}: {url}")
            html = self.fetcher.get(url)
            if not html:
                db.log_failure(conn, url, err="brand page fetch failed")
                continue

            ids = parsers.parse_search_page(html)
            logger.info(f"  → {len(ids)}개 매물 ID 발견")

            for pid in ids:
                conn.execute(
                    """INSERT OR IGNORE INTO listing
                       (product_id, seller_id, crawled_at)
                       VALUES (?, ?, ?)""",
                    (pid, "_pending_", db.now()),
                )
            db.set_state(conn, state_key, "1")
            seeded += len(ids)
        return seeded

    # ============================================================
    # 2단계: 매물 상세 → 본문, 가격, 셀러 ID 채우기
    # ============================================================
    def crawl_listings(self, conn, limit: int = 1000):
        """seller_id가 _pending_인 매물부터 상세 페이지 크롤링."""
        rows = conn.execute(
            """SELECT product_id FROM listing
               WHERE seller_id = '_pending_'
               LIMIT ?""",
            (limit,),
        ).fetchall()

        logger.info(f"[LISTINGS] {len(rows)}개 매물 상세 크롤링 시작")
        success = fail = 0

        for row in rows:
            pid = row["product_id"]
            # 정확한 slug를 모르므로 placeholder로 요청
            # 후르츠 라우팅이 product_id만 맞으면 slug 무시하고 정상 응답함을 가정
            url = f"{config.BASE_URL}/product/{pid}/x"
            html = self.fetcher.get(url)
            if not html:
                db.log_failure(conn, url, err="product fetch failed")
                conn.commit()
                fail += 1
                continue

            try:
                data = parsers.parse_product_page(html, url)
            except Exception as e:
                # 파싱 예외는 격리 — 한 매물 실패가 batch 전체를 죽이지 않게
                logger.exception(f"파싱 예외 {pid}: {e}")
                db.log_failure(conn, url, err=f"parse exception: {type(e).__name__}: {e}")
                conn.commit()
                fail += 1
                continue

            if not data or not data.get("seller_id"):
                db.log_failure(conn, url, err="product parse incomplete (no seller_id)")
                conn.commit()
                fail += 1
                continue

            try:
                db.upsert_listing(conn, data)
                conn.commit()
                success += 1
            except Exception as e:
                logger.exception(f"DB 저장 예외 {pid}: {e}")
                db.log_failure(conn, url, err=f"db exception: {type(e).__name__}: {e}")
                conn.commit()
                fail += 1
                continue

            if success % 10 == 0:
                logger.info(f"  진행: {success} 성공 / {fail} 실패")

        logger.info(f"[LISTINGS 완료] {success} 성공 / {fail} 실패")

    # ============================================================
    # 3단계: 셀러 상세
    # ============================================================
    def crawl_sellers(self, conn, limit: int = 500):
        """매물에서 발견됐지만 메타 안 채워진 셀러들 처리."""
        seller_ids = db.list_sellers_to_crawl(conn, limit=limit)
        logger.info(f"[SELLERS] {len(seller_ids)}명 셀러 크롤링 시작")

        success = fail = 0
        for sid in seller_ids:
            # username을 모르는 상태 → 매물에서 발견한 임의 매물의 셀러 username으로 시도
            # 실제로는 listing 테이블에 seller_username을 같이 저장하도록 스키마 보강 필요
            # 지금은 username 자리에 'x' 넣고 요청 → 라우팅이 ID만으로 응답하는지 검증
            url = f"{config.BASE_URL}/seller/{sid}/x"
            html = self.fetcher.get(url)
            if not html:
                db.log_failure(conn, url, err="seller fetch failed")
                conn.commit()
                fail += 1
                continue

            try:
                data = parsers.parse_seller_page(html, url)
            except Exception as e:
                logger.exception(f"파싱 예외 seller {sid}: {e}")
                db.log_failure(conn, url, err=f"parse exception: {type(e).__name__}: {e}")
                conn.commit()
                fail += 1
                continue

            if not data:
                db.log_failure(conn, url, err="seller parse incomplete")
                conn.commit()
                fail += 1
                continue

            try:
                db.upsert_seller(conn, data)
                # 셀러 페이지에서 발견한 추가 매물도 저장
                for L in data.get("listings", []):
                    # 매물 상세는 이미 있을 수 있으므로 INSERT OR IGNORE 동작
                    conn.execute(
                        """INSERT OR IGNORE INTO listing
                           (product_id, seller_id, brand, price_final, price_original,
                            discount_pct, is_sold, crawled_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            L["product_id"], L["seller_id"], L.get("brand"),
                            L.get("price_final"), L.get("price_original"),
                            L.get("discount_pct", 0),
                            int(L.get("is_sold", False)),
                            db.now(),
                        ),
                    )
                conn.commit()
                success += 1
            except Exception as e:
                logger.exception(f"DB 저장 예외 seller {sid}: {e}")
                db.log_failure(conn, url, err=f"db exception: {type(e).__name__}: {e}")
                conn.commit()
                fail += 1
                continue

            if success % 10 == 0:
                logger.info(f"  진행: {success} 성공 / {fail} 실패")

            if success % 25 == 0:
                logger.info(f"  진행: {success} 성공 / {fail} 실패")

        logger.info(f"[SELLERS 완료] {success} 성공 / {fail} 실패")

    # ============================================================
    # 2b단계: Apollo 필드 채우기 (condition/like_count/view_count 없는 매물)
    # ============================================================
    def fill_listing_details(self, conn, limit: int = 5000, shard_id: int = 0, n_shards: int = 1):
        """condition IS NULL인 기존 매물의 상세 페이지를 개별 재크롤링.

        셀러 페이지 카드에서 발견된 매물은 condition/like_count/view_count가 없음.
        이 단계에서 각 product_id별 상세 페이지를 요청해 Apollo 필드를 채운다.
        중단 후 재개해도 이미 채워진 행은 스킵된다.
        """
        rows = conn.execute(
            """SELECT product_id, seller_id FROM listing
               WHERE seller_id != '_pending_' AND condition IS NULL
                 AND (rowid % ? = ?)
               LIMIT ?""",
            (n_shards, shard_id, limit),
        ).fetchall()

        logger.info(f"[FILL shard {shard_id}/{n_shards}] {len(rows)}개 매물 Apollo 필드 채우기 시작")
        success = fail = 0

        for row in rows:
            pid = row["product_id"]
            existing_seller_id = row["seller_id"]
            url = f"{config.BASE_URL}/product/{pid}/x"
            html = self.fetcher.get(url)
            if not html:
                db.log_failure(conn, url, err="fill fetch failed")
                conn.commit()
                fail += 1
                continue

            try:
                data = parsers.parse_product_page(html, url)
            except Exception as e:
                logger.exception(f"파싱 예외 {pid}: {e}")
                fail += 1
                conn.commit()
                continue

            if not data:
                fail += 1
                conn.commit()
                continue

            # seller_id가 파싱 안 됐으면 기존 DB 값 보존
            if not data.get("seller_id"):
                data["seller_id"] = existing_seller_id

            try:
                db.upsert_listing(conn, data)
                conn.commit()
                success += 1
            except Exception as e:
                logger.exception(f"DB 저장 예외 {pid}: {e}")
                fail += 1
                conn.commit()
                continue

            if success % 100 == 0:
                logger.info(f"  진행: {success} 완료 / {fail} 실패")

        logger.info(f"[FILL 완료] {success} 성공 / {fail} 실패")

    # ============================================================
    # 5단계: 위시리스트 (셀러가 공개 찜한 매물)
    # ============================================================
    def crawl_wishlists(self, conn, limit: int = 1000):
        """각 셀러의 공개 위시리스트 첫 페이지(최대 40건)를 수집.

        - 매핑 실패/빈 위시리스트는 done 표시만 하고 넘어감 (재요청 방지).
        - product_id는 다른 셀러의 매물일 수 있음 — listing 테이블엔 안 넣고
          wishlist 테이블에 페어만 저장 (필요시 후속 단계에서 발견된 신규
          product_id를 listing 시드로 끌어올 수 있음).
        """
        # crawl_state로 이미 처리한 셀러 제외
        rows = conn.execute(
            """SELECT s.seller_id FROM seller s
               LEFT JOIN crawl_state cs ON cs.key = 'wishlist:' || s.seller_id || ':done'
               WHERE cs.value IS NULL
               LIMIT ?""",
            (limit,),
        ).fetchall()

        logger.info(f"[WISHLISTS] {len(rows)}명 셀러 위시리스트 크롤링 시작")
        success = fail = empty = 0

        for row in rows:
            sid = row["seller_id"]
            url = f"{config.BASE_URL}/seller/{sid}/x/like"
            html = self.fetcher.get(url)
            if not html:
                db.log_failure(conn, url, err="wishlist fetch failed")
                conn.commit()
                fail += 1
                continue

            try:
                items = parsers.parse_wishlist_page(html, sid)
            except Exception as e:
                logger.exception(f"위시 파싱 예외 {sid}: {e}")
                db.log_failure(conn, url, err=f"wishlist parse: {type(e).__name__}: {e}")
                conn.commit()
                fail += 1
                continue

            if not items:
                # 빈 위시리스트 또는 매핑 실패 — done 표시 후 다음으로
                empty += 1
            else:
                for it in items:
                    conn.execute(
                        """INSERT OR IGNORE INTO wishlist
                           (owner_seller_id, product_id, rank, crawled_at)
                           VALUES (?, ?, ?, ?)""",
                        (it["owner_seller_id"], it["product_id"], it["rank"], db.now()),
                    )
                success += 1

            db.set_state(conn, f"wishlist:{sid}:done", "1")
            conn.commit()

            if (success + empty + fail) % 25 == 0:
                logger.info(f"  진행: {success} 항목수집 / {empty} 빈위시/실패매핑 / {fail} 요청실패")

        logger.info(f"[WISHLISTS 완료] {success} 항목수집 / {empty} 빈위시/실패매핑 / {fail} 요청실패")

    # ============================================================
    # 4단계: 리뷰
    # ============================================================
    def crawl_reviews(self, conn, min_sales: int = 5, limit: int = 200):
        """일정 거래수 이상의 셀러에 한해 리뷰 수집."""
        rows = conn.execute(
            """SELECT seller_id, username_hash FROM seller
               WHERE total_sales >= ? AND n_reviews = 0
               LIMIT ?""",
            (min_sales, limit),
        ).fetchall()

        logger.info(f"[REVIEWS] {len(rows)}명 셀러 리뷰 크롤링 시작")
        for row in rows:
            sid = row["seller_id"]
            url = f"{config.BASE_URL}/seller/{sid}/x/review"
            html = self.fetcher.get(url)
            if not html:
                continue
            reviews = parsers.parse_review_page(html, sid)
            for r in reviews:
                db.insert_review(conn, r)
            conn.execute(
                "UPDATE seller SET n_reviews = ? WHERE seller_id = ?",
                (len(reviews), sid),
            )
            conn.commit()
