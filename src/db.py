"""SQLite 기반 데이터 저장 + 체크포인팅.

설계 철학:
- 모든 raw 데이터는 즉시 DB에 INSERT. 메모리에 들고 있다가 죽으면 손실.
- INSERT OR IGNORE 사용. 같은 매물 다시 만나면 조용히 스킵.
- 'crawl_state' 테이블로 어디까지 처리했는지 추적 (재개용).
"""
import sqlite3
import hashlib
from datetime import datetime
from contextlib import contextmanager
from . import config


SCHEMA = """
-- 셀러 (1행 = 1명)
CREATE TABLE IF NOT EXISTS seller (
    seller_id TEXT PRIMARY KEY,
    username_hash TEXT NOT NULL,    -- 익명화된 식별자
    followers INTEGER,
    total_sales INTEGER,
    rating REAL,
    n_reviews INTEGER DEFAULT 0,
    is_vintage_shop INTEGER DEFAULT 0,
    crawled_at TEXT NOT NULL
);

-- 매물 (1행 = 1건)
CREATE TABLE IF NOT EXISTS listing (
    product_id TEXT PRIMARY KEY,
    seller_id TEXT NOT NULL,
    title TEXT,
    description TEXT,
    brand TEXT,
    category_l1 TEXT,           -- 남자/여자/라이프/굿즈
    category_l2 TEXT,           -- 아우터/상의/...
    subcategory_id INTEGER,     -- URL의 subcategoryIds 값
    size TEXT,
    price_original INTEGER,
    price_final INTEGER,
    discount_pct INTEGER DEFAULT 0,
    likes INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    n_photos INTEGER DEFAULT 0,
    is_sold INTEGER DEFAULT 0,
    posted_relative TEXT,
    condition TEXT,             -- NEW/GOOD_CONDITION/LIGHTLY_WORN/WORN (Apollo)
    like_count INTEGER DEFAULT 0,
    view_count INTEGER DEFAULT 0,
    created_at TEXT,            -- ISO 등록일시 (Apollo createdAt)
    gender TEXT,                -- MALE/FEMALE/UNISEX
    crawled_at TEXT NOT NULL,
    FOREIGN KEY (seller_id) REFERENCES seller(seller_id)
);

-- 셀러가 받은 구매자 리뷰
CREATE TABLE IF NOT EXISTS review (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    seller_id TEXT NOT NULL,
    review_text TEXT,
    review_rating INTEGER,
    crawled_at TEXT NOT NULL,
    FOREIGN KEY (seller_id) REFERENCES seller(seller_id)
);

-- 크롤링 상태 추적 — 재개용
-- key 예: 'category:MEN:26:page', 'seller:nni4:done'
CREATE TABLE IF NOT EXISTS crawl_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT NOT NULL
);

-- 실패 로그 — 끊긴 페이지 재시도 추적
CREATE TABLE IF NOT EXISTS fetch_failure (
    failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    status_code INTEGER,
    error_msg TEXT,
    failed_at TEXT NOT NULL
);

-- 자주 쓰는 쿼리 인덱스
CREATE INDEX IF NOT EXISTS idx_listing_seller ON listing(seller_id);
CREATE INDEX IF NOT EXISTS idx_listing_brand ON listing(brand);
CREATE INDEX IF NOT EXISTS idx_listing_sold ON listing(is_sold);
CREATE INDEX IF NOT EXISTS idx_review_seller ON review(seller_id);
"""


def init_db(db_path=None):
    """DB 파일과 스키마 생성. 이미 있으면 그대로 둠."""
    db_path = db_path or config.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    _migrate(conn)
    return conn


def _migrate(conn):
    """기존 DB에 새 컬럼 추가 (없는 경우만)."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(listing)")}
    new_cols = [
        ("condition",   "TEXT"),
        ("like_count",  "INTEGER DEFAULT 0"),
        ("view_count",  "INTEGER DEFAULT 0"),
        ("created_at",  "TEXT"),
        ("gender",      "TEXT"),
    ]
    for col, definition in new_cols:
        if col not in existing:
            conn.execute(f"ALTER TABLE listing ADD COLUMN {col} {definition}")
    conn.commit()


@contextmanager
def get_conn(db_path=None):
    """with 블록으로 안전하게 커넥션 사용."""
    db_path = db_path or config.DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def hash_username(username: str) -> str:
    """익명화. 솔트 + SHA256."""
    salted = (config.ANONYMIZATION_SALT + username).encode("utf-8")
    return hashlib.sha256(salted).hexdigest()[:16]


def now() -> str:
    return datetime.utcnow().isoformat()


# ============================================================
# Upsert 함수들 — INSERT OR IGNORE/REPLACE 정책
# ============================================================

def upsert_seller(conn, seller: dict):
    """셀러 레코드 INSERT. 이미 있으면 메타만 업데이트."""
    conn.execute(
        """
        INSERT INTO seller (seller_id, username_hash, followers, total_sales, rating,
                            n_reviews, is_vintage_shop, crawled_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(seller_id) DO UPDATE SET
            followers = excluded.followers,
            total_sales = excluded.total_sales,
            rating = excluded.rating,
            n_reviews = excluded.n_reviews,
            crawled_at = excluded.crawled_at
        """,
        (
            seller["seller_id"],
            hash_username(seller["username"]),
            seller.get("followers"),
            seller.get("total_sales"),
            seller.get("rating"),
            seller.get("n_reviews", 0),
            int(seller.get("is_vintage_shop", False)),
            now(),
        ),
    )


def upsert_listing(conn, listing: dict):
    """매물 레코드 INSERT 또는 UPDATE.

    Upsert 정책:
    - seed 단계가 만든 (seller_id='_pending_') placeholder는 listings 단계에서
      받은 진짜 데이터로 모든 필드를 덮어쓴다.
    - 이미 진짜 데이터가 있는 행에 다시 진짜 데이터가 들어오면 그것도 덮어쓴다
      (재크롤링 시 sold 상태 갱신 등).
    - 단, 빈 새 데이터(NULL)가 기존 값을 NULL로 만들지 않도록 COALESCE로 보호.
    """
    conn.execute(
        """
        INSERT INTO listing (product_id, seller_id, title, description, brand,
                             category_l1, category_l2, subcategory_id, size,
                             price_original, price_final, discount_pct,
                             likes, comments, n_photos, is_sold, posted_relative,
                             condition, like_count, view_count, created_at, gender,
                             crawled_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(product_id) DO UPDATE SET
            seller_id      = CASE WHEN excluded.seller_id != '_pending_'
                                  THEN excluded.seller_id ELSE listing.seller_id END,
            title          = COALESCE(excluded.title,          listing.title),
            description    = COALESCE(excluded.description,    listing.description),
            brand          = COALESCE(excluded.brand,          listing.brand),
            category_l1    = COALESCE(excluded.category_l1,    listing.category_l1),
            category_l2    = COALESCE(excluded.category_l2,    listing.category_l2),
            subcategory_id = COALESCE(excluded.subcategory_id, listing.subcategory_id),
            size           = COALESCE(excluded.size,           listing.size),
            price_original = COALESCE(excluded.price_original, listing.price_original),
            price_final    = COALESCE(excluded.price_final,    listing.price_final),
            discount_pct   = excluded.discount_pct,
            likes          = excluded.likes,
            comments       = excluded.comments,
            n_photos       = MAX(excluded.n_photos, listing.n_photos),
            is_sold        = excluded.is_sold,
            posted_relative= COALESCE(excluded.posted_relative, listing.posted_relative),
            condition      = COALESCE(excluded.condition,      listing.condition),
            like_count     = COALESCE(excluded.like_count,     listing.like_count),
            view_count     = COALESCE(excluded.view_count,     listing.view_count),
            created_at     = COALESCE(excluded.created_at,     listing.created_at),
            gender         = COALESCE(excluded.gender,         listing.gender),
            crawled_at     = excluded.crawled_at
        """,
        (
            listing["product_id"],
            listing["seller_id"],
            listing.get("title"),
            listing.get("description"),
            listing.get("brand"),
            listing.get("category_l1"),
            listing.get("category_l2"),
            listing.get("subcategory_id"),
            listing.get("size"),
            listing.get("price_original"),
            listing.get("price_final"),
            listing.get("discount_pct", 0),
            listing.get("likes", 0),
            listing.get("comments", 0),
            listing.get("n_photos", 0),
            int(listing.get("is_sold", False)),
            listing.get("posted_relative"),
            listing.get("condition"),
            listing.get("like_count", 0),
            listing.get("view_count", 0),
            listing.get("created_at"),
            listing.get("gender"),
            now(),
        ),
    )


def insert_review(conn, review: dict):
    conn.execute(
        """
        INSERT INTO review (seller_id, review_text, review_rating, crawled_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            review["seller_id"],
            review.get("review_text"),
            review.get("review_rating"),
            now(),
        ),
    )


def log_failure(conn, url: str, status: int = None, err: str = ""):
    conn.execute(
        "INSERT INTO fetch_failure (url, status_code, error_msg, failed_at) VALUES (?, ?, ?, ?)",
        (url, status, err[:500], now()),
    )


def set_state(conn, key: str, value: str):
    conn.execute(
        """
        INSERT INTO crawl_state (key, value, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """,
        (key, value, now()),
    )


def get_state(conn, key: str, default=None):
    row = conn.execute("SELECT value FROM crawl_state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


# ============================================================
# 통계 쿼리 (진행 상황 모니터링용)
# ============================================================

def stats(conn) -> dict:
    return {
        "n_sellers": conn.execute("SELECT COUNT(*) FROM seller").fetchone()[0],
        "n_listings": conn.execute("SELECT COUNT(*) FROM listing").fetchone()[0],
        "n_sold": conn.execute("SELECT COUNT(*) FROM listing WHERE is_sold = 1").fetchone()[0],
        "n_reviews": conn.execute("SELECT COUNT(*) FROM review").fetchone()[0],
        "n_failures": conn.execute("SELECT COUNT(*) FROM fetch_failure").fetchone()[0],
    }


def list_sellers_to_crawl(conn, limit: int = 100):
    """수집된 매물에서 발견됐지만 아직 셀러 상세 안 받은 셀러 ID 목록."""
    return [
        row["seller_id"]
        for row in conn.execute(
            """
            SELECT DISTINCT l.seller_id
            FROM listing l
            LEFT JOIN seller s ON s.seller_id = l.seller_id
            WHERE s.seller_id IS NULL
               OR s.followers IS NULL
            LIMIT ?
            """,
            (limit,),
        )
    ]
