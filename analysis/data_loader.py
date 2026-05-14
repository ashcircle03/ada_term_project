"""DB → DataFrame 로더.

분석 단계의 모든 모듈은 이 모듈 하나만 import하면 데이터에 접근 가능.
SQL은 여기서만 쓰고, 다른 모듈은 pandas DataFrame만 다룬다.

캐시 정책:
  data/cache/listings.parquet 등에 캐시 → 두 번째 실행부터 빠름
  --refresh 인자로 강제 재로딩
"""
import sqlite3
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fruitsfamily.db"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"


def _conn():
    return sqlite3.connect(DB_PATH)


# ============================================================
# 매물 (listings) — 분석의 기본 단위
# ============================================================

def load_listings(real_only: bool = True, refresh: bool = False) -> pd.DataFrame:
    """매물 테이블 → DataFrame.

    Args:
        real_only: True면 seller_id != '_pending_'인 행만 (분석에서는 항상 True)
        refresh: 캐시 무시하고 DB에서 다시 읽기
    """
    cache = CACHE_DIR / "listings.parquet"
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
    else:
        with _conn() as conn:
            df = pd.read_sql_query(
                """
                SELECT product_id, seller_id, title, description, brand,
                       category_l1, category_l2, subcategory_id, size,
                       price_original, price_final, discount_pct,
                       likes, comments, n_photos, is_sold, posted_relative,
                       condition, like_count, view_count, created_at, gender,
                       crawled_at
                FROM listing
                """,
                conn,
            )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)

    if real_only:
        df = df[df["seller_id"] != "_pending_"].copy()

    # 타입 보정
    df["is_sold"] = df["is_sold"].astype(bool)
    df["price_final"] = pd.to_numeric(df["price_final"], errors="coerce")
    df["price_original"] = pd.to_numeric(df["price_original"], errors="coerce")

    return df.reset_index(drop=True)


# ============================================================
# 셀러 (sellers) — 메타 정보
# ============================================================

def load_sellers(refresh: bool = False) -> pd.DataFrame:
    cache = CACHE_DIR / "sellers.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    with _conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT seller_id, username_hash, followers, total_sales, rating,
                   n_reviews, is_vintage_shop, crawled_at
            FROM seller
            """,
            conn,
        )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


# ============================================================
# 리뷰
# ============================================================

def load_reviews(refresh: bool = False) -> pd.DataFrame:
    cache = CACHE_DIR / "reviews.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    with _conn() as conn:
        df = pd.read_sql_query(
            "SELECT review_id, seller_id, review_text, review_rating, crawled_at FROM review",
            conn,
        )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


# ============================================================
# 통합 조인 — 분석에 가장 자주 쓰이는 형태
# ============================================================

def load_listings_with_seller(refresh: bool = False) -> pd.DataFrame:
    """매물 + 셀러 메타 조인된 wide format.

    H2·H3 분석의 기본 입력. 각 매물에 셀러 팔로워·누적판매수·평점이 붙은 형태.
    """
    listings = load_listings(real_only=True, refresh=refresh)
    sellers = load_sellers(refresh=refresh)
    if sellers.empty:
        # 셀러 단계가 아직 안 돌았을 때 — 셀러 컬럼 NaN으로 채워서 반환
        for col in ["followers", "total_sales", "rating", "is_vintage_shop"]:
            listings[col] = pd.NA
        return listings
    return listings.merge(
        sellers[["seller_id", "followers", "total_sales", "rating", "is_vintage_shop"]],
        on="seller_id",
        how="left",
    )


# ============================================================
# 진단
# ============================================================

def summary() -> None:
    """현재 데이터 상태 요약 — 분석 진입 전 sanity check 용."""
    print("=" * 60)
    print("데이터 로드 가능성 확인")
    print("=" * 60)

    listings = load_listings(real_only=True, refresh=True)
    sellers = load_sellers(refresh=True)
    reviews = load_reviews(refresh=True)

    print(f"  매물 (real seller): {len(listings):,}건")
    print(f"  셀러 (seller table): {len(sellers):,}명")
    print(f"  리뷰: {len(reviews):,}건")
    print()

    if not listings.empty:
        print(f"  unique seller in listings: {listings['seller_id'].nunique()}명")
        print(f"  unique brand: {listings['brand'].nunique()}개")
        print(f"  sold 비율: {listings['is_sold'].mean():.1%}")
        print(f"  가격 중앙값: {listings['price_final'].median():,.0f}원")
        print(f"  가격 IQR: {listings['price_final'].quantile(0.25):,.0f}~{listings['price_final'].quantile(0.75):,.0f}원")


if __name__ == "__main__":
    summary()
