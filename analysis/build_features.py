"""공용 feature 파이프라인 — 현 DB(288k)에서 분석용 parquet 생성.

산출:
  data/cache/features_listing.parquet  (매물단위, 필터 후 ~280k)
  data/cache/features_seller.parquet   (셀러단위, ~11k)

설계 결정 (계획서):
  - 결과변수 is_sold. view_count/like_count는 누수 → 피처 테이블에서 아예 제외.
  - gender는 category_l1과 중복 → 제외.
  - _pending_ placeholder, created_at/description NULL 행 제외 (N 로그 출력).

실행:  python -m analysis.build_features
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import featurelib as fl

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fruitsfamily.db"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
TODAY = pd.Timestamp("2026-05-29", tz="UTC")  # 연령 기준일

# 등록 후 누적값 → 예측에 쓰면 누수. 피처 테이블에 절대 포함 금지.
LEAKAGE_COLS = ["view_count", "like_count", "likes", "comments", "crawled_at"]

TOP_K_BRANDS = 30
PRICE_TIER_BINS = [0, 30_000, 80_000, 200_000, np.inf]
PRICE_TIER_LABELS = ["1_low<3만", "2_mid<8만", "3_high<20만", "4_top20만+"]


def load_raw() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT product_id, seller_id, title, description, brand,
                   category_l1, category_l2, size,
                   price_original, price_final, discount_pct,
                   n_photos, is_sold, condition, created_at
            FROM listing
            """,
            conn,
        )
    return df


def build_listing_features(raw: pd.DataFrame) -> pd.DataFrame:
    n0 = len(raw)
    df = raw[raw["seller_id"] != "_pending_"].copy()
    n1 = len(df)
    df = df[df["created_at"].notna() & df["description"].notna()].copy()
    df = df[pd.to_numeric(df["price_final"], errors="coerce").fillna(0) > 0].copy()
    n2 = len(df)
    print(f"[listing] raw={n0:,} → drop _pending_ ={n1:,} → drop NULL created_at/desc/price ={n2:,}")

    df["is_sold"] = df["is_sold"].astype(int)
    df["price_final"] = pd.to_numeric(df["price_final"], errors="coerce")
    df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce").fillna(0)

    # --- 구조 피처 (통제) ---
    df["log_price"] = np.log1p(df["price_final"])
    df["price_tier"] = pd.cut(df["price_final"], bins=PRICE_TIER_BINS,
                              labels=PRICE_TIER_LABELS, right=False)
    created = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["age_days"] = (TODAY - created).dt.total_seconds() / 86400.0
    df["created_year"] = created.dt.year
    # brand top-K + OTHER (고카디널리티 6,765종 → 회귀용 축소)
    top_brands = df["brand"].value_counts().head(TOP_K_BRANDS).index
    df["brand_top"] = df["brand"].where(df["brand"].isin(top_brands), "OTHER")

    # --- 통제가능(표현) 피처 ---
    txt = fl.text_features(df["description"])
    df = pd.concat([df, txt], axis=1)
    df["relative_price_z"] = fl.relative_price_z(df)
    df["has_discount"] = (df["discount_pct"] > 0).astype(int)

    # 누수 가드 — 절대 들어오면 안 되는 컬럼
    leaked = [c for c in LEAKAGE_COLS if c in df.columns]
    assert not leaked, f"누수 컬럼이 피처 테이블에 있음: {leaked}"

    keep = [
        "product_id", "seller_id", "is_sold", "created_at", "created_year",
        # 통제가능(표현)
        "n_photos", "desc_len", "n_lines", "n_hashtag", "n_emoji", "has_desc",
        "kw_measure", "kw_flaw", "kw_material", "kw_purchase", "kw_usage", "kw_wash",
        "discount_pct", "has_discount", "relative_price_z",
        # 구조(통제)
        "brand", "brand_top", "category_l1", "category_l2", "condition",
        "price_final", "log_price", "price_tier", "age_days",
    ]
    return df[keep].reset_index(drop=True)


def build_seller_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["seller_id"] != "_pending_"].copy()
    df["is_sold"] = df["is_sold"].astype(int)
    df["price_final"] = pd.to_numeric(df["price_final"], errors="coerce")

    sig = fl.seller_signature(df)
    text = fl.build_seller_text(df)
    out = sig.merge(text, on="seller_id", how="left")

    # 셀러 메타 조인
    with sqlite3.connect(DB_PATH) as conn:
        meta = pd.read_sql_query(
            "SELECT seller_id, followers, total_sales, rating FROM seller", conn
        )
    out = out.merge(meta, on="seller_id", how="left")
    print(f"[seller] {len(out):,}명  (메타 매칭 {out['followers'].notna().sum():,})")
    return out


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()

    listing = build_listing_features(raw)
    listing.to_parquet(CACHE_DIR / "features_listing.parquet", index=False)

    seller = build_seller_features(raw)
    seller.to_parquet(CACHE_DIR / "features_seller.parquet", index=False)

    # --- sanity check (계획서 EDA 수치와 대조) ---
    print("\n=== SANITY ===")
    print(f"  전체 전환율: {listing['is_sold'].mean():.1%}  (기대 ~21%)")
    print(f"  relative_price_z 계산된 비율: {listing['relative_price_z'].notna().mean():.1%}")
    print(f"  age_days 중앙값: {listing['age_days'].median():.0f}일")
    pg = pd.cut(listing["n_photos"], [-1, 2, 5, 10], labels=["<=2", "3-5", "6+"])
    print("  사진수 구간별 전환율 (기대: <=2 최고):")
    print(listing.groupby(pg, observed=True)["is_sold"].mean().round(3).to_string())
    print(f"  zero-sale 셀러 비율: {(seller['n_sold']==0).mean():.1%}  (기대 ~16%)")


if __name__ == "__main__":
    main()
