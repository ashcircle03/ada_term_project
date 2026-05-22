"""분석용 가공 변수 생성.

원본 listing/seller 테이블에서 가설 1·2·3의 입력 변수를 만든다.

핵심 가공물:
  signature_text   — 셀러의 모든 매물 텍스트 결합 (H1 클러스터링 입력)
  signature_consistency — 셀러 매물 브랜드 분포의 엔트로피 기반 일관성 (H2 보조)
  matched_pairs    — (브랜드, 카테고리, 사이즈) 동일 매물 매칭 (H2 통제)
  listing_features — 매물 단위 모델 입력 (H3 지도학습)
"""
import math
from collections import Counter
import pandas as pd
import numpy as np


# ============================================================
# 셀러 단위 가공
# ============================================================

def build_seller_text(listings: pd.DataFrame, brand_weight: int = 5,
                      title_weight: int = 2, desc_weight: int = 1) -> pd.DataFrame:
    """셀러별로 모든 매물의 제목+본문+브랜드를 하나의 문서로 결합.

    가중치 정책 (보일러플레이트 오염을 줄이기 위한 설계):
      - brand_weight (기본 5): 브랜드는 시그니처의 가장 강한 신호.
        텍스트에 N번 반복하여 TF-IDF 가중치를 자연스럽게 높인다.
      - title_weight (기본 2): 제목도 본문보다 시그니처 신호가 강함.
      - desc_weight  (기본 1): 본문은 보일러플레이트가 많아 노이즈 비율 높음.
        보일러플레이트 잔존 시 desc_weight=0 으로 본문 제외 가능.

    Returns:
        DataFrame[seller_id, signature_text, n_listings]
    """
    if listings.empty:
        return pd.DataFrame(columns=["seller_id", "signature_text", "n_listings"])

    def repeat_join(series, weight):
        if weight <= 0:
            return ""
        joined = " ".join(series.dropna().astype(str))
        return (joined + " ") * weight

    grouped = listings.groupby("seller_id").agg(
        titles=("title", lambda s: repeat_join(s, title_weight)),
        descs=("description", lambda s: repeat_join(s, desc_weight)),
        brands=("brand", lambda s: repeat_join(s, brand_weight)),
        n_listings=("product_id", "count"),
    )
    grouped["signature_text"] = (
        grouped["brands"] + " " + grouped["titles"] + " " + grouped["descs"]
    ).str.strip()
    return grouped[["signature_text", "n_listings"]].reset_index()


def signature_consistency(listings: pd.DataFrame) -> pd.DataFrame:
    """셀러별 브랜드 분포 엔트로피 기반 일관성 점수.

    consistency = 1 - H/H_max
      H        = -Σ p_i log p_i
      H_max    = log(unique 브랜드 수)
    값:
      1.0  → 한 브랜드만 (시그니처 명확)
      ~0.0 → 균등 분산 (잡탕형)
    """
    rows = []
    for sid, group in listings.groupby("seller_id"):
        brands = group["brand"].dropna().tolist()
        unique_brands = len(set(brands))
        if len(brands) < 2:
            consistency = 1.0
        else:
            counts = Counter(brands)
            total = sum(counts.values())
            probs = [c / total for c in counts.values()]
            H = -sum(p * math.log(p) for p in probs)
            H_max = math.log(unique_brands) if unique_brands > 1 else 1
            consistency = 1 - (H / H_max) if H_max > 0 else 1.0
        rows.append({
            "seller_id": sid,
            "signature_consistency": consistency,
            "unique_brands": unique_brands,
        })
    return pd.DataFrame(rows)


def seller_aggregates(listings: pd.DataFrame) -> pd.DataFrame:
    """셀러 단위 집계 피처 — H2·H3 모두에서 셀러 컨텍스트로 활용."""
    if listings.empty:
        return pd.DataFrame()

    agg_dict = {
        "n_listings":  ("product_id", "count"),
        "n_sold":      ("is_sold", "sum"),
        "avg_price":   ("price_final", "mean"),
        "median_price":("price_final", "median"),
        "avg_discount":("discount_pct", "mean"),
        "median_likes":("likes", "median"),
        "avg_n_photos":("n_photos", "mean"),
    }
    if "like_count" in listings.columns:
        agg_dict["avg_like_count"] = ("like_count", "mean")
    if "view_count" in listings.columns:
        agg_dict["avg_view_count"] = ("view_count", "mean")

    agg = listings.groupby("seller_id").agg(**agg_dict).reset_index()
    agg["sold_rate"] = agg["n_sold"] / agg["n_listings"]
    return agg


# ============================================================
# 매물 단위 가공
# ============================================================

def listing_features(listings: pd.DataFrame, seller_aggs: pd.DataFrame = None) -> pd.DataFrame:
    """매물 단위 모델 입력 (H3).

    각 매물에 셀러 컨텍스트(평균 가격, 매물 수 등)를 붙이고
    log price 같은 변환 변수도 만든다.
    """
    df = listings.copy()

    # 결손 처리: price_original이 없으면 final로 채움
    df["price_original_filled"] = df["price_original"].fillna(df["price_final"])
    df["log_price"] = np.log1p(df["price_final"].fillna(0))
    df["title_len"] = df["title"].fillna("").str.len()
    df["desc_len"] = df["description"].fillna("").str.len()
    df["has_size"] = df["size"].notna().astype(int)
    df["has_discount"] = (df["discount_pct"].fillna(0) > 0).astype(int)

    # 셀러 컨텍스트 join
    if seller_aggs is not None and not seller_aggs.empty:
        df = df.merge(
            seller_aggs.add_prefix("seller_").rename(columns={"seller_seller_id": "seller_id"}),
            on="seller_id",
            how="left",
        )

    return df


# ============================================================
# H2: 매칭 — 동일 조건 매물 쌍 찾기
# ============================================================

def matched_pairs(listings: pd.DataFrame, by=("brand", "category_l2", "size")) -> pd.DataFrame:
    """동일 (브랜드, 카테고리, 사이즈)를 가진 매물끼리 묶기.

    H2 가설 검정에서 "옷의 본질 가치 통제"의 핵심.
    같은 그룹 내에서 셀러 시그니처 클러스터별 가격 차이를 본다.

    Returns:
        listings 행 + 'match_group_id' 컬럼 (그룹 멤버 2 이상인 그룹만)
    """
    df = listings.copy()

    # NULL 처리 — 카테고리/브랜드는 NULL이면 매칭 불가, size는 NO_SIZE 토큰으로
    # (jewelry/액세서리는 size NULL이 정상이라 매칭에 포함시킴)
    required = [c for c in by if c != "size"]
    df = df.dropna(subset=required)
    if "size" in by:
        df["size"] = df["size"].fillna("NO_SIZE")

    df["match_group_id"] = df.groupby(list(by)).ngroup()

    # 멤버 2 이상인 그룹만 — 비교가 가능한 매물
    counts = df.groupby("match_group_id").size()
    valid_groups = counts[counts >= 2].index
    df = df[df["match_group_id"].isin(valid_groups)].copy()

    return df


# ============================================================
# 진단
# ============================================================

def summary(listings: pd.DataFrame) -> None:
    print("=" * 60)
    print("가공 변수 생성 결과")
    print("=" * 60)

    if listings.empty:
        print("  매물 데이터 없음 — 수집 후 다시 실행")
        return

    text_df = build_seller_text(listings)
    cons_df = signature_consistency(listings)
    aggs = seller_aggregates(listings)
    features = listing_features(listings, aggs)
    matched = matched_pairs(listings)

    print(f"  signature_text 행: {len(text_df)}")
    print(f"    텍스트 길이 중앙값: {text_df['signature_text'].str.len().median():.0f} 글자")
    print(f"    매물 수 중앙값:    {text_df['n_listings'].median():.0f}")
    print()
    print(f"  signature_consistency 분포:")
    print(f"    중앙값: {cons_df['signature_consistency'].median():.3f}")
    print(f"    > 0.7 (시그니처 명확): {(cons_df['signature_consistency'] > 0.7).sum()}명")
    print(f"    < 0.3 (잡탕형):      {(cons_df['signature_consistency'] < 0.3).sum()}명")
    print()
    print(f"  listing_features 행: {len(features)}")
    print(f"    추가 컬럼: log_price, title_len, desc_len, has_size, has_discount")
    print()
    print(f"  matched_pairs:")
    print(f"    매칭 가능한 매물: {len(matched)}건")
    print(f"    매칭 그룹 수: {matched['match_group_id'].nunique() if not matched.empty else 0}")


if __name__ == "__main__":
    from analysis.data_loader import load_listings
    df = load_listings()
    summary(df)
