"""가설 2: 매물의 브랜드·카테고리·사이즈를 통제했을 때, 셀러가 속한 시그니처
클러스터에 따라 매물 가격 및 판매 소요시간 평균에 유의미한 차이가 존재한다.

전략:
  단순 ANOVA가 아니라 "동일 조건 매물 매칭 후 클러스터 비교" 가 핵심.
  같은 브랜드·카테고리·사이즈인 매물끼리 묶고, 그 안에서 클러스터별 가격 차이.

방법:
  1. listings + seller_clusters 조인 (H1 결과)
  2. matched_pairs로 (브랜드, 카테고리, 사이즈) 동일 그룹 형성
  3. 각 그룹 내 클러스터 간 가격 차이 검증 — Kruskal-Wallis (정규성 미가정)
  4. 보조 분석: 시그니처 일관성 → 가격 프리미엄 회귀
"""
import numpy as np
import pandas as pd

from analysis import utils
from analysis.data_loader import load_listings, CACHE_DIR
from analysis.features import (
    matched_pairs, signature_consistency, seller_aggregates,
)


def load_seller_clusters() -> pd.DataFrame | None:
    """H1 결과에서 셀러-클러스터 라벨 불러오기."""
    path = CACHE_DIR / "seller_clusters.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


# ============================================================
# 통계 검정
# ============================================================

def kruskal_by_cluster(df: pd.DataFrame, group_col: str = "cluster",
                      value_col: str = "price_final") -> dict:
    """클러스터별 가격 분포에 차이가 있는지 비모수 검정.

    표본이 정규분포를 따른다는 가정을 못 함 (가격은 right-skewed).
    Kruskal-Wallis는 ANOVA의 비모수 대안.
    """
    from scipy import stats
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
    groups = [g for g in groups if len(g) >= 5]  # 최소 표본 보장
    if len(groups) < 2:
        return {"valid": False, "reason": "충분한 그룹 없음"}
    stat, p = stats.kruskal(*groups)
    return {
        "valid": True,
        "statistic": float(stat),
        "p_value": float(p),
        "n_groups": len(groups),
        "n_samples": [int(len(g)) for g in groups],
        "medians": [float(np.median(g)) for g in groups],
    }


def dunn_post_hoc(df: pd.DataFrame, group_col: str = "cluster",
                  value_col: str = "price_final") -> pd.DataFrame:
    """Kruskal에서 유의했을 때 어느 그룹쌍이 다른지 사후검정."""
    try:
        import scikit_posthocs as sp
        return sp.posthoc_dunn(df, val_col=value_col, group_col=group_col,
                               p_adjust="bonferroni")
    except ImportError:
        # 라이브러리 없으면 pairwise Mann-Whitney 직접 수행
        from scipy import stats
        groups = list(df.groupby(group_col))
        n = len(groups)
        result = pd.DataFrame(np.ones((n, n)),
                              index=[g[0] for g in groups],
                              columns=[g[0] for g in groups])
        for i in range(n):
            for j in range(i + 1, n):
                _, p = stats.mannwhitneyu(
                    groups[i][1][value_col].dropna(),
                    groups[j][1][value_col].dropna(),
                    alternative="two-sided",
                )
                # Bonferroni 보정
                p_adj = min(1.0, p * (n * (n - 1) / 2))
                result.iat[i, j] = p_adj
                result.iat[j, i] = p_adj
        return result


# ============================================================
# 매칭 분석 — 동일 조건 매물 내 클러스터 비교
# ============================================================

def matched_premium_analysis(df: pd.DataFrame) -> dict:
    """같은 (브랜드, 카테고리, 사이즈) 그룹 내에서 클러스터별 가격 비교.

    각 매칭 그룹에서 클러스터별 평균 가격 → 그룹 평균 대비 % 차이로 표준화.
    그 표준화 값들을 클러스터별로 집계 → 시그니처 가격 프리미엄.
    """
    matched = matched_pairs(df)
    if matched.empty:
        return {"valid": False, "reason": "매칭 그룹 없음"}

    # 그룹 내 평균 대비 가격 비율
    matched["group_mean_price"] = matched.groupby("match_group_id")["price_final"].transform("mean")
    matched["price_ratio"] = matched["price_final"] / matched["group_mean_price"]

    # 클러스터별 가격 비율 집계
    if "cluster" not in matched.columns:
        return {"valid": False, "reason": "cluster 컬럼 없음 (H1 먼저 실행)"}

    summary = matched.groupby("cluster").agg(
        n_listings=("product_id", "count"),
        median_price_ratio=("price_ratio", "median"),
        mean_price_ratio=("price_ratio", "mean"),
    ).reset_index()
    return {
        "valid": True,
        "n_matched_groups": int(matched["match_group_id"].nunique()),
        "n_matched_listings": int(len(matched)),
        "by_cluster": summary.to_dict(orient="records"),
    }


# ============================================================
# 메인
# ============================================================

def run() -> dict:
    utils.section("H2: 시그니처 클러스터별 가격 차이 검정")
    utils.setup_korean_font()

    listings = load_listings()
    clusters = load_seller_clusters()

    if listings.empty:
        print("  ✗ 매물 데이터 없음")
        return {}
    if clusters is None:
        print("  ✗ seller_clusters 없음 — H1 먼저 실행 필요")
        return {}

    # 매물에 클러스터 라벨 join
    df = listings.merge(clusters, on="seller_id", how="inner")
    utils.bullet("분석 대상 매물", f"{len(df)}건 (클러스터 라벨 보유)")

    if df.empty:
        return {"status": "no_overlap"}

    # ---------------------------------------------------------
    # 분석 1: 전체 클러스터별 가격 분포 차이
    # ---------------------------------------------------------
    utils.section("분석 1: 클러스터별 전체 가격 분포 차이")
    kw = kruskal_by_cluster(df, "cluster", "price_final")
    if kw.get("valid"):
        utils.bullet("Kruskal-Wallis statistic", f"{kw['statistic']:.4f}")
        utils.bullet("p-value", f"{kw['p_value']:.6f}")
        utils.bullet("결론", "유의 (H0 기각)" if kw["p_value"] < 0.05 else "비유의")
        utils.bullet("클러스터별 표본 수", kw["n_samples"])
        utils.bullet("클러스터별 가격 중앙값",
                     [f"{m:,.0f}" for m in kw["medians"]])
    else:
        utils.bullet("스킵", kw.get("reason"))

    # ---------------------------------------------------------
    # 분석 2: 매칭 분석 — 동일 조건 통제 후
    # ---------------------------------------------------------
    utils.section("분석 2: 동일 (브랜드·카테고리·사이즈) 매칭 후 가격 프리미엄")
    matched_result = matched_premium_analysis(df)
    if matched_result.get("valid"):
        utils.bullet("매칭 그룹 수", matched_result["n_matched_groups"])
        utils.bullet("매칭된 매물 수", matched_result["n_matched_listings"])
        print()
        for row in matched_result["by_cluster"]:
            premium = (row["median_price_ratio"] - 1) * 100
            utils.bullet(
                f"클러스터 {row['cluster']}",
                f"매물 {row['n_listings']}건  "
                f"중앙 가격비 {row['median_price_ratio']:.3f}  "
                f"({premium:+.1f}%)",
            )
    else:
        utils.bullet("스킵", matched_result.get("reason"))

    # ---------------------------------------------------------
    # 분석 3: 시그니처 일관성 → 판매율 회귀 (보조)
    # ---------------------------------------------------------
    utils.section("분석 3: 시그니처 일관성과 판매 성공률")
    cons = signature_consistency(listings)
    aggs = seller_aggregates(listings)
    seller_df = cons.merge(aggs, on="seller_id", how="inner")
    seller_df = seller_df[seller_df["n_listings"] >= 3]

    if len(seller_df) >= 10:
        from scipy import stats
        corr_sold = stats.spearmanr(
            seller_df["signature_consistency"],
            seller_df["sold_rate"],
        )
        corr_price = stats.spearmanr(
            seller_df["signature_consistency"],
            seller_df["median_price"],
        )
        utils.bullet("일관성 vs 판매율 (Spearman)",
                     f"ρ={corr_sold.statistic:.4f}  p={corr_sold.pvalue:.4f}")
        utils.bullet("일관성 vs 가격 (Spearman)",
                     f"ρ={corr_price.statistic:.4f}  p={corr_price.pvalue:.4f}")
    else:
        utils.bullet("스킵", f"셀러 수 부족 ({len(seller_df)})")

    # 결과 저장
    payload = {
        "kruskal_overall": kw,
        "matched_premium": matched_result,
        "n_listings_in_analysis": len(df),
    }
    utils.save_result("h2_anova", payload)
    return payload


if __name__ == "__main__":
    run()
