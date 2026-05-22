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
                      value_col: str = "price_final",
                      min_group_size: int = 5) -> dict:
    """클러스터별 가격 분포에 차이가 있는지 비모수 검정.

    표본이 정규분포를 따른다는 가정을 못 함 (가격은 right-skewed).
    Kruskal-Wallis는 ANOVA의 비모수 대안.
    η² (eta-squared) = (H - k + 1) / (n - k) 로 효과 크기도 보고.
    """
    from scipy import stats
    grouped = [(name, g[value_col].dropna().values)
               for name, g in df.groupby(group_col)]
    grouped = [(name, g) for name, g in grouped if len(g) >= min_group_size]
    if len(grouped) < 2:
        return {"valid": False, "reason": "충분한 그룹 없음"}
    names = [name for name, _ in grouped]
    groups = [g for _, g in grouped]
    stat, p = stats.kruskal(*groups)
    k = len(groups)
    n = sum(len(g) for g in groups)
    eta2 = (stat - k + 1) / (n - k) if n > k else float("nan")
    return {
        "valid": True,
        "statistic": float(stat),
        "p_value": float(p),
        "eta_squared": float(eta2),
        "n_groups": k,
        "group_names": [int(nm) if isinstance(nm, (int, np.integer)) else nm
                        for nm in names],
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

    설계:
    - 매칭 그룹 안에서 2개 이상의 다른 클러스터가 존재하는 그룹만 사용
      (같은 클러스터끼리만 있는 그룹은 클러스터 효과를 분리할 수 없음)
    - 그룹 내 중앙값 대비 가격 비율로 표준화 → 상품 가치 차이 제거
    - 클러스터별 가격 비율 분포 집계 → 시그니처 가격 프리미엄
    """
    if "cluster" not in df.columns:
        return {"valid": False, "reason": "cluster 컬럼 없음 (H1 먼저 실행)"}

    matched = matched_pairs(df)
    if matched.empty:
        return {"valid": False, "reason": "매칭 그룹 없음"}

    # 클러스터가 2개 이상 섞인 그룹만 — 그래야 클러스터 효과가 식별 가능
    cluster_diversity = matched.groupby("match_group_id")["cluster"].nunique()
    valid_groups = cluster_diversity[cluster_diversity >= 2].index
    matched = matched[matched["match_group_id"].isin(valid_groups)].copy()

    if len(matched) < 10:
        return {"valid": False, "reason": f"클러스터 간 비교 가능한 매물 부족 ({len(matched)}건)"}

    # 그룹 내 중앙값 대비 가격 비율로 표준화
    matched["group_median_price"] = matched.groupby("match_group_id")["price_final"].transform("median")
    matched["price_ratio"] = matched["price_final"] / matched["group_median_price"]

    # 전문형 vs 잡화형 분리
    specialist = matched[matched["cluster"] != -1]
    generalist = matched[matched["cluster"] == -1]

    # 클러스터별 가격 비율 집계 (전문형만 — 각 시그니처의 프리미엄)
    by_cluster = specialist.groupby("cluster").agg(
        n_listings=("product_id", "count"),
        median_price_ratio=("price_ratio", "median"),
        mean_price_ratio=("price_ratio", "mean"),
    ).reset_index().sort_values("median_price_ratio", ascending=False)

    # 전문형 vs 잡화형 전체 비교
    spec_ratio = specialist["price_ratio"].median() if len(specialist) > 0 else float("nan")
    gen_ratio = generalist["price_ratio"].median() if len(generalist) > 0 else float("nan")

    # Mann-Whitney: 전문형이 잡화형보다 높은 가격 비율을 갖는지
    mw_result = {}
    if len(specialist) >= 5 and len(generalist) >= 5:
        from scipy import stats
        u_stat, p_val = stats.mannwhitneyu(
            specialist["price_ratio"].dropna(),
            generalist["price_ratio"].dropna(),
            alternative="greater",
        )
        # 효과 크기: rank-biserial correlation
        n1, n2 = len(specialist["price_ratio"].dropna()), len(generalist["price_ratio"].dropna())
        r = (2 * u_stat) / (n1 * n2) - 1
        mw_result = {
            "statistic": float(u_stat),
            "p_value": float(p_val),
            "rank_biserial_r": float(r),
        }

    return {
        "valid": True,
        "n_cross_cluster_groups": int(len(valid_groups)),
        "n_matched_listings": int(len(matched)),
        "specialist_median_ratio": float(spec_ratio),
        "generalist_median_ratio": float(gen_ratio),
        "specialist_vs_generalist_mw": mw_result,
        "by_cluster": by_cluster.to_dict(orient="records"),
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

    # 매물에 클러스터 라벨 join (가격 결측·0원 제외)
    df = listings.merge(clusters, on="seller_id", how="inner")
    df = df[df["price_final"].notna() & (df["price_final"] > 0)]
    utils.bullet("분석 대상 매물", f"{len(df):,}건 (클러스터 라벨 보유, 가격 유효)")

    spec = df[df["cluster"] != -1]
    gen = df[df["cluster"] == -1]
    utils.bullet("전문형 매물", f"{len(spec):,}건  중앙값 {spec['price_final'].median():,.0f}원")
    utils.bullet("잡화형 매물", f"{len(gen):,}건  중앙값 {gen['price_final'].median():,.0f}원")

    if df.empty:
        return {"status": "no_overlap"}

    # ---------------------------------------------------------
    # 분석 1: 전문형 클러스터 간 가격 차이 (잡화형 제외)
    # ---------------------------------------------------------
    utils.section("분석 1: 전문형 클러스터 간 가격 분포 차이 (Kruskal-Wallis)")
    kw_spec = kruskal_by_cluster(spec, "cluster", "price_final")
    if kw_spec.get("valid"):
        utils.bullet("Kruskal-Wallis H", f"{kw_spec['statistic']:.2f}")
        utils.bullet("p-value", f"{kw_spec['p_value']:.2e}")
        utils.bullet("η² (효과 크기)", f"{kw_spec['eta_squared']:.4f}")
        sig = "유의 (H0 기각)" if kw_spec["p_value"] < 0.05 else "비유의"
        utils.bullet("결론", sig)
        # 상위/하위 클러스터 출력
        paired = sorted(zip(kw_spec["group_names"], kw_spec["medians"]),
                        key=lambda x: x[1], reverse=True)
        utils.bullet("가격 상위 3개 클러스터",
                     "  ".join(f"C{c}={m:,.0f}원" for c, m in paired[:3]))
        utils.bullet("가격 하위 3개 클러스터",
                     "  ".join(f"C{c}={m:,.0f}원" for c, m in paired[-3:]))
    else:
        utils.bullet("스킵", kw_spec.get("reason"))

    # ---------------------------------------------------------
    # 분석 1b: 전문형 vs 잡화형 전체 비교
    # ---------------------------------------------------------
    utils.section("분석 1b: 전문형 vs 잡화형 가격 비교 (Mann-Whitney)")
    kw_all = kruskal_by_cluster(df, "cluster", "price_final")
    from scipy import stats as scipy_stats
    if len(spec) >= 5 and len(gen) >= 5:
        u, p_sv = scipy_stats.mannwhitneyu(
            spec["price_final"].dropna(),
            gen["price_final"].dropna(),
            alternative="greater",
        )
        n1, n2 = len(spec["price_final"].dropna()), len(gen["price_final"].dropna())
        r_sv = (2 * u) / (n1 * n2) - 1
        utils.bullet("전문형 중앙값", f"{spec['price_final'].median():,.0f}원")
        utils.bullet("잡화형 중앙값", f"{gen['price_final'].median():,.0f}원")
        utils.bullet("Mann-Whitney p (전문형 > 잡화형)", f"{p_sv:.2e}")
        utils.bullet("효과 크기 r (rank-biserial)", f"{r_sv:.4f}")
        spec_vs_gen = {"u_statistic": float(u), "p_value": float(p_sv),
                       "rank_biserial_r": float(r_sv)}
    else:
        spec_vs_gen = {}

    # ---------------------------------------------------------
    # 분석 2: 매칭 분석 — 동일 조건 통제 후 프리미엄
    # ---------------------------------------------------------
    utils.section("분석 2: 동일 (브랜드·카테고리·사이즈) 통제 후 가격 프리미엄")
    matched_result = matched_premium_analysis(df)
    if matched_result.get("valid"):
        utils.bullet("클러스터 간 비교 가능 그룹", f"{matched_result['n_cross_cluster_groups']:,}개")
        utils.bullet("해당 매물", f"{matched_result['n_matched_listings']:,}건")
        utils.bullet("전문형 중앙 가격비", f"{matched_result['specialist_median_ratio']:.3f}")
        utils.bullet("잡화형 중앙 가격비", f"{matched_result['generalist_median_ratio']:.3f}")
        mw = matched_result.get("specialist_vs_generalist_mw", {})
        if mw:
            utils.bullet("Mann-Whitney p (전문형 > 잡화형, 매칭 내)",
                         f"{mw['p_value']:.2e}  r={mw['rank_biserial_r']:.4f}")
        print()
        # 클러스터별 프리미엄 상위/하위
        by_c = matched_result["by_cluster"]
        for row in by_c[:5]:
            premium = (row["median_price_ratio"] - 1) * 100
            utils.bullet(
                f"C{row['cluster']} ({row['n_listings']}건)",
                f"가격비 {row['median_price_ratio']:.3f}  ({premium:+.1f}%)",
            )
    else:
        utils.bullet("스킵", matched_result.get("reason"))

    # ---------------------------------------------------------
    # 분석 3: 시그니처 일관성 → 가격·판매율 상관 (보조)
    # ---------------------------------------------------------
    utils.section("분석 3: 시그니처 일관성 vs 가격·판매율 (Spearman)")
    cons = signature_consistency(listings)
    aggs = seller_aggregates(listings)
    seller_df = cons.merge(aggs, on="seller_id", how="inner")
    seller_df = seller_df[seller_df["n_listings"] >= 5]

    corr_sold = corr_price = None
    if len(seller_df) >= 10:
        corr_sold = scipy_stats.spearmanr(
            seller_df["signature_consistency"], seller_df["sold_rate"]
        )
        corr_price = scipy_stats.spearmanr(
            seller_df["signature_consistency"], seller_df["median_price"]
        )
        utils.bullet("일관성 vs 판매율",
                     f"ρ={corr_sold.statistic:.4f}  p={corr_sold.pvalue:.4f}")
        utils.bullet("일관성 vs 가격 중앙값",
                     f"ρ={corr_price.statistic:.4f}  p={corr_price.pvalue:.4f}")
    else:
        utils.bullet("스킵", f"셀러 수 부족 ({len(seller_df)})")

    # 결과 저장
    payload = {
        "n_listings_in_analysis": len(df),
        "n_specialist": len(spec),
        "n_generalist": len(gen),
        "specialist_median_price": float(spec["price_final"].median()),
        "generalist_median_price": float(gen["price_final"].median()),
        "kruskal_specialist_clusters": kw_spec,
        "specialist_vs_generalist": spec_vs_gen,
        "matched_premium": matched_result,
        "consistency_correlations": {
            "sold_rate": {"rho": float(corr_sold.statistic), "p": float(corr_sold.pvalue)} if corr_sold else None,
            "median_price": {"rho": float(corr_price.statistic), "p": float(corr_price.pvalue)} if corr_price else None,
        },
    }
    utils.save_result("h2_anova", payload)
    return payload


if __name__ == "__main__":
    run()
