"""가설 3: 매물 메타데이터만 사용한 가격 예측 모델보다, 셀러 시그니처를
결합한 모델의 예측 성능이 유의미하게 우수하다.

방법:
  Model A: 매물 피처만 (브랜드, 카테고리, 사이즈, 사진 수, 본문 길이 등)
  Model B: + 셀러 시그니처 클러스터, 일관성, 셀러 메타(팔로워, 누적판매, 평점)
  → XGBoost 회귀, 5-fold cross-validation
  → RMSE / R² / MAE 비교, paired t-test 로 유의성 검정
  → SHAP 으로 시그니처 변수의 기여도 분해
"""
import numpy as np
import pandas as pd

from analysis import utils
from analysis.data_loader import load_listings_with_seller, CACHE_DIR
from analysis.features import (
    listing_features, signature_consistency, seller_aggregates,
)


def load_seller_clusters() -> pd.DataFrame | None:
    path = CACHE_DIR / "seller_clusters.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def prepare_features(df: pd.DataFrame, with_signature: bool) -> tuple:
    """모델 입력 매트릭스 + 종속변수 준비.

    LEAKAGE 방지 정책:
      - seller_avg_price, seller_median_price 같은 가격 집계 변수는 절대 제외.
        → 셀러의 평균 가격은 종속변수(매물 가격) 자체에서 계산된 것이라
          self-prediction이 됨. 학습 시 R²가 0.95+ 같이 비현실적으로 나옴.
      - 가격이 아닌 셀러 메타만 사용 (followers, rating, n_listings, sold_rate 등).

    Returns:
        X (DataFrame), y (Series), feature_names (list)
    """
    # 종속변수: log 가격 (right-skewed 보정)
    y = np.log1p(df["price_final"])

    # 공통 피처
    common_cols = [
        "discount_pct", "title_len", "desc_len", "has_size",
        "has_discount", "n_photos",
    ]
    # condition 더미 (매물 등록 시 셀러가 직접 선택 → 가격 결정 요인)
    if "condition" in df.columns:
        cat_cols = ["brand", "category_l1", "category_l2", "size", "condition"]
    else:
        cat_cols = ["brand", "category_l1", "category_l2", "size"]

    base_df = df[common_cols].copy()
    cat_dummies = pd.get_dummies(df[cat_cols].astype(str), drop_first=False)

    if with_signature:
        sig_cols = []
        if "cluster" in df.columns:
            cluster_dummies = pd.get_dummies(df["cluster"].astype(str), prefix="cluster")
            sig_cols.append(cluster_dummies)

        # ⚠️ LEAKAGE 변수는 명시적 차단:
        # seller_avg_price, seller_median_price = 가격에서 계산된 값 → 제외
        # seller_avg_discount = 가격에서 파생 → 제외
        leakage_cols = {
            "seller_avg_price", "seller_median_price", "seller_avg_discount",
            "avg_price", "median_price",  # 접두사 없는 형태도 차단
        }
        safe_seller_cols = [
            "signature_consistency",  # 브랜드 분포 엔트로피 — 가격 미사용
            "unique_brands",
            "followers", "total_sales", "rating",  # 셀러 메타
            "seller_n_listings", "seller_sold_rate",  # 셀러 활동량
            "seller_median_likes", "seller_avg_n_photos",  # 매물 품질 시그널
            "seller_avg_like_count", "seller_avg_view_count",  # 셀러 평균 참여도
        ]
        for col in safe_seller_cols:
            if col in df.columns and col not in leakage_cols:
                sig_cols.append(df[[col]].fillna(df[col].median()))
        X = pd.concat([base_df, cat_dummies] + sig_cols, axis=1)
    else:
        X = pd.concat([base_df, cat_dummies], axis=1)

    # 결측 처리
    X = X.fillna(0)
    return X, y, list(X.columns)


def cv_evaluate(X: pd.DataFrame, y: pd.Series, n_splits: int = 10) -> dict:
    """XGBoost 회귀 + 10-fold CV. fold별 RMSE/MAE/R² 반환.

    5-fold에서 10-fold로 변경한 이유: 표본이 26,000+으로 충분하고,
    paired t-test가 fold 수에 민감 (5-fold는 자유도 4로 검정력 매우 낮음).
    10-fold는 자유도 9로 실질적 개선이 있을 때 탐지 가능.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        # XGBoost 없으면 GradientBoosting으로 폴백
        from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor

    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = {"rmse": [], "mae": [], "r2": []}
    feature_importance = np.zeros(X.shape[1])

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # 원래 스케일에서 평가 (log 역변환)
        y_test_orig = np.expm1(y_test)
        pred_orig = np.expm1(pred)
        fold_results["rmse"].append(float(np.sqrt(mean_squared_error(y_test_orig, pred_orig))))
        fold_results["mae"].append(float(mean_absolute_error(y_test_orig, pred_orig)))
        fold_results["r2"].append(float(r2_score(y_test, pred)))  # log scale 에서

        if hasattr(model, "feature_importances_"):
            feature_importance += model.feature_importances_

    feature_importance /= n_splits
    return {
        "rmse_mean": float(np.mean(fold_results["rmse"])),
        "rmse_std": float(np.std(fold_results["rmse"])),
        "mae_mean": float(np.mean(fold_results["mae"])),
        "r2_mean": float(np.mean(fold_results["r2"])),
        "r2_std": float(np.std(fold_results["r2"])),
        "fold_rmse": fold_results["rmse"],
        "fold_r2": fold_results["r2"],
        "feature_importance": feature_importance.tolist(),
    }


def paired_test(metric_a: list, metric_b: list, alternative="less") -> dict:
    """두 모델의 fold별 metric에 대한 paired t-test.

    alternative='less' 는 Model B의 RMSE가 A보다 낮은지 검증.
    """
    from scipy import stats
    diff = np.array(metric_a) - np.array(metric_b)
    if len(diff) < 2:
        return {"valid": False}
    t, p = stats.ttest_rel(metric_a, metric_b, alternative=alternative)
    return {
        "valid": True,
        "t_statistic": float(t),
        "p_value": float(p),
        "mean_diff": float(diff.mean()),
        "n_folds": len(diff),
    }


# ============================================================
# 메인
# ============================================================

def run(min_listings: int = 100) -> dict:
    utils.section("H3: 셀러 시그니처가 가격 예측 성능을 개선하는가")
    utils.setup_korean_font()

    df = load_listings_with_seller()
    if len(df) < min_listings:
        print(f"  ✗ 매물 부족 ({len(df)} < {min_listings}) — 수집 후 재실행")
        return {"status": "insufficient_data", "n": len(df)}

    # H1 클러스터 + signature_consistency join
    clusters = load_seller_clusters()
    cons = signature_consistency(df)
    aggs = seller_aggregates(df)

    if clusters is not None:
        df = df.merge(clusters, on="seller_id", how="left")
        # H1 분석 제외 셀러(brand unreliable, 매물 5 미만)는 cluster=NULL → -1(잡화형)로 처리
        df["cluster"] = df["cluster"].fillna(-1).astype(int)
    df = df.merge(cons, on="seller_id", how="left")
    df = df.merge(
        aggs.add_prefix("seller_").rename(columns={"seller_seller_id": "seller_id"}),
        on="seller_id", how="left",
    )

    # 가격 결측·0원 매물 제외
    df = df.dropna(subset=["price_final"])
    df = df[df["price_final"] > 0]

    # 매물 단위 파생 변수
    df = listing_features(df)

    n_specialist = int((df["cluster"] != -1).sum()) if "cluster" in df.columns else 0
    utils.bullet("분석 대상 매물", f"{len(df):,}건")
    utils.bullet("가격 중앙값", f"{df['price_final'].median():,.0f}원")
    utils.bullet("전문형 매물 비중", f"{n_specialist:,}건 ({n_specialist/len(df):.1%})")

    # ---------------------------------------------------------
    # Model A vs B
    # ---------------------------------------------------------
    utils.section("Model A: 매물 피처만")
    X_a, y, _ = prepare_features(df, with_signature=False)
    utils.bullet("입력 차원", f"{X_a.shape[0]} × {X_a.shape[1]}")
    res_a = cv_evaluate(X_a, y)
    utils.bullet("RMSE", f"{res_a['rmse_mean']:,.0f} ± {res_a['rmse_std']:,.0f}원")
    utils.bullet("R² (log scale)", f"{res_a['r2_mean']:.4f}")
    utils.bullet("MAE", f"{res_a['mae_mean']:,.0f}원")

    utils.section("Model B: + 셀러 시그니처 (클러스터 + 일관성 + 셀러 메타)")
    X_b, _, _ = prepare_features(df, with_signature=True)
    utils.bullet("입력 차원", f"{X_b.shape[0]} × {X_b.shape[1]}")
    res_b = cv_evaluate(X_b, y)
    utils.bullet("RMSE", f"{res_b['rmse_mean']:,.0f} ± {res_b['rmse_std']:,.0f}원")
    utils.bullet("R² (log scale)", f"{res_b['r2_mean']:.4f}")
    utils.bullet("MAE", f"{res_b['mae_mean']:,.0f}원")

    # ---------------------------------------------------------
    # 차이 검정
    # ---------------------------------------------------------
    utils.section("Model A vs B 차이 — paired t-test (10-fold)")
    rmse_test = paired_test(res_a["fold_rmse"], res_b["fold_rmse"], alternative="greater")
    r2_test = paired_test(res_b["fold_r2"], res_a["fold_r2"], alternative="greater")

    if rmse_test.get("valid"):
        utils.bullet("RMSE 감소 (B < A) p-value",
                     f"{rmse_test['p_value']:.4f}  (Δ={rmse_test['mean_diff']:,.0f}원)")
    if r2_test.get("valid"):
        utils.bullet("R² 증가 (B > A) p-value",
                     f"{r2_test['p_value']:.4f}  (Δ={r2_test['mean_diff']:.4f})")

    rmse_drop_pct = (res_a["rmse_mean"] - res_b["rmse_mean"]) / res_a["rmse_mean"] * 100
    utils.bullet("RMSE 개선율", f"{rmse_drop_pct:+.2f}%")

    # Model B feature importance 상위 시그니처 피처
    _, _, feat_names_b = prepare_features(df, with_signature=True)
    fi = res_b["feature_importance"]
    top_fi = sorted(zip(feat_names_b, fi), key=lambda x: x[1], reverse=True)[:15]
    utils.section("Model B — 상위 15 피처 (평균 feature importance)")
    for name, imp in top_fi:
        bar = "█" * int(imp * 300)
        utils.bullet(name[:35], f"{imp:.4f} {bar}")

    # ---------------------------------------------------------
    # 결과 저장
    # ---------------------------------------------------------
    payload = {
        "n_listings": len(df),
        "n_specialist": n_specialist,
        "n_folds": 10,
        "model_a": {k: v for k, v in res_a.items() if k != "feature_importance"},
        "model_b": {k: v for k, v in res_b.items() if k != "feature_importance"},
        "rmse_test": rmse_test,
        "r2_test": r2_test,
        "rmse_drop_pct": rmse_drop_pct,
        "top_features_b": [{"feature": n, "importance": float(i)} for n, i in top_fi],
    }
    utils.save_result("h3_prediction", payload)
    return payload


if __name__ == "__main__":
    run()
