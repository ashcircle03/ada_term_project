"""재사용 가능한 피처 엔지니어링 함수 모음.

build_features.py(캐시 생성)와 노트북들이 공통으로 import한다.
SQL은 build_features.py에만, 여기서는 DataFrame만 다룬다.

설계 결정:
  - 결과변수는 is_sold (유동성/매칭). 속도 아님.
  - gender(MEN/WOMEN)는 category_l1(남자/여자)과 사실상 동일 → 공선성 회피 위해
    구조 통제에는 category_l1만 사용, gender는 제외.
  - condition은 순서(GOOD_CONDITION vs LIGHTLY_WORN)가 모호 → 순서형 가정 없이 범주형.
  - 한국어 NLP 의존성 회피 → 설명 텍스트는 해석가능한 카운트 피처로만.
  - view_count/like_count는 등록 후 누적 → 누수. 예측 피처로 절대 포함 금지.
"""
import math
import re
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================
# 설명 텍스트 — 해석가능 카운트 피처 (한국어 NLP 의존성 없음)
# ============================================================

# 플랫폼 가이드가 권장하는 "자세한 설명"의 구체 신호.
# 등록률(전체 대비): 실측 12.7%, 하자 13.5%, 소재 6.8%, 구매 32.4%, 사용 18.2%, 세탁 4.1%
DESC_KEYWORDS = {
    "kw_measure": ["실측", "사이즈", "총장", "어깨", "가슴", "허리"],  # 실측 사이즈
    "kw_flaw":    ["하자", "오염", "손상", "데미지", "흠집", "변색"],   # 훼손 고지
    "kw_material":["소재", "혼용", "면 ", "울 ", "코튼", "나일론", "가죽"],  # 소재
    "kw_purchase":["구매", "정품", "구입", "영수증"],                  # 구매처/정품
    "kw_usage":   ["사용", "착용", "보관"],                           # 사용 이력
    "kw_wash":    ["세탁", "드라이", "관리"],                         # 관리법
}

_EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF" "\U00002B00-\U00002BFF" "]",
    flags=re.UNICODE,
)
_HASHTAG_RE = re.compile(r"#\S+")


def text_features(desc: pd.Series) -> pd.DataFrame:
    """설명 텍스트 → 카운트 피처 DataFrame (입력과 같은 인덱스).

    셀러가 '통제 가능한' 표현 노력의 신호:
      desc_len, n_lines, n_hashtag, n_emoji, has_desc, + 키워드 플래그 6종
    """
    s = desc.fillna("").astype(str)
    out = pd.DataFrame(index=desc.index)
    out["desc_len"] = s.str.len()
    out["has_desc"] = (out["desc_len"] > 0).astype(int)
    out["n_lines"] = s.str.count(r"\n") + (out["desc_len"] > 0).astype(int)
    out["n_hashtag"] = s.str.count(_HASHTAG_RE.pattern)
    out["n_emoji"] = s.apply(lambda x: len(_EMOJI_RE.findall(x)))
    for col, kws in DESC_KEYWORDS.items():
        pat = "|".join(re.escape(k) for k in kws)
        out[col] = s.str.contains(pat, regex=True).astype(int)
    return out


# ============================================================
# 상대가격 — 동종 매물 대비 가격 포지셔닝 (셀러 통제가능 레버)
# ============================================================

def relative_price_z(df: pd.DataFrame, price_col: str = "price_final",
                     min_group: int = 5) -> pd.Series:
    """brand×category_l1×condition 그룹 내 log가격 z-score.

    그룹 n<min_group이면 brand×category_l1로 폴백, 그래도 부족하면 NaN.
    "같은 브랜드·종류·상태의 다른 매물보다 비싸게/싸게 내놨나"를 격리.
    """
    log_p = np.log1p(df[price_col].clip(lower=0))

    def group_z(keys):
        g = log_p.groupby([df[k] for k in keys])
        mean = g.transform("mean")
        std = g.transform("std")
        size = g.transform("size")
        z = (log_p - mean) / std
        z[(size < min_group) | (std == 0) | std.isna()] = np.nan
        return z

    z = group_z(["brand", "category_l1", "condition"])
    z_fallback = group_z(["brand", "category_l1"])
    return z.fillna(z_fallback)


# ============================================================
# 셀러 시그니처 — 스타일 아키타입의 구조적 입력 (H3)
# ============================================================

def _entropy(counts) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log(p) for p in probs)


def seller_signature(listings: pd.DataFrame) -> pd.DataFrame:
    """셀러 단위 스타일 시그니처 + 성과 집계.

    구조 피처: 브랜드 HHI/엔트로피, 브랜드 일관성, 카테고리 분포,
               가격대(중앙값/분위), condition 믹스, n_listings, sell_through.
    """
    rows = []
    for sid, g in listings.groupby("seller_id"):
        brands = g["brand"].dropna()
        bc = Counter(brands)
        n_b = len(bc)
        total_b = sum(bc.values())
        hhi = sum((c / total_b) ** 2 for c in bc.values()) if total_b else np.nan
        ent = _entropy(bc.values())
        ent_norm = ent / math.log(n_b) if n_b > 1 else 0.0  # 0=한브랜드 집중, 1=균등분산

        cat = g["category_l1"].dropna()
        share_men = (cat == "남자").mean() if len(cat) else np.nan

        cond = g["condition"].dropna()
        share_new = (cond == "NEW").mean() if len(cond) else np.nan

        prices = g["price_final"].dropna()
        rows.append({
            "seller_id": sid,
            "n_listings": len(g),
            "n_sold": int(g["is_sold"].sum()),
            "sell_through": g["is_sold"].mean(),
            "n_brands": n_b,
            "brand_hhi": hhi,                  # 높을수록 소수 브랜드 집중
            "brand_entropy_norm": ent_norm,    # 낮을수록 시그니처 뚜렷
            "share_men": share_men,
            "share_new": share_new,
            "median_price": prices.median() if len(prices) else np.nan,
            "log_median_price": np.log1p(prices.median()) if len(prices) else np.nan,
            "avg_n_photos": g["n_photos"].mean(),
        })
    return pd.DataFrame(rows)


def build_seller_text(listings: pd.DataFrame, brand_weight: int = 5,
                      title_weight: int = 2, desc_weight: int = 0) -> pd.DataFrame:
    """셀러별 시그니처 텍스트 (선행 analysis/features.py 레시피 재사용).

    브랜드를 5회 반복해 TF-IDF 가중을 높이고, 본문(보일러플레이트)은 기본 제외(0).
    Returns: DataFrame[seller_id, signature_text]
    """
    def repeat_join(series, weight):
        if weight <= 0:
            return ""
        return (" ".join(series.dropna().astype(str)) + " ") * weight

    grouped = listings.groupby("seller_id").agg(
        titles=("title", lambda s: repeat_join(s, title_weight)),
        descs=("description", lambda s: repeat_join(s, desc_weight)),
        brands=("brand", lambda s: repeat_join(s, brand_weight)),
    )
    grouped["signature_text"] = (
        grouped["brands"] + " " + grouped["titles"] + " " + grouped["descs"]
    ).str.strip()
    return grouped[["signature_text"]].reset_index()


def cluster_sellers(seller_text: pd.DataFrame, n_svd: int = 100,
                    min_cluster_size: int = 30, random_state: int = 42) -> pd.DataFrame:
    """TF-IDF(char n-gram) → TruncatedSVD → HDBSCAN 아키타입 군집.

    선행 레시피 재사용. K 미지정 밀도 기반 → noise(-1)='잡화형' 셀러 자연 분리.
    char n-gram 사용으로 한국어 토크나이저 의존성 회피.
    Returns: DataFrame[seller_id, cluster]
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    import hdbscan

    texts = seller_text["signature_text"].fillna("")
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                            min_df=5, max_features=20000)
    X = tfidf.fit_transform(texts)
    n_comp = min(n_svd, X.shape[1] - 1, max(2, X.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    Xr = svd.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(Xr)
    out = seller_text[["seller_id"]].copy()
    out["cluster"] = labels
    return out


# ============================================================
# 성향점수매칭 (PSM) — H1 준인과 robustness
# ============================================================

def propensity_match(df: pd.DataFrame, treatment: str, covariates: list[str],
                     caliper: float = 0.05, random_state: int = 42):
    """로지스틱 성향점수 → 1:1 최근접 매칭(캘리퍼) → ATT.

    treatment: 0/1 컬럼 (예: 가이드 준수 여부)
    covariates: 매칭에 쓸 구조 공변량 (이미 수치화/더미화된 컬럼들)
    Returns: (att, dict 진단)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    d = df.dropna(subset=covariates + [treatment, "is_sold"]).copy()
    X = StandardScaler().fit_transform(d[covariates].values)
    t = d[treatment].astype(int).values
    ps = LogisticRegression(max_iter=1000).fit(X, t).predict_proba(X)[:, 1]
    d["_ps"] = ps

    treated = d[d[treatment] == 1]
    control = d[d[treatment] == 0]
    nn = NearestNeighbors(n_neighbors=1).fit(control[["_ps"]].values)
    dist, idx = nn.kneighbors(treated[["_ps"]].values)

    keep = dist.ravel() <= caliper
    matched_treated = treated[keep]
    matched_control = control.iloc[idx.ravel()[keep]]
    att = matched_treated["is_sold"].mean() - matched_control["is_sold"].mean()
    return att, {
        "n_treated": int(len(treated)),
        "n_matched": int(keep.sum()),
        "match_rate": float(keep.mean()),
        "treated_sold": float(matched_treated["is_sold"].mean()),
        "control_sold": float(matched_control["is_sold"].mean()),
        "caliper": caliper,
    }
