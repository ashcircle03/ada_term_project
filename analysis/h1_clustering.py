"""가설 1: 셀러는 매물 텍스트·브랜드 분포에 따라 의미 있는 N개의 스타일 시그니처로 분리된다.

방법:
  1. 셀러별 signature_text를 TF-IDF 벡터화
  2. K-means + HDBSCAN 비교
  3. 실루엣 계수로 K 결정
  4. 각 클러스터의 대표 키워드·브랜드 추출 → 시그니처 라벨 부여

출력:
  results/h1_clustering.json — k, silhouette, 클러스터별 키워드/대표브랜드/셀러수
  results/figures/h1_*.png  — 클러스터 시각화 (PCA 2D)
  data/cache/seller_clusters.parquet — 셀러별 클러스터 라벨 (H2·H3에서 사용)
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import utils
from analysis.data_loader import load_listings, CACHE_DIR
from analysis.features import build_seller_text, signature_consistency


# ============================================================
# 한국어 + 영문 토크나이저 — TF-IDF 입력
# ============================================================

# 의미 없는 단어 제거 (분석에 노이즈)
STOPWORDS = {
    # ============================================================
    # 빈티지 마켓 보일러플레이트 — 셀러들이 매물 본문에 복붙하는 문구
    # 클러스터링을 "복붙 셀러 그룹"으로 왜곡함
    # ============================================================
    "배송", "배송이", "배송하며", "배송시작", "시작됩니다", "이내", "평균적으로",
    "측에서", "직접", "택배", "택배거래", "직거래", "택배비", "택포",
    "상품은", "상품", "제품은", "제품", "제품입니다", "입니다", "있습니다",
    "판매", "판매합니다", "판매중", "판매중입니다", "구매", "구매시", "거래",
    "정품", "정사", "실착", "실착1회", "보관", "보관품",
    "사진으로", "사진", "확인", "문의", "문의주세요", "문의하세요",
    "있으니", "있고", "있는", "없는", "있어요", "없어요", "합니다", "니다",
    "주세요", "드립니다", "되었습니다", "이며", "이고", "이라",
    # 추가 — 1차 결과에서 잔존 발견된 보일러플레이트
    "판매자", "시작", "평균적", "가능합니다", "가능", "착용", "있을",
    "교환", "환불", "특성상", "네고", "네고불가", "쿨거래",
    "예민하신분은", "예민", "약간", "많이", "지금", "이쁜",
    "사이트", "있어", "없이", "있음", "없음",

    # ============================================================
    # 어미·조사 (한국어 형태소 분석 미수행으로 토큰에 붙어있음)
    # ============================================================
    "에서", "으로", "에게", "에서부터", "까지",
    "하지만", "그래서", "그리고", "하고", "라서",

    # ============================================================
    # 일반적이라 시그니처 차이를 만들지 않는 어휘
    # ============================================================
    "사이즈", "상태", "색상", "컬러", "사용", "사용감", "오프", "빠른",
    "신상", "새상", "중고", "관련", "매물", "후르츠", "후르츠패밀리",
    "가지고", "위해", "정도", "조금", "그냥",
    "너무", "가로", "세로", "총장", "어깨", "기장", "암홀", "허리",  # 측정 어휘

    # ============================================================
    # 보편 색상 — 모든 매물에 등장해 시그니처 차이를 못 만듦
    # ============================================================
    "블랙", "화이트", "네이비", "그레이", "그린", "레드", "블루",
    "베이지", "브라운", "옐로우", "퍼플", "핑크", "오렌지",
    "black", "white", "navy", "gray", "grey", "green", "red", "blue",
    "beige", "brown", "yellow", "purple", "pink", "orange",

    # ============================================================
    # 너무 흔한 영문 단어
    # ============================================================
    "the", "and", "for", "with", "this", "that", "from", "your", "have",
    "size", "color", "item", "items", "good", "great", "nice",
    "made", "new", "used", "free",

    # ============================================================
    # 빈티지 마켓 공통 어휘 — 시그니처 차이를 만들지 않음
    # ============================================================
    "vintage", "빈티지", "archive", "아카이브", "rare", "희귀",
    "ss", "fw", "aw", "spring", "summer", "fall", "winter",  # 시즌은 보편
    "cm", "kg", "ml",  # 단위
}

KOREAN_RE = re.compile(r"[가-힣]+")
ENGLISH_RE = re.compile(r"[A-Za-z]{2,}")

# 한국어 어미·조사 — 토큰 끝에서 제거 (단순 휴리스틱, KoNLPy 대안)
KOREAN_ENDINGS = (
    "하며", "하면", "하고", "해서", "이라", "이며", "이고",
    "입니다", "이에요", "예요", "이다", "이며", "이라는",
    "됩니다", "되어", "되며", "됐다", "됐어요",
    "습니다", "어요", "아요",
    "에서", "으로", "에게", "까지", "에서는", "으로는",
    "들이", "들은", "들에",
    "이라서", "라서", "에서도",
)


def _trim_ending(token: str) -> str:
    """한국어 토큰의 흔한 어미·조사를 끝에서 제거.
    e.g., '배송하며' → '배송', '시작됩니다' → '시작', '측에서' → '측'
    너무 짧은 결과(1자)는 빈 문자열로 (단일 음절은 의미 없음).
    """
    for ending in KOREAN_ENDINGS:
        if token.endswith(ending) and len(token) > len(ending) + 1:
            return token[: -len(ending)]
    return token


def korean_tokenizer(text: str) -> list[str]:
    """한국어 명사·영문 단어를 단순 추출.

    KoNLPy 형태소 분석기 없이도 동작 — 의존성 최소화.
    빈티지 패션 키워드는 명사·고유명사·브랜드명이 많아서 단순 추출도 효과적.
    어미·조사를 단순 트리밍하여 명사 형태에 가깝게 정규화.
    """
    if not text:
        return []
    raw = KOREAN_RE.findall(text) + ENGLISH_RE.findall(text)
    out = []
    for t in raw:
        t = t.lower()
        # 한국어는 어미 트리밍, 영문은 그대로
        if KOREAN_RE.fullmatch(t):
            t = _trim_ending(t)
        if len(t) < 2:
            continue
        if t in STOPWORDS:
            continue
        out.append(t)
    return out


# ============================================================
# 클러스터링
# ============================================================

def vectorize_sellers(seller_text_df: pd.DataFrame, max_features: int = 1000):
    """signature_text → TF-IDF 매트릭스."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(
        tokenizer=korean_tokenizer,
        max_features=max_features,
        min_df=2,        # 2명 미만 셀러만 쓰는 어휘 제외
        max_df=0.8,      # 80% 이상 셀러가 쓰는 흔한 어휘 제외
        token_pattern=None,
    )
    X = vec.fit_transform(seller_text_df["signature_text"].fillna(""))
    return X, vec


def kmeans_with_optimal_k(X, k_range=range(3, 12)):
    """실루엣 계수로 최적 K 찾기."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if X.shape[0] < max(k_range):
        # 셀러 수가 너무 적으면 K 후보를 축소
        k_range = range(2, min(X.shape[0], 6))

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(X, labels) if X.shape[0] > k else float("nan")
        results.append({"k": k, "silhouette": sil, "labels": labels, "model": km})

    if not results:
        return None, None, []
    best = max(results, key=lambda r: r["silhouette"])
    return best["model"], best["labels"], results


def hdbscan_clusters(X):
    """HDBSCAN — K 지정 없이 자연스러운 군집 발견."""
    try:
        import hdbscan
    except ImportError:
        return None, None
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean")
    labels = clusterer.fit_predict(X.toarray())
    return clusterer, labels


# ============================================================
# 클러스터 해석 — 각 클러스터의 대표 키워드 / 브랜드
# ============================================================

def cluster_top_keywords(X, labels, vectorizer, top_n: int = 10) -> dict:
    """각 클러스터의 TF-IDF 평균이 높은 키워드 top_n."""
    feature_names = vectorizer.get_feature_names_out()
    result = {}
    for c in sorted(set(labels)):
        if c == -1:  # HDBSCAN noise
            continue
        mask = labels == c
        # 클러스터 내 셀러들의 평균 TF-IDF
        mean_tfidf = X[mask].mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[::-1][:top_n]
        result[int(c)] = [feature_names[i] for i in top_idx]
    return result


def cluster_top_brands(seller_text_df: pd.DataFrame, listings: pd.DataFrame,
                       labels: np.ndarray, top_n: int = 5) -> dict:
    """각 클러스터의 대표 브랜드 (셀러들이 가장 많이 다루는 브랜드)."""
    # seller_id → cluster 매핑
    seller_cluster = pd.DataFrame({
        "seller_id": seller_text_df["seller_id"].values,
        "cluster": labels,
    })
    df = listings.merge(seller_cluster, on="seller_id", how="inner")
    result = {}
    for c, group in df.groupby("cluster"):
        if c == -1:
            continue
        brand_counts = group["brand"].value_counts().head(top_n)
        result[int(c)] = list(brand_counts.index)
    return result


# ============================================================
# 메인
# ============================================================

def run(min_listings_per_seller: int = 3,
        brand_weight: int = 5, title_weight: int = 2,
        desc_weight: int = 1) -> dict:
    """H1 분석 메인 흐름.

    Args:
        min_listings_per_seller: 매물이 N건 미만인 셀러는 제외 (시그니처 추정 불가)
        brand_weight: 브랜드 텍스트 반복 횟수 (기본 5)
        title_weight: 제목 반복 횟수 (기본 2)
        desc_weight: 본문 반복 횟수 (기본 1, 0 으로 본문 제외 가능)
    """
    utils.section("H1: 셀러 시그니처 클러스터링")
    utils.setup_korean_font()

    # 1. 데이터 로드
    listings = load_listings()
    if listings.empty:
        print("  ✗ 매물 데이터 없음 — 수집 후 재실행")
        return {}

    seller_text = build_seller_text(
        listings,
        brand_weight=brand_weight,
        title_weight=title_weight,
        desc_weight=desc_weight,
    )
    seller_text = seller_text[seller_text["n_listings"] >= min_listings_per_seller]
    utils.bullet(
        "분석 대상 셀러",
        f"{len(seller_text)}명 (매물 {min_listings_per_seller}+ 보유)",
    )
    utils.bullet(
        "시그니처 가중치",
        f"brand×{brand_weight}, title×{title_weight}, desc×{desc_weight}",
    )

    if len(seller_text) < 10:
        print(f"  ✗ 셀러 수 부족 ({len(seller_text)} < 10) — 수집 더 필요")
        return {"status": "insufficient_data", "n_sellers": len(seller_text)}

    # 2. 벡터화
    X, vectorizer = vectorize_sellers(seller_text)
    utils.bullet("TF-IDF 차원", f"{X.shape[0]} sellers × {X.shape[1]} features")

    # 3. K-means + 최적 K
    utils.section("K-means 최적 K 탐색")
    km_model, km_labels, all_results = kmeans_with_optimal_k(X)
    for r in all_results:
        utils.bullet(f"k={r['k']}", f"silhouette = {r['silhouette']:.4f}")

    if km_model is None:
        return {"status": "kmeans_failed"}

    best_k = km_model.n_clusters
    utils.bullet("최적 K", best_k)

    # 4. 클러스터 해석
    utils.section("클러스터별 대표 키워드/브랜드")
    keywords = cluster_top_keywords(X, km_labels, vectorizer)
    brands = cluster_top_brands(seller_text, listings, km_labels)
    cluster_summary = {}
    for c in sorted(keywords.keys()):
        n_sellers = int((km_labels == c).sum())
        utils.bullet(f"클러스터 {c} ({n_sellers}명)",
                     f"브랜드: {', '.join(brands.get(c, [])[:3])}")
        utils.bullet(f"  → 키워드", ", ".join(keywords[c][:8]))
        cluster_summary[c] = {
            "n_sellers": n_sellers,
            "keywords": keywords[c],
            "top_brands": brands.get(c, []),
        }

    # 5. 셀러-클러스터 라벨 저장 → H2·H3에서 사용
    seller_clusters = pd.DataFrame({
        "seller_id": seller_text["seller_id"].values,
        "cluster": km_labels,
    })
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    seller_clusters.to_parquet(CACHE_DIR / "seller_clusters.parquet", index=False)
    utils.bullet("셀러-클러스터 라벨 저장",
                 str(CACHE_DIR / "seller_clusters.parquet"))

    # 6. 결과 저장
    payload = {
        "n_sellers": len(seller_text),
        "best_k": best_k,
        "best_silhouette": next(r["silhouette"] for r in all_results if r["k"] == best_k),
        "all_k_silhouette": [
            {"k": r["k"], "silhouette": r["silhouette"]} for r in all_results
        ],
        "clusters": cluster_summary,
    }
    utils.save_result("h1_clustering", payload)

    # 7. PCA 시각화
    try:
        plot_clusters(X, km_labels, seller_text)
    except Exception as e:
        print(f"  시각화 스킵: {e}")

    return payload


def plot_clusters(X, labels, seller_text_df):
    """PCA 2D 산점도. 한글 폰트 설정 후 호출."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=2, random_state=42)
    X2 = svd.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 7))
    for c in sorted(set(labels)):
        mask = labels == c
        ax.scatter(X2[mask, 0], X2[mask, 1], label=f"cluster {c}",
                   alpha=0.7, s=50)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("Seller Signature Clusters (TruncatedSVD 2D)")
    ax.legend()
    utils.save_figure(fig, "h1_clusters_2d")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--min-listings", type=int, default=5,
                   help="셀러당 최소 매물 수 (기본 5). 5+: 시그니처 안정. 3: 표본 더 많이.")
    p.add_argument("--brand-weight", type=int, default=5,
                   help="브랜드 텍스트 반복 횟수 (기본 5). 보일러플레이트 잔존 시 7~10으로 상향.")
    p.add_argument("--title-weight", type=int, default=2,
                   help="제목 반복 횟수 (기본 2)")
    p.add_argument("--desc-weight", type=int, default=1,
                   help="본문 반복 횟수 (기본 1). 0=본문 제외, 보일러플레이트 잔존 시 권장.")
    args = p.parse_args()
    run(
        min_listings_per_seller=args.min_listings,
        brand_weight=args.brand_weight,
        title_weight=args.title_weight,
        desc_weight=args.desc_weight,
    )
