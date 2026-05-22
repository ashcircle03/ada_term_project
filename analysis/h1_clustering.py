"""가설 1: 셀러 집단에서 의미 있는 스타일 시그니처 그룹이 존재한다.
단, 모든 셀러가 시그니처를 갖지는 않는다 — 특정 스타일에 집중하는 '전문형' 셀러와
여러 브랜드를 혼합하는 '잡화형' 셀러가 공존할 것이다.

방법:
  1. 셀러별 signature_text를 TF-IDF 벡터화 (desc_weight=0 — 본문 보일러플레이트 제거)
  2. TruncatedSVD(100차원)로 고차원 스파스 행렬 축소
  3. HDBSCAN — K 지정 없이 밀도 기반 군집 발견, noise(-1)로 시그니처 없는 셀러 처리
  4. 각 클러스터의 대표 키워드·브랜드 추출 → 시그니처 라벨 부여

설계 근거:
  - K-means: 모든 셀러를 반드시 군집에 배정 → 시그니처 없는 셀러가 쓰레기 클러스터 형성
    (실루엣 0.063, k=19에서도 수렴 없음 — 자연적 군집 구조 부재 의미)
  - consistency 필터: brand 엔트로피 기반으로 다중 브랜드 스타일 셀러를 잘못 제외
    (예: Rick Owens+Chrome Hearts+Yohji = 명확한 아방가르드 시그니처지만 consistency=0.063)
  - HDBSCAN+SVD: 자연적 고밀도 영역만 클러스터로 지정, 나머지는 noise(-1) = '잡화형'

출력:
  results/h1_clustering.json — 클러스터별 키워드/대표브랜드/셀러수, 잡화형 셀러 수
  results/figures/h1_*.png  — 클러스터 시각화 (SVD 2D)
  data/cache/seller_clusters.parquet — 셀러별 클러스터 라벨 (-1=잡화형, H2·H3에서 사용)
"""
import re
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

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
    # 범용 의류 카테고리 어휘 — 모든 셀러가 사용해 시그니처 차이 미생성
    # (top feature 분석에서 확인: 클러스터 분리에 기여하지 않음)
    # ============================================================
    "티셔츠", "팬츠", "자켓", "후드티", "후드", "맨투맨", "셔츠", "코트",
    "바지", "청바지", "데님", "원피스", "스커트", "니트", "가디건", "점퍼",
    "집업", "스웨터", "블라우스", "슬랙스", "조거", "레깅스", "트레이닝",
    "가슴", "어깨너비", "밑단", "소매", "허리단", "힙", "밑위",  # 치수 어휘 보강
    "jacket", "shirt", "pants", "hoodie", "coat", "dress", "skirt",
    "sweater", "cardigan", "tshirt", "denim", "jeans", "top", "bottom",

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


def _load_brand_dict() -> dict[str, str]:
    """DB에서 공백 포함 브랜드명을 로드해 치환 사전 생성.

    'Stone Island' → 'Stone_Island' 형태로 보호.
    공백 없는 브랜드는 Kiwi가 자연스럽게 처리하므로 제외.
    긴 브랜드명 먼저 치환해야 부분 치환 오류 방지.
    """
    db_path = Path(__file__).parent.parent / "data" / "fruitsfamily.db"
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    brands = pd.read_sql(
        "SELECT DISTINCT brand FROM listing WHERE brand IS NOT NULL", conn
    )["brand"].tolist()
    conn.close()

    mapping = {}
    for b in brands:
        key = b.strip()
        if len(key) < 3:
            continue
        # 공백 포함 브랜드 (Stone Island → Stone_Island)
        if " " in key:
            mapping[key] = key.replace(" ", "_")
        # . 포함 브랜드 (A.Presse → A_Presse)
        elif "." in key:
            mapping[key] = key.replace(".", "_")
    # 긴 브랜드명 먼저 치환 (부분 치환 방지)
    return dict(sorted(mapping.items(), key=lambda x: -len(x[0])))


# 모듈 로드 시 1회 초기화
_BRAND_DICT: dict[str, str] = _load_brand_dict()
# 역변환: 'Stone_Island' → 'stone island' (소문자 정규화)
_BRAND_DICT_INV: dict[str, str] = {
    v.lower(): k.lower().replace(" ", "_") for k, v in _BRAND_DICT.items()
}


def _protect_brands(text: str) -> str:
    """브랜드명 공백을 _ 로 치환해 Kiwi 분리 방지."""
    for brand, protected in _BRAND_DICT.items():
        text = text.replace(brand, protected)
    return text


def korean_tokenizer(text: str) -> list[str]:
    """Kiwi 형태소 분석 기반 한국어 명사 + 영문 고유명사 추출.

    브랜드명 보호 흐름:
      1. 공백 포함 브랜드명을 _ 연결형으로 치환 (Stone Island → Stone_Island)
      2. Kiwi로 토큰화 — SL 토큰이 _ 기호 사이에 분리되므로
      3. 연속 SL + SW(_) + SL 패턴을 다시 합쳐 브랜드 토큰 복원
      4. NNG/NNP/SL 태그 명사만 남기고 스탑워드·2자 미만 제거
    """
    if not text:
        return []

    try:
        _get_kiwi()
    except ImportError:
        return _regex_tokenizer(text)

    protected = _protect_brands(text)
    raw_tokens = _get_kiwi().tokenize(protected)

    # SL + SW(_) + SL 연속 패턴 합치기 (브랜드명 복원)
    merged = []
    i = 0
    while i < len(raw_tokens):
        t = raw_tokens[i]
        # 영문 토큰 뒤에 _ 기호가 오면 다음 영문과 합침
        if (
            t.tag == "SL"
            and i + 2 < len(raw_tokens)
            and raw_tokens[i + 1].form == "_"
            and raw_tokens[i + 2].tag == "SL"
        ):
            combined = t.form + "_" + raw_tokens[i + 2].form
            # 추가로 이어지는 _+SL 패턴도 합침 (3단어 이상 브랜드)
            j = i + 3
            while j + 1 < len(raw_tokens) and raw_tokens[j].form == "_" and raw_tokens[j + 1].tag == "SL":
                combined += "_" + raw_tokens[j + 1].form
                j += 2
            merged.append(combined.lower())
            i = j
        elif t.tag in ("NNG", "NNP", "SL"):
            merged.append(t.form.lower())
            i += 1
        else:
            i += 1

    out = []
    for form in merged:
        if len(form) < 2:
            continue
        if form in STOPWORDS:
            continue
        out.append(form)
    return out


def _regex_tokenizer(text: str) -> list[str]:
    """kiwipiepy 미설치 환경용 정규식 폴백."""
    ENGLISH_RE = re.compile(r"[A-Za-z_]{2,}")
    raw = KOREAN_RE.findall(text) + ENGLISH_RE.findall(text)
    out = []
    for t in raw:
        t = t.lower()
        if len(t) < 2 or t in STOPWORDS:
            continue
        out.append(t)
    return out


# Kiwi 인스턴스 싱글톤 (초기화 비용 절감)
_kiwi_instance = None


def _get_kiwi():
    global _kiwi_instance
    if _kiwi_instance is None:
        from kiwipiepy import Kiwi
        _kiwi_instance = Kiwi()
    return _kiwi_instance


# ============================================================
# 클러스터링
# ============================================================

def vectorize_sellers(seller_text_df: pd.DataFrame, max_features: int = 3000):
    """signature_text → TF-IDF 매트릭스."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    n = len(seller_text_df)
    # min_df: 전체 셀러의 1% 이상이 사용한 어휘만 포함
    # (n=883일 때 min_df≈9 → 브랜드별 소수 매니아 어휘는 제거, 시그니처 공유 어휘만 유지)
    min_df = max(2, int(n * 0.01))
    vec = TfidfVectorizer(
        tokenizer=korean_tokenizer,
        max_features=max_features,
        min_df=min_df,
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


def hdbscan_clusters(X_svd, min_cluster_size: int = 5):
    """HDBSCAN on SVD-reduced matrix — K 지정 없이 밀도 기반 군집 발견.

    고차원 TF-IDF에 직접 euclidean 거리를 쓰면 차원의 저주로 거리 구분이 무의미해짐.
    TruncatedSVD로 100차원 축소 후 적용.
    noise(-1) = 시그니처 없는 잡화형 셀러.
    """
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        return None, None
    clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(X_svd)
    return clusterer, labels


def reduce_dimensions(X, n_components: int = 100):
    """TruncatedSVD로 스파스 TF-IDF 행렬을 밀집 저차원 공간으로 축소."""
    from sklearn.decomposition import TruncatedSVD
    n_comp = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_svd = svd.fit_transform(X)
    explained = svd.explained_variance_ratio_.sum()
    return X_svd, svd, explained


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

def build_brand_matrix(listings: pd.DataFrame,
                       min_listings_per_seller: int = 5,
                       min_sellers_per_brand: int = 3):
    """셀러 × 브랜드 공동출현 행렬 구성.

    TF-IDF 가중치 적용:
      TF = log(1 + 매물수) — 단순 카운트의 scale 효과 완화
      IDF = log(N / 브랜드 취급 셀러수 + 1) + 1 — 흔한 브랜드 다운웨이팅
    L2 정규화로 셀러별 매물 수 차이 제거.
    """
    from sklearn.preprocessing import normalize

    df = listings[listings["brand"].notna()].copy()
    df = df.groupby(["seller_id", "brand"]).size().reset_index(name="n")

    # 셀러 필터
    seller_total = df.groupby("seller_id")["n"].sum()
    valid_sellers = seller_total[seller_total >= min_listings_per_seller].index
    df = df[df["seller_id"].isin(valid_sellers)]

    # 브랜드 필터 (희귀 브랜드 제거)
    brand_seller_cnt = df.groupby("brand")["seller_id"].nunique()
    valid_brands = brand_seller_cnt[brand_seller_cnt >= min_sellers_per_brand].index
    df = df[df["brand"].isin(valid_brands)]

    pivot = df.pivot_table(index="seller_id", columns="brand",
                           values="n", fill_value=0)
    X_raw = pivot.values.astype(float)
    N = X_raw.shape[0]

    TF = np.log1p(X_raw)
    IDF = np.log(N / ((X_raw > 0).sum(axis=0) + 1)) + 1
    X_tfidf = normalize(TF * IDF, norm="l2")

    return X_tfidf, pivot.index.tolist(), pivot.columns.tolist(), X_raw


def run(min_listings_per_seller: int = 5,
        min_sellers_per_brand: int = 3,
        svd_components: int = 15,
        umap_components: int = 10,
        min_cluster_size: int = 5) -> dict:
    """H1 분석 메인 흐름 — 셀러×브랜드 공동출현 행렬 기반.

    텍스트 TF-IDF 대신 브랜드 포트폴리오 행렬을 직접 구성:
      - 투명성: 피처가 브랜드명 그 자체 → 클러스터 해석 직관적
      - 정확성: 브랜드 반복 텍스트 trick 없이 실제 취급 패턴 반영
      - 파이프라인: 브랜드 행렬 → SVD → UMAP → HDBSCAN
    """
    import warnings
    warnings.filterwarnings("ignore")

    utils.section("H1: 셀러 시그니처 클러스터링 (브랜드 공동출현 행렬)")
    utils.setup_korean_font()

    # 1. 데이터 로드
    listings = load_listings()
    if listings.empty:
        print("  ✗ 매물 데이터 없음 — 수집 후 재실행")
        return {}

    # brand NULL 50%+ 셀러 제외
    brand_null_rate = listings.groupby("seller_id")["brand"].apply(
        lambda x: x.isna().mean()
    )
    brand_unreliable = set(brand_null_rate[brand_null_rate > 0.5].index)
    if brand_unreliable:
        utils.bullet("brand 신뢰 불가 셀러 제외", f"{len(brand_unreliable)}명")
        listings = listings[~listings["seller_id"].isin(brand_unreliable)]

    # 2. 브랜드 공동출현 행렬
    X_tfidf, sellers, brand_names, X_raw = build_brand_matrix(
        listings, min_listings_per_seller, min_sellers_per_brand
    )
    n_sellers = len(sellers)
    utils.bullet("분석 대상 셀러", f"{n_sellers}명 (매물 {min_listings_per_seller}+ 보유)")
    utils.bullet("브랜드 피처", f"{len(brand_names)}개 (셀러 {min_sellers_per_brand}명+ 취급)")
    utils.bullet("행렬 밀도", f"{(X_raw > 0).mean():.3f}")

    if n_sellers < 10:
        print(f"  ✗ 셀러 수 부족 ({n_sellers} < 10)")
        return {"status": "insufficient_data", "n_sellers": n_sellers}

    # 3. SVD 차원 축소
    utils.section("SVD 차원 축소")
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)
    explained = svd.explained_variance_ratio_.sum()
    utils.bullet(f"SVD {svd_components}차원", f"설명 분산 {explained:.1%}")

    # 4. HDBSCAN — SVD 공간에서 직접 클러스터링
    # UMAP → HDBSCAN 파이프라인은 UMAP이 로컬 구조를 과도하게 확대해
    # 실제로 응집되지 않은 셀러도 전문형으로 분류하는 과클러스터링 문제 발생
    # (SVD 공간 실루엣 0.117로 확인) → SVD 공간에서 직접 클러스터링
    utils.section("HDBSCAN 클러스터링 (SVD 공간)")
    _, hdb_labels = hdbscan_clusters(X_svd, min_cluster_size=min_cluster_size)
    if hdb_labels is None:
        return {"status": "hdbscan_unavailable"}

    # 5. 내부 응집도 낮은 클러스터 강등 (잡화형으로 재분류)
    # 브랜드 행렬이 sparse해 HDBSCAN이 밀도 희박 클러스터를 만들 수 있음
    # 내부 cosine similarity < 0.3 → 실제 공유 브랜드 없는 인위적 클러스터로 판단
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
    hdb_labels = np.array(hdb_labels)
    demoted = 0
    for cid in sorted(set(hdb_labels)):
        if cid == -1:
            continue
        mask = hdb_labels == cid
        if mask.sum() < 3:
            continue
        X_c = X_svd[mask]
        sim = _cos_sim(X_c)
        np.fill_diagonal(sim, np.nan)
        mean_sim = float(np.nanmean(sim))
        if mean_sim < 0.3:
            hdb_labels[mask] = -1
            demoted += 1
    if demoted:
        utils.bullet("응집도 미달 클러스터 잡화형 강등", f"{demoted}개 (내부 유사도 < 0.3)")

    n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    n_generalist = int((hdb_labels == -1).sum())
    n_specialist = n_sellers - n_generalist
    utils.bullet("발견된 클러스터 수", n_clusters)
    utils.bullet("전문형 셀러", f"{n_specialist}명 ({n_specialist/n_sellers:.1%})")
    utils.bullet("잡화형 셀러 (noise)", f"{n_generalist}명 ({n_generalist/n_sellers:.1%})")

    # 6. 클러스터 해석 — 브랜드 TF-IDF 평균으로 대표 브랜드 추출
    utils.section("클러스터별 대표 브랜드")
    cluster_summary = {}
    label_arr = np.array(hdb_labels)
    X_tfidf_arr = np.array(X_tfidf)

    for c in sorted(set(hdb_labels)):
        if c == -1:
            continue
        mask = label_arr == c
        n_c = int(mask.sum())

        # 클러스터 내 평균 TF-IDF → 상위 브랜드
        mean_vec = X_tfidf_arr[mask].mean(axis=0)
        top_idx = mean_vec.argsort()[::-1][:10]
        top_brands = [brand_names[i] for i in top_idx]

        # top-3 브랜드 집중도 (매물 기준)
        cluster_sellers = [sellers[i] for i in range(len(sellers)) if mask[i]]
        sub = listings[listings["seller_id"].isin(cluster_sellers)]["brand"].dropna()
        top3_share = (sub.value_counts().head(3).sum() / len(sub)) if len(sub) else 0

        utils.bullet(
            f"클러스터 {c} ({n_c}명, top3={top3_share:.0%})",
            f"{', '.join(top_brands[:4])}"
        )
        cluster_summary[int(c)] = {
            "n_sellers": n_c,
            "top3_brand_share": round(top3_share, 3),
            "top_brands": top_brands,
        }

    # 7. 셀러-클러스터 라벨 저장
    seller_clusters = pd.DataFrame({
        "seller_id": sellers,
        "cluster": hdb_labels,
    })
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    seller_clusters.to_parquet(CACHE_DIR / "seller_clusters.parquet", index=False)
    utils.bullet("셀러-클러스터 라벨 저장",
                 str(CACHE_DIR / "seller_clusters.parquet"))

    # 8. 결과 저장
    payload = {
        "n_sellers": n_sellers,
        "n_clusters": n_clusters,
        "n_specialist": n_specialist,
        "n_generalist": n_generalist,
        "specialist_rate": round(n_specialist / n_sellers, 4),
        "svd_explained_variance": round(explained, 4),
        "clusters": cluster_summary,
    }
    utils.save_result("h1_clustering", payload)

    # 9. SVD 2D 시각화
    try:
        plot_clusters(X_svd, hdb_labels,
                      pd.DataFrame({"seller_id": sellers}))
    except Exception as e:
        print(f"  시각화 스킵: {e}")

    return payload


def plot_clusters(X_svd, labels, seller_text_df):
    """SVD 2D 산점도 (run()이 이미 SVD를 수행한 행렬을 전달).

    noise(-1) 셀러는 회색 반투명으로 표시해 잡화형 셀러 비중을 시각화.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import TruncatedSVD

    # 시각화용 2차원 추가 축소
    svd2 = TruncatedSVD(n_components=2, random_state=42)
    X2 = svd2.fit_transform(X_svd)

    fig, ax = plt.subplots(figsize=(11, 8))
    # noise 먼저 그려서 클러스터가 위에 오게
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(X2[noise_mask, 0], X2[noise_mask, 1],
                   color="lightgray", alpha=0.3, s=20, label="잡화형 (noise)")
    unique_clusters = sorted(c for c in set(labels) if c != -1)
    for c in unique_clusters:
        mask = labels == c
        ax.scatter(X2[mask, 0], X2[mask, 1], label=f"C{c}", alpha=0.8, s=60)
    ax.set_xlabel("SVD Component 1")
    ax.set_ylabel("SVD Component 2")
    n_clusters = len(unique_clusters)
    n_noise = noise_mask.sum()
    ax.set_title(f"Seller Signature Clusters — HDBSCAN\n"
                 f"{n_clusters}개 클러스터 / 잡화형 {n_noise}명 ({n_noise/len(labels):.0%})")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    utils.save_figure(fig, "h1_clusters_2d")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--min-listings", type=int, default=5)
    p.add_argument("--min-sellers-per-brand", type=int, default=3)
    p.add_argument("--svd-components", type=int, default=15)
    p.add_argument("--umap-components", type=int, default=10)
    p.add_argument("--min-cluster-size", type=int, default=5)
    args = p.parse_args()
    run(
        min_listings_per_seller=args.min_listings,
        min_sellers_per_brand=args.min_sellers_per_brand,
        svd_components=args.svd_components,
        umap_components=args.umap_components,
        min_cluster_size=args.min_cluster_size,
    )
