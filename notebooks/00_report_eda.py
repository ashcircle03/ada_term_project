# %% [markdown]
# # 00 · EDA · 문제 정의 (FruitsFamily의 낮은 판매 전환)
#
# **문제:** 빈티지 1점물 C2C에서 매물 대부분이 판매로 전환되지 않는다. 그런데 플랫폼은
# 전 셀러에게 동일한 수량형 등록 안내("사진 많이, 설명 길게")를 제공한다. 이 노트북은
# 낮은 판매 전환의 규모와, 수량형 안내가 원시 자료에서 기대와 반대로 보이는 현상을 확인한다.
# 이는 인과 결론이 아니라 H1/H2에서 검증할 문제 제기다.
#
# 결과변수: `is_sold` (매칭 성공). 속도 지표는 아니며, 연령으로도 안 팔리는 매물이 다수다.

# %% [markdown]
# ## 분석 설계: 변수 정의·전처리·선택 근거 및 유의수준
#
# 본 분석에서 사용한 변수와 그 전처리·선택 이유를 정리한다. 구체 구현은
# `analysis/build_features.py`, `analysis/featurelib.py` 참조.
#
# | 변수 | 정의 · 전처리 | 선택 / 처리 근거 |
# |---|---|---|
# | `is_sold` (결과) | 판매 완료 여부(0/1) | 유동성 문제의 직접 지표. 판매 시점(`sold_at`)이 없어 *속도* 대신 *전환*을 결과로 둠 |
# | `n_photos` | 사진 장수(0–10) | 셀러가 통제하는 표현 노력이자 플랫폼 가이드의 핵심 권고 |
# | `desc_len` | 설명 글자 수 | 표현 노력. '자세함'을 길이로 근사 |
# | `kw_*` (실측·하자·소재·구매·사용·세탁) | 해당 키워드 포함 여부(0/1) | 가이드가 권하는 '자세한 설명' 항목을 구체화. 단 '하자' 등은 매물 상태를 반영하는 내생 변수라 인과 해석에 주의 |
# | `relative_price_z` | 동종(brand×category_l1×condition) 내 `log1p(price_final)` z-score. 표본 n<5 또는 표준편차 0이면 brand×category_l1로 폴백, 산출 불가 시 결측표시 변수(`rel_price_missing`) 동반 | 정확한 시세가 아니라 관측 가능한 동종 그룹 기준의 가격 포지셔닝 근사. 결측은 미싱 인디케이터로 보존하여 표본 손실 방지 |
# | `brand` | 빈도 상위 30개 + 기타 | 6,765종의 고카디널리티 → 회귀/더미화를 위해 상위 빈도로 축약 |
# | `category_l1`, `category_l2` | 남/여, 하위 카테고리. H2 더미화 단계에서 `category_l2`만 상위 20 + 기타로 축약 | 매물 종류라는 구조 요인 통제. H3 토픽은 원 `category_l2`를 사용 |
# | `condition` | NEW/GOOD_CONDITION/LIGHTLY_WORN/WORN, 결측은 UNK | 등급 간 순서가 모호하여 순서형 대신 범주형으로 처리 |
# | `price_final`, `log_price` | 판매가와 그 로그 | 가격 분포의 우측 왜도가 커 로그 변환. 매물 가치(구조) 통제 |
# | `price_tier` | <3만 / <8만 / <20만 / 20만+ | 가격대별 이질성(셀러 통제력 차이) 분석용 구간 |
# | `age_days` | 관측일 − 등록일 | 오래된 매물일수록 노출 기간이 길었던 절단을 보정하기 위한 필수 통제. 모든 모형에 포함 |
# | `gender` | 분석에서 제외 | `category_l1`(남/여)과 사실상 동일하여 공선성 회피 |
# | `view_count`, `like_count` | 피처 테이블과 예측에서 제외 | 등록 이후 누적되는 사후 변수(누수)이므로 모델 입력으로 사용하지 않음 |
# | wishlist | owner→product, 브랜드 매핑 | 취향 네트워크·매칭 가능성. 위시 주체는 셀러이며 브랜드 단위 근사라는 한계 명시 |
#
# **표본 구성(행 드롭 회계).** 크롤링 수집 288,903건에서 (i) 상세 미수집 placeholder(`seller_id='_pending_'`)와
# (ii) 핵심 항목(등록일·설명·가격) 결측 행을 제외해 분석 표본 284,654건을 확정한다. 핵심 항목이 없으면
# 회귀·예측에 쓸 수 없어 행을 버리되, 그 외 변수의 결측은 위 표처럼 미싱 인디케이터·`UNK`로 보존해 표본
# 손실을 최소화했다.
#
# **표집·대표성 근거.** 시드는 카테고리·브랜드 페이지를 **최신 등록순(RECENT)** 으로 모았다. 인기순
# (POPULAR)은 동일 핫브랜드가 전 카테고리를 도배해 브랜드 다양성과 셀러 시그니처 군집을 무너뜨리기 때문이다
# (config.py `SEARCH_SORT`). 그 대가로 최근 등록이 과표집되므로, 78.5% 미판매가 표집·절단의 산물이 아님을
# 1b절 코호트 분석으로 별도 검증한다.
#
# **유의수준.** 표본이 약 28만 건으로 매우 커 작은 효과도 쉽게 유의해지므로, 유의수준은
# α=0.05로 두되 유의성만으로 판단하지 않고 효과크기(오즈비·AUC 차·ε²)와 신뢰구간을 함께
# 보고한다. 가설별 귀무가설과 기각 방법: (H1) 회귀계수=0을 Wald 검정으로, (H2) 모형 AUC=0.5를
# 라벨 순열검정으로·두 모형 ΔAUC=0을 부트스트랩 신뢰구간으로, (H3) NMF 토픽 간 전환율 동일을
# 크러스컬–월리스 검정과 효과크기 ε²로 검정하고, 위시-셀링 정렬은 무작위 짝지음 null과 비교한다.

# %%
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "AppleGothic"   # macOS 한글 폰트
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
CACHE = ROOT / "data" / "cache"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

lst = pd.read_parquet(CACHE / "features_listing.parquet")
sel = pd.read_parquet(CACHE / "features_seller.parquet")
print("listing:", lst.shape, "| seller:", sel.shape)
print("전체 전환율:", f"{lst['is_sold'].mean():.1%}")

# %% [markdown]
# ## 1. 낮은 판매 전환. 5개 중 4개는 미판매로 남는다
#
# 연령이 쌓여도 전환율은 ~23%에서 정체 → *속도*가 아니라 **매칭** 문제.

# %%
bins = [0, 7, 30, 90, 365, np.inf]
labels = ["0-7d", "7-30d", "30-90d", "90-365d", "365d+"]
lst["age_bucket"] = pd.cut(lst["age_days"], bins=bins, labels=labels, right=False)
by_age = lst.groupby("age_bucket", observed=True)["is_sold"].agg(["mean", "size"])
print(by_age.assign(mean=lambda d: (d["mean"]*100).round(1)))

dead = lst[lst["is_sold"] == 0]
dead90 = (dead["age_days"] > 90).mean()
print(f"\n전체 미판매율: {1-lst['is_sold'].mean():.1%}")
print(f"미판매 중 90일+ 방치: {dead90:.1%}")

# 보고서에는 수치만 사용한다. 보조 시각화는 archive/non_report/notebooks/pre_cleanup에 보존했다.

# %% [markdown]
# ## 1b. 코호트 강건성. 미판매율이 표집·절단의 산물인가
#
# RECENT 정렬로 최근 매물이 과표집됐을 수 있다(아래 2026 코호트가 57%). 이때 78.5% 미판매가
# 단지 "최근 등록이라 아직 안 팔린 것(절단)"의 산물일 가능성을 배제해야 한다. 등록 코호트별
# 미판매율이 일정하고, 1년 이상 노출된 성숙 매물에서도 전환율이 더 오르지 않는다면, 미판매는
# 표집·절단이 아닌 실제 유동성 문제다.

# %%
import sqlite3
with sqlite3.connect(ROOT / "data" / "fruitsfamily.db") as conn:
    cdf = pd.read_sql_query(
        "SELECT is_sold, created_at, "
        "julianday('2026-05-29')-julianday(created_at) AS age_days "
        "FROM listing WHERE seller_id!='_pending_' AND created_at IS NOT NULL "
        "AND price_final IS NOT NULL AND price_final>0", conn)
cdf["yr"] = cdf["created_at"].str[:4]
by_yr = cdf.groupby("yr")["is_sold"].agg(n="size", unsold=lambda s: 1 - s.mean())
print("등록 코호트별 미판매율:"); print((by_yr.assign(unsold=lambda d: (d.unsold*100).round(1))).to_string())
mat = cdf[cdf.age_days >= 365]; rec = cdf[cdf.age_days < 365]
print(f"\n성숙(>=1년, n={len(mat):,}) 미판매 {1-mat.is_sold.mean():.3f} "
      f"vs 최근(<1년, n={len(rec):,}) 미판매 {1-rec.is_sold.mean():.3f} (거의 동일)")
mm = mat.copy(); mm["ab"] = pd.cut(mm.age_days, [365, 730, 1095, 1e9], labels=["1~2년", "2~3년", "3년+"])
plateau = mm.groupby("ab", observed=True)["is_sold"].mean()
print("성숙 매물 연령대별 전환율(노출 더 길어도 정체):"); print((plateau*100).round(1).to_string())

# 보고서에는 코호트별 미판매율과 성숙 매물 수치만 사용한다.

cohort_summary = {
    "unsold_by_year": {k: round(v*100, 1) for k, v in by_yr["unsold"].items()},
    "matured_ge1y_unsold": round(float(1-mat.is_sold.mean()), 3),
    "recent_lt1y_unsold": round(float(1-rec.is_sold.mean()), 3),
    "recent_lt1y_share": round(float(len(rec)/len(cdf)), 3),
    "matured_sellthrough_by_age": {str(k): round(v, 3) for k, v in plateau.items()},
}

# %% [markdown]
# ## 2. 셀러 불평등. 16%는 단 한 건도 못 판다

# %%
zero = (sel["n_sold"] == 0).mean()
sold_sorted = np.sort(sel["n_sold"].values)[::-1]
top10_share = sold_sorted[:int(len(sel)*0.1)].sum() / sold_sorted.sum()
print(f"zero-sale 셀러: {zero:.1%}")
print(f"상위 10% 셀러가 전체 판매의 {top10_share:.1%} 차지")

# Lorenz curve 수치는 판매 집중도 확인용이다. 보고서에는 zero-sale 비율만 직접 사용한다.
cum = np.cumsum(np.sort(sel["n_sold"].values)) / sel["n_sold"].sum()
x = np.linspace(0, 1, len(cum))

# %% [markdown]
# ## 3. 핵심 모순. 수량형 안내가 원시 자료에서 기대와 반대로 보인다
#
# "사진 많이"는 전 가격대에서 **역방향**, "설명 길게"도 마찬가지.
# 단, 이 결과는 raw 상관이다. 역선택(안 팔릴 매물에 노력 집중) 가능성은 H1/H2에서 분리한다.

# %%
lst["photo_grp"] = pd.cut(lst["n_photos"], [-1, 2, 5, 10], labels=["≤2", "3-5", "6+"])
lst["desc_grp"] = pd.cut(lst["desc_len"], [-1, 50, 150, 300, np.inf],
                         labels=["<50", "50-150", "150-300", "300+"])

print("사진수 구간 × 가격대 전환율:")
piv = lst.pivot_table("is_sold", "price_tier", "photo_grp", observed=True)
print((piv*100).round(1))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
(piv*100).plot(kind="bar", ax=ax[0], rot=0)
ax[0].set(title="사진수↑ → 전환율↓ (모든 가격대)", ylabel="sold %", xlabel="price tier")
ax[0].legend(title="photos", fontsize=8)
(lst.groupby("desc_grp", observed=True)["is_sold"].mean()*100).plot(
    kind="bar", ax=ax[1], rot=0, color="#7a5")
ax[1].set(title="설명 길이↑ → 전환율↓", ylabel="sold %", xlabel="desc length")
fig.tight_layout(); fig.savefig(FIG / "eda_paradox.png", bbox_inches="tight"); plt.close(fig)

# %% [markdown]
# ## 4. 보고서 EDA 요약 저장

# %%
summary = {
    "n_listings": int(len(lst)),
    "n_sellers": int(len(sel)),
    "overall_sell_through": round(float(lst["is_sold"].mean()), 4),
    "unsold_pct": round(float(1 - lst["is_sold"].mean()), 4),
    "dead_stock_90d_pct": round(float(dead90), 4),
    "cohort_robustness": cohort_summary,
    "zero_sale_seller_pct": round(float(zero), 4),
    "top10pct_sales_share": round(float(top10_share), 4),
    "sold_by_age_bucket": {k: round(v*100, 1) for k, v in by_age["mean"].items()},
    "sold_by_photo_grp": {str(k): round(v*100, 1) for k, v in
                          lst.groupby("photo_grp", observed=True)["is_sold"].mean().items()},
    "sold_by_desc_grp": {str(k): round(v*100, 1) for k, v in
                         lst.groupby("desc_grp", observed=True)["is_sold"].mean().items()},
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "eda.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n보고서 핵심 EDA: 미판매율, 90일 초과 미판매, zero-sale 셀러, 사진·설명 원시 역설")
