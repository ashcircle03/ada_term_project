# %% [markdown]
# # 07 · 신규 셀러 초기 안내 PoC · 위시리스트로 '유사 + 성공' 셀러 참조군 산출
#
# **목적.** 빈티지 1점물은 동일상품 시세·매칭이 불가능하다(당근·KREAM식 표준화 공식 적용 불가).
# 대신 FF만의 자산인 **공개 위시리스트(수요 측 취향 그래프)** 로, 셀링 이력이 0인 **신규 진입 셀러**에게
# "취향이 비슷하면서 잘 파는 기존 셀러"를 참조군으로 제시할 수 있는지
# 시연한다. 단면 자료 기반 프록시 검증이므로 실제 신규 셀러의 미래 판매 효과를 보인 것은 아니다.
#
# **전제(03 키스톤).** 위시-브랜드 분포와 셀링-브랜드 분포가 동일 사용자 내 횡단면에서 강하게
# 정렬돼 있다(코사인 무작위 대비 8.9×, `results/h3.json:wish_sell_taste`). 그래서 위시는 셀링 이력
# 부재 시 사용할 후보 신호가 될 수 있다.
#
# **무엇을 보이고 무엇을 안 보이나(정직).** (i) 수요 취향이 또렷한가, (ii) 취향 이웃으로 프로파일을
# 이용할 수 있는가, (iii) **위시로 검색한 유사 셀러가 실제 그 사람의 셀링 취향과도 가까운가(leave-one-out,
# 무작위 대비)**, (iv) 신규 셀러마다 '유사+성공' 레퍼런스가 실제로 산출되는가(도달률)이다.
# 즉 본 분석이 보이는 것은 **메커니즘 feasibility** 까지다. 단면 자료라 '진짜 신규 셀러의 위시→이후 판매'를 종단으로 본 건 아니며(기존
# 셀러로 프록시 타당성만 검증), 위시 주체는 셀러이고 지표는 브랜드 단위 근사다. 전환 효과는 A/B 몫.

# %%
import json, os, sqlite3
from pathlib import Path
import numpy as np, pandas as pd
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
matplotlib.rcParams["font.family"] = "AppleGothic"; matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
FIG = ROOT / "results" / "figures"; CACHE = ROOT / "data" / "cache"

lst = pd.read_parquet(CACHE / "features_listing.parquet")
sel = pd.read_parquet(CACHE / "features_seller.parquet")
with sqlite3.connect(ROOT / "data" / "fruitsfamily.db") as conn:
    wl = pd.read_sql_query("SELECT owner_seller_id, product_id FROM wishlist", conn)
brand_of = lst.set_index("product_id")["brand"]
wl["brand"] = wl["product_id"].map(brand_of)
wlm = wl.dropna(subset=["brand"]).copy()
print(f"위시 {len(wl):,}건 중 브랜드 매핑 {len(wlm):,} ({len(wlm)/len(wl):.0%}) | owner {wlm.owner_seller_id.nunique():,}명")

# %% [markdown]
# ## 1. 수요 취향은 또렷한가
#
# 매칭이 의미 있으려면 사용자의 '원하는 것'이 분산돼 있지 않고 집중돼 있어야 한다. 셀러가 *파는*
# 브랜드는 다양한데(공급), 사용자가 *찜하는* 브랜드는 집중적인지(수요)를 브랜드 집중도(HHI)로 비교한다.

# %%
def hhi(s):
    p = s.value_counts(normalize=True); return float((p**2).sum())
wish_hhi = wlm.groupby("owner_seller_id")["brand"].agg(hhi)
sell_hhi = sel.loc[sel.n_listings >= 5, "brand_hhi"].dropna()
print(f"위시(수요) 브랜드 HHI 중앙값: {wish_hhi.median():.3f}  (높을수록 취향 집중)")
print(f"판매(공급) 브랜드 HHI 중앙값: {sell_hhi.median():.3f}")
print(f"→ 공급은 다양({sell_hhi.median():.2f})하나 수요 취향은 또렷({wish_hhi.median():.2f}): 매칭의 신호가 존재")

# %% [markdown]
# ## 2. 위시로 찾은 유사 셀러가 실제 셀링과도 가까운가 (leave-one-out)
#
# 신규 셀러는 셀링이 없으므로 **위시 벡터**로 기존 셀러를 검색한다(= 내 *위시*와 닮은 *셀링*을 하는 셀러).
# 검증: 위시·셀링을 모두 가진 셀러를 신규처럼 취급해 자기 셀링을 가린 뒤, 위시로 찾은 상위 K명의
# **실제 셀링 취향**이 그 사람 본인의 셀링과 가까운지를 무작위 K명과 비교한다(둘 다 본인 셀링은 ground
# truth로만 사용, 검색엔 미사용 = 누수 없음). 또 그 K명 중 **전환율 상위 1/3(성공 셀러)** 이 몇 명
# 잡히는지로 '유사+성공' 레퍼런스 도달률을 본다.
#
# **설계 근거.** 진짜 신규 셀러의 '이후 판매'는 단면 자료라 관측 불가하므로, **위시·셀링을 모두 가진 기존
# 셀러(각 5건↑)** 를 신규처럼 취급해 위시→셀링 매핑의 타당성을 *프록시*로 검증한다(한계: 진짜 신규 셀러의
# 종단 관찰은 아님). 검색 키는 위시 벡터, 평가 기준(ground truth)은 본인 셀링 벡터로 분리해 누수를 막는다.
# K=30은 레퍼런스 카드에 보일 후보 풀 크기이며, '성공 셀러'는 참조 후보 풀의 전환율 상위 1/3
# (본 데이터 cutoff≈0.263)로 정의한다. 이 기준은 참조군 선별 규칙이지 신규 셀러의 기대 전환율
# 상승폭을 의미하지 않는다.

# %%
sell_cnt = lst.groupby(["seller_id", "brand"]).size().rename("n").reset_index()
wish_cnt = wlm.groupby(["owner_seller_id", "brand"]).size().rename("n").reset_index()
s_tot = lst.groupby("seller_id").size(); w_tot = wlm.groupby("owner_seller_id").size()
est = sorted(s_tot[s_tot >= 5].index)                              # 레퍼런스 후보(기존 셀러)
both = sorted(set(est) & set(w_tot[w_tot >= 5].index))            # 위시도 있는 셀러 = 신규로 시뮬
vocab = sorted(set(sell_cnt["brand"]) | set(wish_cnt["brand"]))
vi = {b: i for i, b in enumerate(vocab)}; ei = {s: i for i, s in enumerate(est)}
sc = sell_cnt[sell_cnt.seller_id.isin(ei)]
S = normalize(csr_matrix((sc.n, (sc.seller_id.map(ei), sc.brand.map(vi))), shape=(len(est), len(vocab))))
bidx2 = {s: i for i, s in enumerate(both)}
wc = wish_cnt[wish_cnt.owner_seller_id.isin(bidx2)]
W = normalize(csr_matrix((wc.n, (wc.owner_seller_id.map(bidx2), wc.brand.map(vi))), shape=(len(both), len(vocab))))
print(f"레퍼런스 후보(기존 셀러) {len(est):,}명 | 신규로 시뮬할 셀러(위시+셀링) {len(both):,}명")

st_est = sel.set_index("seller_id").reindex(est)["sell_through"].values
hi_bar = float(np.nanquantile(st_est, 2 / 3))                    # 전환율 상위 1/3 = '성공 셀러'
is_hi = st_est >= hi_bar
both_in_est = np.array([ei[s] for s in both])                    # 각 신규-시뮬 셀러의 est 위치(self)
print(f"성공 셀러 cutoff(참조 후보 전환율 상위 1/3): {hi_bar:.3f}")

K = 30
ST = S.T.tocsr()
rng = np.random.RandomState(0)
neigh_cos, rand_cos, hi_counts = [], [], []
for start in range(0, len(both), 512):
    Wc = W[start:start + 512]
    sims = (Wc @ ST).toarray()                                   # (chunk × est) 위시→셀링 유사도
    rows = np.arange(start, min(start + 512, len(both)))
    for rl, rg in enumerate(rows):
        sims[rl, both_in_est[rg]] = -1.0                         # self 제외
    topk = np.argpartition(-sims, K, axis=1)[:, :K]
    for rl, rg in enumerate(rows):
        my = both_in_est[rg]; nb = topk[rl]
        neigh_cos.append(float((S[my] @ S[nb].T).toarray().mean()))   # 이웃의 셀링 vs 내 셀링
        hi_counts.append(int(is_hi[nb].sum()))
        rd = rng.choice(len(est), K + 1, replace=False); rd = rd[rd != my][:K]
        rand_cos.append(float((S[my] @ S[rd].T).toarray().mean()))    # 무작위 K명 vs 내 셀링

neigh_cos = np.array(neigh_cos); rand_cos = np.array(rand_cos); hi_counts = np.array(hi_counts)
lift = float(neigh_cos.mean() / rand_cos.mean())
reach = {m: float((hi_counts >= m).mean()) for m in [1, 3, 5]}
print(f"위시로 찾은 상위{K} 이웃의 셀링↔내 셀링 코사인: {neigh_cos.mean():.3f}  vs  무작위 {rand_cos.mean():.3f}  (lift {lift:.1f}×)")
print(f"'유사+성공(전환 상위1/3, cutoff {hi_bar:.3f})' 레퍼런스 도달률: ≥1명 {reach[1]:.1%} · ≥3명 {reach[3]:.1%} · ≥5명 {reach[5]:.1%}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].bar(["무작위 K명", "위시-검색 K명"], [rand_cos.mean(), neigh_cos.mean()],
          color=["#aab", "#4a8c5f"])
ax[0].set(title=f"위시로 찾은 유사 셀러의 셀링 일치도 (무작위 대비 {lift:.1f}×)",
          ylabel="이웃 셀링 ↔ 본인 셀링 코사인(평균)")
for i, v in enumerate([rand_cos.mean(), neigh_cos.mean()]):
    ax[0].text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)
xs = ["≥1명", "≥3명", "≥5명"]
ax[1].bar(xs, [reach[1] * 100, reach[3] * 100, reach[5] * 100], color="#3a6ea5")
ax[1].set(title=f"'유사+성공' 레퍼런스 도달률 (성공 cutoff {hi_bar:.3f})", ylabel="% of 신규-시뮬 셀러", ylim=(0, 100))
for i, m in enumerate([1, 3, 5]):
    ax[1].text(i, reach[m] * 100 + 1.5, f"{reach[m]*100:.0f}%", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "p7_onboarding_reach.png", bbox_inches="tight"); plt.close(fig)

# %% [markdown]
# ## 3. 가격 밴드 커버리지 + 저장
#
# 가격 밴드(brand×category×condition 동종 분포)는 표준화 마켓도 하는 **기본기**. 5건 이상 동종 그룹으로
# 산출 가능한 매물 비율만 확인한다.

# %%
g = lst.dropna(subset=["brand", "category_l1"]).copy()
g["cond"] = g["condition"].fillna("NA")
grp_n  = g.groupby(["brand", "category_l1", "cond"])["product_id"].transform("size")
grp_bc = g.groupby(["brand", "category_l1"])["product_id"].transform("size")
band_cov = float((grp_n >= 5).mean())
band_cov_bc = float((grp_bc >= 5).mean())   # condition 없이 brand×cat 폴백
print(f"가격 밴드 산출 가능: brand×cat×cond {band_cov:.1%} / brand×cat {band_cov_bc:.1%}")

res = {
    "premise": "위시-셀링 브랜드 코사인 lift 8.9× (results/h3.json:wish_sell_taste). 동일 사용자 내 횡단면 정렬",
    "demand_taste_hhi_median": round(float(wish_hhi.median()), 3),
    "supply_sell_hhi_median": round(float(sell_hhi.median()), 3),
    "onboarding_poc": {
        "k_pool": K,
        "n_reference_pool_sellers": len(est),
        "n_simulated_new_sellers": len(both),
        "neighbor_sell_cos_mean": round(float(neigh_cos.mean()), 3),
        "random_sell_cos_mean": round(float(rand_cos.mean()), 3),
        "validation_lift": round(lift, 2),
        "success_bar_sellthrough_top33": round(hi_bar, 3),
        "reach_similar_and_successful_ref": {f"ge{m}": round(reach[m], 3) for m in [1, 3, 5]},
    },
    "price_band_coverage_bcc": round(band_cov, 3),
    "price_band_coverage_bc": round(band_cov_bc, 3),
    "caveats": "단면 자료(종단 아님; 기존 셀러로 프록시 검증); 성공 셀러 cutoff는 참조 후보 전환율 상위 1/3 규칙이지 신규 셀러 기대 전환율 상승폭 아님; 위시 주체=셀러(구매자 아님); 브랜드 단위 근사; comp 밴드는 시세 투명성 도구(가격위치는 sell-through 거의 못 가름, p3_staleness.json). 가격·할인 탄력성은 A/B 필요",
}
(ROOT / "results" / "taste_matching.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
