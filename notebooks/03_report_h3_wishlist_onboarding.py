# %% [markdown]
# # 03 · H3 · NMF 기반 셀러 취향 토픽과 초기 안내 신호
#
# **가설.** 셀러를 선명한 군집으로 나누기보다, 브랜드와 하위 카테고리 조합에서 반복되는
# 취향 토픽을 추정하면 판매 전환 차이를 더 해석 가능하게 설명할 수 있다. 이 토픽은 신규
# 셀러에게 유사 참조군을 찾는 후보 신호가 된다. 단 위시리스트의 찜 시각이 없으므로,
# 이 노트북은 콜드스타트를 해결했다고 주장하지 않고 내부 실험으로 검증할 초기 안내 가설을 만든다.
#
# **방법.** 매물 5건 이상 셀러의 `brand`와 `brand|category_l2` 토큰을 셀러별 sparse vector로
# 만들고, TF-IDF 변환 뒤 NMF를 적합한다. 토픽 수 선택 규칙에는 판매 전환율을 사용하지 않는다.
# 각 셀러는 가장 큰 topic weight의 토픽으로 요약하되,
# dominance를 함께 저장해 혼합 취향 셀러가 많다는 점을 드러낸다. 결과변수 `sell_through`는
# 토픽 학습 피처에서 제외하고 사후 검정에만 사용한다. leave-one-out 검증은 실제 신규 셀러가
# 아니라 위시와 셀링을 모두 가진 기존 셀러를 프록시로 쓰므로, 적용 범위는 별도 위시 보유율로 확인한다.
#
# 산출은 `results/h3.json`, `data/cache/seller_clusters.parquet`, H3 관련 그림이다.

# %%
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ada-matplotlib-cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.stats import kruskal
import statsmodels.api as sm
from sklearn.decomposition import NMF
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
FIG = ROOT / "results" / "figures"
CACHE = ROOT / "data" / "cache"

sel = pd.read_parquet(CACHE / "features_seller.parquet")
lst = pd.read_parquet(CACHE / "features_listing.parquet")

sel_c = sel[sel["n_listings"] >= 5].copy()
lst_c0 = lst[lst["seller_id"].isin(sel_c["seller_id"])].copy()
print(f"전체 셀러 {len(sel):,} → 토픽 대상(매물5+) {len(sel_c):,} ({len(sel_c)/len(sel):.0%})")

# %% [markdown]
# ## 1. 브랜드×하위카테고리 NMF 토픽
#
# 빈티지 셀러의 취향은 하나의 배타적 집단보다 여러 브랜드·카테고리 묶음의 조합에
# 가깝다. 비음수 토픽 가중치로 한 셀러가 여러 취향을 얼마나 섞어 갖는지 표현하고,
# 각 토픽은 상위 브랜드로 해석한다. 토픽 수는
# 전환율을 보지 않는 균형·주도성 기준으로만 정해 결과 누수를 막는다.

# %%
def _safe_token_series(s):
    return s.fillna("UNK").astype(str).str.strip().str.replace(" ", "_", regex=False).str.slice(0, 80)


def _token_dicts(frame, id_col, brand_col="brand", category_col="category_l2"):
    base = frame[[id_col, brand_col, category_col]].copy()
    brand = _safe_token_series(base[brand_col])
    cat2 = _safe_token_series(base[category_col])
    long = pd.concat(
        [
            pd.DataFrame({id_col: base[id_col].values, "token": "brand=" + brand}),
            pd.DataFrame({id_col: base[id_col].values, "token": brand + "|" + cat2}),
        ],
        ignore_index=True,
    )
    counts = long.groupby([id_col, "token"], sort=True).size().rename("n").reset_index()
    return [
        (sid, dict(zip(g["token"], g["n"])))
        for sid, g in counts.groupby(id_col, sort=True)
    ]


def _seller_token_dicts(listings):
    rows = _token_dicts(listings, "seller_id")
    seller_ids = [r[0] for r in rows]
    vec = DictVectorizer(dtype=float)
    counts = vec.fit_transform([r[1] for r in rows])
    tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)
    X = tfidf.fit_transform(counts)
    return seller_ids, np.array(vec.get_feature_names_out()), X, vec, tfidf


seller_ids, token_names, X_topic, topic_vectorizer, topic_tfidf = _seller_token_dicts(lst_c0)
sell_y = sel_c.set_index("seller_id").loc[seller_ids, "sell_through"].values


def _topic_eval(k):
    model = NMF(
        n_components=k,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=800,
        random_state=42,
    )
    W = model.fit_transform(X_topic)
    labels = W.argmax(axis=1)
    dominance = W.max(axis=1) / (W.sum(axis=1) + 1e-12)
    groups = [sell_y[labels == t] for t in range(k) if (labels == t).sum() >= 2]
    H, p = kruskal(*groups)
    eps2 = (H - len(groups) + 1) / (len(sell_y) - len(groups))
    st = [float(g.mean()) for g in groups]
    counts = np.bincount(labels, minlength=k)
    sil = silhouette_score(W, labels, sample_size=min(5000, W.shape[0]), random_state=42)
    max_share = counts.max() / len(labels)
    return {
        "k": int(k),
        "model": model,
        "W": W,
        "labels": labels,
        "dominance": dominance,
        "silhouette_w": round(float(sil), 3),
        "kw_H": float(H),
        "kw_p": float(p),
        "kw_epsilon_sq": round(float(eps2), 3),
        "sellthrough_range": round(float(max(st) - min(st)), 3),
        "dominance_median": round(float(np.median(dominance)), 3),
        "min_topic_size": int(counts.min()),
        "max_topic_size": int(counts.max()),
        "max_topic_share": round(float(max_share), 3),
        "reconstruction_error": round(float(model.reconstruction_err_), 3),
    }


topic_runs = [_topic_eval(k) for k in range(3, 9)]
selection_rule = {
    "min_topic_size": 300,
    "max_topic_share": 0.35,
    "min_dominance_median": 0.55,
    "tie_break": "smallest k satisfying unsupervised balance and dominance constraints",
}
eligible = [
    r for r in topic_runs
    if r["min_topic_size"] >= selection_rule["min_topic_size"]
    and r["max_topic_share"] <= selection_rule["max_topic_share"]
    and r["dominance_median"] >= selection_rule["min_dominance_median"]
]
if not eligible:
    eligible = [
        r for r in topic_runs
        if r["min_topic_size"] >= selection_rule["min_topic_size"]
        and r["dominance_median"] >= selection_rule["min_dominance_median"]
    ]
selected = sorted(eligible, key=lambda r: r["k"])[0]

nmf = selected["model"]
W = selected["W"]
labels = selected["labels"]
dominance = selected["dominance"]
best_k = selected["k"]
sel_c = sel_c.set_index("seller_id").loc[seller_ids].reset_index()
sel_c["archetype"] = labels
sel_c["topic_dominance"] = dominance

print("NMF topic candidates:")
for r in topic_runs:
    print(
        f"  k={r['k']} silW={r['silhouette_w']:.3f} eps={r['kw_epsilon_sq']:.3f} "
        f"range={r['sellthrough_range']:.3f} dom_med={r['dominance_median']:.3f} "
        f"min={r['min_topic_size']} max={r['max_topic_size']} max_share={r['max_topic_share']:.3f}"
    )
print(f"selected k={best_k} by unsupervised rule: {selection_rule}")

topic_top_terms = {}
for t, comp in enumerate(nmf.components_):
    topic_top_terms[int(t)] = token_names[np.argsort(comp)[-8:][::-1]].tolist()

# %% [markdown]
# ## 2. 토픽 프로필과 전환율 차이

# %%
lst_c = lst.merge(sel_c[["seller_id", "archetype", "topic_dominance"]], on="seller_id", how="inner")
profile = sel_c.groupby("archetype").agg(
    n_sellers=("seller_id", "size"),
    sell_through=("sell_through", "mean"),
    dominance=("topic_dominance", "median"),
    brand_hhi=("brand_hhi", "mean"),
    share_men=("share_men", "mean"),
    share_new=("share_new", "mean"),
    median_price=("median_price", "median"),
    avg_n_photos=("avg_n_photos", "mean"),
).round(3)
profile["top_terms"] = profile.index.map(lambda a: ", ".join(topic_top_terms.get(int(a), [])[:6]))
print(profile.to_string())

groups = [g["sell_through"].values for _, g in sel_c.groupby("archetype")]
H, p = kruskal(*groups)
eps2 = (H - best_k + 1) / (len(sel_c) - best_k)
print(f"Kruskal-Wallis H={H:.1f}, p={p:.2e}, eps2={eps2:.3f}")


def _short_topic_label(a):
    terms = topic_top_terms.get(int(a), [])[:3]
    terms = [t.replace("brand=", "").replace("_", " ") for t in terms]
    return "\n".join(terms)


fig, ax = plt.subplots(figsize=(8.4, 4.4))
order = profile.sort_values("sell_through").index
bars = ax.bar(
    [f"T{a}\n{_short_topic_label(a)}" for a in order],
    profile.loc[order, "sell_through"] * 100,
    color="#4a8c5f",
)
for i, a in enumerate(order):
    ax.text(
        i,
        profile.loc[a, "sell_through"] * 100 + 0.4,
        f"{profile.loc[a, 'sell_through']*100:.1f}%\nn={profile.loc[a, 'n_sellers']}",
        ha="center",
        fontsize=8,
    )
ax.set(
    title=f"NMF 셀러 취향 토픽별 sell-through (k={best_k}, eps2={eps2:.3f})",
    ylabel="sell-through %",
    xlabel="",
)
ax.tick_params(axis="x", labelsize=7)
fig.tight_layout()
fig.savefig(FIG / "h3_nmf_topics.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 3. 위시 취향이 셀링 취향과 정렬되는가
#
# 공개 위시리스트는 신규 셀러의 초기 안내에 사용할 후보 신호다. 단 찜 시각이 없어 시간적
# 선행성을 보장하지 못하므로, 여기서는 인과가 아니라 같은 사용자 안에서 위시 브랜드 분포와
# 셀링 브랜드 분포가 무작위 짝지음보다 얼마나 가까운지만 확인한다.

# %%
with sqlite3.connect(ROOT / "data" / "fruitsfamily.db") as conn:
    wl = pd.read_sql_query("SELECT owner_seller_id, product_id FROM wishlist", conn)

brand_map = lst.set_index("product_id")["brand"]
wl["brand"] = wl["product_id"].map(brand_map)
wlm = wl.dropna(subset=["brand"])
n_owners = wlm["owner_seller_id"].nunique()
brand_cov = len(wlm) / len(wl)
print(f"wishlist owner {n_owners:,} | 브랜드 매핑된 찜 {len(wlm):,} ({brand_cov:.0%})")

sell_cnt = lst.groupby(["seller_id", "brand"]).size().rename("n").reset_index()
wish_cnt = wlm.groupby(["owner_seller_id", "brand"]).size().rename("n").reset_index()
s_tot = lst.groupby("seller_id").size()
w_tot = wlm.groupby("owner_seller_id").size()
both = sorted(set(s_tot[s_tot >= 5].index) & set(w_tot[w_tot >= 5].index))
vocab = sorted(set(sell_cnt["brand"]) | set(wish_cnt["brand"]))
vi = {b: i for i, b in enumerate(vocab)}
bi = {s: i for i, s in enumerate(both)}
sc = sell_cnt[sell_cnt.seller_id.isin(bi)]
wc = wish_cnt[wish_cnt.owner_seller_id.isin(bi)]
S = normalize(csr_matrix((sc.n, (sc.seller_id.map(bi), sc.brand.map(vi))), shape=(len(both), len(vocab))))
Ww = normalize(csr_matrix((wc.n, (wc.owner_seller_id.map(bi), wc.brand.map(vi))), shape=(len(both), len(vocab))))
cos = np.asarray(S.multiply(Ww).sum(axis=1)).ravel()
rng = np.random.RandomState(0)
null_means = [
    np.asarray(S.multiply(Ww[rng.permutation(Ww.shape[0])]).sum(axis=1)).ravel().mean()
    for _ in range(20)
]
null_mean = float(np.mean(null_means))
lift = float(cos.mean() / null_mean)
print(f"위시-셀링 브랜드 코사인: 중앙값 {np.median(cos):.3f}, 평균 {cos.mean():.3f}, >0 {100*(cos>0).mean():.1f}%")
print(f"무작위 null 평균 {null_mean:.3f} → lift {lift:.1f}배 (n={len(both):,})")

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.hist(cos, bins=40, color="#4a8c5f", alpha=0.85, label="관측(같은 셀러의 위시 vs 셀링)")
ax.axvline(null_mean, color="#c0504d", ls="--", lw=2, label=f"무작위 짝지음 평균 {null_mean:.2f}")
ax.axvline(np.median(cos), color="#333", ls=":", lw=1.5, label=f"관측 중앙값 {np.median(cos):.2f}")
ax.set(title=f"위시 취향 ↔ 셀링 취향 유사도 (브랜드, 무작위 대비 {lift:.1f}×)",
       xlabel="코사인 유사도", ylabel="셀러 수")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "h3_wish_sell_taste.png", bbox_inches="tight")
plt.close(fig)

WST = {
    "n_sellers_both": int(len(both)),
    "brand_map_coverage": round(float(brand_cov), 3),
    "cosine_median": round(float(np.median(cos)), 3),
    "cosine_mean": round(float(cos.mean()), 3),
    "frac_positive": round(float((cos > 0).mean()), 3),
    "null_mean": round(null_mean, 3),
    "lift_over_null": round(lift, 2),
}

# 같은 NMF 토픽 공간으로 위시와 셀링을 투영해 취향 정렬을 재점검한다.
# 이는 brand-level cosine보다 해석 가능한 저차원 취향 신호가 참조군 산출에 쓸 수 있는지 보는 보조 검증이다.
product_meta = lst.set_index("product_id")[["brand", "category_l2"]]
wl_topic = wl.join(product_meta, on="product_id", rsuffix="_mapped").dropna(subset=["brand_mapped"])

def _wish_topic_dicts(wish_rows):
    rows = _token_dicts(wish_rows, "owner_seller_id", brand_col="brand_mapped")
    ids = [r[0] for r in rows]
    Xc = topic_vectorizer.transform([r[1] for r in rows])
    Xt = topic_tfidf.transform(Xc)
    return ids, Xt

wish_topic_ids, X_wish_topic = _wish_topic_dicts(wl_topic)
wish_topic_W = nmf.transform(X_wish_topic)
sell_topic_index = {sid: i for i, sid in enumerate(seller_ids)}
wish_topic_index = {sid: i for i, sid in enumerate(wish_topic_ids)}
topic_both = sorted(
    set(s for s in both if s in sell_topic_index)
    & set(s for s in both if s in wish_topic_index)
)
sell_topic_mat = normalize(W[[sell_topic_index[s] for s in topic_both]])
wish_topic_mat = normalize(wish_topic_W[[wish_topic_index[s] for s in topic_both]])
topic_cos = np.asarray((sell_topic_mat * wish_topic_mat).sum(axis=1)).ravel()
rt = np.random.RandomState(0)
topic_null = float(np.mean([
    np.asarray((sell_topic_mat * wish_topic_mat[rt.permutation(wish_topic_mat.shape[0])]).sum(axis=1)).ravel().mean()
    for _ in range(20)
]))
topic_lift = float(topic_cos.mean() / topic_null)
print(
    f"NMF topic 위시-셀링 코사인: 중앙값 {np.median(topic_cos):.3f}, "
    f"무작위 null {topic_null:.3f} → lift {topic_lift:.1f}배 (n={len(topic_both):,})"
)
WST.update({
    "topic_n_sellers_both": int(len(topic_both)),
    "topic_cosine_median": round(float(np.median(topic_cos)), 3),
    "topic_cosine_mean": round(float(topic_cos.mean()), 3),
    "topic_null_mean": round(topic_null, 3),
    "topic_lift_over_null": round(topic_lift, 2),
})

# %% [markdown]
# ## 4. 신규·저이력 셀러의 위시 신호 보유율
#
# 위시 기반 참조군은 실제 신규 셀러에게 위시리스트가 있어야 적용할 수 있다. leave-one-out
# 검증 대상은 위시와 셀링을 모두 가진 기존 셀러이므로, 별도로 판매 0건과 저이력 셀러의
# 브랜드 매핑 가능 위시 수를 확인한다. 위시가 부족한 매우 초기 셀러에게는 온보딩 취향 입력이
# 보완 신호가 되어야 한다.

# %%
seller_wish = sel[["seller_id", "n_listings", "n_sold"]].set_index("seller_id").copy()
seller_wish["brand_mapped_wishes"] = (
    w_tot.reindex(seller_wish.index).fillna(0).astype(int)
)


def _wishlist_availability(mask):
    g = seller_wish.loc[mask].copy()
    w = g["brand_mapped_wishes"]
    return {
        "n_sellers": int(len(g)),
        "any_wish_rate": round(float((w >= 1).mean()), 3),
        "ge5_wish_rate": round(float((w >= 5).mean()), 3),
        "wish_count_median": round(float(w.median()), 1),
        "wish_count_p75": round(float(w.quantile(0.75)), 1),
        "wish_count_p90": round(float(w.quantile(0.90)), 1),
    }


low_history_wishlist = {
    "sold_eq0": _wishlist_availability(seller_wish["n_sold"] == 0),
    "sold_le2": _wishlist_availability(seller_wish["n_sold"] <= 2),
    "listings_le2": _wishlist_availability(seller_wish["n_listings"] <= 2),
}
print("신규·저이력 셀러 위시 보유율:")
for k, v in low_history_wishlist.items():
    print(
        f"  {k:12s} n={v['n_sellers']:,} | any={v['any_wish_rate']:.1%} "
        f"| >=5={v['ge5_wish_rate']:.1%} | median={v['wish_count_median']:.1f}"
    )

# %% [markdown]
# ## 5. 위시 기반 참조군 검색 PoC
#
# 신규 셀러의 실제 미래 판매는 단면 자료로 관측할 수 없다. 따라서 위시와 셀링을 모두 가진
# 기존 셀러를 신규처럼 취급해 셀링을 가리고, 위시 벡터만으로 유사한 기존 셀러를 찾는다.
# 검색 결과의 실제 셀링 취향이 본인 셀링과 가까운지, 그리고 전환율 상위 1/3 참조 셀러가
# 도달 가능한지를 본다. 다만 상위 1/3 셀러를 30명 안에 1명 이상 포함하는 도달률은 무작위
# 기준에서도 거의 100%에 가까우므로 성과 지표가 아니다. 핵심 검증은 검색 이웃의 실제 셀링
# 취향이 무작위 이웃보다 얼마나 가까운지다.

# %%
def hhi(s):
    p = s.value_counts(normalize=True)
    return float((p ** 2).sum())

wish_hhi = wlm.groupby("owner_seller_id")["brand"].agg(hhi)
sell_hhi = sel.loc[sel.n_listings >= 5, "brand_hhi"].dropna()
print(f"위시 HHI 중앙값 {wish_hhi.median():.3f} | 셀링 HHI 중앙값 {sell_hhi.median():.3f}")

est = sorted(s_tot[s_tot >= 5].index)
both_poc = sorted(set(est) & set(w_tot[w_tot >= 5].index))
ei = {s: i for i, s in enumerate(est)}
sc_ref = sell_cnt[sell_cnt.seller_id.isin(ei)]
S_ref = normalize(csr_matrix((sc_ref.n, (sc_ref.seller_id.map(ei), sc_ref.brand.map(vi))),
                             shape=(len(est), len(vocab))))
bidx2 = {s: i for i, s in enumerate(both_poc)}
wc_ref = wish_cnt[wish_cnt.owner_seller_id.isin(bidx2)]
W_ref = normalize(csr_matrix((wc_ref.n, (wc_ref.owner_seller_id.map(bidx2), wc_ref.brand.map(vi))),
                             shape=(len(both_poc), len(vocab))))
print(f"참조 후보 {len(est):,}명 | 신규 시뮬레이션 대상 {len(both_poc):,}명")

st_est = sel.set_index("seller_id").reindex(est)["sell_through"].values
hi_bar = float(np.nanquantile(st_est, 2 / 3))
is_hi = st_est >= hi_bar
both_in_est = np.array([ei[s] for s in both_poc])
K = 30
ST = S_ref.T.tocsr()
rng = np.random.RandomState(0)
neigh_cos, rand_cos, hi_counts = [], [], []
for start in range(0, len(both_poc), 512):
    Wc = W_ref[start:start + 512]
    sims = (Wc @ ST).toarray()
    rows = np.arange(start, min(start + 512, len(both_poc)))
    for rl, rg in enumerate(rows):
        sims[rl, both_in_est[rg]] = -1.0
    topk = np.argpartition(-sims, K, axis=1)[:, :K]
    for rl, rg in enumerate(rows):
        my = both_in_est[rg]
        nb = topk[rl]
        neigh_cos.append(float((S_ref[my] @ S_ref[nb].T).toarray().mean()))
        hi_counts.append(int(is_hi[nb].sum()))
        rd = rng.choice(len(est), K + 1, replace=False)
        rd = rd[rd != my][:K]
        rand_cos.append(float((S_ref[my] @ S_ref[rd].T).toarray().mean()))

neigh_cos = np.array(neigh_cos)
rand_cos = np.array(rand_cos)
hi_counts = np.array(hi_counts)
validation_lift = float(neigh_cos.mean() / rand_cos.mean())
reach = {m: float((hi_counts >= m).mean()) for m in [1, 3, 5]}
print(f"위시 검색 이웃 셀링 일치도 {neigh_cos.mean():.3f} vs 무작위 {rand_cos.mean():.3f} | lift {validation_lift:.1f}x")
print(f"유사+성공 참조군 도달률: ≥1명 {reach[1]:.1%}, ≥3명 {reach[3]:.1%}, ≥5명 {reach[5]:.1%}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].bar(["무작위 K명", "위시-검색 K명"], [rand_cos.mean(), neigh_cos.mean()],
          color=["#aab", "#4a8c5f"])
ax[0].set(title=f"위시 기반 유사 셀러의 셀링 일치도 (무작위 대비 {validation_lift:.1f}×)",
          ylabel="이웃 셀링 ↔ 본인 셀링 코사인(평균)")
for i, v in enumerate([rand_cos.mean(), neigh_cos.mean()]):
    ax[0].text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)
xs = ["≥1명", "≥3명", "≥5명"]
ax[1].bar(xs, [reach[1] * 100, reach[3] * 100, reach[5] * 100], color="#3a6ea5")
ax[1].set(title=f"'유사+성공' 참조군 도달률 (성공 cutoff {hi_bar:.3f})", ylabel="% of 신규-시뮬 셀러", ylim=(0, 100))
for i, m in enumerate([1, 3, 5]):
    ax[1].text(i, reach[m] * 100 + 1.5, f"{reach[m]*100:.0f}%", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "p7_onboarding_reach.png", bbox_inches="tight")
plt.close(fig)

onboarding_poc = {
    "k_pool": K,
    "n_reference_pool_sellers": int(len(est)),
    "n_simulated_new_sellers": int(len(both_poc)),
    "neighbor_sell_cos_mean": round(float(neigh_cos.mean()), 3),
    "random_sell_cos_mean": round(float(rand_cos.mean()), 3),
    "validation_lift": round(validation_lift, 2),
    "success_bar_sellthrough_top33": round(hi_bar, 3),
    "reach_similar_and_successful_ref": {f"ge{m}": round(reach[m], 3) for m in [1, 3, 5]},
}

# %% [markdown]
# ## 6. 가격군 보조 검증
#
# 가격군은 신규 셀러의 시세 불확실성을 줄이는 참고 정보다. 정확한 브랜드 기준 가격 위치와,
# 화면에 보여주기 쉬운 넓은 시장 기준 가격 위치를 함께 확인한다. 전자는 엄밀하지만 1점물 특성상
# 비교군이 작고, 후자는 덜 정밀하지만 신규 셀러에게 가격대 기준을 제공하기 쉽다.

# %%
price_df = lst.copy()
g_price = price_df.groupby(["brand", "category_l1", "condition"])
price_df["grp_n"] = g_price["price_final"].transform("size")
price_df["grp_pct"] = g_price["price_final"].rank(pct=True)
in_group = price_df[price_df["grp_n"] >= 5].copy()
in_group["price_bin"] = pd.cut(
    in_group["grp_pct"],
    [0, 0.25, 0.5, 0.75, 1.0],
    labels=["하위25%(저가)", "25-50", "50-75", "상위25%(고가)"],
)
sellthrough_by_bin = in_group.groupby("price_bin", observed=True)["is_sold"].mean()
sold_pct = float(in_group.loc[in_group["is_sold"] == 1, "grp_pct"].mean())
unsold_pct = float(in_group.loc[in_group["is_sold"] == 0, "grp_pct"].mean())
print("동종군 내 가격 백분위별 전환율:")
print((sellthrough_by_bin * 100).round(1).to_string())

band_df = lst.dropna(subset=["brand", "category_l1"]).copy()
band_df["cond"] = band_df["condition"].fillna("NA")
grp_n_bcc = band_df.groupby(["brand", "category_l1", "cond"])["product_id"].transform("size")
grp_n_bc = band_df.groupby(["brand", "category_l1"])["product_id"].transform("size")
price_band_coverage = {
    "brand_category_condition": round(float((grp_n_bcc >= 5).mean()), 3),
    "brand_category": round(float((grp_n_bc >= 5).mean()), 3),
}
price_position_check = {
    "group_definition": "brand × category_l1 × condition; valid group n>=5",
    "n": int(len(price_df)),
    "n_with_valid_group": int(len(in_group)),
    "coverage_bcc_raw_condition": round(float((price_df["grp_n"] >= 5).mean()), 3),
    "sellthrough_by_within_group_price_bin": {
        str(k): round(float(v), 3) for k, v in sellthrough_by_bin.items()
    },
    "sold_vs_unsold_mean_pct": [round(sold_pct, 3), round(unsold_pct, 3)],
    "conclusion": (
        "현재의 브랜드·대분류·컨디션 기준 동종 가격 위치는 전환율을 거의 가르지 못한다. "
        "따라서 가격군은 판매 확률 처방이 아니라 시세 불확실성 완화 정보로 해석한다."
    ),
}

market_df = lst.copy()
market_df["cond"] = market_df["condition"].fillna("NA")
market_cols = ["brand_top", "category_l2", "cond"]
g_market = market_df.groupby(market_cols)
market_df["market_grp_n"] = g_market["price_final"].transform("size")
market_df["market_grp_pct"] = g_market["price_final"].rank(pct=True)
market_valid = market_df[market_df["market_grp_n"] >= 20].copy()
market_valid["market_price_bin"] = pd.cut(
    market_valid["market_grp_pct"],
    [0, 0.25, 0.5, 0.75, 1.0],
    labels=["하위25%(저가)", "25-50", "50-75", "상위25%(고가)"],
)
market_sellthrough_by_bin = market_valid.groupby("market_price_bin", observed=True)["is_sold"].mean()
market_sold_pct = float(market_valid.loc[market_valid["is_sold"] == 1, "market_grp_pct"].mean())
market_unsold_pct = float(market_valid.loc[market_valid["is_sold"] == 0, "market_grp_pct"].mean())
q1 = market_valid.loc[market_valid["market_price_bin"].eq("하위25%(저가)"), "is_sold"]
q4 = market_valid.loc[market_valid["market_price_bin"].eq("상위25%(고가)"), "is_sold"]
market_q4_vs_q1_or = float((q4.sum() / (len(q4) - q4.sum())) / (q1.sum() / (len(q1) - q1.sum())))

category_price_effect = {}
for cat, g in market_valid.groupby("category_l2"):
    if len(g) < 3000:
        continue
    rates = g.groupby("market_price_bin", observed=True)["is_sold"].mean()
    if "하위25%(저가)" in rates and "상위25%(고가)" in rates:
        category_price_effect[str(cat)] = {
            "n": int(len(g)),
            "low_q1": round(float(rates["하위25%(저가)"]), 3),
            "high_q4": round(float(rates["상위25%(고가)"]), 3),
            "q4_minus_q1": round(float(rates["상위25%(고가)"] - rates["하위25%(저가)"]), 3),
        }
category_price_effect = dict(
    sorted(category_price_effect.items(), key=lambda kv: kv[1]["q4_minus_q1"])
)
market_price_position_check = {
    "group_definition": "brand_top × category_l2 × condition; valid group n>=20",
    "n_with_valid_group": int(len(market_valid)),
    "coverage": round(float(len(market_valid) / len(market_df)), 3),
    "sellthrough_by_market_price_bin": {
        str(k): round(float(v), 3) for k, v in market_sellthrough_by_bin.items()
    },
    "sold_vs_unsold_mean_pct": [round(market_sold_pct, 3), round(market_unsold_pct, 3)],
    "q4_vs_q1_OR": round(market_q4_vs_q1_or, 3),
    "category_l2_effects_n_ge3000": category_price_effect,
}
print("시장 기준 가격 백분위별 전환율:")
print((market_sellthrough_by_bin * 100).round(1).to_string())
print("시장 기준 Q4 vs Q1 OR:", round(market_q4_vs_q1_or, 3))

# %% [markdown]
# ## 7. 토픽 효과의 통제 검증
#
# 가장 낮은 전환율을 보인 토픽이 가격이나 카테고리 구성 차이만으로 낮아졌는지 확인한다.
# 매물 단위 로지스틱 회귀에 토픽 더미와 가격, 카테고리, 컨디션, 등록 경과일을 함께 넣는다.
# 기준 토픽은 매물 단위 최빈 토픽이다.

# %%
def z_num(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd else s * 0

topic_st = sel_c.groupby("archetype").sell_through.mean()
focal = int(topic_st.idxmin())
ml = lst.merge(sel_c[["seller_id", "archetype"]], on="seller_id", how="inner")
ml["condition"] = ml["condition"].fillna("UNK")
ml["category_l1"] = ml["category_l1"].fillna("UNK")
arch_dummies = pd.get_dummies(ml.archetype, prefix="arch", drop_first=False)
baseline_col = f"arch_{ml.archetype.mode()[0]}"
if baseline_col in arch_dummies:
    arch_dummies = arch_dummies.drop(columns=[baseline_col])
Xa = pd.concat([
    arch_dummies,
    pd.DataFrame({"z_logp": z_num(ml.log_price), "z_age": z_num(ml.age_days)}),
    pd.get_dummies(ml.category_l1, prefix="c", drop_first=True),
    pd.get_dummies(ml.condition, prefix="cond", drop_first=True),
], axis=1).astype(float)
ma = sm.Logit(ml.is_sold.values, sm.add_constant(Xa)).fit(disp=0, method="lbfgs", maxiter=300)
focal_col = f"arch_{focal}"
topic_control = {
    "low_liquidity_archetype": focal,
    "low_liquidity_sellthrough": round(float(topic_st.loc[focal]), 3),
    "low_liquidity_OR_controlled": round(float(np.exp(ma.params[focal_col])), 3) if focal_col in ma.params else None,
    "low_liquidity_p_controlled": round(float(ma.pvalues[focal_col]), 4) if focal_col in ma.params else None,
}
print("토픽 통제 검증:", topic_control)

# %% [markdown]
# ## 8. 저장

# %%
sel_c[["seller_id", "archetype", "topic_dominance"]].to_parquet(
    CACHE / "seller_clusters.parquet", index=False
)

topic_metric_records = [
    {k: v for k, v in r.items() if k not in {"model", "W", "labels", "dominance"}}
    for r in topic_runs
]

h3 = {
    "n_sellers_clustered": int(len(sel_c)),
    "coverage_of_all_sellers": round(float(len(sel_c) / len(sel)), 3),
    "clustering_strategy": {
        "selected_method": "NMF",
        "selected_config": f"brand+brand_category_l2 tfidf, k={best_k}",
        "rationale": "NMF gives interpretable soft taste topics; k is selected by unsupervised topic balance and dominance, not sell-through.",
        "selection_rule": selection_rule,
    },
    "best_k": int(best_k),
    "topic_token_space": "brand + brand|category_l2",
    "topic_model_comparison": topic_metric_records,
    "topic_top_terms": topic_top_terms,
    "archetype_profile": json.loads(profile.reset_index().to_json(orient="records", force_ascii=False)),
    "kruskal": {"H": round(float(H), 2), "p": float(p)},
    "inference": {
        "selected_method": "NMF",
        "selected_k": int(best_k),
        "silhouette_w": selected["silhouette_w"],
        "kw_epsilon_sq": round(float(eps2), 3),
        "topic_dominance_median": selected["dominance_median"],
        "min_topic_size": selected["min_topic_size"],
        "max_topic_size": selected["max_topic_size"],
        "max_topic_share": selected["max_topic_share"],
    },
    "sellthrough_range_across_archetypes": [
        round(float(profile["sell_through"].min()), 3),
        round(float(profile["sell_through"].max()), 3),
    ],
    "wish_sell_taste": WST,
    "low_history_wishlist_availability": low_history_wishlist,
    "topic_control": topic_control,
    "onboarding_poc": onboarding_poc,
    "price_band_coverage": price_band_coverage,
    "price_position_check": price_position_check,
    "market_price_position_check": market_price_position_check,
}

(ROOT / "results" / "h3.json").write_text(
    json.dumps(h3, ensure_ascii=False, indent=2), encoding="utf-8"
)
(ROOT / "results" / "taste_matching.json").write_text(
    json.dumps({
        "premise": "위시-셀링 취향은 횡단면에서 무작위보다 높게 정렬된다. 브랜드 공간 8.9×는 보조 수치이고, 원 브랜드 검색의 핵심 검증은 이웃 셀링 일치도 5.3×다.",
        "demand_taste_hhi_median": round(float(wish_hhi.median()), 3),
        "supply_sell_hhi_median": round(float(sell_hhi.median()), 3),
        "low_history_wishlist_availability": low_history_wishlist,
        "onboarding_poc": onboarding_poc,
        "price_band_coverage_bcc": price_band_coverage["brand_category_condition"],
        "price_band_coverage_bc": price_band_coverage["brand_category"],
        "caveats": (
            "단면 자료(종단 아님; 기존 셀러로 프록시 검증); 성공 셀러 cutoff는 참조 후보 전환율 "
            "상위 1/3 규칙이지 신규 셀러 기대 전환율 상승폭 아님; 위시 주체=셀러(구매자 아님); "
            "브랜드 단위 근사; comp 밴드는 시세 투명성 도구이며 넓은 시장 기준 가격 위치는 "
            "고가 구간의 낮은 전환 신호를 일부 포착함(p3_staleness.json). 가격·할인 탄력성은 A/B 필요"
        ),
    }, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
(ROOT / "results" / "p3_staleness.json").write_text(
    json.dumps(
        {
            "exact_price_position_check": price_position_check,
            "market_price_position_check": market_price_position_check,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print("saved. NMF topics:", best_k, "| sell-through range:", h3["sellthrough_range_across_archetypes"])
