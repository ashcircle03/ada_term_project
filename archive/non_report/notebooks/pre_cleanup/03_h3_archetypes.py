# %% [markdown]
# # 03 · H3 — NMF 기반 셀러 취향 토픽과 초기 안내 신호
#
# **가설.** 셀러를 선명한 군집으로 나누기보다, 브랜드와 하위 카테고리 조합에서 반복되는
# 취향 토픽을 추정하면 판매 전환 차이를 더 해석 가능하게 설명할 수 있다. 이 토픽은 신규
# 셀러에게 유사 참조군을 찾는 보조 신호가 된다.
#
# **방법.** 매물 5건 이상 셀러의 `brand`와 `brand|category_l2` 토큰을 셀러별 sparse vector로
# 만들고, TF-IDF 변환 뒤 NMF를 적합한다. 각 셀러는 가장 큰 topic weight의 토픽으로 요약하되,
# dominance를 함께 저장해 혼합 취향 셀러가 많다는 점을 드러낸다. 결과변수 `sell_through`는
# 토픽 학습 피처에서 제외하고 사후 검정에만 사용한다.
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
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import scikit_posthocs as sp

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

# %%
def _safe_token(x):
    if pd.isna(x):
        return "UNK"
    return str(x).strip().replace(" ", "_")[:80]


def _seller_token_dicts(listings):
    rows = []
    for sid, g in listings.groupby("seller_id"):
        d = {}
        for _, r in g.iterrows():
            brand = _safe_token(r["brand"])
            cat2 = _safe_token(r["category_l2"])
            for tok in (f"brand={brand}", f"{brand}|{cat2}"):
                d[tok] = d.get(tok, 0) + 1
        rows.append((sid, d))
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
dunn = sp.posthoc_dunn(sel_c, val_col="sell_through", group_col="archetype", p_adjust="bonferroni")
print("\nDunn posthoc (p, Bonferroni):")
print(dunn.round(4))


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
# ## 3. 유형 내 상위 vs 하위 셀러의 표현 격차

# %%
ctrl_cols = ["n_photos", "desc_len", "kw_measure", "kw_flaw", "kw_material",
             "kw_usage", "relative_price_z"]
seller_repr = lst_c.groupby("seller_id")[ctrl_cols].mean()
sel_b = sel_c.merge(seller_repr, on="seller_id", how="left")

bench = {}
for a, g in sel_b.groupby("archetype"):
    if len(g) < 30:
        continue
    hi = g[g["sell_through"] >= g["sell_through"].quantile(0.67)]
    lo = g[g["sell_through"] <= g["sell_through"].quantile(0.33)]
    bench[int(a)] = {
        c: {
            "top": round(float(hi[c].mean()), 3),
            "bottom": round(float(lo[c].mean()), 3),
            "gap": round(float(hi[c].mean() - lo[c].mean()), 3),
        }
        for c in ctrl_cols
    }

gap_df = pd.DataFrame({a: {c: v[c]["gap"] for c in ctrl_cols} for a, v in bench.items()}).T
print("토픽별 상위−하위 셀러 표현 격차 (양수=성공셀러가 더 많이 함):")
print(gap_df.round(3).to_string())

# %% [markdown]
# ## 4. 키스톤 — 위시 취향이 셀링 취향을 예측하는가

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
    rows = []
    for oid, g in wish_rows.groupby("owner_seller_id"):
        d = {}
        for _, r in g.iterrows():
            brand = _safe_token(r["brand_mapped"])
            cat2 = _safe_token(r["category_l2"])
            for tok in (f"brand={brand}", f"{brand}|{cat2}"):
                d[tok] = d.get(tok, 0) + 1
        rows.append((oid, d))
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
# ## 4b. 키스톤 강건성

# %%
popb = sc.brand.value_counts()
top10 = list(popb.head(10).index)
top30 = list(popb.head(30).index)
top100 = list(popb.head(100).index)


def _kw_lift(scheme):
    s, w = sc.copy(), wc.copy()
    if scheme == "drop_top10":
        s = s[~s.brand.isin(top10)]
        w = w[~w.brand.isin(top10)]
    if scheme == "drop_generic":
        generic = ["Vintage", "Japanese Vintage"]
        s = s[~s.brand.isin(generic)]
        w = w[~w.brand.isin(generic)]
    if scheme == "bucket30":
        s = s.assign(b=np.where(s.brand.isin(top30), s.brand, "__OTHER__"))
        w = w.assign(b=np.where(w.brand.isin(top30), w.brand, "__OTHER__"))
    elif scheme in ("top30", "top100"):
        keep = set(top30 if scheme == "top30" else top100)
        s = s[s.brand.isin(keep)].assign(b=lambda d: d.brand)
        w = w[w.brand.isin(keep)].assign(b=lambda d: d.brand)
    else:
        s = s.assign(b=s.brand)
        w = w.assign(b=w.brand)
    voc = sorted(set(s.b) | set(w.b))
    v2 = {b: i for i, b in enumerate(voc)}
    sa = s.groupby(["seller_id", "b"], as_index=False).n.sum()
    wa = w.groupby(["owner_seller_id", "b"], as_index=False).n.sum()
    Sx = csr_matrix((sa.n, (sa.seller_id.map(bi), sa.b.map(v2))), shape=(len(both), len(voc)), dtype=float)
    Wx = csr_matrix((wa.n, (wa.owner_seller_id.map(bi), wa.b.map(v2))), shape=(len(both), len(voc)), dtype=float)
    if scheme == "svd100":
        from scipy.sparse import vstack as spvstack
        kc = min(100, min(Sx.shape) - 1)
        svd = TruncatedSVD(n_components=kc, random_state=0).fit(spvstack([Sx, Wx]))
        A = normalize(svd.transform(Sx))
        B = normalize(svd.transform(Wx))
        c = (A * B).sum(1)
        nz = (np.abs(A).sum(1) > 0) & (np.abs(B).sum(1) > 0)
        r = np.random.RandomState(0)
        nm = float(np.mean([(A * B[r.permutation(B.shape[0])]).sum(1)[nz].mean() for _ in range(10)]))
        return {"dims": int(kc), "median": round(float(np.median(c[nz])), 3),
                "lift": round(float(c[nz].mean() / nm), 1)}
    if scheme == "idf":
        dfb = np.asarray((Sx > 0).sum(0)).ravel() + np.asarray((Wx > 0).sum(0)).ravel()
        idfw = np.log((2 * len(both) + 1) / (dfb + 1)) + 1
        Sx = Sx.multiply(idfw).tocsr()
        Wx = Wx.multiply(idfw).tocsr()
    Sx = normalize(Sx)
    Wx = normalize(Wx)
    c = np.asarray(Sx.multiply(Wx).sum(1)).ravel()
    nz = (np.asarray((Sx != 0).sum(1)).ravel() > 0) & (np.asarray((Wx != 0).sum(1)).ravel() > 0)
    r = np.random.RandomState(0)
    nm = float(np.mean([
        np.asarray(Sx.multiply(Wx[r.permutation(Wx.shape[0])]).sum(1)).ravel()[nz].mean()
        for _ in range(10)
    ]))
    return {"dims": int(len(voc)), "median": round(float(np.median(c[nz])), 3),
            "lift": round(float(c[nz].mean() / nm), 1)}


ROBUST = {s: _kw_lift(s) for s in ["full", "bucket30", "top30", "top100", "idf", "svd100", "drop_top10", "drop_generic"]}
for k, v in ROBUST.items():
    print(f"  {k:12s} dims={v['dims']:5d} | 중앙 {v['median']:.3f} | lift {v['lift']}x")


def _gini(v):
    v = np.sort(np.asarray(v, dtype=float))
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum())) if v.sum() else 0.0


wish_gini = _gini(wc.groupby("brand")["n"].sum().values)
print(f"  위시-브랜드 분포 Gini = {wish_gini:.3f}")

fig, ax = plt.subplots(figsize=(6.8, 3.4))
labels_kr = {"full": "전체 raw\n(현재)", "bucket30": "상위30+기타\n버킷", "top30": "상위30만",
             "top100": "상위100만", "idf": "IDF 가중", "svd100": "SVD\n100차원",
             "drop_top10": "상위10\n제거", "drop_generic": "Vintage류\n제거"}
ks = list(ROBUST)
vals = [ROBUST[k]["lift"] for k in ks]
ax.bar([labels_kr[k] for k in ks], vals, color=["#c0504d" if k == "bucket30" else "#4a8c5f" for k in ks])
ax.axhline(1.0, color="#888", ls="--", lw=1)
for i, vv in enumerate(vals):
    ax.text(i, vv + 0.2, f"{vv}x", ha="center", fontsize=8)
ax.set(title="키스톤 강건성: 벡터공간 설계별 무작위 대비 lift", ylabel="lift (관측/무작위)")
ax.tick_params(axis="x", labelsize=7)
fig.tight_layout()
fig.savefig(FIG / "h3_keystone_robust.png", bbox_inches="tight")
plt.close(fig)

cof = lst.set_index("product_id")["category_l2"]
slc = lst.dropna(subset=["brand", "category_l2"]).copy()
slc["tok"] = slc["brand"].astype(str) + "|" + slc["category_l2"].astype(str)
wlc = wlm.assign(cat=wlm["product_id"].map(cof)).dropna(subset=["cat"])
wlc = wlc.assign(tok=wlc["brand"].astype(str) + "|" + wlc["cat"].astype(str))
sbc = slc[slc.seller_id.isin(bi)].groupby(["seller_id", "tok"], as_index=False).size().rename(columns={"size": "n"})
wbc = wlc[wlc.owner_seller_id.isin(bi)].groupby(["owner_seller_id", "tok"], as_index=False).size().rename(columns={"size": "n"})
vocbc = sorted(set(sbc.tok) | set(wbc.tok))
vbc = {t: i for i, t in enumerate(vocbc)}
Sbc = normalize(csr_matrix((sbc.n, (sbc.seller_id.map(bi), sbc.tok.map(vbc))), shape=(len(both), len(vocbc))))
Wbc = normalize(csr_matrix((wbc.n, (wbc.owner_seller_id.map(bi), wbc.tok.map(vbc))), shape=(len(both), len(vocbc))))
cbc = np.asarray(Sbc.multiply(Wbc).sum(1)).ravel()
nzbc = (np.asarray((Sbc != 0).sum(1)).ravel() > 0) & (np.asarray((Wbc != 0).sum(1)).ravel() > 0)
rbc = np.random.RandomState(0)
nmbc = float(np.mean([
    np.asarray(Sbc.multiply(Wbc[rbc.permutation(Wbc.shape[0])]).sum(1)).ravel()[nzbc].mean()
    for _ in range(10)
]))
ROBUST["brand_cat"] = {"dims": int(len(vocbc)), "median": round(float(np.median(cbc[nzbc])), 3),
                       "lift": round(float(cbc[nzbc].mean() / nmbc), 1)}
print(f"  brand×cat    dims={len(vocbc):5d} | 중앙 {ROBUST['brand_cat']['median']:.3f} | lift {ROBUST['brand_cat']['lift']}x")

nlist = sel.set_index("seller_id")["n_listings"]
low = [s for s in both if 5 <= nlist.get(s, 0) <= 8]
lb = {s: i for i, s in enumerate(low)}
scl = sc[sc.seller_id.isin(lb)]
wcl = wc[wc.owner_seller_id.isin(lb)]
Sl = normalize(csr_matrix((scl.n, (scl.seller_id.map(lb), scl.brand.map(vi))), shape=(len(low), len(vocab))))
Wl = normalize(csr_matrix((wcl.n, (wcl.owner_seller_id.map(lb), wcl.brand.map(vi))), shape=(len(low), len(vocab))))
cl = np.asarray(Sl.multiply(Wl).sum(1)).ravel()
rl = np.random.RandomState(0)
nml = float(np.mean([np.asarray(Sl.multiply(Wl[rl.permutation(Wl.shape[0])]).sum(1)).ravel().mean() for _ in range(10)]))
ROBUST["low_listing_5_8"] = {"n": int(len(low)), "median": round(float(np.median(cl)), 3),
                             "lift": round(float(cl.mean() / nml), 1)}
print(f"  저이력5~8건 n={len(low)} | 중앙 {ROBUST['low_listing_5_8']['median']} | lift {ROBUST['low_listing_5_8']['lift']}x")

# %% [markdown]
# ## 5. 저장

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
    "benchmark_gap": bench,
    "wish_sell_taste": WST,
    "wish_sell_taste_robustness": ROBUST,
    "wish_brand_gini": round(float(wish_gini), 3),
}

(ROOT / "results" / "h3.json").write_text(
    json.dumps(h3, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("saved. NMF topics:", best_k, "| sell-through range:", h3["sellthrough_range_across_archetypes"])
