# %% [markdown]
# # 06 · 보고서 보조 검증
#
# 본 노트북은 `REPORT.md`에 직접 쓰인 강건성 수치를 재현한다. 새 데이터를 추가하지 않고,
# 관측 자료 기반 결론이 어느 조건에서 약해지는지와 어떤 범위까지 해석 가능한지를 확인한다.
#
# - R1 성숙 코호트: 노출 기간이 비슷한 오래된 매물에서도 사진·설명 계수가 유지되는가
# - R2 PSM 민감도: 가이드 준수 ATT가 미관측 교란에 얼마나 취약한가
# - R3 AUC 분해: 구조 우위가 등록 경과 시간만의 산물인가
# - R4 NMF 토픽 효과: 저유동성 취향 토픽이 가격·카테고리 통제 후에도 낮은가

# %%
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ada-matplotlib-cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

FIG = ROOT / "results" / "figures"
CACHE = ROOT / "data" / "cache"
R = {}

lst = pd.read_parquet(CACHE / "features_listing.parquet")
lst["rel_price_missing"] = lst["relative_price_z"].isna().astype(int)
lst["relative_price_z"] = lst["relative_price_z"].fillna(0.0)
lst["condition"] = lst["condition"].fillna("UNK")
lst["category_l1"] = lst["category_l1"].fillna("UNK")
y = lst["is_sold"].astype(int).values
KW = ["kw_measure", "kw_flaw", "kw_material", "kw_purchase", "kw_usage", "kw_wash"]

def z(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd else s * 0

# %% [markdown]
# ## R1. 성숙 코호트에서 표현 변수 계수 재검증
#
# 판매 시점이 없기 때문에 오래 걸린 매물과 빨리 팔린 매물을 구분할 수 없다. 이를 완전히
# 해결하지는 못하지만, 등록 후 1~2년이 지난 성숙 매물만 따로 보면 노출 기간 차이에서 오는
# 절단 편향을 일부 줄일 수 있다. 이 검증에서 사진 OR이 1에 가까워지면, 사진 수가 판매를
# 낮춘다는 강한 인과 해석은 피해야 한다.

# %%
coh = lst[(lst.age_days >= 365) & (lst.age_days <= 730)].copy()
print(f"성숙 코호트 n={len(coh):,} | sold={coh.is_sold.mean():.3f}")

def photo_table(df):
    g = pd.cut(df.n_photos, [-1, 2, 5, 10], labels=["le2", "p3_5", "p6plus"])
    return df.groupby(g, observed=True).is_sold.mean()

def fit_or(df):
    base = pd.DataFrame({
        "z_photos": z(df.n_photos),
        "z_desc": z(df.desc_len),
        "z_logp": z(df.log_price),
        "z_relp": z(df.relative_price_z),
        "rel_price_missing": df.rel_price_missing,
        "z_age": z(df.age_days),
    })
    for k in KW:
        base[k] = df[k]
    X = pd.concat([
        base,
        pd.get_dummies(df.brand_top, prefix="b", drop_first=True),
        pd.get_dummies(df.category_l1, prefix="c", drop_first=True),
        pd.get_dummies(df.condition, prefix="cond", drop_first=True),
    ], axis=1).astype(float)
    m = sm.Logit(df.is_sold.values, sm.add_constant(X)).fit(disp=0, method="lbfgs", maxiter=300)
    return np.exp(m.params["z_photos"]), np.exp(m.params["z_desc"])

or_full = fit_or(lst)
or_coh = fit_or(coh)
print(f"통제 후 OR(per SD) 사진 full={or_full[0]:.3f}, cohort={or_coh[0]:.3f}")
print(f"통제 후 OR(per SD) 설명 full={or_full[1]:.3f}, cohort={or_coh[1]:.3f}")

R["R1_cohort"] = {
    "n": int(len(coh)),
    "sold_rate": round(float(coh.is_sold.mean()), 3),
    "photo_sellthrough": {
        "full": {str(k): round(float(v), 3) for k, v in photo_table(lst).items()},
        "cohort": {str(k): round(float(v), 3) for k, v in photo_table(coh).items()},
    },
    "photo_OR": {"full": round(float(or_full[0]), 3), "cohort": round(float(or_coh[0]), 3)},
    "desc_OR": {"full": round(float(or_full[1]), 3), "cohort": round(float(or_coh[1]), 3)},
}

# %% [markdown]
# ## R2. PSM 균형과 E-value 민감도
#
# PSM은 관측된 공변량의 차이를 줄이는 도구일 뿐, 심미성·핏·희소성 같은 미관측 교란은
# 제거하지 못한다. 따라서 ATT와 함께 E-value를 계산해, 어느 정도의 숨은 교란이면 결론이
# 뒤집힐 수 있는지 확인한다.

# %%
lst["compliant"] = ((lst.n_photos >= 3) & (lst.desc_len >= 150)).astype(int)
cov = pd.concat([
    pd.DataFrame({"z_logp": z(lst.log_price), "z_age": z(lst.age_days), "z_relp": z(lst.relative_price_z)}),
    pd.get_dummies(lst.brand_top, prefix="b", drop_first=True),
    pd.get_dummies(lst.category_l1, prefix="c", drop_first=True),
    pd.get_dummies(lst.condition, prefix="cond", drop_first=True),
], axis=1).astype(float)

t = lst.compliant.values
scaler = StandardScaler()
cov_z = scaler.fit_transform(cov)
ps = LogisticRegression(max_iter=1000).fit(cov_z, t).predict_proba(cov_z)[:, 1]
tr = np.where(t == 1)[0]
ct = np.where(t == 0)[0]
nn = NearestNeighbors(n_neighbors=1).fit(ps[ct].reshape(-1, 1))
dist, idx = nn.kneighbors(ps[tr].reshape(-1, 1))
caliper = 0.01
keep = dist.ravel() <= caliper
m_tr = tr[keep]
m_ct = ct[idx.ravel()[keep]]

def smd(col):
    a = cov[col].values[m_tr]
    b = cov[col].values[m_ct]
    a0 = cov[col].values[tr]
    b0 = cov[col].values[ct]
    sd = np.sqrt((a0.var() + b0.var()) / 2) or 1
    return abs(a0.mean() - b0.mean()) / sd, abs(a.mean() - b.mean()) / sd

key_covariates = ["z_logp", "z_age", "z_relp"]
smd_pre = {c: round(float(smd(c)[0]), 3) for c in key_covariates}
smd_post = {c: round(float(smd(c)[1]), 3) for c in key_covariates}
att = y[m_tr].mean() - y[m_ct].mean()
rr = y[m_tr].mean() / y[m_ct].mean()
inv_rr = 1 / rr
evalue = inv_rr + np.sqrt(inv_rr * (inv_rr - 1))
print(f"매칭률 {keep.mean():.3f} | ATT {att:+.4f} | RR {rr:.3f} | E-value {evalue:.2f}")
print(f"SMD pre {smd_pre} -> post {smd_post}")

R["R2_psm"] = {
    "caliper": caliper,
    "match_rate": round(float(keep.mean()), 3),
    "att": round(float(att), 4),
    "rr": round(float(rr), 3),
    "evalue": round(float(evalue), 2),
    "smd_pre": smd_pre,
    "smd_post": smd_post,
}

# %% [markdown]
# ## R3. AUC 분해에서 등록 경과 시간 분리
#
# 구조 변수의 예측력이 단지 오래 노출된 매물이 더 팔렸다는 시간 효과라면 `age_days`만으로도
# 높은 AUC가 나와야 한다. 따라서 통제가능 변수, 매물 구조, age-only, 구조+age, 전체 모형을
# 같은 XGBoost 사양으로 비교한다.

# %%
CONTROLLABLE = [
    "n_photos", "desc_len", "n_lines", "n_hashtag", "n_emoji",
    "kw_measure", "kw_flaw", "kw_material", "kw_purchase", "kw_usage", "kw_wash",
    "discount_pct", "has_discount", "relative_price_z", "rel_price_missing",
]
top_l2 = lst.category_l2.fillna("UNK").value_counts().head(20).index
lst["category_l2x"] = lst.category_l2.fillna("UNK").where(lst.category_l2.fillna("UNK").isin(top_l2), "OTHER")
STRUCT_ITEM = pd.concat([
    lst[["log_price"]].astype(float),
    pd.get_dummies(lst.brand_top, prefix="b", drop_first=True),
    pd.get_dummies(lst.category_l1, prefix="c1", drop_first=True),
    pd.get_dummies(lst.category_l2x, prefix="c2", drop_first=True),
    pd.get_dummies(lst.condition, prefix="cond", drop_first=True),
], axis=1).astype(float)
AGE = lst[["age_days"]].astype(float)
CTRLX = lst[CONTROLLABLE].astype(float)

def xgb():
    return XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )

cv = StratifiedKFold(5, shuffle=True, random_state=42)
sets = {
    "controllable": CTRLX,
    "item(no age)": STRUCT_ITEM,
    "age_only": AGE,
    "item+age": pd.concat([STRUCT_ITEM, AGE], axis=1),
    "full": pd.concat([CTRLX, STRUCT_ITEM, AGE], axis=1),
}
auc = {
    k: round(float(cross_val_score(xgb(), X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()), 4)
    for k, X in sets.items()
}
print("AUC 분해:")
for k, v in auc.items():
    print(f"  {k:14s} {v}")
R["R3_auc_decomp"] = auc

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.bar(list(auc.keys()), list(auc.values()), color=["#48a", "#c44", "#999", "#c84", "#333"])
ax.axhline(0.5, ls="--", c="gray")
ax.set(title="AUC 분해: 통제가능 vs 아이템구조 vs 시간(age)", ylabel="5-fold AUC")
ax.set_ylim(0.5, 0.82)
plt.xticks(rotation=20)
for i, v in enumerate(auc.values()):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "r3_auc_decomp.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## R4. NMF 토픽의 저유동성 효과 통제 검증
#
# NMF 토픽별 전환율 차이가 가격이나 카테고리 구성 차이만으로 생긴 것인지 확인한다. 매물 단위
# 로지스틱 회귀에 토픽 더미와 가격, 카테고리, 컨디션, 등록 경과일을 함께 넣고 저유동성 토픽의
# 오즈비를 확인한다.

# %%
sel = pd.read_parquet(CACHE / "features_seller.parquet")
clusters = pd.read_parquet(CACHE / "seller_clusters.parquet")
h3_meta = json.loads((ROOT / "results" / "h3.json").read_text(encoding="utf-8"))
strategy = h3_meta.get("clustering_strategy", {})
print(f"H3 선택 방식: {strategy.get('selected_method')} {strategy.get('selected_config')}")

sb = sel[sel.n_listings >= 5].merge(clusters, on="seller_id", how="inner")
topic_st = sb.groupby("archetype").sell_through.mean()
focal = int(topic_st.idxmin())
print(f"저유동성 topic={focal} | sell-through={topic_st.loc[focal]:.3f}")

ml = lst.merge(clusters, on="seller_id", how="inner")
arch_dummies = pd.get_dummies(ml.archetype, prefix="arch", drop_first=False)
baseline_col = f"arch_{ml.archetype.mode()[0]}"
if baseline_col in arch_dummies:
    arch_dummies = arch_dummies.drop(columns=[baseline_col])

Xa = pd.concat([
    arch_dummies,
    pd.DataFrame({"z_logp": z(ml.log_price), "z_age": z(ml.age_days)}),
    pd.get_dummies(ml.category_l1, prefix="c", drop_first=True),
    pd.get_dummies(ml.condition, prefix="cond", drop_first=True),
], axis=1).astype(float)
ma = sm.Logit(ml.is_sold.values, sm.add_constant(Xa)).fit(disp=0, method="lbfgs", maxiter=300)
focal_col = f"arch_{focal}"
focal_or = float(np.exp(ma.params[focal_col])) if focal_col in ma.params else None
focal_p = float(ma.pvalues[focal_col]) if focal_col in ma.params else None
print(
    f"가격·카테고리·컨디션·age 통제 후 저유동성 topic OR={focal_or:.3f}, p={focal_p:.4f}"
    if focal_or is not None else "저유동성 topic이 기준범주로 흡수됨"
)

R["R4_topic_control"] = {
    "selected_method": strategy.get("selected_method"),
    "selected_config": strategy.get("selected_config"),
    "low_liquidity_archetype": int(focal),
    "low_liquidity_sellthrough": round(float(topic_st.loc[focal]), 3),
    "low_liquidity_OR_controlled": round(focal_or, 3) if focal_or is not None else None,
    "low_liquidity_p_controlled": round(focal_p, 4) if focal_p is not None else None,
    "topic_dominance_median": h3_meta.get("inference", {}).get("topic_dominance_median"),
}

# %% [markdown]
# ## 저장

# %%
(ROOT / "results" / "robustness.json").write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(R, ensure_ascii=False, indent=2))
