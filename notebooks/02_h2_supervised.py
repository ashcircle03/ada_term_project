# %% [markdown]
# # 02 · H2 — 통제가능 vs 구조적 기여 분해 (지도학습)
#
# **가설:** 전환은 셀러가 *바꿀 수 있는* 표현(사진·설명·상대가격)보다 *못 바꾸는*
# 구조(브랜드수요·가격대·카테고리)에 더 좌우된다 → 일반 가이드가 "노력→판매"를
# 과대포장한다. 또한 통제가능 레버의 효과는 세그먼트별로 이질적이다.
#
# **방법:** (1) 통제가능-only vs 전체 피처 AUC 비교(셀러 통제력의 상한),
# (2) permutation importance 그룹 기여 분해, (3) 가격대별 이질성.
# view_count/like_count는 누수 → 제외.

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
FIG = ROOT / "results" / "figures"

lst = pd.read_parquet(ROOT / "data" / "cache" / "features_listing.parquet")
lst["rel_price_missing"] = lst["relative_price_z"].isna().astype(int)
lst["relative_price_z"] = lst["relative_price_z"].fillna(0.0)
lst["condition"] = lst["condition"].fillna("UNK")
lst["category_l1"] = lst["category_l1"].fillna("UNK")
lst["category_l2"] = lst["category_l2"].fillna("UNK")
y = lst["is_sold"].astype(int).values
print("N =", len(lst), "| sold rate =", f"{y.mean():.3f}")

# %% [markdown]
# ## 1. 피처 그룹 정의 (통제가능 vs 구조)
#
# 누수 가드: view_count/like_count는 데이터에 없음(피처 테이블에서 이미 제외).

# %%
CONTROLLABLE = ["n_photos", "desc_len", "n_lines", "n_hashtag", "n_emoji",
                "kw_measure", "kw_flaw", "kw_material", "kw_purchase", "kw_usage", "kw_wash",
                "discount_pct", "has_discount", "relative_price_z", "rel_price_missing"]
STRUCT_NUM = ["log_price", "age_days"]
STRUCT_CAT = ["brand_top", "category_l1", "category_l2", "condition"]

assert not ({"view_count", "like_count", "likes"} & set(lst.columns)), "누수 컬럼 존재!"

# category_l2 고카디널리티 → top20 + OTHER
top_l2 = lst["category_l2"].value_counts().head(20).index
lst["category_l2"] = lst["category_l2"].where(lst["category_l2"].isin(top_l2), "OTHER")

def make_X(which):
    parts = []
    if which in ("controllable", "full"):
        parts.append(lst[CONTROLLABLE].astype(float))
    if which in ("structural", "full"):
        parts.append(lst[STRUCT_NUM].astype(float))
        for c in STRUCT_CAT:
            parts.append(pd.get_dummies(lst[c], prefix=c, drop_first=True).astype(float))
    return pd.concat(parts, axis=1)

X_ctrl, X_struct, X_full = make_X("controllable"), make_X("structural"), make_X("full")
print("dims:", X_ctrl.shape[1], X_struct.shape[1], X_full.shape[1])

# %% [markdown]
# ## 2. AUC 비교 — 셀러 통제력의 상한
#
# 통제가능-only AUC가 0.5에 가까우면 셀러가 움직일 수 있는 레버의 예측력이 약하다는 뜻.

# %%
cv = StratifiedKFold(5, shuffle=True, random_state=42)
def xgb():
    return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         eval_metric="logloss", n_jobs=-1, random_state=42)

auc = {}
for name, X in [("controllable", X_ctrl), ("structural", X_struct), ("full", X_full)]:
    s = cross_val_score(xgb(), X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    auc[name] = {"auc_mean": round(s.mean(), 4), "auc_std": round(s.std(), 4)}
    print(f"  {name:12s} AUC = {s.mean():.4f} ± {s.std():.4f}")

# 로지스틱 baseline (통제가능-only)
log_auc = cross_val_score(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
    X_ctrl, y, cv=cv, scoring="roc_auc").mean()
print(f"  (logistic controllable-only AUC = {log_auc:.4f})")

# %% [markdown]
# ## 2b. 추론 — 귀무가설 기각 (지도학습의 검정 틀)
#
# 지도학습엔 모수적 t-검정이 없으므로, 분포 가정 없이 **재표집/순열**로 검정한다.
# (i) **라벨 순열검정**: y를 무작위로 섞어 만든 null AUC 분포와 관측 AUC를 비교 →
#     귀무 "예측력=우연(AUC 0.5)"을 기각. (ii) **부트스트랩 ΔAUC 95% CI**: 동일 테스트셋을
#     재표집해 두 모델의 AUC 차 분포를 구함 → CI가 0을 제외하면 "통제가능=구조" 귀무를 기각.

# %%
idx = np.arange(len(y))
i_tr, i_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
yte_ = y[i_te]
def fit_pred(X):
    return xgb().fit(X.iloc[i_tr], y[i_tr]).predict_proba(X.iloc[i_te])[:, 1]
p_ctrl, p_str, p_full = fit_pred(X_ctrl), fit_pred(X_struct), fit_pred(X_full)
obs_full = roc_auc_score(yte_, p_full)

rng = np.random.RandomState(1); B = 1000
null_auc = np.array([roc_auc_score(rng.permutation(yte_), p_full) for _ in range(B)])
perm_p = (1 + np.sum(null_auc >= obs_full)) / (B + 1)
print(f"  full AUC(test)={obs_full:.4f} | 순열 null {null_auc.mean():.3f}±{null_auc.std():.3f} | p={perm_p:.4f}")

def boot_dauc(pa, pb, B=1000):
    n = len(yte_); d = np.empty(B)
    for b in range(B):
        s = rng.randint(0, n, n)
        d[b] = roc_auc_score(yte_[s], pa[s]) - roc_auc_score(yte_[s], pb[s])
    return np.percentile(d, [2.5, 97.5])
ci_str = boot_dauc(p_str, p_ctrl); ci_full = boot_dauc(p_full, p_ctrl)
print(f"  ΔAUC 구조−통제가능 95%CI [{ci_str[0]:.3f}, {ci_str[1]:.3f}]")
print(f"  ΔAUC 전체−통제가능 95%CI [{ci_full[0]:.3f}, {ci_full[1]:.3f}]")
H2_INF = {"full_test_auc": round(float(obs_full), 4), "perm_p": float(perm_p),
          "perm_null_mean": round(float(null_auc.mean()), 3),
          "dAUC_struct_minus_ctrl_CI95": [round(float(ci_str[0]), 3), round(float(ci_str[1]), 3)],
          "dAUC_full_minus_ctrl_CI95": [round(float(ci_full[0]), 3), round(float(ci_full[1]), 3)]}

# %% [markdown]
# ## 3. 기여 분해 — permutation importance (그룹 합산)
#
# 전체 모델에서 피처를 셔플했을 때 AUC 하락폭. 통제가능 vs 구조 그룹별 합산.

# %%
Xtr, Xte, ytr, yte = train_test_split(X_full, y, test_size=0.15, stratify=y, random_state=42)
clf = xgb().fit(Xtr, ytr)
print("holdout AUC:", round(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]), 4))

# 속도 위해 holdout 일부로 permutation
idx = np.random.RandomState(0).choice(len(Xte), size=min(30000, len(Xte)), replace=False)
perm = permutation_importance(clf, Xte.iloc[idx], yte[idx], scoring="roc_auc",
                              n_repeats=5, random_state=0, n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=X_full.columns).sort_values(ascending=False)

def group_of(col):
    if col in CONTROLLABLE:
        return "통제가능(표현)"
    return "구조(아이템)"

grp = imp.groupby(imp.index.map(group_of)).sum()
grp_share = (grp / grp.sum()).round(3)
print("\n그룹별 기여 합 (permutation, AUC 하락):")
print(grp.round(4))
print("비율:", grp_share.to_dict())
print("\nTop 12 피처:")
print(imp.head(12).round(4))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
pd.Series(grp_share).plot(kind="bar", ax=ax[0], rot=0, color=["#48a", "#c44"])
ax[0].set(title="기여 비율: 통제가능 vs 구조", ylabel="permutation importance share")
imp.head(12)[::-1].plot(kind="barh", ax=ax[1],
                        color=[("#48a" if group_of(c)=="통제가능(표현)" else "#c44") for c in imp.head(12).index[::-1]])
ax[1].set(title="Top 12 피처 (파랑=통제가능, 빨강=구조)", xlabel="AUC drop")
fig.tight_layout(); fig.savefig(FIG / "h2_contribution.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 4. 이질성 — 가격대별 통제가능 레버의 예측력

# %%
het = {}
for tier, sub in lst.groupby("price_tier", observed=True):
    if len(sub) < 2000:
        continue
    Xc = sub[CONTROLLABLE].astype(float)
    ys = sub["is_sold"].astype(int).values
    if ys.mean() in (0, 1):
        continue
    a = cross_val_score(xgb(), Xc, ys, cv=3, scoring="roc_auc", n_jobs=1).mean()
    het[str(tier)] = round(a, 4)
    print(f"  {tier}: controllable-only AUC = {a:.4f}  (n={len(sub):,}, sold={ys.mean():.2f})")

# %% [markdown]
# ## 5. 결과 저장

# %%
h2 = {
    "n": int(len(lst)),
    "auc": auc,
    "logistic_controllable_auc": round(float(log_auc), 4),
    "auc_gap_full_minus_controllable": round(auc["full"]["auc_mean"] - auc["controllable"]["auc_mean"], 4),
    "inference": H2_INF,
    "contribution_share": {k: float(v) for k, v in grp_share.items()},
    "top_features": {k: round(float(v), 4) for k, v in imp.head(12).items()},
    "controllable_auc_by_price_tier": het,
}
(ROOT / "results" / "h2.json").write_text(
    json.dumps(h2, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(h2, ensure_ascii=False, indent=2))
