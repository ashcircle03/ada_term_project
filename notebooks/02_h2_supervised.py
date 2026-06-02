# %% [markdown]
# # 02 · H2 — 셀러 통제요인과 구조적 요인 중 무엇이 판매를 가르나 (지도학습)
#
# **가설.** 판매 전환은 셀러가 *바꿀 수 있는* 표현(사진·설명·상대가격)보다 *못 바꾸는*
# 구조(브랜드 수요·가격대·카테고리)에 더 좌우된다. 따라서 일반 가이드는 "노력→판매"를
# 과대포장하며, 통제가능 레버의 효과는 가격대별로 이질적이다.
#
# **왜 이 방법인가.** 비선형·상호작용이 많은 자료에서 *예측 가능성의 상한*을 재기 위해
# 유연한 **그래디언트 부스팅**을 학습한다. 성능은 임의로 고른 팔린 매물과 안 팔린 매물 중
# 모형이 팔린 쪽에 더 높은 점수를 줄 확률, 곧 둘을 얼마나 잘 구분하는지를 0.5(무작위)~1(완벽)
# 로 나타내는 **AUC**로 측정한다. 표현 변수만 넣은 모형과 구조까지 넣은 모형의 AUC 격차가
# 곧 셀러가 통제할 수 있는 여지의 상한이다. 등록 후 누적되는 조회·찜 수는 결과를 미리
# 반영하는 **누수** 변수라 예측에서 제외한다.
#
# **유의성(2b절).** 모수적 검정이 없으므로 재표집으로 귀무를 기각한다 — 라벨 순열검정으로
# "AUC=0.5(우연)"을, 부트스트랩 신뢰구간으로 "두 모형의 AUC 차=0"을. 유의수준 α=0.05.
#
# **구성:** (1) 피처군 정의, (2) AUC 비교, (2b) 순열·부트스트랩 검정, (3) 기여 분해,
# (4) 가격대별 이질성. 산출 수치는 results/h2.json.

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
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb_core
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
lst["created_at_dt"] = pd.to_datetime(lst["created_at"], utc=True, errors="coerce")
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
# ### 피처군 분류 근거 (H2의 조작적 핵심)
#
# 두 군의 경계는 **"셀러가 등록 시점에 바꿀 수 있는가"** 라는 조작적 정의다 — 사진·설명·키워드·할인/
# 상대가격은 *통제가능(표현)*, 브랜드·카테고리·컨디션·가격은 사실상 고정된 *매물 구조*. 이 분할 자체가 H2의
# 검정 대상이라 명시적으로 고정한다. **누수 가드**: 조회·찜 수는 등록 *이후* 누적되는 사후 변수라 결과를 미리
# 반영(누수)하므로 두 군 모두에서 제외한다(피처 테이블에 부재, 위 assert로 재확인). 고카디널리티
# `brand_top`(상위30+기타)·`category_l2`(상위20+기타)는 더미 폭발·과적합을 막는 축약이며, 뒤에서
# `brand_top_OTHER`(비주류 브랜드)가 최상위 피처로 나오는 것은 '무명일수록 안 팔린다'는 구조 신호로 읽는다.
# `age_days`는 절단을 분리하려 구조군에 두되 'age만'의 AUC를 따로 재 시간 효과와 구분한다. 모델은 비선형·
# 상호작용 상한을 재기 위해 XGBoost를, 성능은 불균형에 견고한 순위 지표 AUC를 쓴다.

# %% [markdown]
# ## 2. AUC 비교 — 셀러 통제력의 상한
#
# 통제가능-only AUC가 0.5(무작위)에 가까울수록 셀러가 움직일 수 있는 레버만으로는 판매를
# 거의 예측하지 못한다는 뜻이다. 구조를 더했을 때의 AUC 상승폭이 곧 '무엇을 파는가'의 힘이다.

# %%
cv = StratifiedKFold(5, shuffle=True, random_state=42)
def xgb():
    return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         eval_metric="logloss", n_jobs=-1, random_state=42)

auc = {}; pr_auc = {}
for name, X in [("controllable", X_ctrl), ("structural", X_struct), ("full", X_full)]:
    s = cross_val_score(xgb(), X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    pr = cross_val_score(xgb(), X, y, cv=cv, scoring="average_precision", n_jobs=1)  # PR-AUC(불균형 보완)
    auc[name] = {"auc_mean": round(s.mean(), 4), "auc_std": round(s.std(), 4)}
    pr_auc[name] = round(float(pr.mean()), 4)
    print(f"  {name:12s} AUC = {s.mean():.4f} ± {s.std():.4f} | PR-AUC = {pr.mean():.4f}")
print(f"  (baseline PR-AUC = 양성률 {y.mean():.3f}; ROC-AUC는 임계 무관 순위 지표라 불균형에 견고하나 PR-AUC를 병기)")

# 로지스틱 baseline (통제가능-only)
log_auc = cross_val_score(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
    X_ctrl, y, cv=cv, scoring="roc_auc").mean()
print(f"  (logistic controllable-only AUC = {log_auc:.4f})")

# %% [markdown]
# ## 2a. Out-of-time 검증 — 미래 자료로 성능 확인
#
# 무작위 K-fold는 내부 변별력을 재는 데 유용하지만, 패션 데이터는 트렌드·계절성이 있어 미래 등록 매물을
# 과거와 섞으면 시간 누수 비판을 받는다. 따라서 등록일 기준 마지막 연도를 테스트셋으로 분리해
# "과거 등록 매물로 학습 → 미래 등록 매물 예측" 성능을 별도로 보고한다. 판매 시점이 없으므로 이는
# 완전한 생존/전환 시점 검증이 아니라 등록 시점 기준의 보수적 out-of-time 검증이다.

# %%
def time_holdout_indices():
    valid = lst["created_at_dt"].notna()
    years = sorted(lst.loc[valid, "created_at_dt"].dt.year.dropna().unique())
    for year in reversed(years):
        test = valid & (lst["created_at_dt"].dt.year == year)
        train = valid & (lst["created_at_dt"].dt.year < year)
        if train.sum() >= 5000 and test.sum() >= 1000 and 0 < y[test].mean() < 1:
            return np.flatnonzero(train), np.flatnonzero(test), f"train < {int(year)}, test = {int(year)}"
    ordered = np.flatnonzero(valid)
    ordered = ordered[np.argsort(lst.iloc[ordered]["created_at_dt"].values)]
    cut = int(len(ordered) * 0.8)
    return ordered[:cut], ordered[cut:], "train first 80%, test last 20%"

time_tr, time_te, time_rule = time_holdout_indices()
print(f"  split: {time_rule} | train={len(time_tr):,}, test={len(time_te):,}, test sold={y[time_te].mean():.3f}")

time_auc = {}; time_pr_auc = {}
for name, X in [("controllable", X_ctrl), ("structural", X_struct), ("full", X_full)]:
    m = xgb().fit(X.iloc[time_tr], y[time_tr])
    p = m.predict_proba(X.iloc[time_te])[:, 1]
    time_auc[name] = round(float(roc_auc_score(y[time_te], p)), 4)
    time_pr_auc[name] = round(float(average_precision_score(y[time_te], p)), 4)
    print(f"  {name:12s} out-of-time AUC = {time_auc[name]:.4f} | PR-AUC = {time_pr_auc[name]:.4f}")

time_split = {
    "rule": time_rule,
    "n_train": int(len(time_tr)),
    "n_test": int(len(time_te)),
    "test_positive_rate": round(float(y[time_te].mean()), 4),
    "auc": time_auc,
    "pr_auc": time_pr_auc,
}

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
ci_str = boot_dauc(p_str, p_ctrl); ci_full = boot_dauc(p_full, p_ctrl); ci_fs = boot_dauc(p_full, p_str)
print(f"  ΔAUC 구조−통제가능 95%CI [{ci_str[0]:.3f}, {ci_str[1]:.3f}]")
print(f"  ΔAUC 전체−통제가능 95%CI [{ci_full[0]:.3f}, {ci_full[1]:.3f}]")
print(f"  ΔAUC 전체−구조(표현의 한계기여) 95%CI [{ci_fs[0]:.3f}, {ci_fs[1]:.3f}]")
H2_INF = {"full_test_auc": round(float(obs_full), 4), "perm_p": float(perm_p),
          "perm_null_mean": round(float(null_auc.mean()), 3),
          "dAUC_struct_minus_ctrl_CI95": [round(float(ci_str[0]), 3), round(float(ci_str[1]), 3)],
          "dAUC_full_minus_ctrl_CI95": [round(float(ci_full[0]), 3), round(float(ci_full[1]), 3)],
          "dAUC_full_minus_struct_CI95": [round(float(ci_fs[0]), 3), round(float(ci_fs[1]), 3)]}

# %% [markdown]
# ## 3. 기여 분해 — permutation importance (그룹 합산)
#
# 전체 모델에서 피처를 셔플했을 때 AUC 하락폭. 통제가능 vs 구조 그룹별 합산.

# %%
Xtr, Xte, ytr, yte = train_test_split(X_full, y, test_size=0.15, stratify=y, random_state=42)
clf = xgb().fit(Xtr, ytr)
print("holdout AUC:", round(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]), 4))

# 속도 위해 holdout 일부로 permutation
idx = np.random.RandomState(0).choice(len(Xte), size=min(10000, len(Xte)), replace=False)
perm = permutation_importance(clf, Xte.iloc[idx], yte[idx], scoring="roc_auc",
                              n_repeats=3, random_state=0, n_jobs=1)
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
fig.tight_layout(); fig.savefig(FIG / "h2_contribution.png", bbox_inches="tight"); plt.close(fig)

# %% [markdown]
# ## 4. Tree SHAP 기여 — 개별 매물 단위 해석
#
# 별도 SHAP 패키지 없이 XGBoost의 `pred_contribs=True`를 사용한다. 평균 절대 기여도로
# 구조/표현의 전체 비중을 다시 확인하고, 사진 수·설명 길이·상대가격의 국소 효과가 가격대별로
# 어떻게 달라지는지 dependence plot으로 본다.

# %%
shap_idx = np.random.RandomState(2).choice(len(Xte), size=min(20000, len(Xte)), replace=False)
dm = xgb_core.DMatrix(Xte.iloc[shap_idx], feature_names=list(X_full.columns))
shap_values = clf.get_booster().predict(dm, pred_contribs=True)
shap_abs = pd.Series(np.abs(shap_values[:, :-1]).mean(axis=0), index=X_full.columns).sort_values(ascending=False)
shap_grp = shap_abs.groupby(shap_abs.index.map(group_of)).sum()
shap_grp_share = (shap_grp / shap_grp.sum()).round(3)
print("\nTree SHAP 평균 절대 기여 비율:")
print(shap_grp_share.to_dict())
print("\nTree SHAP Top 12:")
print(shap_abs.head(12).round(4))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
pd.Series(shap_grp_share).plot(kind="bar", ax=ax[0], rot=0, color=["#c44", "#48a"])
ax[0].set(title="Tree SHAP 기여 비율", ylabel="mean |contribution| share")
shap_abs.head(12)[::-1].plot(
    kind="barh",
    ax=ax[1],
    color=[("#48a" if group_of(c) == "통제가능(표현)" else "#c44") for c in shap_abs.head(12).index[::-1]],
)
ax[1].set(title="Tree SHAP Top 12", xlabel="mean |contribution|")
fig.tight_layout(); fig.savefig(FIG / "h2_shap_summary.png", bbox_inches="tight"); plt.close(fig)

plot_df = Xte.iloc[shap_idx][["n_photos", "desc_len", "relative_price_z"]].copy()
plot_df["price_tier"] = lst.iloc[Xte.index[shap_idx]]["price_tier"].astype(str).values
for col in ["n_photos", "desc_len", "relative_price_z"]:
    plot_df[f"shap_{col}"] = shap_values[:, list(X_full.columns).index(col)]

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for i, col in enumerate(["n_photos", "desc_len", "relative_price_z"]):
    sample = plot_df.sample(min(6000, len(plot_df)), random_state=10 + i)
    ax[i].scatter(sample[col], sample[f"shap_{col}"], s=5, alpha=0.25)
    ax[i].axhline(0, color="black", lw=0.8)
    ax[i].set(title=f"{col} contribution", xlabel=col, ylabel="Tree SHAP contribution")
fig.tight_layout(); fig.savefig(FIG / "h2_shap_dependence.png", bbox_inches="tight"); plt.close(fig)

# %% [markdown]
# ## 5. 이질성 — 가격대별 통제가능 레버의 예측력

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
# ## 6. 결과 저장

# %%
h2 = {
    "n": int(len(lst)),
    "auc": auc,
    "pr_auc": pr_auc,
    "logistic_controllable_auc": round(float(log_auc), 4),
    "auc_gap_full_minus_controllable": round(auc["full"]["auc_mean"] - auc["controllable"]["auc_mean"], 4),
    "time_split": time_split,
    "inference": H2_INF,
    "contribution_share": {k: float(v) for k, v in grp_share.items()},
    "top_features": {k: round(float(v), 4) for k, v in imp.head(12).items()},
    "shap_group_share": {k: float(v) for k, v in shap_grp_share.items()},
    "shap_top_features": {k: round(float(v), 4) for k, v in shap_abs.head(12).items()},
    "controllable_auc_by_price_tier": het,
}
(ROOT / "results" / "h2.json").write_text(
    json.dumps(h2, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(h2, ensure_ascii=False, indent=2))
