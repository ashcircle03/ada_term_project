# %% [markdown]
# # 01 · H1 · 일반 등록 가이드는 판매를 돕는가 (통계·준인과)
#
# **가설.** "사진을 많이, 설명을 자세히"라는 일반 권고와 판매의 음(-)의 상관은,
# 어려운 매물일수록 사진·설명에 공을 들이는 역선택 등 교란 때문이며, 매물 속성을
# 통제하면 권고를 지지하는 독립적 효과는 확인하기 어렵다.
#
# **왜 이 방법인가.** 결과(판매 여부)가 0/1이므로 **로지스틱 회귀**로 각 변수가 판매
# 성사의 *승산(odds)* 을 몇 배로 바꾸는지를 **오즈비(OR)** 로 추정한다(OR>1이면 판매에
# 유리, <1이면 불리). 가격·브랜드·카테고리·컨디션·등록연령을 함께 넣어 *관측 가능한*
# 교란을 통제하고, 각 계수가 0인지(효과 없음)는 **Wald 검정**으로 판정한다. 회귀는
# 자료에 있는 변수만 보정하므로, 권고를 따른 매물과 안 따른 매물이 가격·브랜드 등에서
# 비슷해지도록 한 건씩 짝지어 비교하는 **성향점수매칭(PSM)** 을 병행한다. PSM도
# *미관측* 교란은 잡지 못하므로, 그 취약성은 본 노트북에서 **E-value**(추정 효과를
# 없애려면 숨은 요인이 처치·결과와 각각 몇 배나 강해야 하는지)로 점검한다.
#
# **유의수준.** 표본이 28만여 건으로 매우 커 작은 효과도 쉽게 유의해진다. 따라서
# α=0.05로 두되 p-값만 보지 않고 **효과 크기(OR이 1에서 얼마나 떨어졌는가)** 를 함께 본다.
#
# **구성:** (1) raw vs 통제 계수 비교, (2) 표현 레버별 OR, (3) "사진 3장+" 임계 검정,
# (4) PSM 처치효과(ATT), (5) 성숙 코호트와 E-value 민감도. 산출 수치는 results/h1.json.

# %%
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import sys
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from analysis import featurelib as fl

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
FIG = ROOT / "results" / "figures"

lst = pd.read_parquet(ROOT / "data" / "cache" / "features_listing.parquet")

# 결측 처리: 미싱 인디케이터 방식으로 N 보존
lst["rel_price_missing"] = lst["relative_price_z"].isna().astype(int)
lst["relative_price_z"] = lst["relative_price_z"].fillna(0.0)
lst["condition"] = lst["condition"].fillna("UNK")
lst["category_l1"] = lst["category_l1"].fillna("UNK")
print("N =", len(lst))

# %% [markdown]
# ## 0. 전처리·설계 근거 (H1 고유 결정)
#
# - **결측 처리.** `relative_price_z`는 동종(brand×category_l1×condition) 표본 n<5 또는 표준편차 0이면
#   로그가격 z-score 산출이 불안정하다.
#   평균대치(편향)·행 삭제(표본 손실) 대신 **미싱 인디케이터 `rel_price_missing`** 로 '산출 불가'라는 사실
#   자체를 변수화하고 값은 0으로 채워 N을 보존한다. `condition`·`category_l1`의 결측은 별도 범주 `UNK`로 둔다.
# - **공변량 선택.** 가격(log)·브랜드·카테고리·컨디션·등록연령은 '어려운 매물일수록 사진·설명에 공들이는'
#   역선택과 '오래 걸어 둔 매물이 더 팔리는' 절단은 노력↔판매의 음상관을 만들 수 있는 *관측 가능한*
#   교란이라 통제에 넣는다.
# - **처치 정의.** PSM 처치 `compliant = 사진≥3 & 설명≥150자`는 사진 가이드의 "3장 이상" 권고와
#   설명을 충분히 작성했다는 조작적 기준을 결합한 것이다. 설명 150자는 플랫폼의 명시 임계값이 아니라
#   짧은 설명과 중간 이상 설명을 가르는 분석 기준이므로, 처치효과는 "일반 권고 준수"의 근사로만 해석한다.
# - **caliper 0.02 · 표준화.** 공변량은 표준화한 뒤 성향점수를 추정하고, 성향점수 확률거리 0.02 안에서
#   최근접 매칭한다. 표현 변수는 z-표준화해 계수를 'SD 1 증가당 OR'로 해석한다.

# %% [markdown]
# ## 1. 표현 피처 표준화 + 설계행렬

# %%
def z(s):
    return (s - s.mean()) / s.std()

lst["z_photos"] = z(lst["n_photos"])
lst["z_desclen"] = z(lst["desc_len"])
lst["z_logprice"] = z(lst["log_price"])
lst["z_age"] = z(lst["age_days"])
lst["z_relprice"] = z(lst["relative_price_z"])

kw = ["kw_measure", "kw_flaw", "kw_material", "kw_purchase", "kw_usage", "kw_wash"]
y = lst["is_sold"].astype(int)

def design(cols, cats=()):
    X = lst[cols].copy()
    for c in cats:
        X = pd.concat([X, pd.get_dummies(lst[c], prefix=c, drop_first=True)], axis=1)
    return sm.add_constant(X.astype(float))

def fit(X):
    return sm.Logit(y, X).fit(disp=0, method="lbfgs", maxiter=200)

# %% [markdown]
# ## 2. raw vs 통제. 사진수·설명길이 계수가 어떻게 변하나
#
# 권고 변수만 넣은 모형(raw)과 매물 구조를 통제한 모형의 계수를 비교한다.
# raw에서 음(-)이던 계수가 통제 후 1(효과 없음)에 가까워질수록, 원래의 음의 상관이
# *교란* 때문이었음을 뜻한다. 통제 후에도 OR이 1을 넘지 못하면 "많을수록 좋다"는 지지되지 않는다.

# %%
m_raw = fit(design(["z_photos", "z_desclen"]))
m_ctrl = fit(design(
    ["z_photos", "z_desclen", *kw, "z_logprice", "z_relprice", "rel_price_missing", "z_age"],
    cats=["brand_top", "category_l1", "condition"],
))

def coef_row(m, name):
    return {"coef": round(m.params[name], 4), "or": round(np.exp(m.params[name]), 3),
            "p": round(m.pvalues[name], 4)}

contrast = {
    "n_photos(per SD)": {"raw": coef_row(m_raw, "z_photos"), "controlled": coef_row(m_ctrl, "z_photos")},
    "desc_len(per SD)": {"raw": coef_row(m_raw, "z_desclen"), "controlled": coef_row(m_ctrl, "z_desclen")},
}
print(json.dumps(contrast, ensure_ascii=False, indent=2))
print(f"\nPseudo R² raw={m_raw.prsquared:.4f}  controlled={m_ctrl.prsquared:.4f}")

# %% [markdown]
# ## 3. 가이드 항목별 통제 후 효과 (OR, 95% CI)
#
# 통제 모델에서 셀러가 통제 가능한 표현 레버들의 방향·유의성.

# %%
levers = {"z_photos": "사진수", "z_desclen": "설명길이", "z_relprice": "상대가격(동종대비, +=고가)",
          "kw_measure": "실측", "kw_flaw": "하자고지", "kw_material": "소재",
          "kw_purchase": "구매/정품", "kw_usage": "사용이력", "kw_wash": "관리법"}
conf = m_ctrl.conf_int()
rows = []
for k, lab in levers.items():
    rows.append({"lever": lab, "OR": np.exp(m_ctrl.params[k]),
                 "lo": np.exp(conf.loc[k, 0]), "hi": np.exp(conf.loc[k, 1]),
                 "p": m_ctrl.pvalues[k]})
eff = pd.DataFrame(rows).set_index("lever")
print(eff.round(4))

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(eff["OR"], range(len(eff)),
            xerr=[eff["OR"]-eff["lo"], eff["hi"]-eff["OR"]], fmt="o", color="#333")
ax.axvline(1.0, ls="--", c="red"); ax.set_yticks(range(len(eff))); ax.set_yticklabels(eff.index)
ax.set(title="통제 후 표현 레버의 판매 오즈비 (OR>1=판매↑)", xlabel="Odds Ratio")
fig.tight_layout(); fig.savefig(FIG / "h1_levers_or.png", bbox_inches="tight"); plt.close(fig)

# %% [markdown]
# ## 4. "사진 3장+" 임계 검정. 가이드의 핵심 권고
#
# ≤2장을 기준으로 3-5장·6+장의 OR. 가이드대로면 양수여야 하지만…

# %%
lst["photo_grp"] = pd.cut(lst["n_photos"], [-1, 2, 5, 10], labels=["le2", "p3_5", "p6plus"])
X_thr = sm.add_constant(pd.concat([
    pd.get_dummies(lst["photo_grp"], prefix="ph", drop_first=True),
    lst[["z_desclen", *kw, "z_logprice", "z_relprice", "rel_price_missing", "z_age"]],
    pd.get_dummies(lst["brand_top"], prefix="b", drop_first=True),
    pd.get_dummies(lst["category_l1"], prefix="c", drop_first=True),
    pd.get_dummies(lst["condition"], prefix="cond", drop_first=True),
], axis=1).astype(float))
m_thr = fit(X_thr)
thr = {g: {"OR": round(np.exp(m_thr.params[g]), 3), "p": round(m_thr.pvalues[g], 4)}
       for g in ["ph_p3_5", "ph_p6plus"]}
print("기준=≤2장. 통제 후 OR:")
print(json.dumps(thr, ensure_ascii=False, indent=2))

# %% [markdown]
# ## 5. 성향점수매칭(PSM). 가이드 준수의 처치효과(ATT)
#
# 회귀가 변수를 '보정'한다면, PSM은 권고를 따른 매물(처치군)마다 가격·브랜드 등이 비슷한
# 안 따른 매물(대조군)을 짝지어, *비교 가능한 쌍* 안에서 전환율 차이(ATT)를 본다.
# 단순 차이(naive)와 매칭 후 차이(ATT)를 비교하면 교란이 차이를 얼마나 만들었는지 드러난다.
# 공변량 균형 점검과 E-value 민감도는 아래에서 함께 수행한다.

# %%
lst["compliant"] = ((lst["n_photos"] >= 3) & (lst["desc_len"] >= 150)).astype(int)
cov = pd.concat([
    lst[["z_logprice", "z_age", "z_relprice"]],
    pd.get_dummies(lst["brand_top"], prefix="b", drop_first=True),
    pd.get_dummies(lst["category_l1"], prefix="c", drop_first=True),
    pd.get_dummies(lst["condition"], prefix="cond", drop_first=True),
], axis=1).astype(float)
psm_df = pd.concat([lst[["is_sold", "compliant"]], cov], axis=1)
att, diag = fl.propensity_match(psm_df, "compliant", list(cov.columns), caliper=0.02)

naive = lst[lst.compliant == 1]["is_sold"].mean() - lst[lst.compliant == 0]["is_sold"].mean()
print(f"순진한 차이(naive): {naive:+.4f}")
print(f"PSM ATT: {att:+.4f}")
print(json.dumps(diag, ensure_ascii=False, indent=2))

# %% [markdown]
# ## 6. 성숙 코호트와 E-value 민감도
#
# 판매 완료 시점이 없어 생존분석을 수행할 수 없으므로, 등록 후 1~2년이 지난 매물만 따로
# 보아 노출 기간 차이에서 오는 절단 편향을 일부 줄인다. 또한 PSM은 관측된 공변량만 맞출 수
# 있으므로, E-value로 숨은 교란에 대한 취약성을 수치화한다.

# %%
coh = lst[(lst.age_days >= 365) & (lst.age_days <= 730)].copy()

def fit_or_for(df):
    base = pd.DataFrame({
        "z_photos": z(df.n_photos),
        "z_desc": z(df.desc_len),
        "z_logp": z(df.log_price),
        "z_relp": z(df.relative_price_z),
        "rel_price_missing": df.rel_price_missing,
        "z_age": z(df.age_days),
    })
    for k in kw:
        base[k] = df[k]
    X = pd.concat([
        base,
        pd.get_dummies(df.brand_top, prefix="b", drop_first=True),
        pd.get_dummies(df.category_l1, prefix="c", drop_first=True),
        pd.get_dummies(df.condition, prefix="cond", drop_first=True),
    ], axis=1).astype(float)
    m = sm.Logit(df.is_sold.values, sm.add_constant(X)).fit(disp=0, method="lbfgs", maxiter=300)
    return float(np.exp(m.params["z_photos"])), float(np.exp(m.params["z_desc"]))

def photo_table(df):
    g = pd.cut(df.n_photos, [-1, 2, 5, 10], labels=["le2", "p3_5", "p6plus"])
    return df.groupby(g, observed=True).is_sold.mean()

or_full = fit_or_for(lst)
or_coh = fit_or_for(coh)
cohort_robustness = {
    "n": int(len(coh)),
    "sold_rate": round(float(coh.is_sold.mean()), 3),
    "photo_sellthrough": {
        "full": {str(k): round(float(v), 3) for k, v in photo_table(lst).items()},
        "cohort_365_730": {str(k): round(float(v), 3) for k, v in photo_table(coh).items()},
    },
    "or": {
        "full": {"photo": round(or_full[0], 3), "desc": round(or_full[1], 3)},
        "cohort_365_730": {"photo": round(or_coh[0], 3), "desc": round(or_coh[1], 3)},
    },
}
print("성숙 코호트:", json.dumps(cohort_robustness, ensure_ascii=False, indent=2))

cov_z = StandardScaler().fit_transform(cov)
ps = LogisticRegression(max_iter=1000).fit(cov_z, lst["compliant"].values).predict_proba(cov_z)[:, 1]
tr = np.where(lst["compliant"].values == 1)[0]
ct = np.where(lst["compliant"].values == 0)[0]
nn = NearestNeighbors(n_neighbors=1).fit(ps[ct].reshape(-1, 1))
dist, idx = nn.kneighbors(ps[tr].reshape(-1, 1))
keep = dist.ravel() <= 0.02
m_tr = tr[keep]
m_ct = ct[idx.ravel()[keep]]

def smd(col):
    a = cov[col].values[m_tr]
    b = cov[col].values[m_ct]
    a0 = cov[col].values[tr]
    b0 = cov[col].values[ct]
    sd = np.sqrt((a0.var() + b0.var()) / 2) or 1
    return abs(a0.mean() - b0.mean()) / sd, abs(a.mean() - b.mean()) / sd

key_covariates = ["z_logprice", "z_age", "z_relprice"]
smd_pre = {c: round(float(smd(c)[0]), 3) for c in key_covariates}
smd_post = {c: round(float(smd(c)[1]), 3) for c in key_covariates}
rr = y.values[m_tr].mean() / y.values[m_ct].mean()
inv_rr = 1 / rr
evalue = inv_rr + np.sqrt(inv_rr * (inv_rr - 1))
psm_sensitivity = {
    "match_rate": round(float(keep.mean()), 3),
    "rr": round(float(rr), 3),
    "evalue": round(float(evalue), 2),
    "smd_pre": smd_pre,
    "smd_post": smd_post,
}
print("PSM sensitivity:", json.dumps(psm_sensitivity, ensure_ascii=False, indent=2))

# %% [markdown]
# ## 7. 결과 저장

# %%
h1 = {
    "n": int(len(lst)),
    "raw_vs_controlled": contrast,
    "pseudo_r2": {"raw": round(m_raw.prsquared, 4), "controlled": round(m_ctrl.prsquared, 4)},
    "lever_odds_ratios": {lab: {"OR": round(r["OR"], 3), "p": round(r["p"], 4)}
                          for lab, r in eff.iterrows()},
    "photo_threshold_vs_le2": thr,
    "psm": {"naive_diff": round(float(naive), 4), "att": round(float(att), 4), **diag},
    "cohort_robustness": cohort_robustness,
    "psm_sensitivity": psm_sensitivity,
}
(ROOT / "results" / "h1.json").write_text(
    json.dumps(h1, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(h1, ensure_ascii=False, indent=2))
