# %% [markdown]
# # 04 · 상대 가격 위치 보조 검증
#
# **목적.** 보고서 3.3절의 "동종 비교 가격군은 산출 가능하지만 현재 기준의 가격 위치는
# 전환을 거의 가르지 못했다"는 문장을 재현한다.
#
# **왜 이 분석인가.** 빈티지 1점물에는 동일 상품 시세가 없으므로, 본 연구는
# `brand × category_l1 × condition` 그룹 안에서 가격 위치를 근사한다. 이 기준은
# 등록 화면의 참고 가격군으로는 쓸 수 있지만, 판매 확률 처방으로 쓰려면 같은 그룹 안에서
# 낮은 가격이 더 높은 전환을 보여야 한다. 따라서 동종군 내 가격 백분위와 전환율의 관계를
# 별도 보조 검증으로 확인한다.

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
CACHE = ROOT / "data" / "cache"

df = pd.read_parquet(CACHE / "features_listing.parquet").copy()

group_cols = ["brand", "category_l1", "condition"]
g = df.groupby(group_cols)
df["grp_n"] = g["price_final"].transform("size")
df["grp_pct"] = g["price_final"].rank(pct=True)
df["grp_median_price"] = g["price_final"].transform("median")

in_group = df[df["grp_n"] >= 5].copy()
coverage_bcc = float((df["grp_n"] >= 5).mean())
print(f"분석 표본 {len(df):,} | 동종군 n>=5 산출 가능 {len(in_group):,} ({coverage_bcc:.1%})")

# %% [markdown]
# ## 1. 동종군 내 가격 위치와 전환율
#
# 네 분위 가격 위치의 전환율이 거의 같으면, 현재의 동종군 기준 가격 위치는 판매 확률을
# 직접 처방하는 변수로 보기 어렵다. 이 경우 가격군은 "얼마가 비슷한 매물의 범위인가"를
# 보여주는 불확실성 완화 도구로만 해석한다.

# %%
in_group["price_bin"] = pd.cut(
    in_group["grp_pct"],
    [0, 0.25, 0.5, 0.75, 1.0],
    labels=["하위25%(저가)", "25-50", "50-75", "상위25%(고가)"],
)
sellthrough_by_bin = in_group.groupby("price_bin", observed=True)["is_sold"].mean()
print("동종군 내 가격 백분위별 전환율:")
print((sellthrough_by_bin * 100).round(1).to_string())

sold_pct = float(in_group.loc[in_group["is_sold"] == 1, "grp_pct"].mean())
unsold_pct = float(in_group.loc[in_group["is_sold"] == 0, "grp_pct"].mean())
print(f"판매 vs 미판매 평균 동종 백분위: {sold_pct:.3f} vs {unsold_pct:.3f}")

# %% [markdown]
# ## 2. 결과 저장
#
# 저장값은 보고서의 가격군 해석 한계를 재현하기 위한 최소 지표만 담는다. 묵은재고 개입
# 시뮬레이션과 가격 인하 실험 설계는 최종 보고서 본문에서 제외했으므로 archive에 보존한다.

# %%
result = {
    "group_definition": "brand × category_l1 × condition; valid group n>=5",
    "n": int(len(df)),
    "n_with_valid_group": int(len(in_group)),
    "coverage_bcc": round(coverage_bcc, 3),
    "sellthrough_by_within_group_price_bin": {
        str(k): round(float(v), 3) for k, v in sellthrough_by_bin.items()
    },
    "sold_vs_unsold_mean_pct": [round(sold_pct, 3), round(unsold_pct, 3)],
    "conclusion": (
        "현재의 브랜드·대분류·컨디션 기준 동종 가격 위치는 전환율을 거의 가르지 못한다. "
        "따라서 가격군은 판매 확률 처방이 아니라 시세 불확실성 완화 정보로 해석한다."
    ),
}

(ROOT / "results" / "p3_staleness.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
