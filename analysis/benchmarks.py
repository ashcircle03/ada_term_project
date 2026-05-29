"""P2·P5 프로토타입 엔진 — 그룹별 기준분포(가격 밴드 + 조회 기준값).

두 제안이 같은 "동종 그룹의 기준분포" 인프라를 공유한다:
  - P2 가격 밴드: brand × category × condition 의 가격 p25/p50/p75 (+폴백)
  - P5 조회 진단: category × price_tier × age_bucket 의 조회수 기준분포 → 원인 라우팅

조회/하트는 등록 후 누적값이라 *예측*엔 안 쓰지만(누수), *이미 떠 있는 매물의 진단*엔
정당하게 쓴다. 단 조회/하트의 판매 변별력은 약하므로(engagement≠conversion) 목표는
"숫자 키우기"가 아니라 **원인 라우팅**.

실행:  python -m analysis.benchmarks   (데모 출력)
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "cache"
DB = ROOT / "data" / "fruitsfamily.db"
MIN_N = 5  # 그룹 최소 표본
AGE_BINS = [0, 7, 30, 90, 365, np.inf]
AGE_LABELS = ["0-7d", "7-30d", "30-90d", "90-365d", "365d+"]


# ============================================================
# 데이터 로드 (진단엔 view/heart 필요 → DB에서 함께)
# ============================================================

def load() -> pd.DataFrame:
    lst = pd.read_parquet(CACHE / "features_listing.parquet")
    with sqlite3.connect(DB) as conn:
        eng = pd.read_sql_query(
            "SELECT product_id, view_count, like_count FROM listing", conn)
    df = lst.merge(eng, on="product_id", how="left")
    df["age_bucket"] = pd.cut(df["age_days"], AGE_BINS, labels=AGE_LABELS, right=False)
    return df


# ============================================================
# P2 — 가격 밴드 (폴백 계층)
# ============================================================

def _band_table(df, keys):
    g = df.groupby(keys, observed=True)["price_final"]
    t = g.agg(n="size", p25=lambda s: s.quantile(.25),
              p50="median", p75=lambda s: s.quantile(.75))
    return t[t["n"] >= MIN_N]


def build_price_bands(df):
    return {
        "L1": _band_table(df, ["brand", "category_l1", "condition"]),
        "L2": _band_table(df, ["brand", "category_l1"]),
        "L3": _band_table(df, ["category_l1", "condition"]),
        "global": df["price_final"].agg(["size", "median"]),
    }


def lookup_price_band(bands, brand, cat, cond):
    """폴백 순서: brand×cat×cond → brand×cat → cat×cond → 전역."""
    for level, key in [("L1", (brand, cat, cond)), ("L2", (brand, cat)),
                       ("L3", (cat, cond))]:
        t = bands[level]
        if key in t.index:
            r = t.loc[key]
            return {"level": level, "n": int(r["n"]),
                    "p25": int(r["p25"]), "p50": int(r["p50"]), "p75": int(r["p75"])}
    return {"level": "global", "n": int(bands["global"]["size"]),
            "p25": None, "p50": int(bands["global"]["median"]), "p75": None}


# ============================================================
# P5 — 조회 기준분포 + 진단 라우팅
# ============================================================

def build_view_baselines(df):
    g = df.groupby(["category_l1", "price_tier", "age_bucket"], observed=True)["view_count"]
    t = g.agg(n="size", v25=lambda s: s.quantile(.25),
              v50="median", v75=lambda s: s.quantile(.75))
    return t[t["n"] >= MIN_N]


def lookup_view_baseline(baselines, cat, tier, age_bucket):
    key = (cat, tier, age_bucket)
    if key in baselines.index:
        r = baselines.loc[key]
        return {"n": int(r["n"]), "v25": int(r["v25"]), "v50": int(r["v50"]), "v75": int(r["v75"])}
    return None


def diagnose(row, baselines, heart_med):
    """이미 떠 있는(미판매) 매물의 원인 라우팅."""
    bl = lookup_view_baseline(baselines, row["category_l1"], row["price_tier"], row["age_bucket"])
    if bl is None:
        return {"stage": "정보부족", "msg": "동종 기준 표본 부족 — 진단 보류"}
    v = row["view_count"]
    if v < bl["v25"]:
        return {"stage": "A. 저노출(발견 안 됨)", "baseline_v50": bl["v50"], "your_view": int(v),
                "msg": f"조회 {int(v)}회 = 동종({row['category_l1']}·{row['age_bucket']}) 하위25%(중앙값 {bl['v50']}) "
                       "→ 발견 자체가 안 됨. 가격·제목·노출 점검. ⚠️사진/설명 추가로는 안 풀림.",
                "route": ["P2 가격밴드", "P4 매칭/노출"]}
    # 노출은 충분한데 미판매
    heart_note = ""
    if row["like_count"] >= heart_med:
        heart_note = f" 찜 {int(row['like_count'])}개로 관심은 있으나 구매 미전환 → 가격이 결정타일 가능성."
    return {"stage": "B. 충분노출+미판매", "baseline_v50": bl["v50"], "your_view": int(v),
            "msg": f"조회 {int(v)}회 = 동종 정상범위(중앙값 {bl['v50']}). 충분히 보였음 "
                   f"→ 가격·상태 신뢰를 점검.{heart_note}",
            "route": ["P2 가격밴드", "P3 묵은재고개입"]}


# ============================================================
# 데모 — 셀러가 볼 화면 목업
# ============================================================

def _price_card(band, price):
    pos = "" if band["p50"] is None else f"  (동종 중앙값 대비 {((price/band['p50'])-1)*100:+.0f}%)"
    inside = (band["p25"] is not None and band["p25"] <= price <= band["p75"])
    rng = "표본부족(전역중앙값)" if band["p25"] is None else f"{band['p25']:,}~{band['p75']:,}원"
    flag = "✅ 적정 범위" if inside else "⚠️ 동종 범위 밖" if band["p25"] is not None else "—"
    return (f"  [P2 가격] 동종 시세대(p25~p75): {rng} · 중앙값 {band['p50']:,}원 "
            f"[lvl {band['level']}, n={band['n']}]\n"
            f"           내 등록가 {price:,}원{pos}  {flag}")


def demo():
    df = load()
    bands = build_price_bands(df)
    baselines = build_view_baselines(df)
    heart_med = df["like_count"].median()

    # 핸드오프 테이블 저장
    bands["L1"].reset_index().to_parquet(CACHE / "price_bands.parquet", index=False)
    cov = len(df[df.set_index(["brand", "category_l1", "condition"]).index.isin(bands["L1"].index)]) / len(df)
    print(f"price_bands.parquet 저장: {len(bands['L1']):,} 그룹 (L1 직접 커버 {cov:.1%})\n")

    rng = np.random.RandomState(7)
    examples = []
    # 미판매 저조회 / 미판매 정상조회 고하트 / 판매됨
    unsold = df[(df.is_sold == 0) & df.view_count.notna() & df.price_tier.notna()]
    examples.append(("미판매·저조회 예시", unsold[unsold.view_count < 30].sample(1, random_state=1).iloc[0]))
    hi = unsold[(unsold.view_count > 200) & (unsold.like_count > heart_med)]
    examples.append(("미판매·충분노출+찜많음", hi.sample(1, random_state=2).iloc[0]))
    examples.append(("판매완료(가격밴드만)", df[df.is_sold == 1].sample(1, random_state=3).iloc[0]))

    out = []
    for label, r in examples:
        print("=" * 66)
        print(f"[{label}]  {r['brand']} / {r['category_l1']}·{r['category_l2']} / {r['condition']} / {int(r['age_days'])}일")
        band = lookup_price_band(bands, r["brand"], r["category_l1"], r["condition"])
        print(_price_card(band, int(r["price_final"])))
        rec = {"label": label, "brand": r["brand"], "price": int(r["price_final"]),
               "price_band": band}
        if r["is_sold"] == 0:
            dg = diagnose(r, baselines, heart_med)
            print(f"  [P5 진단] {dg['stage']}")
            print(f"           {dg['msg']}")
            if "route" in dg:
                print(f"           ↳ 연결: {', '.join(dg['route'])}")
            rec["diagnosis"] = {k: dg[k] for k in dg if k != "msg"}
            rec["diagnosis_msg"] = dg["msg"]
        out.append(rec)
        print()

    import json
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "prototype_examples.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("results/prototype_examples.json 저장")


if __name__ == "__main__":
    demo()
