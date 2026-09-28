from __future__ import annotations

import numpy as np
import pandas as pd

CORE_FEATURES = [
    "ret5","ret20","ret60","rsi14","vol_ratio",
    "atr_pct","ma20_bias","vol20_ann","macd_hist",
]

def _safe_auc(x: pd.Series, y: pd.Series) -> float:
    """Rank-based univariate AUC without sklearn."""
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if d.empty or d["y"].nunique() < 2:
        return np.nan
    n1 = int((d["y"] == 1).sum())
    n0 = int((d["y"] == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = d["x"].rank(method="average")
    r1 = ranks[d["y"] == 1].sum()
    auc = (r1 - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(auc, 1 - auc))  # strength, direction reported separately


def build_precision_recall_table(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    d = results.copy()
    d["hit50"] = pd.to_numeric(d["hit50"], errors="coerce")
    d["rank"] = pd.to_numeric(d["rank"], errors="coerce")
    d = d[d["hit50"].notna() & d["rank"].notna()].copy()
    if d.empty:
        return pd.DataFrame()

    rows = []
    cutoffs = [5, 10, 20, 30, 50, 100]
    for k in cutoffs:
        per_day = []
        for dt, g in d.groupby("date"):
            actual = int((g["hit50"] == 1).sum())
            picked = g[g["rank"] <= k]
            tp = int((picked["hit50"] == 1).sum())
            n_pick = int(len(picked))
            per_day.append({
                "actual": actual,
                "tp": tp,
                "picked": n_pick,
                "pool": int(len(g)),
            })
        q = pd.DataFrame(per_day)
        actual_total = int(q["actual"].sum())
        tp_total = int(q["tp"].sum())
        picked_total = int(q["picked"].sum())
        pool_total = int(q["pool"].sum())
        base_rate = actual_total / pool_total if pool_total else np.nan
        precision = tp_total / picked_total if picked_total else np.nan
        recall = tp_total / actual_total if actual_total else np.nan

        rows.append({
            "範圍": f"Top {k}",
            "K": k,
            "測試截面數": int(len(q)),
            "選中樣本數": picked_total,
            "實際+50樣本數": actual_total,
            "抓到+50樣本數": tp_total,
            "Precision": precision * 100 if pd.notna(precision) else np.nan,
            "Recall": recall * 100 if pd.notna(recall) else np.nan,
            "全池基準命中率": base_rate * 100 if pd.notna(base_rate) else np.nan,
            "Lift": precision / base_rate if pd.notna(precision) and pd.notna(base_rate) and base_rate > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def build_feature_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    d = results.copy()
    d["hit50"] = pd.to_numeric(d["hit50"], errors="coerce")
    d = d[d["hit50"].notna()].copy()
    base = d["hit50"].mean()
    rows = []

    for f in CORE_FEATURES:
        if f not in d.columns:
            continue
        x = pd.to_numeric(d[f], errors="coerce")
        z = pd.DataFrame({"x": x, "hit": d["hit50"]}).dropna()
        if len(z) < 100 or z["x"].nunique() < 10:
            continue

        hit = z[z["hit"] == 1]["x"]
        no = z[z["hit"] == 0]["x"]
        std = z["x"].std()
        smd = (hit.mean() - no.mean()) / std if pd.notna(std) and std > 0 else np.nan

        q25 = z["x"].quantile(0.25)
        q75 = z["x"].quantile(0.75)
        low = z[z["x"] <= q25]["hit"].mean()
        high = z[z["x"] >= q75]["hit"].mean()

        rows.append({
            "特徵": f,
            "+50成功中位數": hit.median(),
            "未達+50中位數": no.median(),
            "標準化差異": smd,
            "單變量區分力AUC": _safe_auc(z["x"], z["hit"]),
            "低25%命中率": low * 100 if pd.notna(low) else np.nan,
            "高25%命中率": high * 100 if pd.notna(high) else np.nan,
            "高低25%命中率差": (high - low) * 100 if pd.notna(high) and pd.notna(low) else np.nan,
            "全池命中率": base * 100 if pd.notna(base) else np.nan,
            "方向": "越高越有利" if pd.notna(smd) and smd >= 0 else "越低越有利",
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["單變量區分力AUC","高低25%命中率差"], ascending=[False,False])


def build_false_negative_table(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    d = results.copy()
    d["hit50"] = pd.to_numeric(d["hit50"], errors="coerce")
    d["rank"] = pd.to_numeric(d["rank"], errors="coerce")
    hits = d[(d["hit50"] == 1) & d["rank"].notna()].copy()
    if hits.empty:
        return pd.DataFrame()

    groups = [
        ("Top10抓到", hits["rank"] <= 10),
        ("Top20才抓到", (hits["rank"] > 10) & (hits["rank"] <= 20)),
        ("Top50才抓到", (hits["rank"] > 20) & (hits["rank"] <= 50)),
        ("Top50之外漏掉", hits["rank"] > 50),
    ]
    rows = []
    for label, mask in groups:
        g = hits[mask].copy()
        if g.empty:
            continue
        row = {
            "類型": label,
            "樣本數": int(len(g)),
            "占全部+50樣本%": len(g) / len(hits) * 100,
            "排名中位數": pd.to_numeric(g["rank"], errors="coerce").median(),
            "模型分數中位數": pd.to_numeric(g["score_hit50"], errors="coerce").median(),
            "40日MFE中位數": pd.to_numeric(g["mfe40"], errors="coerce").median(),
            "到+50中位天數": pd.to_numeric(g.get("days_to_hit50"), errors="coerce").median() if "days_to_hit50" in g else np.nan,
        }
        for f in CORE_FEATURES:
            if f in g:
                row[f] = pd.to_numeric(g[f], errors="coerce").median()
        rows.append(row)
    return pd.DataFrame(rows)


def build_runup_buckets(results: pd.DataFrame) -> pd.DataFrame:
    """Check whether +50 winners also exist after a prior large run; do not exclude second-leg candidates."""
    if results.empty or "prior60_runup" not in results.columns:
        return pd.DataFrame()
    d = results.copy()
    d["hit50"] = pd.to_numeric(d["hit50"], errors="coerce")
    d["prior60_runup"] = pd.to_numeric(d["prior60_runup"], errors="coerce")
    d = d[d["hit50"].notna() & d["prior60_runup"].notna()].copy()
    if d.empty:
        return pd.DataFrame()

    d["前60日已漲幅層級"] = pd.cut(
        d["prior60_runup"],
        bins=[-np.inf, 10, 20, 35, 50, np.inf],
        labels=["≤10%","10–20%","20–35%","35–50%",">50%"],
        ordered=True,
    ).astype(str)

    rows = []
    total_hits = int((d["hit50"] == 1).sum())
    for label, g in d.groupby("前60日已漲幅層級", dropna=False):
        h = int((g["hit50"] == 1).sum())
        rows.append({
            "前60日已漲幅層級": str(label),
            "樣本數": int(len(g)),
            "+50樣本數": h,
            "未來40日+50命中率": g["hit50"].mean() * 100,
            "占全部+50成功樣本": h / total_hits * 100 if total_hits else np.nan,
            "模型分數中位數": pd.to_numeric(g["score_hit50"], errors="coerce").median(),
            "未來40日MFE中位數": pd.to_numeric(g["mfe40"], errors="coerce").median(),
            "20日乖離中位數": pd.to_numeric(g.get("ma20_bias"), errors="coerce").median(),
        })
    return pd.DataFrame(rows)


def build_ignition_timing(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "days_to_hit50" not in results.columns:
        return pd.DataFrame()
    d = results.copy()
    d["hit50"] = pd.to_numeric(d["hit50"], errors="coerce")
    d["rank"] = pd.to_numeric(d["rank"], errors="coerce")
    d = d[(d["hit50"] == 1)].copy()
    if d.empty:
        return pd.DataFrame()

    rows = []
    for label, max_rank in [("全部+50成功股", None),("Top10成功股",10),("Top20成功股",20),("Top50成功股",50)]:
        g = d if max_rank is None else d[d["rank"] <= max_rank]
        x = pd.to_numeric(g["days_to_hit50"], errors="coerce").dropna()
        if x.empty:
            continue
        rows.append({
            "群組": label,
            "樣本數": int(len(x)),
            "到+50中位天數": float(x.median()),
            "25分位天數": float(x.quantile(0.25)),
            "75分位天數": float(x.quantile(0.75)),
            "10日內達+50": float((x <= 10).mean() * 100),
            "20日內達+50": float((x <= 20).mean() * 100),
            "30日內達+50": float((x <= 30).mean() * 100),
        })
    return pd.DataFrame(rows)
