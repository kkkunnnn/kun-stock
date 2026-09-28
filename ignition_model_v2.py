from __future__ import annotations

import numpy as np
import pandas as pd

DEV_CUTOFF = pd.Timestamp("2025-01-01")

IGNITION_V2_FEATURES = [
    "atr_pct",
    "atr_accel_5_20",
    "vol20_ann",
    "vol_accel_5_20",
    "volume_ma5_vs20",
    "volume_1_vs5",
    "range_compression_10_40",
    "bb_width20",
    "bb_width_delta5",
    "ma5_slope5",
    "ma20_slope5",
    "rsi_delta5",
    "macd_hist_delta5",
    "dist_prev20_high",
    "ret5",
    "ret20",
    "ma20_bias",
]

FUSION_WEIGHTS = {
    "I2_F0_S2_ONLY": 0.00,
    "I2_F10": 0.10,
    "I2_F20": 0.20,
    "I2_F30": 0.30,
}


def _smooth_prob(train: pd.DataFrame, test: pd.DataFrame, feature: str, target: str, bins: int = 12) -> np.ndarray:
    y0 = pd.to_numeric(train[target], errors="coerce")
    base = float(y0.mean())
    tr = pd.DataFrame({
        "x": pd.to_numeric(train.get(feature), errors="coerce"),
        "y": y0,
    }).dropna()
    tx = pd.to_numeric(test.get(feature), errors="coerce")
    if len(tr) < 500 or tr["x"].nunique() < 8 or pd.isna(base):
        return np.full(len(test), 0.0 if pd.isna(base) else base * 100)

    qs = np.unique(np.nanquantile(tr["x"].to_numpy(float), np.linspace(0, 1, bins + 1)))
    if len(qs) < 4:
        return np.full(len(test), base * 100)
    qs[0] = -np.inf
    qs[-1] = np.inf

    b = pd.cut(tr["x"], bins=qs, include_lowest=True, labels=False, duplicates="drop")
    stats = pd.DataFrame({"bin": b, "y": tr["y"]}).dropna().groupby("bin")["y"].agg(["mean", "count"])
    # Shrink noisy bins toward the train-set target rate.
    stats["p"] = (stats["mean"] * stats["count"] + base * 150.0) / (stats["count"] + 150.0)

    tb = pd.cut(tx, bins=qs, include_lowest=True, labels=False, duplicates="drop")
    return pd.Series(tb).map(stats["p"]).fillna(base).to_numpy(float) * 100


def _evidence(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    parts = []
    for f in IGNITION_V2_FEATURES:
        if f in train.columns and f in test.columns:
            parts.append(_smooth_prob(train, test, f, target))
    if not parts:
        base = pd.to_numeric(train[target], errors="coerce").mean()
        return np.full(len(test), float(base * 100) if pd.notna(base) else 0.0)
    return np.nanmean(np.vstack(parts), axis=0)


def add_ignition_v2_components(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Leakage-safe Ignition 2.0 score.
    Focuses on under-extended names and predicts fast future expansion:
    +50% within 30 trading days and +30% within 20 trading days.
    """
    out = test.copy().reset_index(drop=True)
    tr = train.copy()

    # Targets are known only in historical training rows that satisfy the embargo upstream.
    tr["ignite50_30"] = (
        (pd.to_numeric(tr["hit50"], errors="coerce") == 1)
        & (pd.to_numeric(tr["days_to_hit50"], errors="coerce") <= 30)
    ).astype(float)
    tr["ignite30_20"] = (
        (pd.to_numeric(tr["hit30"], errors="coerce") == 1)
        & (pd.to_numeric(tr["days_to_hit30"], errors="coerce") <= 20)
    ).astype(float)

    # Deliberately separate "not yet extended" ignition candidates from second-leg names.
    gate_train = (
        pd.to_numeric(tr.get("ret20"), errors="coerce").between(-15, 20)
        & pd.to_numeric(tr.get("ma20_bias"), errors="coerce").between(-10, 10)
        & (pd.to_numeric(tr.get("prior60_runup"), errors="coerce") < 35)
    )
    itr = tr[gate_train.fillna(False)].copy()
    if len(itr) < 2500:
        itr = tr

    p50fast = _evidence(itr, out, "ignite50_30")
    p30fast = _evidence(itr, out, "ignite30_20")
    raw = 0.70 * p50fast + 0.30 * p30fast

    gate_test = (
        pd.to_numeric(out.get("ret20"), errors="coerce").between(-15, 20)
        & pd.to_numeric(out.get("ma20_bias"), errors="coerce").between(-10, 10)
        & (pd.to_numeric(out.get("prior60_runup"), errors="coerce") < 35)
    ).fillna(False).to_numpy()

    base_fast = float(pd.to_numeric(itr["ignite50_30"], errors="coerce").mean() * 100)
    out["ignition_v2_raw"] = np.where(gate_test, raw, base_fast * 0.50)
    out["ignition_v2_fast50_raw"] = np.where(gate_test, p50fast, base_fast * 0.50)
    out["ignition_v2_fast30_raw"] = np.where(gate_test, p30fast, base_fast * 0.50)
    out["ignition_v2_gate"] = gate_test
    return out


def _pct(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rank(pct=True, method="average") * 100


def add_ignition_v2_rankings(results: pd.DataFrame, s2_selected: str) -> pd.DataFrame:
    d = results.copy()
    blocks = []
    s2_score_col = f"{s2_selected}_score"
    s2_rank_col = f"{s2_selected}_rank"

    for _, g in d.groupby("date", sort=False):
        g = g.copy()
        if s2_score_col in g.columns:
            s2_pct = _pct(g[s2_score_col])
        else:
            # Fallback to inverse rank percentile.
            r = pd.to_numeric(g[s2_rank_col], errors="coerce")
            s2_pct = (1 - (r - 1) / max(len(g) - 1, 1)) * 100

        ign_pct = _pct(g["ignition_v2_raw"])
        g["ignition_v2_pct"] = ign_pct

        # Independent ignition ranking for diagnostic use.
        g["ignition_v2_rank"] = pd.to_numeric(g["ignition_v2_raw"], errors="coerce").rank(
            ascending=False, method="first"
        )

        # Only development data will choose the fusion weight.
        for name, w_ign in FUSION_WEIGHTS.items():
            score = (1 - w_ign) * s2_pct + w_ign * ign_pct
            g[f"{name}_score"] = score
            g[f"{name}_rank"] = score.rank(ascending=False, method="first")
        blocks.append(g)

    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def _metrics(d: pd.DataFrame, rank_col: str, k: int) -> dict:
    x = d[d["hit50"].notna() & d[rank_col].notna()].copy()
    picked = x[pd.to_numeric(x[rank_col], errors="coerce") <= k]
    actual = int((pd.to_numeric(x["hit50"], errors="coerce") == 1).sum())
    tp = int((pd.to_numeric(picked["hit50"], errors="coerce") == 1).sum())
    n = int(len(picked))
    base = actual / len(x) if len(x) else np.nan
    precision = tp / n if n else np.nan
    recall = tp / actual if actual else np.nan
    return {
        "Precision": precision * 100 if pd.notna(precision) else np.nan,
        "Recall": recall * 100 if pd.notna(recall) else np.nan,
        "Lift": precision / base if pd.notna(precision) and pd.notna(base) and base > 0 else np.nan,
        "TP": tp,
        "選中樣本": n,
        "實際+50": actual,
    }


def select_ignition_v2_fusion(results: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    dev = results[pd.to_datetime(results["date"]) < DEV_CUTOFF].copy()
    rows = []
    for name, w in FUSION_WEIGHTS.items():
        rcol = f"{name}_rank"
        m10 = _metrics(dev, rcol, 10)
        m20 = _metrics(dev, rcol, 20)
        m50 = _metrics(dev, rcol, 50)
        objective = 0.45 * m10["Precision"] + 0.20 * m20["Precision"] + 0.35 * m50["Recall"]
        rows.append({
            "config": name,
            "Ignition權重": w,
            "開發期目標分數": objective,
            "Top10 Precision": m10["Precision"],
            "Top10 Recall": m10["Recall"],
            "Top10 Lift": m10["Lift"],
            "Top20 Precision": m20["Precision"],
            "Top20 Recall": m20["Recall"],
            "Top50 Recall": m50["Recall"],
        })
    t = pd.DataFrame(rows).sort_values(
        ["開發期目標分數", "Top10 Precision", "Top50 Recall"],
        ascending=False,
    ).reset_index(drop=True)
    t["開發期排名"] = np.arange(1, len(t) + 1)
    return str(t.iloc[0]["config"]), t


def compare_ignition_v2(results: pd.DataFrame, selected: str, s2_selected: str) -> pd.DataFrame:
    d = results.copy()
    periods = {
        "開發期": d[pd.to_datetime(d["date"]) < DEV_CUTOFF],
        "OOS期": d[pd.to_datetime(d["date"]) >= DEV_CUTOFF],
        "全期間": d,
    }
    models = [
        ("S1原模型", "rank"),
        ("S2 leading", f"{s2_selected}_rank"),
        ("S2+Ignition2", f"{selected}_rank"),
        ("Ignition2單獨", "ignition_v2_rank"),
    ]
    rows = []
    for period_name, p in periods.items():
        for model_name, rcol in models:
            if rcol not in p.columns:
                continue
            for k in [5,10,20,30,50,100]:
                m = _metrics(p, rcol, k)
                rows.append({
                    "期間": period_name,
                    "模型": model_name,
                    "設定": selected if model_name == "S2+Ignition2" else (
                        s2_selected if model_name == "S2 leading" else model_name
                    ),
                    "K": k,
                    **m,
                })
    return pd.DataFrame(rows)


def false_negative_recovery(results: pd.DataFrame, selected: str) -> pd.DataFrame:
    """
    Dedicated test on actual +50 winners missed by S1 Top50.
    Reports how many are promoted by Ignition 2.0 / fused model.
    """
    d = results.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["hit50"] = pd.to_numeric(d["hit50"], errors="coerce")
    d["rank"] = pd.to_numeric(d["rank"], errors="coerce")
    misses = d[(d["hit50"] == 1) & (d["rank"] > 50)].copy()
    rows = []

    for period_name, p in [
        ("開發期", misses[misses["date"] < DEV_CUTOFF]),
        ("OOS期", misses[misses["date"] >= DEV_CUTOFF]),
        ("全期間", misses),
    ]:
        total = len(p)
        if total == 0:
            continue
        for model_name, rcol in [
            ("Ignition2單獨", "ignition_v2_rank"),
            ("S2+Ignition2", f"{selected}_rank"),
        ]:
            rr = pd.to_numeric(p[rcol], errors="coerce")
            rows.append({
                "期間": period_name,
                "模型": model_name,
                "S1原本Top50外+50樣本": int(total),
                "救回Top20": int((rr <= 20).sum()),
                "救回Top50": int((rr <= 50).sum()),
                "救回Top100": int((rr <= 100).sum()),
                "Top50救回率": float((rr <= 50).mean() * 100),
                "Top100救回率": float((rr <= 100).mean() * 100),
                "新排名中位數": float(rr.median()),
            })
    return pd.DataFrame(rows)


def feature_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    d = results.copy()
    d["ignite_target"] = (
        (pd.to_numeric(d["hit50"], errors="coerce") == 1)
        & (pd.to_numeric(d["days_to_hit50"], errors="coerce") <= 30)
        & (pd.to_numeric(d["prior60_runup"], errors="coerce") < 35)
    ).astype(float)

    rows = []
    for f in IGNITION_V2_FEATURES:
        if f not in d.columns:
            continue
        z = pd.DataFrame({
            "x": pd.to_numeric(d[f], errors="coerce"),
            "y": d["ignite_target"],
        }).dropna()
        if len(z) < 100 or z["x"].nunique() < 10:
            continue
        q25, q75 = z["x"].quantile([0.25, 0.75])
        low = z[z["x"] <= q25]["y"].mean()
        high = z[z["x"] >= q75]["y"].mean()
        rows.append({
            "特徵": f,
            "成功中位數": z[z["y"] == 1]["x"].median(),
            "其他中位數": z[z["y"] == 0]["x"].median(),
            "低25%快速爆發率": low * 100,
            "高25%快速爆發率": high * 100,
            "高低差": (high - low) * 100,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("高低差", ascending=False).reset_index(drop=True)
