from __future__ import annotations

import numpy as np
import pandas as pd

DEV_CUTOFF = pd.Timestamp("2025-01-01")

GENERAL_FEATURES = [
    "atr_pct","vol20_ann","ret60","ret20","rsi14",
    "ma20_bias","prior60_runup","dd20_high",
]
IGNITION_FEATURES = [
    "atr_pct","vol20_ann","ret5","ret20","rsi14",
    "ma20_bias","dd20_high","vol_ratio",
]
SECOND_LEG_FEATURES = [
    "prior60_runup","dd20_high","dd60_high",
    "atr_pct","vol20_ann","ret20","vol_ratio",
]

CANDIDATE_CONFIGS = {
    "S2_A_BASE": {"base":1.00,"general":0.00,"ignition":0.00,"second":0.00},
    "S2_B_GENERAL": {"base":0.70,"general":0.30,"ignition":0.00,"second":0.00},
    "S2_C_IGNITION": {"base":0.55,"general":0.25,"ignition":0.20,"second":0.00},
    "S2_D_SECOND": {"base":0.55,"general":0.25,"ignition":0.00,"second":0.20},
    "S2_E_BALANCED": {"base":0.45,"general":0.25,"ignition":0.15,"second":0.15},
    "S2_F_DISCOVERY": {"base":0.35,"general":0.25,"ignition":0.20,"second":0.20},
}


def _smoothed_bin_probability(train: pd.DataFrame, test: pd.DataFrame, feature: str, bins: int = 12) -> np.ndarray:
    base = float(pd.to_numeric(train["hit50"], errors="coerce").mean())
    tr = pd.DataFrame({
        "x": pd.to_numeric(train.get(feature), errors="coerce"),
        "y": pd.to_numeric(train["hit50"], errors="coerce"),
    }).dropna()
    tx = pd.to_numeric(test.get(feature), errors="coerce")
    if len(tr) < 500 or tr["x"].nunique() < 8:
        return np.full(len(test), base * 100)

    qs = np.unique(np.nanquantile(tr["x"].to_numpy(float), np.linspace(0,1,bins+1)))
    if len(qs) < 4:
        return np.full(len(test), base * 100)
    qs[0] = -np.inf
    qs[-1] = np.inf

    b = pd.cut(tr["x"], bins=qs, include_lowest=True, labels=False, duplicates="drop")
    stats = pd.DataFrame({"bin":b, "y":tr["y"]}).dropna().groupby("bin")["y"].agg(["mean","count"])
    # Bayesian-style shrinkage toward the training base rate.
    stats["p"] = (stats["mean"] * stats["count"] + base * 120.0) / (stats["count"] + 120.0)

    tb = pd.cut(tx, bins=qs, include_lowest=True, labels=False, duplicates="drop")
    out = pd.Series(tb).map(stats["p"]).fillna(base).to_numpy(float) * 100
    return out


def _evidence_score(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    pieces = []
    for f in features:
        if f in train.columns and f in test.columns:
            pieces.append(_smoothed_bin_probability(train, test, f))
    if not pieces:
        base = float(pd.to_numeric(train["hit50"], errors="coerce").mean() * 100)
        return np.full(len(test), base)
    return np.nanmean(np.vstack(pieces), axis=0)


def add_selection_v2_components(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe component scores using only training rows available at each test date."""
    out = test.copy().reset_index(drop=True)
    base_rate = float(pd.to_numeric(train["hit50"], errors="coerce").mean() * 100)

    out["s2_general_raw"] = _evidence_score(train, out, GENERAL_FEATURES)

    # Ignition branch: seek high future-explosion capacity before the recent trend is fully extended.
    ignition_train = train[
        pd.to_numeric(train.get("ret20"), errors="coerce").between(-12, 15)
        & pd.to_numeric(train.get("ma20_bias"), errors="coerce").between(-8, 8)
    ].copy()
    if len(ignition_train) < 2500:
        ignition_train = train
    ign = _evidence_score(ignition_train, out, IGNITION_FEATURES)
    ignition_gate = (
        pd.to_numeric(out.get("ret20"), errors="coerce").between(-12, 15)
        & pd.to_numeric(out.get("ma20_bias"), errors="coerce").between(-8, 8)
    ).fillna(False).to_numpy()
    out["s2_ignition_raw"] = np.where(ignition_gate, ign, base_rate * 0.75)

    # Second-leg branch: do not automatically punish stocks that already ran strongly.
    runup = pd.to_numeric(train.get("prior60_runup"), errors="coerce")
    second_train = train[runup >= 35].copy()
    if len(second_train) < 1500:
        second_train = train
    sec = _evidence_score(second_train, out, SECOND_LEG_FEATURES)
    second_gate = (pd.to_numeric(out.get("prior60_runup"), errors="coerce") >= 35).fillna(False).to_numpy()
    out["s2_second_raw"] = np.where(second_gate, sec, base_rate * 0.75)

    return out


def _pct_score(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average") * 100


def add_candidate_rankings(results: pd.DataFrame) -> pd.DataFrame:
    d = results.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    blocks = []
    for dt, g in d.groupby("date", sort=False):
        g = g.copy()
        g["s2_base_pct"] = _pct_score(pd.to_numeric(g["score_hit50"], errors="coerce"))
        g["s2_general_pct"] = _pct_score(pd.to_numeric(g["s2_general_raw"], errors="coerce"))
        g["s2_ignition_pct"] = _pct_score(pd.to_numeric(g["s2_ignition_raw"], errors="coerce"))
        g["s2_second_pct"] = _pct_score(pd.to_numeric(g["s2_second_raw"], errors="coerce"))

        for name, w in CANDIDATE_CONFIGS.items():
            score = (
                g["s2_base_pct"] * w["base"]
                + g["s2_general_pct"] * w["general"]
                + g["s2_ignition_pct"] * w["ignition"]
                + g["s2_second_pct"] * w["second"]
            )
            g[f"{name}_score"] = score
            g[f"{name}_rank"] = score.rank(ascending=False, method="first")
        blocks.append(g)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def _metrics(d: pd.DataFrame, rank_col: str, k: int) -> dict:
    x = d[d["hit50"].notna() & d[rank_col].notna()].copy()
    picked = x[pd.to_numeric(x[rank_col], errors="coerce") <= k]
    actual = int((pd.to_numeric(x["hit50"], errors="coerce") == 1).sum())
    tp = int((pd.to_numeric(picked["hit50"], errors="coerce") == 1).sum())
    n = len(picked)
    pool = len(x)
    p = tp / n if n else np.nan
    r = tp / actual if actual else np.nan
    base = actual / pool if pool else np.nan
    return {
        "Precision": p * 100 if pd.notna(p) else np.nan,
        "Recall": r * 100 if pd.notna(r) else np.nan,
        "Lift": p / base if pd.notna(p) and pd.notna(base) and base > 0 else np.nan,
        "TP": tp,
        "選中樣本": n,
        "實際+50": actual,
    }


def select_config_on_development(results_ranked: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    dev = results_ranked[pd.to_datetime(results_ranked["date"]) < DEV_CUTOFF].copy()
    rows = []
    for name in CANDIDATE_CONFIGS:
        rcol = f"{name}_rank"
        m10 = _metrics(dev, rcol, 10)
        m20 = _metrics(dev, rcol, 20)
        m50 = _metrics(dev, rcol, 50)
        # Development-only objective; OOS is never used for configuration selection.
        objective = (
            0.45 * m10["Precision"]
            + 0.20 * m20["Precision"]
            + 0.35 * m50["Recall"]
        )
        rows.append({
            "config": name,
            "開發期目標分數": objective,
            "Top10 Precision": m10["Precision"],
            "Top10 Recall": m10["Recall"],
            "Top10 Lift": m10["Lift"],
            "Top20 Precision": m20["Precision"],
            "Top20 Recall": m20["Recall"],
            "Top50 Recall": m50["Recall"],
            **{f"w_{k}":v for k,v in CANDIDATE_CONFIGS[name].items()},
        })
    table = pd.DataFrame(rows).sort_values(
        ["開發期目標分數","Top10 Precision","Top50 Recall"],
        ascending=False,
    ).reset_index(drop=True)
    table["開發期排名"] = np.arange(1, len(table)+1)
    selected = str(table.iloc[0]["config"])
    return selected, table


def compare_selected_vs_baseline(results_ranked: pd.DataFrame, selected: str) -> pd.DataFrame:
    d = results_ranked.copy()
    d["baseline_rank"] = pd.to_numeric(d["rank"], errors="coerce")
    selected_rank = f"{selected}_rank"
    periods = {
        "開發期": d[pd.to_datetime(d["date"]) < DEV_CUTOFF],
        "OOS期": d[pd.to_datetime(d["date"]) >= DEV_CUTOFF],
        "全期間": d,
    }
    rows = []
    for period_name, p in periods.items():
        for model_name, rcol in [("S1原模型","baseline_rank"),("S2候選模型",selected_rank)]:
            for k in [5,10,20,30,50,100]:
                m = _metrics(p, rcol, k)
                rows.append({
                    "期間": period_name,
                    "模型": model_name,
                    "S2設定": selected if model_name == "S2候選模型" else "S1",
                    "K": k,
                    **m,
                })
    return pd.DataFrame(rows)


def branch_capture_summary(results_ranked: pd.DataFrame, selected: str) -> pd.DataFrame:
    d = results_ranked.copy()
    d["hit50"] = pd.to_numeric(d["hit50"], errors="coerce")
    rcol = f"{selected}_rank"
    rows = []
    for period_name, p in [
        ("開發期", d[pd.to_datetime(d["date"]) < DEV_CUTOFF]),
        ("OOS期", d[pd.to_datetime(d["date"]) >= DEV_CUTOFF]),
    ]:
        hits = p[p["hit50"] == 1].copy()
        for branch, col in [
            ("General","s2_general_pct"),
            ("Ignition","s2_ignition_pct"),
            ("SecondLeg","s2_second_pct"),
        ]:
            if hits.empty:
                continue
            rows.append({
                "期間": period_name,
                "分支": branch,
                "+50樣本數": int(len(hits)),
                "成功股分支分數中位數": pd.to_numeric(hits[col], errors="coerce").median(),
                "成功股分支分數75分位": pd.to_numeric(hits[col], errors="coerce").quantile(0.75),
                "S2 Top20內成功股數": int(((pd.to_numeric(hits[rcol], errors="coerce") <= 20)).sum()),
            })
    return pd.DataFrame(rows)
