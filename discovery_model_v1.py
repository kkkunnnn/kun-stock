from __future__ import annotations

import numpy as np
import pandas as pd

DEV_CUTOFF = pd.Timestamp("2025-01-01")

# Discovery is a second list, not a replacement for the S2 primary ranking.
# It only searches names outside the S2 Top50.
DISCOVERY_CONFIGS = {
    "D1_IGN100_GATE": {"ign": 1.00, "general": 0.00, "s2": 0.00, "gate": True},
    "D2_IGN80_GEN20_GATE": {"ign": 0.80, "general": 0.20, "s2": 0.00, "gate": True},
    "D3_IGN70_GEN20_S210_GATE": {"ign": 0.70, "general": 0.20, "s2": 0.10, "gate": True},
    "D4_IGN80_S220_GATE": {"ign": 0.80, "general": 0.00, "s2": 0.20, "gate": True},
    "D5_IGN100_ALL": {"ign": 1.00, "general": 0.00, "s2": 0.00, "gate": False},
}


def _pct(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rank(pct=True, method="average") * 100


def add_discovery_rankings(results: pd.DataFrame, s2_selected: str) -> pd.DataFrame:
    d = results.copy()
    s2_rank_col = f"{s2_selected}_rank"
    s2_score_col = f"{s2_selected}_score"
    blocks = []

    for _, g in d.groupby("date", sort=False):
        g = g.copy()
        s2_rank = pd.to_numeric(g[s2_rank_col], errors="coerce")
        pool = s2_rank > 50

        ign_pct = _pct(g["ignition_v2_raw"])
        gen_pct = _pct(g["s2_general_raw"])
        if s2_score_col in g.columns:
            s2_pct = _pct(g[s2_score_col])
        else:
            s2_pct = (1 - (s2_rank - 1) / max(len(g) - 1, 1)) * 100

        gate = g.get("ignition_v2_gate", pd.Series(False, index=g.index))
        gate = pd.Series(gate, index=g.index).fillna(False).astype(bool)

        for name, cfg in DISCOVERY_CONFIGS.items():
            eligible = pool & (gate if cfg["gate"] else True)
            score = (
                cfg["ign"] * ign_pct
                + cfg["general"] * gen_pct
                + cfg["s2"] * s2_pct
            )
            score = score.where(eligible, np.nan)
            g[f"{name}_score"] = score
            g[f"{name}_rank"] = score.rank(ascending=False, method="first")

        blocks.append(g)

    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def _metrics(d: pd.DataFrame, rank_col: str, k: int) -> dict:
    x = d[d["hit50"].notna()].copy()
    picked = x[pd.to_numeric(x[rank_col], errors="coerce") <= k]
    actual = int((pd.to_numeric(x["hit50"], errors="coerce") == 1).sum())
    tp = int((pd.to_numeric(picked["hit50"], errors="coerce") == 1).sum())
    n = int(len(picked))
    precision = tp / n if n else np.nan
    recall = tp / actual if actual else np.nan
    return {
        "Precision": precision * 100 if pd.notna(precision) else np.nan,
        "Recall_all50": recall * 100 if pd.notna(recall) else np.nan,
        "TP": tp,
        "選中樣本": n,
        "實際+50": actual,
    }


def _miss_metrics(d: pd.DataFrame, rank_col: str, s2_selected: str, k: int) -> dict:
    s2_rank = pd.to_numeric(d[f"{s2_selected}_rank"], errors="coerce")
    hit = pd.to_numeric(d["hit50"], errors="coerce")
    misses = d[(hit == 1) & (s2_rank > 50)].copy()
    caught = misses[pd.to_numeric(misses[rank_col], errors="coerce") <= k]
    return {
        "S2Top50外+50樣本": int(len(misses)),
        "救回數": int(len(caught)),
        "MissRecall": (len(caught) / len(misses) * 100) if len(misses) else np.nan,
    }


def select_discovery_on_development(results: pd.DataFrame, s2_selected: str) -> tuple[str, pd.DataFrame]:
    dev = results[pd.to_datetime(results["date"]) < DEV_CUTOFF].copy()
    rows = []

    for name in DISCOVERY_CONFIGS:
        rcol = f"{name}_rank"
        m10 = _metrics(dev, rcol, 10)
        m20 = _metrics(dev, rcol, 20)
        miss10 = _miss_metrics(dev, rcol, s2_selected, 10)
        miss20 = _miss_metrics(dev, rcol, s2_selected, 20)

        # Main objective: a useful SECOND list must still have precision,
        # while recovering winners that the S2 Top50 missed.
        objective = (
            0.50 * m10["Precision"]
            + 0.20 * m20["Precision"]
            + 0.20 * miss10["MissRecall"]
            + 0.10 * miss20["MissRecall"]
        )
        rows.append({
            "config": name,
            "開發期目標分數": objective,
            "DiscoveryTop10 Precision": m10["Precision"],
            "DiscoveryTop10 全體Recall": m10["Recall_all50"],
            "DiscoveryTop10 MissRecall": miss10["MissRecall"],
            "DiscoveryTop20 Precision": m20["Precision"],
            "DiscoveryTop20 MissRecall": miss20["MissRecall"],
            "gate_only": DISCOVERY_CONFIGS[name]["gate"],
            "w_ignition": DISCOVERY_CONFIGS[name]["ign"],
            "w_general": DISCOVERY_CONFIGS[name]["general"],
            "w_s2": DISCOVERY_CONFIGS[name]["s2"],
        })

    table = pd.DataFrame(rows).sort_values(
        ["開發期目標分數", "DiscoveryTop10 Precision", "DiscoveryTop10 MissRecall"],
        ascending=False,
    ).reset_index(drop=True)
    table["開發期排名"] = np.arange(1, len(table) + 1)
    return str(table.iloc[0]["config"]), table


def evaluate_discovery(results: pd.DataFrame, selected: str, s2_selected: str) -> pd.DataFrame:
    d = results.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    periods = {
        "開發期": d[d["date"] < DEV_CUTOFF],
        "OOS期": d[d["date"] >= DEV_CUTOFF],
        "全期間": d,
    }

    rows = []
    for period_name, p in periods.items():
        for k in [5, 10, 20, 30, 50]:
            rcol = f"{selected}_rank"
            m = _metrics(p, rcol, k)
            mm = _miss_metrics(p, rcol, s2_selected, k)
            rows.append({
                "期間": period_name,
                "模型": "Discovery V1",
                "設定": selected,
                "K": k,
                **m,
                **mm,
            })
    return pd.DataFrame(rows)


def compare_primary_plus_discovery(results: pd.DataFrame, selected: str, s2_selected: str) -> pd.DataFrame:
    """
    Compare S2 primary Top10 versus a 20-name combined watchlist:
    S2 Top10 + Discovery Top10 (zero overlap by construction).
    Also report S2 Top20 so the extra-list benefit has a fair same-size benchmark.
    """
    d = results.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    periods = {
        "開發期": d[d["date"] < DEV_CUTOFF],
        "OOS期": d[d["date"] >= DEV_CUTOFF],
        "全期間": d,
    }
    rows = []

    for period_name, p in periods.items():
        for model_name, mask_builder in [
            ("S2 Top10", lambda g: pd.to_numeric(g[f"{s2_selected}_rank"], errors="coerce") <= 10),
            ("S2 Top20", lambda g: pd.to_numeric(g[f"{s2_selected}_rank"], errors="coerce") <= 20),
            ("S2 Top10 + Discovery Top10", lambda g: (
                (pd.to_numeric(g[f"{s2_selected}_rank"], errors="coerce") <= 10)
                | (pd.to_numeric(g[f"{selected}_rank"], errors="coerce") <= 10)
            )),
        ]:
            picks = []
            for _, g in p.groupby("date"):
                picks.append(g[mask_builder(g)])
            picked = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
            tp = int((pd.to_numeric(picked.get("hit50"), errors="coerce") == 1).sum()) if not picked.empty else 0
            actual = int((pd.to_numeric(p["hit50"], errors="coerce") == 1).sum())
            n = len(picked)
            precision = tp / n * 100 if n else np.nan
            recall = tp / actual * 100 if actual else np.nan
            rows.append({
                "期間": period_name,
                "名單": model_name,
                "選中樣本": int(n),
                "+50成功數": tp,
                "Precision": precision,
                "Recall": recall,
            })
    return pd.DataFrame(rows)


def discovery_stock_samples(results: pd.DataFrame, selected: str, s2_selected: str) -> pd.DataFrame:
    """Diagnostic characteristics of Discovery Top10 winners vs misses."""
    d = results.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["hit50"] = pd.to_numeric(d["hit50"], errors="coerce")
    rcol = f"{selected}_rank"

    rows = []
    for period_name, p in [
        ("開發期", d[d["date"] < DEV_CUTOFF]),
        ("OOS期", d[d["date"] >= DEV_CUTOFF]),
    ]:
        top = p[pd.to_numeric(p[rcol], errors="coerce") <= 10].copy()
        for label, z in [
            ("Discovery Top10成功", top[top["hit50"] == 1]),
            ("Discovery Top10未成功", top[top["hit50"] == 0]),
        ]:
            if z.empty:
                continue
            rows.append({
                "期間": period_name,
                "類型": label,
                "樣本數": int(len(z)),
                "S2排名中位數": pd.to_numeric(z[f"{s2_selected}_rank"], errors="coerce").median(),
                "Ignition分數中位數": pd.to_numeric(z["ignition_v2_raw"], errors="coerce").median(),
                "ATR%中位數": pd.to_numeric(z["atr_pct"], errors="coerce").median(),
                "20日波動中位數": pd.to_numeric(z["vol20_ann"], errors="coerce").median(),
                "20日報酬中位數": pd.to_numeric(z["ret20"], errors="coerce").median(),
                "前60日漲幅中位數": pd.to_numeric(z["prior60_runup"], errors="coerce").median(),
                "40日MFE中位數": pd.to_numeric(z["mfe40"], errors="coerce").median(),
                "到+50中位天數": pd.to_numeric(z["days_to_hit50"], errors="coerce").median(),
            })
    return pd.DataFrame(rows)
