from __future__ import annotations

import numpy as np
import pandas as pd

DEV_CUTOFF = pd.Timestamp("2025-01-01")

# Pre-declared, deliberately coarse threshold set to limit overfitting.
# None means a policy baseline rather than a threshold rule.
CANDIDATES = {
    "OH_EXCLUDE_ALL": None,
    "OH_KEEP_ALL": {"s2": 0, "general": 0, "second": 0, "risk_max": 999},
    "OH_A_LOOSE": {"s2": 65, "general": 55, "second": 65, "risk_max": 35},
    "OH_B_MID": {"s2": 70, "general": 55, "second": 65, "risk_max": 35},
    "OH_C_CURRENT": {"s2": 70, "general": 60, "second": 70, "risk_max": 35},
    "OH_D_BALANCED": {"s2": 70, "general": 65, "second": 75, "risk_max": 30},
    "OH_E_HIGH_S2": {"s2": 75, "general": 60, "second": 70, "risk_max": 35},
    "OH_F_STRICT": {"s2": 75, "general": 65, "second": 75, "risk_max": 30},
    "OH_G_VERY_STRICT": {"s2": 80, "general": 70, "second": 80, "risk_max": 25},
}


def _num(d: pd.DataFrame, col: str) -> pd.Series:
    if col not in d.columns:
        return pd.Series(np.nan, index=d.index, dtype=float)
    return pd.to_numeric(d[col], errors="coerce")


def add_overheat_proxy(results: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the live overheat/risk logic from historical point-in-time features."""
    d = results.copy()
    rsi = _num(d, "rsi14")
    ret5 = _num(d, "ret5")
    bias = _num(d, "ma20_bias")
    atr = _num(d, "atr_pct")
    vol = _num(d, "vol20_ann")
    dd20 = _num(d, "dd20_high")

    d["overheat_proxy"] = (
        (rsi >= 75)
        | (bias >= 12)
        | (ret5 >= 15)
    )

    risk = pd.Series(0.0, index=d.index)
    risk += np.where(rsi >= 75, 12, np.where(rsi >= 70, 5, 0))
    risk += np.where(ret5 >= 15, 12, np.where(ret5 >= 10, 5, 0))
    risk += np.where(bias >= 12, 15, np.where(bias >= 8, 7, 0))
    risk += np.where(atr >= 6, 10, np.where(atr >= 4.5, 5, 0))
    risk += np.where(vol >= 60, 10, np.where(vol >= 45, 5, 0))
    risk += np.where(dd20 <= -15, 8, 0)
    d["overheat_risk_proxy"] = risk.clip(upper=40)
    return d


def _healthy_mask(d: pd.DataFrame, config: str, s2_score_col: str) -> pd.Series:
    hot = d["overheat_proxy"].fillna(False).astype(bool)
    if config == "OH_EXCLUDE_ALL":
        return pd.Series(False, index=d.index)
    if config == "OH_KEEP_ALL":
        return hot

    cfg = CANDIDATES[config]
    return (
        hot
        & (_num(d, s2_score_col) >= cfg["s2"])
        & (_num(d, "s2_general_pct") >= cfg["general"])
        & (_num(d, "s2_second_pct") >= cfg["second"])
        & (_num(d, "overheat_risk_proxy") < cfg["risk_max"])
    )


def add_policy_rankings(results: pd.DataFrame, s2_selected: str) -> pd.DataFrame:
    d = add_overheat_proxy(results)
    s2_score_col = f"{s2_selected}_score"
    if s2_score_col not in d.columns:
        raise KeyError(f"missing {s2_score_col}")

    blocks = []
    for _, g in d.groupby("date", sort=False):
        g = g.copy()
        base_score = _num(g, s2_score_col)
        hot = g["overheat_proxy"].fillna(False).astype(bool)

        for name in CANDIDATES:
            healthy = _healthy_mask(g, name, s2_score_col)
            eligible = (~hot) | healthy
            rank = pd.Series(np.nan, index=g.index, dtype=float)
            if eligible.any():
                rank.loc[eligible] = base_score.loc[eligible].rank(ascending=False, method="first")
            g[f"{name}_healthy_hot"] = healthy
            g[f"{name}_rank"] = rank
        blocks.append(g)

    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def _metrics(d: pd.DataFrame, rank_col: str, k: int) -> dict:
    x = d[d["hit50"].notna() & d[rank_col].notna()].copy()
    picked = x[_num(x, rank_col) <= k]
    hit = _num(x, "hit50")
    picked_hit = _num(picked, "hit50")
    actual = int((hit == 1).sum())
    tp = int((picked_hit == 1).sum())
    n = int(len(picked))
    pool = int(len(x))
    precision = tp / n if n else np.nan
    recall = tp / actual if actual else np.nan
    base = actual / pool if pool else np.nan
    return {
        "Precision": precision * 100 if pd.notna(precision) else np.nan,
        "Recall": recall * 100 if pd.notna(recall) else np.nan,
        "Lift": precision / base if pd.notna(precision) and pd.notna(base) and base > 0 else np.nan,
        "TP": tp,
        "選中樣本": n,
        "實際+50": actual,
    }


def _hot_winner_recall(d: pd.DataFrame, config: str) -> float:
    hot_winners = d[
        d["overheat_proxy"].fillna(False).astype(bool)
        & (_num(d, "hit50") == 1)
    ]
    if hot_winners.empty:
        return np.nan
    kept = d.loc[hot_winners.index, f"{config}_healthy_hot"].fillna(False).astype(bool)
    return float(kept.mean() * 100)


def select_on_development(results_ranked: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    dev = results_ranked[pd.to_datetime(results_ranked["date"]) < DEV_CUTOFF].copy()
    rows = []
    for name, cfg in CANDIDATES.items():
        rcol = f"{name}_rank"
        m10 = _metrics(dev, rcol, 10)
        m20 = _metrics(dev, rcol, 20)
        m50 = _metrics(dev, rcol, 50)
        hot_recall = _hot_winner_recall(dev, name)

        # Keep the same goal structure as S2 model selection:
        # prioritize Top10 precision, then Top20 precision and deeper recall.
        # Hot-winner retention is reported and used only as a tie-break, not rewarded directly.
        objective = (
            0.45 * m10["Precision"]
            + 0.20 * m20["Precision"]
            + 0.35 * m50["Recall"]
        )
        row = {
            "config": name,
            "開發期目標分數": objective,
            "Top10 Precision": m10["Precision"],
            "Top10 Recall": m10["Recall"],
            "Top20 Precision": m20["Precision"],
            "Top50 Recall": m50["Recall"],
            "過熱+50成功股保留率": hot_recall,
        }
        if cfg:
            row.update({
                "S2最低": cfg["s2"],
                "General最低": cfg["general"],
                "SecondLeg最低": cfg["second"],
                "風險上限": cfg["risk_max"],
            })
        rows.append(row)

    table = pd.DataFrame(rows).sort_values(
        ["開發期目標分數", "Top10 Precision", "Top50 Recall", "過熱+50成功股保留率"],
        ascending=False,
    ).reset_index(drop=True)
    table["開發期排名"] = np.arange(1, len(table) + 1)
    return str(table.iloc[0]["config"]), table


def compare_selected(results_ranked: pd.DataFrame, selected: str) -> pd.DataFrame:
    d = results_ranked.copy()
    periods = {
        "開發期": d[pd.to_datetime(d["date"]) < DEV_CUTOFF],
        "OOS期": d[pd.to_datetime(d["date"]) >= DEV_CUTOFF],
        "全期間": d,
    }
    policies = ["OH_EXCLUDE_ALL", selected, "OH_KEEP_ALL"]
    policies = list(dict.fromkeys(policies))

    rows = []
    for period_name, p in periods.items():
        for config in policies:
            for k in [5, 10, 20, 50]:
                m = _metrics(p, f"{config}_rank", k)
                rows.append({
                    "期間": period_name,
                    "政策": config,
                    "K": k,
                    **m,
                    "過熱+50成功股保留率": _hot_winner_recall(p, config),
                })
    return pd.DataFrame(rows)


def hot_cohort_summary(results_ranked: pd.DataFrame, selected: str, s2_selected: str) -> pd.DataFrame:
    d = results_ranked.copy()
    rows = []
    for period_name, p in [
        ("開發期", d[pd.to_datetime(d["date"]) < DEV_CUTOFF]),
        ("OOS期", d[pd.to_datetime(d["date"]) >= DEV_CUTOFF]),
    ]:
        hot = p[p["overheat_proxy"].fillna(False).astype(bool)].copy()
        if hot.empty:
            continue
        healthy = hot[hot[f"{selected}_healthy_hot"].fillna(False).astype(bool)]
        terminal = hot[~hot[f"{selected}_healthy_hot"].fillna(False).astype(bool)]
        for label, g in [("全部過熱", hot), ("強勢延續型", healthy), ("末端過熱", terminal)]:
            if g.empty:
                continue
            rows.append({
                "期間": period_name,
                "分類": label,
                "樣本數": int(len(g)),
                "+50命中率": float((_num(g, "hit50") == 1).mean() * 100),
                "40日MFE中位數": float(_num(g, "mfe40").median()),
                "40日MAE中位數": float(_num(g, "mae40").median()),
                "S2中位數": float(_num(g, f"{s2_selected}_score").median()),
                "SecondLeg PR中位數": float(_num(g, "s2_second_pct").median()),
                "風險代理中位數": float(_num(g, "overheat_risk_proxy").median()),
            })
    return pd.DataFrame(rows)


def selected_config_frame(selected: str, s2_selected: str) -> pd.DataFrame:
    cfg = CANDIDATES[selected]
    row = {
        "selected_config": selected,
        "s2_model": s2_selected,
        "selection_rule": "只用2025前開發期選過熱門檻；2025起OOS只驗證不參與選擇",
        "target": "40交易日內最高漲幅>=50%",
        "overheat_proxy": "RSI>=75 或 MA20乖離>=12% 或 5日漲幅>=15%",
    }
    if cfg is None:
        row.update({"s2_min": np.nan, "general_min": np.nan, "second_min": np.nan, "risk_max": np.nan})
    else:
        row.update({
            "s2_min": cfg["s2"],
            "general_min": cfg["general"],
            "second_min": cfg["second"],
            "risk_max": cfg["risk_max"],
        })
    return pd.DataFrame([row])
