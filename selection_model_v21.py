from __future__ import annotations

import numpy as np
import pandas as pd

DEV_CUTOFF = pd.Timestamp("2025-01-01")

# Branch-specific ensemble patterns. These are NOT fixed weighted sums.
# Each pattern interleaves independently ranked branch lists, skips duplicates,
# then fills any remaining names with the original S1 ranking.
ENSEMBLE_PATTERNS = {
    "S21_A_BASE": ["base"],
    "S21_B_SECOND_HEAVY": ["second","general","second","ignition"],
    "S21_C_BALANCED": ["general","second","ignition"],
    "S21_D_IGNITION_HEAVY": ["ignition","general","ignition","second"],
    "S21_E_DUAL_EXPLOSION": ["second","ignition","general"],
    "S21_F_GENERAL_SECOND": ["general","second","general","ignition"],
}

BRANCH_SCORE_COL = {
    "base": "score_hit50",
    "general": "s2_general_raw",
    "ignition": "s2_ignition_raw",
    "second": "s2_second_raw",
}


def _eligible_mask(g: pd.DataFrame, branch: str) -> pd.Series:
    if branch == "ignition":
        ret20 = pd.to_numeric(g.get("ret20"), errors="coerce")
        bias = pd.to_numeric(g.get("ma20_bias"), errors="coerce")
        return ret20.between(-12, 15) & bias.between(-8, 8)
    if branch == "second":
        runup = pd.to_numeric(g.get("prior60_runup"), errors="coerce")
        return runup >= 35
    return pd.Series(True, index=g.index)


def _branch_order(g: pd.DataFrame, branch: str) -> list[int]:
    col = BRANCH_SCORE_COL[branch]
    x = g.loc[_eligible_mask(g, branch)].copy()
    if x.empty:
        return []
    x["_score"] = pd.to_numeric(x[col], errors="coerce")
    x["_base"] = pd.to_numeric(x["score_hit50"], errors="coerce")
    x = x.sort_values(["_score","_base"], ascending=[False,False], na_position="last")
    return x.index.tolist()


def _interleaved_rank(g: pd.DataFrame, pattern: list[str]) -> pd.Series:
    """
    Build one full cross-sectional ranking by round-robin branch selection.
    Each branch is an independent model/list. No branch score is averaged with another.
    """
    orders = {b: _branch_order(g, b) for b in set(pattern + ["base"])}
    ptr = {b: 0 for b in orders}
    selected = []
    used = set()

    # Round-robin until every stock has been assigned or branch lists are exhausted.
    max_steps = max(len(g) * max(len(pattern), 1) * 3, 100)
    steps = 0
    while len(selected) < len(g) and steps < max_steps:
        progressed = False
        for branch in pattern:
            arr = orders.get(branch, [])
            while ptr.get(branch, 0) < len(arr):
                idx = arr[ptr[branch]]
                ptr[branch] += 1
                if idx not in used:
                    selected.append(idx)
                    used.add(idx)
                    progressed = True
                    break
        if not progressed:
            break
        steps += 1

    # Fill unselected names using original S1 order to ensure a complete ranking.
    for idx in orders.get("base", []):
        if idx not in used:
            selected.append(idx)
            used.add(idx)

    # Safety fallback for any rows with missing scores.
    for idx in g.index:
        if idx not in used:
            selected.append(idx)
            used.add(idx)

    rank_map = {idx: rank for rank, idx in enumerate(selected, start=1)}
    return pd.Series(g.index.map(rank_map), index=g.index, dtype=float)


def add_s21_rankings(results: pd.DataFrame) -> pd.DataFrame:
    d = results.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    blocks = []
    for _, g in d.groupby("date", sort=False):
        g = g.copy()
        for name, pattern in ENSEMBLE_PATTERNS.items():
            g[f"{name}_rank"] = _interleaved_rank(g, pattern)
        blocks.append(g)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def _metrics(d: pd.DataFrame, rank_col: str, k: int) -> dict:
    x = d[d["hit50"].notna() & d[rank_col].notna()].copy()
    picked = x[pd.to_numeric(x[rank_col], errors="coerce") <= k]
    actual = int((pd.to_numeric(x["hit50"], errors="coerce") == 1).sum())
    tp = int((pd.to_numeric(picked["hit50"], errors="coerce") == 1).sum())
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


def select_s21_on_development(results: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    dev = results[pd.to_datetime(results["date"]) < DEV_CUTOFF].copy()
    rows = []
    for name, pattern in ENSEMBLE_PATTERNS.items():
        rcol = f"{name}_rank"
        m10 = _metrics(dev, rcol, 10)
        m20 = _metrics(dev, rcol, 20)
        m50 = _metrics(dev, rcol, 50)

        # Goal-aligned development score:
        # preserve top-10 precision while increasing discovery/recall deeper in the list.
        objective = (
            0.40 * m10["Precision"]
            + 0.20 * m20["Precision"]
            + 0.40 * m50["Recall"]
        )
        rows.append({
            "config": name,
            "分支輪替": " → ".join(pattern),
            "開發期目標分數": objective,
            "Top10 Precision": m10["Precision"],
            "Top10 Recall": m10["Recall"],
            "Top10 Lift": m10["Lift"],
            "Top20 Precision": m20["Precision"],
            "Top20 Recall": m20["Recall"],
            "Top50 Recall": m50["Recall"],
        })

    table = pd.DataFrame(rows).sort_values(
        ["開發期目標分數","Top10 Precision","Top50 Recall"],
        ascending=False,
    ).reset_index(drop=True)
    table["開發期排名"] = np.arange(1, len(table)+1)
    return str(table.iloc[0]["config"]), table


def compare_s21(results: pd.DataFrame, selected: str, s2_selected: str | None = None) -> pd.DataFrame:
    d = results.copy()
    d["S1_rank"] = pd.to_numeric(d["rank"], errors="coerce")

    models = [("S1原模型","S1_rank")]
    if s2_selected and f"{s2_selected}_rank" in d.columns:
        models.append(("S2加權候選",f"{s2_selected}_rank"))
    models.append(("S2.1分支模型",f"{selected}_rank"))

    periods = {
        "開發期": d[pd.to_datetime(d["date"]) < DEV_CUTOFF],
        "OOS期": d[pd.to_datetime(d["date"]) >= DEV_CUTOFF],
        "全期間": d,
    }

    rows = []
    for period_name, p in periods.items():
        for model_name, rcol in models:
            for k in [5,10,20,30,50,100]:
                m = _metrics(p, rcol, k)
                rows.append({
                    "期間": period_name,
                    "模型": model_name,
                    "設定": selected if model_name == "S2.1分支模型" else (s2_selected if model_name == "S2加權候選" else "S1"),
                    "K": k,
                    **m,
                })
    return pd.DataFrame(rows)


def branch_source_attribution(results: pd.DataFrame, selected: str) -> pd.DataFrame:
    """
    For stocks entering selected S2.1 Top20, report which branch ranked them strongest.
    This is descriptive attribution, not used to select the model.
    """
    d = results.copy()
    rcol = f"{selected}_rank"
    rows = []
    for period_name, p in [
        ("開發期", d[pd.to_datetime(d["date"]) < DEV_CUTOFF]),
        ("OOS期", d[pd.to_datetime(d["date"]) >= DEV_CUTOFF]),
    ]:
        x = p[pd.to_numeric(p[rcol], errors="coerce") <= 20].copy()
        if x.empty:
            continue

        # Percentile within each date gives comparable independent branch strength.
        attrs = []
        for _, g in x.groupby("date"):
            q = g.copy()
            for branch, col in [
                ("General","s2_general_raw"),
                ("Ignition","s2_ignition_raw"),
                ("SecondLeg","s2_second_raw"),
            ]:
                q[f"_{branch}"] = pd.to_numeric(q[col], errors="coerce").rank(pct=True) * 100
            score_cols = ["_General","_Ignition","_SecondLeg"]
            q["主要分支"] = q[score_cols].idxmax(axis=1).str.replace("_","",regex=False)
            attrs.append(q)
        z = pd.concat(attrs, ignore_index=True)

        for branch, g in z.groupby("主要分支"):
            rows.append({
                "期間": period_name,
                "主要分支": branch,
                "Top20樣本數": int(len(g)),
                "占Top20比例": len(g) / len(z) * 100,
                "+50命中率": pd.to_numeric(g["hit50"], errors="coerce").mean() * 100,
                "+50成功數": int((pd.to_numeric(g["hit50"], errors="coerce") == 1).sum()),
            })
    return pd.DataFrame(rows)
