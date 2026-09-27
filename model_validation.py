from __future__ import annotations

import numpy as np
import pandas as pd


def attach_professional_forward_metrics(snapshot_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    對每天實際留下的模型快照做 forward / live OOS 驗證。
    不重新回填舊訊號；只有當未來價格真的發生後才產生成熟樣本。
    """
    if snapshot_df is None or snapshot_df.empty or price_df is None or price_df.empty:
        return pd.DataFrame()

    required = {"股票代號", "快照日期"}
    if not required.issubset(snapshot_df.columns):
        return pd.DataFrame()
    if not {"股票代號", "日期", "收盤價"}.issubset(price_df.columns):
        return pd.DataFrame()

    p = price_df.copy()
    p["股票代號"] = p["股票代號"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
    p["日期"] = pd.to_datetime(p["日期"], errors="coerce")
    p["收盤價"] = pd.to_numeric(p["收盤價"], errors="coerce")
    p = p.dropna(subset=["日期", "收盤價"]).sort_values(["股票代號", "日期"])

    price_map = {
        code: g[["日期", "收盤價"]].reset_index(drop=True)
        for code, g in p.groupby("股票代號")
    }

    rows = []
    for _, r in snapshot_df.iterrows():
        code = str(r.get("股票代號", "")).zfill(4)
        snap_date = pd.to_datetime(r.get("快照日期"), errors="coerce")
        g = price_map.get(code)
        if g is None or pd.isna(snap_date):
            continue

        pos = g.index[g["日期"] >= snap_date]
        if len(pos) == 0:
            continue

        i = int(pos[0])
        base = float(g.iloc[i]["收盤價"])
        if not np.isfinite(base) or base <= 0:
            continue

        rec = r.to_dict()
        rec["驗證基準價"] = base

        for h in (5, 20, 40):
            end = i + h
            rec[f"{h}日成熟"] = bool(end < len(g))
            if end < len(g):
                future = g.iloc[i + 1 : end + 1]["收盤價"].astype(float)
                rec[f"{h}日後報酬率"] = (float(g.iloc[end]["收盤價"]) / base - 1) * 100
                rec[f"{h}日MFE"] = (float(future.max()) / base - 1) * 100
                rec[f"{h}日MAE"] = (float(future.min()) / base - 1) * 100
            else:
                rec[f"{h}日後報酬率"] = np.nan
                rec[f"{h}日MFE"] = np.nan
                rec[f"{h}日MAE"] = np.nan

        mfe40 = rec.get("40日MFE", np.nan)
        rec["40日達20%"] = bool(mfe40 >= 20) if pd.notna(mfe40) else np.nan
        rec["40日達30%"] = bool(mfe40 >= 30) if pd.notna(mfe40) else np.nan
        rec["40日達50%"] = bool(mfe40 >= 50) if pd.notna(mfe40) else np.nan

        rows.append(rec)

    return pd.DataFrame(rows)


def _group_metrics(label: str, d: pd.DataFrame, baseline_hit50: float | None) -> dict | None:
    if d is None or d.empty:
        return None

    mature = d[d.get("40日成熟", False) == True].copy()
    if mature.empty:
        return {
            "群組": label,
            "成熟樣本數": 0,
            "40日+20%命中率": np.nan,
            "40日+30%命中率": np.nan,
            "40日+50%命中率": np.nan,
            "+50% Lift": np.nan,
            "40日MFE中位數": np.nan,
            "40日MAE中位數": np.nan,
            "40日報酬中位數": np.nan,
            "40日正報酬率": np.nan,
        }

    def hit(col):
        x = mature[col].dropna() if col in mature.columns else pd.Series(dtype=float)
        return float(x.astype(float).mean() * 100) if len(x) else np.nan

    hit50 = hit("40日達50%")
    lift = (
        hit50 / baseline_hit50
        if baseline_hit50 is not None and np.isfinite(baseline_hit50) and baseline_hit50 > 0 and np.isfinite(hit50)
        else np.nan
    )

    end_ret = pd.to_numeric(mature.get("40日後報酬率"), errors="coerce")
    mfe = pd.to_numeric(mature.get("40日MFE"), errors="coerce")
    mae = pd.to_numeric(mature.get("40日MAE"), errors="coerce")

    return {
        "群組": label,
        "成熟樣本數": int(len(mature)),
        "40日+20%命中率": hit("40日達20%"),
        "40日+30%命中率": hit("40日達30%"),
        "40日+50%命中率": hit50,
        "+50% Lift": lift,
        "40日MFE中位數": float(mfe.median()) if mfe.notna().any() else np.nan,
        "40日MAE中位數": float(mae.median()) if mae.notna().any() else np.nan,
        "40日報酬中位數": float(end_ret.median()) if end_ret.notna().any() else np.nan,
        "40日正報酬率": float((end_ret.dropna() > 0).mean() * 100) if end_ret.notna().any() else np.nan,
    }


def professional_validation_summary(bt: pd.DataFrame) -> pd.DataFrame:
    """
    專業投資人最重要的 Top-K、命中率、Lift、MFE/MAE。
    Lift 以全部股票池成熟樣本的 +50% 命中率為基準。
    """
    if bt is None or bt.empty:
        return pd.DataFrame()

    baseline = bt[bt.get("40日成熟", False) == True].copy()
    if not baseline.empty and "40日達50%" in baseline.columns:
        x = baseline["40日達50%"].dropna().astype(float)
        baseline_hit50 = float(x.mean() * 100) if len(x) else None
    else:
        baseline_hit50 = None

    rank_col = "主模型排名" if "主模型排名" in bt.columns else "一週模型排名"

    groups = [("全部股票池", bt)]
    if rank_col in bt.columns:
        rank_num = pd.to_numeric(bt[rank_col], errors="coerce")
        groups.extend([
            ("Top 5", bt[rank_num <= 5]),
            ("Top 10", bt[rank_num <= 10]),
            ("Top 20", bt[rank_num <= 20]),
        ])

    if "進場判定" in bt.columns:
        actionable = bt[
            bt["進場判定"].astype(str).str.contains("可觀察進場|等待回檔|等待突破", regex=True, na=False)
        ]
        groups.append(("可操作訊號", actionable))

    rows = []
    for label, d in groups:
        row = _group_metrics(label, d, baseline_hit50)
        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows)


def regime_validation(bt: pd.DataFrame) -> pd.DataFrame:
    if bt is None or bt.empty:
        return pd.DataFrame()

    mature = bt[bt.get("40日成熟", False) == True].copy()
    if mature.empty:
        return pd.DataFrame()

    rows = []
    for col in ["台股環境判定", "全球環境判定"]:
        if col not in mature.columns:
            continue
        for regime, d in mature.groupby(col, dropna=False):
            x = d.get("40日達50%", pd.Series(dtype=float)).dropna()
            mfe = pd.to_numeric(d.get("40日MFE"), errors="coerce")
            mae = pd.to_numeric(d.get("40日MAE"), errors="coerce")
            rows.append({
                "分類": col.replace("判定", ""),
                "環境": str(regime),
                "成熟樣本數": int(len(d)),
                "40日+50%命中率": float(x.astype(float).mean() * 100) if len(x) else np.nan,
                "40日MFE中位數": float(mfe.median()) if mfe.notna().any() else np.nan,
                "40日MAE中位數": float(mae.median()) if mae.notna().any() else np.nan,
            })
    return pd.DataFrame(rows)


def validation_readiness(bt: pd.DataFrame) -> dict:
    if bt is None or bt.empty:
        return {
            "成熟樣本數": 0,
            "成熟交易日數": 0,
            "狀態": "尚未形成可審查樣本",
            "說明": "需要持續累積每日 forward OOS 訊號與後續 40 個交易日價格。",
        }

    mature = bt[bt.get("40日成熟", False) == True].copy()
    days = 0
    if "快照日期" in mature.columns and not mature.empty:
        days = pd.to_datetime(mature["快照日期"], errors="coerce").dt.normalize().nunique()

    n = int(len(mature))
    if n >= 1000 and days >= 60:
        status = "可進行初步專業審查"
        note = "樣本已具一定規模，但仍應繼續累積不同市場環境並做參數敏感度檢查。"
    elif n >= 300 and days >= 30:
        status = "初步觀察階段"
        note = "已有 forward OOS 樣本，但尚不足以做強結論。"
    else:
        status = "樣本累積中"
        note = "目前以資料品質與流程穩定為主，不應過度解讀命中率。"

    return {
        "成熟樣本數": n,
        "成熟交易日數": int(days),
        "狀態": status,
        "說明": note,
    }
