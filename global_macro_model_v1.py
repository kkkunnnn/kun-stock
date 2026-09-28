from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

DEV_CUTOFF = pd.Timestamp("2025-01-01")

MACRO_TICKERS = {
    "^GSPC": "SPX",
    "^NDX": "NDX",
    "^SOX": "SOX",
    "^VIX": "VIX",
    "^TNX": "TNX",
    "TWD=X": "USDTWD",
    "CL=F": "WTI",
    "BZ=F": "BRENT",
    "DX-Y.NYB": "DXY",
    "TSM": "TSM",
    "NVDA": "NVDA",
    "AMD": "AMD",
    "AVGO": "AVGO",
    "MU": "MU",
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "AMZN": "AMZN",
}

MACRO_INTERACTION_FEATURES = [
    "sox_align",
    "ndx_align",
    "techlead_align",
    "usd_align",
    "oil_align",
    "rate_align",
    "riskoff_align",
    "spx_align",
]

MACRO_CONFIGS = {
    "M0_S2_ONLY": 0.00,
    "M10": 0.10,
    "M20": 0.20,
    "M30": 0.30,
}


def download_macro_history(start_date: str) -> pd.DataFrame:
    raw = yf.download(
        list(MACRO_TICKERS.keys()),
        start=start_date,
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    if raw is None or raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        return pd.DataFrame()

    frames = []
    for ticker, name in MACRO_TICKERS.items():
        try:
            if ticker not in raw.columns.get_level_values(1):
                continue
            x = pd.DataFrame({
                "date": raw.index,
                f"{name}_close": raw[("Close", ticker)],
            }).set_index("date")
            frames.append(x)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    m = pd.concat(frames, axis=1).reset_index()
    m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.tz_localize(None)
    m = m.sort_values("date").reset_index(drop=True)

    # Forward-fill only market series across differing market holidays.
    value_cols = [c for c in m.columns if c != "date"]
    m[value_cols] = m[value_cols].ffill(limit=5)

    for name in MACRO_TICKERS.values():
        c = f"{name}_close"
        if c not in m.columns:
            continue
        m[f"{name}_ret1"] = pd.to_numeric(m[c], errors="coerce").pct_change()
        m[f"{name}_ret5"] = pd.to_numeric(m[c], errors="coerce").pct_change(5) * 100
        m[f"{name}_ret20"] = pd.to_numeric(m[c], errors="coerce").pct_change(20) * 100
        m[f"{name}_vol20"] = m[f"{name}_ret1"].rolling(20, min_periods=20).std() * np.sqrt(252) * 100

    # Rates and VIX are more naturally interpreted by level changes.
    if "TNX_close" in m.columns:
        m["TNX_delta20"] = pd.to_numeric(m["TNX_close"], errors="coerce").diff(20)
    if "VIX_close" in m.columns:
        m["VIX_delta20"] = pd.to_numeric(m["VIX_close"], errors="coerce").diff(20)

    tech_cols = [f"{x}_ret20" for x in ["TSM","NVDA","AMD","AVGO","MU"] if f"{x}_ret20" in m.columns]
    if tech_cols:
        m["TECHLEAD_ret20"] = m[tech_cols].mean(axis=1)

    return m


def attach_stock_macro_interactions(dataset: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Make global information stock-specific via rolling exposure × current macro move.
    Common macro variables alone cannot change a same-day cross-sectional ranking.
    """
    if dataset.empty or macro.empty:
        return dataset.copy()

    d = dataset.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    m = macro.copy().sort_values("date")
    keep = ["date"] + [c for c in m.columns if c.endswith("_ret1") or c.endswith("_ret20") or c in {
        "TNX_delta20","VIX_delta20","VIX_close","TECHLEAD_ret20"
    }]
    keep = list(dict.fromkeys([c for c in keep if c in m.columns]))
    d = d.merge(m[keep], on="date", how="left")

    parts = []
    for _, g in d.groupby("stock_id", sort=False):
        g = g.sort_values("date").copy()
        sr = pd.to_numeric(g["close"], errors="coerce").pct_change()

        def corr60(col):
            if col not in g.columns:
                return pd.Series(np.nan, index=g.index)
            return sr.rolling(60, min_periods=40).corr(pd.to_numeric(g[col], errors="coerce"))

        g["corr60_SOX"] = corr60("SOX_ret1")
        g["corr60_NDX"] = corr60("NDX_ret1")
        g["corr60_SPX"] = corr60("SPX_ret1")
        g["corr60_USDTWD"] = corr60("USDTWD_ret1")
        g["corr60_WTI"] = corr60("WTI_ret1")
        g["corr60_TNX"] = corr60("TNX_ret1")
        g["corr60_VIX"] = corr60("VIX_ret1")

        g["sox_align"] = g["corr60_SOX"] * pd.to_numeric(g.get("SOX_ret20"), errors="coerce")
        g["ndx_align"] = g["corr60_NDX"] * pd.to_numeric(g.get("NDX_ret20"), errors="coerce")
        g["spx_align"] = g["corr60_SPX"] * pd.to_numeric(g.get("SPX_ret20"), errors="coerce")
        g["usd_align"] = g["corr60_USDTWD"] * pd.to_numeric(g.get("USDTWD_ret20"), errors="coerce")
        g["oil_align"] = g["corr60_WTI"] * pd.to_numeric(g.get("WTI_ret20"), errors="coerce")
        g["rate_align"] = g["corr60_TNX"] * pd.to_numeric(g.get("TNX_delta20"), errors="coerce")
        # Falling VIX should benefit stocks positively correlated with risk-on / negatively with VIX.
        g["riskoff_align"] = -g["corr60_VIX"] * pd.to_numeric(g.get("VIX_delta20"), errors="coerce")
        tech = pd.to_numeric(g.get("TECHLEAD_ret20"), errors="coerce")
        g["techlead_align"] = g["corr60_SOX"].fillna(g["corr60_NDX"]) * tech

        parts.append(g)

    return pd.concat(parts, ignore_index=True)


def _smoothed_prob(train: pd.DataFrame, test: pd.DataFrame, feature: str, bins: int = 10) -> np.ndarray:
    base = float(pd.to_numeric(train["hit50"], errors="coerce").mean())
    tr = pd.DataFrame({
        "x": pd.to_numeric(train.get(feature), errors="coerce"),
        "y": pd.to_numeric(train["hit50"], errors="coerce"),
    }).dropna()
    tx = pd.to_numeric(test.get(feature), errors="coerce")
    if len(tr) < 500 or tr["x"].nunique() < 8:
        return np.full(len(test), base * 100)

    qs = np.unique(np.nanquantile(tr["x"].to_numpy(float), np.linspace(0, 1, bins + 1)))
    if len(qs) < 4:
        return np.full(len(test), base * 100)
    qs[0] = -np.inf
    qs[-1] = np.inf
    b = pd.cut(tr["x"], bins=qs, include_lowest=True, labels=False, duplicates="drop")
    stats = pd.DataFrame({"bin": b, "y": tr["y"]}).dropna().groupby("bin")["y"].agg(["mean","count"])
    stats["p"] = (stats["mean"] * stats["count"] + base * 150.0) / (stats["count"] + 150.0)
    tb = pd.cut(tx, bins=qs, include_lowest=True, labels=False, duplicates="drop")
    return pd.Series(tb).map(stats["p"]).fillna(base).to_numpy(float) * 100


def add_macro_component(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    out = test.copy().reset_index(drop=True)
    pieces = []
    for f in MACRO_INTERACTION_FEATURES:
        if f in train.columns and f in out.columns:
            pieces.append(_smoothed_prob(train, out, f))
    base = float(pd.to_numeric(train["hit50"], errors="coerce").mean() * 100)
    out["macro_v1_raw"] = np.nanmean(np.vstack(pieces), axis=0) if pieces else base
    return out


def _pct(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rank(pct=True, method="average") * 100


def add_macro_rankings(results: pd.DataFrame, s2_selected: str) -> pd.DataFrame:
    d = results.copy()
    blocks = []
    s2_score = f"{s2_selected}_score"
    for _, g in d.groupby("date", sort=False):
        g = g.copy()
        base_pct = _pct(g[s2_score]) if s2_score in g.columns else (
            100 - _pct(pd.to_numeric(g[f"{s2_selected}_rank"], errors="coerce"))
        )
        macro_pct = _pct(g["macro_v1_raw"])
        g["macro_v1_pct"] = macro_pct
        for name, w in MACRO_CONFIGS.items():
            score = (1 - w) * base_pct + w * macro_pct
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
    base = actual / len(x) if len(x) else np.nan
    p = tp / n if n else np.nan
    r = tp / actual if actual else np.nan
    return {
        "Precision": p * 100 if pd.notna(p) else np.nan,
        "Recall": r * 100 if pd.notna(r) else np.nan,
        "Lift": p / base if pd.notna(p) and pd.notna(base) and base > 0 else np.nan,
        "TP": tp,
        "選中樣本": int(n),
        "實際+50": actual,
    }


def select_macro_on_development(results: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    dev = results[pd.to_datetime(results["date"]) < DEV_CUTOFF].copy()
    rows = []
    for name, w in MACRO_CONFIGS.items():
        rcol = f"{name}_rank"
        m10 = _metrics(dev, rcol, 10)
        m20 = _metrics(dev, rcol, 20)
        m50 = _metrics(dev, rcol, 50)
        objective = 0.45*m10["Precision"] + 0.20*m20["Precision"] + 0.35*m50["Recall"]
        rows.append({
            "config": name,
            "Macro權重": w,
            "開發期目標分數": objective,
            "Top10 Precision": m10["Precision"],
            "Top10 Recall": m10["Recall"],
            "Top10 Lift": m10["Lift"],
            "Top20 Precision": m20["Precision"],
            "Top20 Recall": m20["Recall"],
            "Top50 Recall": m50["Recall"],
        })
    t = pd.DataFrame(rows).sort_values(
        ["開發期目標分數","Top10 Precision","Top50 Recall"], ascending=False
    ).reset_index(drop=True)
    t["開發期排名"] = np.arange(1, len(t)+1)
    return str(t.iloc[0]["config"]), t


def compare_macro(results: pd.DataFrame, selected: str, s2_selected: str) -> pd.DataFrame:
    d = results.copy()
    periods = {
        "開發期": d[pd.to_datetime(d["date"]) < DEV_CUTOFF],
        "OOS期": d[pd.to_datetime(d["date"]) >= DEV_CUTOFF],
        "全期間": d,
    }
    rows = []
    for period_name, p in periods.items():
        for model_name, rcol in [
            ("S2 leading", f"{s2_selected}_rank"),
            ("S2+GlobalMacro", f"{selected}_rank"),
        ]:
            for k in [5,10,20,30,50,100]:
                m = _metrics(p, rcol, k)
                rows.append({
                    "期間": period_name,
                    "模型": model_name,
                    "設定": selected if model_name == "S2+GlobalMacro" else s2_selected,
                    "K": k,
                    **m,
                })
    return pd.DataFrame(rows)


def macro_feature_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    d = results.copy()
    rows = []
    for f in MACRO_INTERACTION_FEATURES:
        if f not in d.columns:
            continue
        z = pd.DataFrame({
            "x": pd.to_numeric(d[f], errors="coerce"),
            "y": pd.to_numeric(d["hit50"], errors="coerce"),
        }).dropna()
        if len(z) < 100 or z["x"].nunique() < 10:
            continue
        q25, q75 = z["x"].quantile([0.25,0.75])
        low = z[z["x"] <= q25]["y"].mean()
        high = z[z["x"] >= q75]["y"].mean()
        rows.append({
            "特徵": f,
            "+50成功中位數": z[z["y"] == 1]["x"].median(),
            "未成功中位數": z[z["y"] == 0]["x"].median(),
            "低25%命中率": low*100,
            "高25%命中率": high*100,
            "高低差": (high-low)*100,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("高低差", ascending=False).reset_index(drop=True) if not out.empty else out
