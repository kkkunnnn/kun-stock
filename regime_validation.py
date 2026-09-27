from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


BENCHMARKS = {
    "^TWII": "TWII",
    "^NDX": "NDX",
    "^SOX": "SOX",
    "^VIX": "VIX",
    "TWD=X": "USDTWD",
}


def download_regime_history(start_date: str) -> pd.DataFrame:
    raw = yf.download(
        list(BENCHMARKS.keys()),
        start=start_date,
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    if raw is None or raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        return pd.DataFrame()

    frames = []
    for ticker, name in BENCHMARKS.items():
        try:
            if ticker not in raw.columns.get_level_values(1):
                continue
            x = pd.DataFrame({
                "date": raw.index,
                f"{name}_close": raw[("Close", ticker)],
            })
            frames.append(x.set_index("date"))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    d = pd.concat(frames, axis=1).reset_index()
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.tz_localize(None)
    return d.sort_values("date").reset_index(drop=True)


def build_breadth(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    p["close"] = pd.to_numeric(p["close"], errors="coerce")
    p = p.dropna(subset=["date","stock_id","close"]).sort_values(["stock_id","date"])

    parts = []
    for _, g in p.groupby("stock_id", sort=False):
        g = g.copy()
        g["ma20"] = g["close"].rolling(20, min_periods=20).mean()
        g["ret20"] = g["close"].pct_change(20)
        g["above_ma20"] = g["close"] > g["ma20"]
        g["up20"] = g["ret20"] > 0
        parts.append(g[["date","above_ma20","up20"]])

    if not parts:
        return pd.DataFrame()

    z = pd.concat(parts, ignore_index=True)
    out = z.groupby("date").agg(
        breadth_ma20=("above_ma20","mean"),
        breadth_up20=("up20","mean"),
        breadth_count=("above_ma20","count"),
    ).reset_index()
    out["breadth_ma20"] *= 100
    out["breadth_up20"] *= 100
    return out


def build_regime_table(
    prices: pd.DataFrame,
    start_date: str,
) -> pd.DataFrame:
    m = download_regime_history(start_date)
    if m.empty:
        return pd.DataFrame()

    b = build_breadth(prices)
    if not b.empty:
        m = m.merge(b, on="date", how="left")

    for prefix in ["TWII","NDX","SOX","USDTWD"]:
        c = f"{prefix}_close"
        if c in m.columns:
            m[f"{prefix}_ma20"] = m[c].rolling(20, min_periods=20).mean()
            m[f"{prefix}_ma60"] = m[c].rolling(60, min_periods=60).mean()
            m[f"{prefix}_ret20"] = m[c].pct_change(20) * 100

    # Domestic regime: trend + breadth, all point-in-time.
    domestic_score = pd.Series(0.0, index=m.index)
    if "TWII_close" in m.columns:
        domestic_score += np.where(m["TWII_close"] > m["TWII_ma20"], 25, 0)
        domestic_score += np.where(m["TWII_close"] > m["TWII_ma60"], 25, 0)
        domestic_score += np.where(m["TWII_ma20"] > m["TWII_ma60"], 15, 0)
    if "breadth_ma20" in m.columns:
        domestic_score += np.where(m["breadth_ma20"] >= 55, 20,
                           np.where(m["breadth_ma20"] >= 45, 10, 0))
    if "breadth_up20" in m.columns:
        domestic_score += np.where(m["breadth_up20"] >= 55, 15,
                           np.where(m["breadth_up20"] >= 45, 7.5, 0))
    m["domestic_score"] = domestic_score.clip(0,100)

    # Global regime: Nasdaq/SOX trend + VIX.
    global_score = pd.Series(0.0, index=m.index)
    for prefix in ["NDX","SOX"]:
        c = f"{prefix}_close"
        if c in m.columns:
            global_score += np.where(m[c] > m[f"{prefix}_ma20"], 20, 0)
            global_score += np.where(m[c] > m[f"{prefix}_ma60"], 15, 0)
    if "VIX_close" in m.columns:
        global_score += np.where(m["VIX_close"] < 20, 30,
                        np.where(m["VIX_close"] < 30, 15, 0))
    m["global_score"] = global_score.clip(0,100)

    combined = m["domestic_score"] * 0.65 + m["global_score"] * 0.35
    m["regime_score"] = combined

    m["regime"] = pd.cut(
        combined,
        bins=[-np.inf, 30, 45, 60, 75, np.inf],
        labels=["偏空逆風","中性偏空","震盪中性","中性偏多","偏多順風"],
        ordered=True,
    ).astype(str)

    return m


def attach_regime(
    results: pd.DataFrame,
    regime: pd.DataFrame,
) -> pd.DataFrame:
    if results.empty or regime.empty:
        return pd.DataFrame()

    r = results.copy()
    r["date"] = pd.to_datetime(r["date"], errors="coerce")
    rg = regime.sort_values("date").copy()

    # asof ensures the signal only sees the latest regime available on or before signal date
    out = pd.merge_asof(
        r.sort_values("date"),
        rg.sort_values("date"),
        on="date",
        direction="backward",
    )
    return out


def summarize_regime_signals(
    results_with_regime: pd.DataFrame,
) -> pd.DataFrame:
    if results_with_regime.empty:
        return pd.DataFrame()

    rows = []
    for max_rank, group_name in [(5,"Top 5"),(10,"Top 10"),(20,"Top 20")]:
        base = results_with_regime[
            pd.to_numeric(results_with_regime["rank"], errors="coerce") <= max_rank
        ].copy()
        base = base[base["hit50"].notna()]

        for regime_name, g in base.groupby("regime", dropna=False):
            if len(g) < 30:
                continue
            hit20 = pd.to_numeric(g["hit20"], errors="coerce").mean() * 100
            hit30 = pd.to_numeric(g["hit30"], errors="coerce").mean() * 100
            hit50 = pd.to_numeric(g["hit50"], errors="coerce").mean() * 100
            rows.append({
                "群組": group_name,
                "市場Regime": str(regime_name),
                "樣本數": int(len(g)),
                "訊號日數": int(g["date"].nunique()),
                "40日+20%命中率": hit20,
                "40日+30%命中率": hit30,
                "40日+50%命中率": hit50,
                "40日MFE中位數": pd.to_numeric(g["mfe40"], errors="coerce").median(),
                "40日MAE中位數": pd.to_numeric(g["mae40"], errors="coerce").median(),
                "40日報酬中位數": pd.to_numeric(g["ret40_end"], errors="coerce").median(),
                "國內分數中位數": pd.to_numeric(g["domestic_score"], errors="coerce").median(),
                "全球分數中位數": pd.to_numeric(g["global_score"], errors="coerce").median(),
                "Regime分數中位數": pd.to_numeric(g["regime_score"], errors="coerce").median(),
            })
    return pd.DataFrame(rows)


def summarize_regime_exit(
    optimizer_trades: pd.DataFrame,
    regime: pd.DataFrame,
    strategy_id: str = "SL10_HOLD60",
) -> pd.DataFrame:
    if optimizer_trades.empty or regime.empty:
        return pd.DataFrame()

    d = optimizer_trades[optimizer_trades["strategy_id"].eq(strategy_id)].copy()
    if d.empty:
        return pd.DataFrame()

    d["signal_date"] = pd.to_datetime(d["signal_date"], errors="coerce")
    rg = regime.rename(columns={"date":"signal_date"}).sort_values("signal_date")
    d = pd.merge_asof(
        d.sort_values("signal_date"),
        rg.sort_values("signal_date"),
        on="signal_date",
        direction="backward",
    )

    rows = []
    for regime_name, g in d.groupby("regime", dropna=False):
        x = pd.to_numeric(g["net_return_pct"], errors="coerce").dropna()
        if len(x) < 30:
            continue
        wins = x[x > 0].sum()
        losses = -x[x < 0].sum()
        pf = wins / losses if losses > 0 else np.nan
        rows.append({
            "策略": strategy_id,
            "市場Regime": str(regime_name),
            "交易數": int(len(x)),
            "平均淨報酬": float(x.mean()),
            "中位數淨報酬": float(x.median()),
            "勝率": float((x > 0).mean() * 100),
            "Profit Factor": float(pf) if pd.notna(pf) else np.nan,
            "10分位淨報酬": float(x.quantile(0.10)),
            "毛+30%實現率": float(g["gross_hit30"].astype(float).mean() * 100),
            "毛+50%實現率": float(g["gross_hit50"].astype(float).mean() * 100),
        })
    return pd.DataFrame(rows)


def build_regime_gate_table(
    signal_summary: pd.DataFrame,
    exit_summary: pd.DataFrame,
) -> pd.DataFrame:
    if signal_summary.empty:
        return pd.DataFrame()

    top10 = signal_summary[signal_summary["群組"].eq("Top 10")].copy()
    if top10.empty:
        return pd.DataFrame()

    out = top10[[
        "市場Regime","樣本數","40日+50%命中率","40日MFE中位數",
        "40日MAE中位數","Regime分數中位數"
    ]].copy()

    if not exit_summary.empty:
        e = exit_summary[[
            "市場Regime","平均淨報酬","Profit Factor","勝率","10分位淨報酬"
        ]].copy()
        out = out.merge(e, on="市場Regime", how="left")

    # Descriptive gate, not a model retraining step.
    pf = pd.to_numeric(out.get("Profit Factor"), errors="coerce")
    hit50 = pd.to_numeric(out["40日+50%命中率"], errors="coerce")
    out["研究判定"] = np.select(
        [
            (pf >= 1.5) & (hit50 >= hit50.median()),
            (pf >= 1.0),
        ],
        [
            "歷史順風",
            "中性觀察",
        ],
        default="歷史逆風",
    )
    return out.sort_values("Regime分數中位數", ascending=False)
