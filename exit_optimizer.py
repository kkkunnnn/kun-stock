from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExitConfig:
    strategy_id: str
    max_hold: int
    stop_pct: float | None = None
    fixed_target_pct: float | None = None
    trail_activate_pct: float | None = None
    trail_pct: float | None = None


def build_exit_grid() -> list[ExitConfig]:
    """Compact grid designed to test robustness without an excessive search space."""
    out: list[ExitConfig] = []

    for hold in (20, 40, 60):
        out.append(ExitConfig(f"HOLD_{hold}", max_hold=hold))

    for stop in (0.10, 0.15, 0.18):
        for hold in (20, 40, 60):
            out.append(
                ExitConfig(
                    f"SL{int(stop*100)}_HOLD{hold}",
                    max_hold=hold,
                    stop_pct=stop,
                )
            )

    for stop in (0.10, 0.15, 0.18):
        for hold in (40, 60):
            out.append(
                ExitConfig(
                    f"SL{int(stop*100)}_TP50_H{hold}",
                    max_hold=hold,
                    stop_pct=stop,
                    fixed_target_pct=0.50,
                )
            )

    for stop in (None, 0.10, 0.15, 0.18):
        stop_tag = "NS" if stop is None else f"SL{int(stop*100)}"
        for activate in (0.30, 0.40):
            for trail in (0.10, 0.15, 0.20):
                for hold in (40, 60):
                    out.append(
                        ExitConfig(
                            f"{stop_tag}_A{int(activate*100)}_T{int(trail*100)}_H{hold}",
                            max_hold=hold,
                            stop_pct=stop,
                            trail_activate_pct=activate,
                            trail_pct=trail,
                        )
                    )

    return out


def _price_map(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    for c in ["open", "high", "low", "close"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.dropna(subset=["date", "open", "high", "low", "close"])
    p = p.sort_values(["stock_id", "date"])
    return {
        str(code): g[["date", "open", "high", "low", "close"]].reset_index(drop=True)
        for code, g in p.groupby("stock_id")
    }


def _signal_paths(
    results: pd.DataFrame,
    prices: pd.DataFrame,
    max_rank: int = 10,
    max_hold: int = 60,
) -> list[dict]:
    pm = _price_map(prices)
    signals = results[pd.to_numeric(results["rank"], errors="coerce") <= max_rank].copy()

    paths = []
    for _, r in signals.iterrows():
        code = str(r["stock_id"])
        signal_date = pd.to_datetime(r["date"], errors="coerce")
        g = pm.get(code)
        if g is None or pd.isna(signal_date):
            continue

        pos = g.index[g["date"] > signal_date]
        if len(pos) < max_hold:
            continue

        start = int(pos[0])
        path = g.iloc[start:start + max_hold].copy()
        if len(path) < max_hold:
            continue

        paths.append({
            "signal_date": signal_date,
            "stock_id": code,
            "rank": float(r.get("rank", np.nan)),
            "score_hit50": float(r.get("score_hit50", np.nan)),
            "dates": path["date"].to_numpy(),
            "open": path["open"].to_numpy(dtype=float),
            "high": path["high"].to_numpy(dtype=float),
            "low": path["low"].to_numpy(dtype=float),
            "close": path["close"].to_numpy(dtype=float),
        })
    return paths


def _simulate_array(path: dict, cfg: ExitConfig, cost_pct: float) -> dict:
    hold = min(cfg.max_hold, len(path["close"]))
    o = path["open"][:hold]
    h = path["high"][:hold]
    l = path["low"][:hold]
    c = path["close"][:hold]
    dates = path["dates"][:hold]

    entry = float(o[0])
    if not np.isfinite(entry) or entry <= 0:
        return {}

    exit_px = float(c[-1])
    exit_idx = hold - 1
    reason = f"{hold}日到期"
    peak = entry

    for i in range(hold):
        oi = float(o[i])
        hi = float(h[i])
        li = float(l[i])

        hard_stop = entry * (1 - cfg.stop_pct) if cfg.stop_pct is not None else None
        if hard_stop is not None:
            if oi <= hard_stop:
                exit_px = oi
                exit_idx = i
                reason = "跳空停損"
                break
            if li <= hard_stop:
                exit_px = hard_stop
                exit_idx = i
                reason = "停損"
                break

        if cfg.fixed_target_pct is not None:
            target = entry * (1 + cfg.fixed_target_pct)
            if oi >= target:
                exit_px = oi
                exit_idx = i
                reason = "跳空停利"
                break
            if hi >= target:
                exit_px = target
                exit_idx = i
                reason = f"+{int(cfg.fixed_target_pct*100)}%停利"
                break

        peak = max(peak, hi)
        if cfg.trail_activate_pct is not None and peak >= entry * (1 + cfg.trail_activate_pct):
            trail_stop = peak * (1 - cfg.trail_pct)
            if oi <= trail_stop:
                exit_px = oi
                exit_idx = i
                reason = "移動停利跳空"
                break
            if li <= trail_stop:
                exit_px = trail_stop
                exit_idx = i
                reason = "移動停利"
                break

    held_h = h[:exit_idx + 1]
    held_l = l[:exit_idx + 1]
    gross = (exit_px / entry - 1) * 100
    net = gross - cost_pct

    return {
        "exit_date": pd.Timestamp(dates[exit_idx]),
        "holding_days": int(exit_idx + 1),
        "gross_return_pct": gross,
        "net_return_pct": net,
        "mfe_pct": (float(np.nanmax(held_h)) / entry - 1) * 100,
        "mae_pct": (float(np.nanmin(held_l)) / entry - 1) * 100,
        "gross_hit20": gross >= 20,
        "gross_hit30": gross >= 30,
        "gross_hit50": gross >= 50,
        "net_hit20": net >= 20,
        "net_hit30": net >= 30,
        "net_hit50": net >= 50,
        "exit_reason": reason,
    }


def simulate_grid(
    results: pd.DataFrame,
    prices: pd.DataFrame,
    cost_pct: float = 0.60,
    max_rank: int = 10,
) -> pd.DataFrame:
    configs = build_exit_grid()
    paths = _signal_paths(results, prices, max_rank=max_rank, max_hold=60)
    rows = []

    for cfg in configs:
        for p in paths:
            sim = _simulate_array(p, cfg, cost_pct)
            if not sim:
                continue
            rows.append({
                "strategy_id": cfg.strategy_id,
                "max_hold": cfg.max_hold,
                "stop_pct": cfg.stop_pct,
                "fixed_target_pct": cfg.fixed_target_pct,
                "trail_activate_pct": cfg.trail_activate_pct,
                "trail_pct": cfg.trail_pct,
                "signal_date": p["signal_date"],
                "stock_id": p["stock_id"],
                "rank": p["rank"],
                "score_hit50": p["score_hit50"],
                **sim,
            })

    return pd.DataFrame(rows)


def _metric_row(d: pd.DataFrame) -> dict:
    x = pd.to_numeric(d["net_return_pct"], errors="coerce").dropna()
    if x.empty:
        return {}

    wins = x[x > 0].sum()
    losses = -x[x < 0].sum()
    pf = wins / losses if losses > 0 else np.nan

    return {
        "交易數": int(len(x)),
        "平均淨報酬": float(x.mean()),
        "中位數淨報酬": float(x.median()),
        "勝率": float((x > 0).mean() * 100),
        "Profit Factor": float(pf) if pd.notna(pf) else np.nan,
        "10分位淨報酬": float(x.quantile(0.10)),
        "最大單筆虧損": float(x.min()),
        "中位持有天數": float(pd.to_numeric(d["holding_days"], errors="coerce").median()),
        "毛+20%實現率": float(d["gross_hit20"].astype(float).mean() * 100),
        "毛+30%實現率": float(d["gross_hit30"].astype(float).mean() * 100),
        "毛+50%實現率": float(d["gross_hit50"].astype(float).mean() * 100),
        "淨+20%實現率": float(d["net_hit20"].astype(float).mean() * 100),
        "淨+30%實現率": float(d["net_hit30"].astype(float).mean() * 100),
        "淨+50%實現率": float(d["net_hit50"].astype(float).mean() * 100),
        "MFE中位數": float(pd.to_numeric(d["mfe_pct"], errors="coerce").median()),
        "MAE中位數": float(pd.to_numeric(d["mae_pct"], errors="coerce").median()),
    }


def summarize_optimizer(
    trades: pd.DataFrame,
    split_date: str = "2025-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    d = trades.copy()
    d["signal_date"] = pd.to_datetime(d["signal_date"], errors="coerce")
    split = pd.Timestamp(split_date)
    d["period"] = np.where(d["signal_date"] < split, "開發期", "OOS期")

    params = [
        "strategy_id", "max_hold", "stop_pct", "fixed_target_pct",
        "trail_activate_pct", "trail_pct",
    ]

    rows = []
    for strategy_id, g in d.groupby("strategy_id"):
        base = {c: g.iloc[0].get(c) for c in params}
        for period, p in g.groupby("period"):
            m = _metric_row(p)
            rows.append({**base, "期間": period, **m})

    long = pd.DataFrame(rows)
    if long.empty:
        return long, pd.DataFrame(), pd.DataFrame()

    dev = long[long["期間"].eq("開發期")].copy()
    oos = long[long["期間"].eq("OOS期")].copy()

    # Development-period-only ranking. This prevents picking the strategy by looking at OOS.
    score_cols = {
        "平均淨報酬": 0.25,
        "Profit Factor": 0.25,
        "10分位淨報酬": 0.20,
        "中位數淨報酬": 0.10,
        "毛+30%實現率": 0.10,
        "最大單筆虧損": 0.10,
    }
    dev["開發期綜合分數"] = 0.0
    for col, weight in score_cols.items():
        vals = pd.to_numeric(dev[col], errors="coerce")
        pct = vals.rank(pct=True, method="average")
        dev["開發期綜合分數"] += pct.fillna(0.5) * weight * 100

    dev["開發期排名"] = dev["開發期綜合分數"].rank(ascending=False, method="min")

    merge_cols = params + ["開發期綜合分數", "開發期排名"]
    all_summary = long.merge(dev[merge_cols], on=params, how="left")

    shortlist_ids = (
        dev.sort_values(["開發期排名", "Profit Factor"], ascending=[True, False])
        .head(10)["strategy_id"]
        .tolist()
    )
    shortlist = all_summary[all_summary["strategy_id"].isin(shortlist_ids)].copy()
    shortlist = shortlist.sort_values(["開發期排名", "期間"])

    yearly_rows = []
    short_trades = d[d["strategy_id"].isin(shortlist_ids)].copy()
    short_trades["年份"] = short_trades["signal_date"].dt.year
    for (strategy_id, year), g in short_trades.groupby(["strategy_id", "年份"]):
        m = _metric_row(g)
        rank_val = dev.loc[dev["strategy_id"].eq(strategy_id), "開發期排名"]
        yearly_rows.append({
            "strategy_id": strategy_id,
            "開發期排名": float(rank_val.iloc[0]) if len(rank_val) else np.nan,
            "年份": int(year),
            **m,
        })

    yearly = pd.DataFrame(yearly_rows).sort_values(["開發期排名", "年份"])
    return all_summary, shortlist, yearly
