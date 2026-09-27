from __future__ import annotations

from pathlib import Path
import math
import os
import numpy as np
import pandas as pd
import yfinance as yf

from exit_optimizer import simulate_grid, summarize_optimizer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

START_DATE = os.environ.get("WF_START_DATE", "2019-01-01")
MODEL_VERSION = "P50-WF-1.1"
ROUND_TRIP_COST_PCT = float(os.environ.get("EXIT_COST_PCT", "0.60"))
FEATURES = [
    "ret5", "ret20", "ret60", "rsi14", "vol_ratio",
    "atr_pct", "ma20_bias", "vol20_ann", "macd_hist",
]


def load_universe() -> list[str]:
    p = DATA_DIR / "weekly_model_latest.csv"
    if not p.exists():
        raise FileNotFoundError("找不到 data/weekly_model_latest.csv")
    d = pd.read_csv(p, dtype={"股票代號": str})
    ids = (
        d["股票代號"].astype(str)
        .str.replace(".0", "", regex=False)
        .str.zfill(4)
        .drop_duplicates()
        .tolist()
    )
    return ids


def download_prices(ids: list[str]) -> pd.DataFrame:
    frames = []
    batch_size = 40
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        tickers = [f"{x}.TW" for x in batch] + [f"{x}.TWO" for x in batch]
        print(f"下載 {start+1}–{min(start+batch_size, len(ids))}/{len(ids)}")
        raw = yf.download(
            tickers=tickers,
            start=START_DATE,
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="column",
        )
        if raw is None or raw.empty:
            continue

        for stock_id in batch:
            chosen = None
            for ticker in (f"{stock_id}.TW", f"{stock_id}.TWO"):
                try:
                    if not isinstance(raw.columns, pd.MultiIndex):
                        continue
                    if ticker not in raw.columns.get_level_values(1):
                        continue
                    x = pd.DataFrame({
                        "date": raw.index,
                        "open": raw[("Open", ticker)],
                        "high": raw[("High", ticker)],
                        "low": raw[("Low", ticker)],
                        "close": raw[("Close", ticker)],
                        "volume": raw[("Volume", ticker)],
                    }).dropna(subset=["close"])
                    if len(x) >= 140:
                        chosen = x.copy()
                        break
                except Exception:
                    continue
            if chosen is None:
                continue
            chosen["stock_id"] = stock_id
            frames.append(chosen)

    if not frames:
        raise RuntimeError("Yahoo Finance 沒有取得可用歷史資料")

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    return out.sort_values(["stock_id", "date"]).reset_index(drop=True)


def compute_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy().reset_index(drop=True)
    c = pd.to_numeric(g["close"], errors="coerce")
    h = pd.to_numeric(g["high"], errors="coerce")
    l = pd.to_numeric(g["low"], errors="coerce")
    v = pd.to_numeric(g["volume"], errors="coerce")

    g["ret5"] = c.pct_change(5) * 100
    g["ret20"] = c.pct_change(20) * 100
    g["ret60"] = c.pct_change(60) * 100

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g["rsi14"] = 100 - 100 / (1 + rs)
    g.loc[(avg_loss == 0) & avg_gain.notna(), "rsi14"] = 100

    ma20 = c.rolling(20, min_periods=20).mean()
    g["ma20_bias"] = (c / ma20 - 1) * 100
    g["vol_ratio"] = v / v.rolling(20, min_periods=20).mean().replace(0, np.nan)

    ret1 = c.pct_change()
    g["vol20_ann"] = ret1.rolling(20, min_periods=20).std() * np.sqrt(252) * 100

    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    g["atr_pct"] = atr14 / c.replace(0, np.nan) * 100

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    g["macd_hist"] = macd - signal

    n = len(g)
    future_mfe = np.full(n, np.nan)
    future_mae = np.full(n, np.nan)
    future_end = np.full(n, np.nan)
    label_end = [pd.NaT] * n

    closes = c.to_numpy(dtype=float)
    dates = g["date"].tolist()
    for i in range(n - 40):
        future = closes[i+1:i+41]
        if len(future) == 40 and np.isfinite(future).all() and np.isfinite(closes[i]) and closes[i] > 0:
            base = closes[i]
            future_mfe[i] = (future.max() / base - 1) * 100
            future_mae[i] = (future.min() / base - 1) * 100
            future_end[i] = (future[-1] / base - 1) * 100
            label_end[i] = dates[i+40]

    g["mfe40"] = future_mfe
    g["mae40"] = future_mae
    g["ret40_end"] = future_end
    g["label_end_date"] = label_end
    g["hit20"] = np.where(pd.notna(g["mfe40"]), g["mfe40"] >= 20, np.nan)
    g["hit30"] = np.where(pd.notna(g["mfe40"]), g["mfe40"] >= 30, np.nan)
    g["hit50"] = np.where(pd.notna(g["mfe40"]), g["mfe40"] >= 50, np.nan)
    return g


def make_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in prices.groupby("stock_id", sort=False):
        parts.append(compute_features(g))
    d = pd.concat(parts, ignore_index=True)
    d = d.dropna(subset=FEATURES).copy()
    return d


def weekly_test_dates(d: pd.DataFrame) -> list[pd.Timestamp]:
    all_dates = pd.Series(sorted(d["date"].dropna().unique()))
    # 每 5 個交易日取一個截面，降低同一行情重複計數與運算量。
    if len(all_dates) <= 200:
        return []
    dates = all_dates.iloc[::5].tolist()
    # 至少保留約兩年資料給第一個訓練窗
    cutoff = pd.Timestamp(START_DATE) + pd.DateOffset(years=2)
    return [pd.Timestamp(x) for x in dates if pd.Timestamp(x) >= cutoff]


def score_date(dataset: pd.DataFrame, test_date: pd.Timestamp, k: int = 250) -> pd.DataFrame:
    test = dataset[dataset["date"].eq(test_date)].copy()
    if test.empty:
        return pd.DataFrame()

    # Embargo：只有 label_end_date 已嚴格早於 test_date 的歷史樣本才能進訓練。
    train = dataset[
        dataset["label_end_date"].notna()
        & (pd.to_datetime(dataset["label_end_date"]) < test_date)
        & dataset["hit50"].notna()
    ].copy()

    if len(train) < 3000:
        return pd.DataFrame()

    means = train[FEATURES].mean()
    stds = train[FEATURES].std().replace(0, 1).fillna(1)

    ztrain = ((train[FEATURES] - means) / stds).clip(-5, 5).to_numpy(dtype=float)
    ztest = ((test[FEATURES] - means) / stds).clip(-5, 5).to_numpy(dtype=float)
    y = train["hit50"].astype(float).to_numpy()

    kk = min(k, len(train))
    out_rows = []
    for pos, (_, r) in enumerate(test.iterrows()):
        v = ztest[pos]
        dist = np.sqrt(((ztrain - v) ** 2).mean(axis=1))
        order = np.argpartition(dist, kk-1)[:kk]
        dsel = dist[order]
        w = 1.0 / (dsel + 0.20)
        hit_rate = float((y[order] * w).sum() / w.sum() * 100)

        out_rows.append({
            "date": test_date,
            "stock_id": r["stock_id"],
            "score_hit50": hit_rate,
            "mfe40": r.get("mfe40"),
            "mae40": r.get("mae40"),
            "ret40_end": r.get("ret40_end"),
            "hit20": r.get("hit20"),
            "hit30": r.get("hit30"),
            "hit50": r.get("hit50"),
            "train_samples": len(train),
            "train_base_hit50": float(train["hit50"].astype(float).mean() * 100),
        })

    out = pd.DataFrame(out_rows)
    out["rank"] = out["score_hit50"].rank(ascending=False, method="first")
    return out


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, max_rank in [("全部股票池", None), ("Top 5", 5), ("Top 10", 10), ("Top 20", 20)]:
        d = results.copy() if max_rank is None else results[results["rank"] <= max_rank].copy()
        d = d[d["hit50"].notna()]
        if d.empty:
            continue

        hit20 = d["hit20"].astype(float).mean() * 100
        hit30 = d["hit30"].astype(float).mean() * 100
        hit50 = d["hit50"].astype(float).mean() * 100
        rows.append({
            "群組": group,
            "成熟樣本數": len(d),
            "測試截面數": d["date"].nunique(),
            "40日+20%命中率": hit20,
            "40日+30%命中率": hit30,
            "40日+50%命中率": hit50,
            "40日MFE中位數": d["mfe40"].median(),
            "40日MAE中位數": d["mae40"].median(),
            "40日報酬中位數": d["ret40_end"].median(),
            "40日正報酬率": (d["ret40_end"] > 0).mean() * 100,
        })

    s = pd.DataFrame(rows)
    if s.empty:
        return s
    base = s.loc[s["群組"].eq("全部股票池"), "40日+50%命中率"]
    baseline = float(base.iloc[0]) if len(base) else np.nan
    s["+50% Lift"] = s["40日+50%命中率"] / baseline if baseline > 0 else np.nan
    return s




def _trade_path(prices_map: dict, stock_id: str, signal_date: pd.Timestamp):
    g = prices_map.get(str(stock_id))
    if g is None or g.empty:
        return None
    pos = g.index[g["date"] > signal_date]
    if len(pos) < 40:
        return None
    start = int(pos[0])
    path = g.iloc[start:start+40].copy().reset_index(drop=True)
    if len(path) < 40:
        return None
    return path


def _simulate_exit(path: pd.DataFrame, strategy: str) -> dict:
    """
    Long-only execution simulation.
    Entry = next trading day's open.
    If stop and target are both touched on the same day, assume stop is hit first
    (conservative because intraday sequence is unknown).
    ROUND_TRIP_COST_PCT is subtracted from gross return as an explicit friction assumption.
    """
    entry = float(path.iloc[0]["open"])
    if not np.isfinite(entry) or entry <= 0:
        return {}

    initial_stop_map = {
        "HOLD40": None,
        "SL8_TP50": 0.08,
        "SL10_TP50": 0.10,
        "SL12_TP50": 0.12,
        "TP15_TRAIL8": 0.08,
        "TP30_TRAIL10": 0.10,
    }
    stop_pct = initial_stop_map[strategy]
    fixed_target = 0.50 if strategy in {"SL8_TP50","SL10_TP50","SL12_TP50"} else None
    trail_activate = 0.15 if strategy == "TP15_TRAIL8" else (0.30 if strategy == "TP30_TRAIL10" else None)
    trail_pct = 0.08 if strategy == "TP15_TRAIL8" else (0.10 if strategy == "TP30_TRAIL10" else None)

    peak = entry
    exit_price = float(path.iloc[-1]["close"])
    exit_date = path.iloc[-1]["date"]
    exit_reason = "40日到期"
    exit_idx = len(path)-1

    for i, row in path.iterrows():
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])

        if strategy == "HOLD40":
            peak = max(peak, h)
            continue

        hard_stop = entry * (1 - stop_pct) if stop_pct is not None else None

        # gap through initial stop
        if hard_stop is not None and o <= hard_stop:
            exit_price = o
            exit_date = row["date"]
            exit_reason = "跳空停損"
            exit_idx = i
            break

        # conservative same-day assumption: hard stop before target
        if hard_stop is not None and l <= hard_stop:
            exit_price = hard_stop
            exit_date = row["date"]
            exit_reason = "停損"
            exit_idx = i
            break

        if fixed_target is not None:
            target = entry * (1 + fixed_target)
            if o >= target:
                exit_price = target
                exit_date = row["date"]
                exit_reason = "+50%停利"
                exit_idx = i
                break
            if h >= target:
                exit_price = target
                exit_date = row["date"]
                exit_reason = "+50%停利"
                exit_idx = i
                break

        old_peak = peak
        peak = max(peak, h)

        if trail_activate is not None and peak >= entry * (1 + trail_activate):
            trail_stop = peak * (1 - trail_pct)

            # if the day's high newly activates trailing and low also breaches it,
            # assume the adverse sequence for robustness.
            if o <= trail_stop:
                exit_price = o
                exit_date = row["date"]
                exit_reason = "移動停利跳空"
                exit_idx = i
                break
            if l <= trail_stop:
                exit_price = trail_stop
                exit_date = row["date"]
                exit_reason = "移動停利"
                exit_idx = i
                break

    held = path.iloc[:exit_idx+1]
    max_high = pd.to_numeric(held["high"], errors="coerce").max()
    min_low = pd.to_numeric(held["low"], errors="coerce").min()
    mfe = (max_high / entry - 1) * 100 if pd.notna(max_high) else np.nan
    mae = (min_low / entry - 1) * 100 if pd.notna(min_low) else np.nan
    gross = (exit_price / entry - 1) * 100
    net = gross - ROUND_TRIP_COST_PCT

    return {
        "entry_date": path.iloc[0]["date"],
        "entry_price": entry,
        "exit_date": exit_date,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "holding_days": int(exit_idx + 1),
        "gross_return_pct": gross,
        "net_return_pct": net,
        "mfe_during_trade": mfe,
        "mae_during_trade": mae,
    }


def run_exit_strategy_backtest(results: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if results.empty or prices.empty:
        return pd.DataFrame()

    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    p = p.sort_values(["stock_id","date"])
    prices_map = {
        str(code): g[["date","open","high","low","close"]].reset_index(drop=True)
        for code, g in p.groupby("stock_id")
    }

    signals = results[
        pd.to_numeric(results["rank"], errors="coerce") <= 20
    ].copy()

    strategies = [
        "HOLD40",
        "SL8_TP50",
        "SL10_TP50",
        "SL12_TP50",
        "TP15_TRAIL8",
        "TP30_TRAIL10",
    ]

    rows = []
    for _, sig in signals.iterrows():
        signal_date = pd.to_datetime(sig["date"], errors="coerce")
        stock_id = str(sig["stock_id"])
        if pd.isna(signal_date):
            continue
        path = _trade_path(prices_map, stock_id, signal_date)
        if path is None:
            continue

        for strategy in strategies:
            sim = _simulate_exit(path, strategy)
            if not sim:
                continue
            rows.append({
                "signal_date": signal_date,
                "stock_id": stock_id,
                "rank": sig.get("rank"),
                "score_hit50": sig.get("score_hit50"),
                "strategy": strategy,
                **sim,
            })

    return pd.DataFrame(rows)


def summarize_exit_strategies(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    rows = []
    for max_rank, group_label in [(5,"Top 5"),(10,"Top 10"),(20,"Top 20")]:
        d0 = trades[pd.to_numeric(trades["rank"], errors="coerce") <= max_rank].copy()
        for strategy, d in d0.groupby("strategy"):
            x = pd.to_numeric(d["net_return_pct"], errors="coerce").dropna()
            if x.empty:
                continue
            wins = x[x > 0].sum()
            losses = -x[x < 0].sum()
            pf = wins / losses if losses > 0 else np.nan
            mfe = pd.to_numeric(d["mfe_during_trade"], errors="coerce")
            capture = np.where(
                mfe > 0,
                pd.to_numeric(d["net_return_pct"], errors="coerce") / mfe * 100,
                np.nan,
            )

            rows.append({
                "群組": group_label,
                "策略": strategy,
                "交易數": int(len(d)),
                "平均淨報酬": float(x.mean()),
                "中位數淨報酬": float(x.median()),
                "勝率": float((x > 0).mean() * 100),
                "Profit Factor": float(pf) if pd.notna(pf) else np.nan,
                "中位持有天數": float(pd.to_numeric(d["holding_days"], errors="coerce").median()),
                "最大單筆虧損": float(x.min()),
                "10分位淨報酬": float(x.quantile(0.10)),
                "+20%實現率": float((x >= 20).mean() * 100),
                "+30%實現率": float((x >= 30).mean() * 100),
                "+50%實現率": float((x >= 50).mean() * 100),
                "MFE捕捉率中位數": float(pd.Series(capture).replace([np.inf,-np.inf],np.nan).median()),
            })

    return pd.DataFrame(rows)


def main():
    ids = load_universe()
    print(f"股票池：{len(ids)} 檔")
    prices = download_prices(ids)
    dataset = make_dataset(prices)
    print(f"可用特徵列：{len(dataset):,}")

    test_dates = weekly_test_dates(dataset)
    print(f"Walk-forward 測試截面：{len(test_dates)}")

    results = []
    for idx, dt in enumerate(test_dates, 1):
        scored = score_date(dataset, dt)
        if not scored.empty:
            results.append(scored)
        if idx % 20 == 0:
            print(f"已完成 {idx}/{len(test_dates)} 個截面")

    if not results:
        raise RuntimeError("Walk-forward 沒有產生有效測試結果")

    res = pd.concat(results, ignore_index=True)
    summary = summarize(res)

    exit_trades = run_exit_strategy_backtest(res, prices)
    exit_summary = summarize_exit_strategies(exit_trades)

    optimizer_trades = simulate_grid(
        res,
        prices,
        cost_pct=ROUND_TRIP_COST_PCT,
        max_rank=10,
    )
    optimizer_summary, optimizer_shortlist, optimizer_yearly = summarize_optimizer(
        optimizer_trades,
        split_date="2025-01-01",
    )

    res.to_csv(DATA_DIR / "walkforward_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(DATA_DIR / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    exit_trades.to_csv(DATA_DIR / "walkforward_exit_trades.csv", index=False, encoding="utf-8-sig")
    exit_summary.to_csv(DATA_DIR / "walkforward_exit_summary.csv", index=False, encoding="utf-8-sig")
    optimizer_trades.to_csv(DATA_DIR / "walkforward_exit_optimizer_trades.csv", index=False, encoding="utf-8-sig")
    optimizer_summary.to_csv(DATA_DIR / "walkforward_exit_optimizer.csv", index=False, encoding="utf-8-sig")
    optimizer_shortlist.to_csv(DATA_DIR / "walkforward_exit_optimizer_shortlist.csv", index=False, encoding="utf-8-sig")
    optimizer_yearly.to_csv(DATA_DIR / "walkforward_exit_optimizer_yearly.csv", index=False, encoding="utf-8-sig")

    metadata = pd.DataFrame([{
        "模型版本": MODEL_VERSION,
        "開始日期": START_DATE,
        "股票池": "目前可交易股票池（存在 survivorship bias）",
        "測試頻率": "每5交易日一個截面",
        "Embargo": "訓練樣本 label_end_date 必須早於測試日",
        "特徵數": len(FEATURES),
        "K": 250,
        "測試截面數": res["date"].nunique(),
        "成熟測試樣本": int(res["hit50"].notna().sum()),
        "出場回測交易摩擦假設": f"每筆往返合計 {ROUND_TRIP_COST_PCT:.2f}%",
        "出場回測進場": "訊號後下一交易日開盤",
        "同日停損與停利皆觸發": "保守假設停損先發生",
        "Exit Optimizer版本": "EO-1.0",
        "Exit Optimizer開發/OOS切點": "2025-01-01",
        "Exit Optimizer原則": "只用開發期排名出場策略，OOS期只驗證不參與挑選",
        "注意": "這是核心價格型態模型驗證，不包含完整歷史法人/基本面因子；不得與完整 live 主模型績效混為一談。",
    }])
    metadata.to_csv(DATA_DIR / "walkforward_metadata.csv", index=False, encoding="utf-8-sig")

    print("✅ Walk-forward 完成")
    print(summary.to_string(index=False))
    print("\n✅ Exit strategy comparison")
    print(exit_summary.to_string(index=False))
    print("\n✅ Exit Optimizer shortlist")
    print(optimizer_shortlist.to_string(index=False))


if __name__ == "__main__":
    main()
