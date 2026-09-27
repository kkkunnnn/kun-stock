from __future__ import annotations

from pathlib import Path
import math
import os
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

START_DATE = os.environ.get("WF_START_DATE", "2019-01-01")
MODEL_VERSION = "P50-WF-1.0"
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

    res.to_csv(DATA_DIR / "walkforward_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(DATA_DIR / "walkforward_summary.csv", index=False, encoding="utf-8-sig")

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
        "注意": "這是核心價格型態模型驗證，不包含完整歷史法人/基本面因子；不得與完整 live 主模型績效混為一談。",
    }])
    metadata.to_csv(DATA_DIR / "walkforward_metadata.csv", index=False, encoding="utf-8-sig")

    print("✅ Walk-forward 完成")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
