from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

from selection_model_v2 import add_selection_v2_components

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_VERSION = "P50-LIVE-S2.0"
MODEL_TARGET = "40交易日內最高報酬達+50%"
LIVE_S2_CONFIG = "S2_E_BALANCED"
NOTEBOOK_PATH = ROOT / "股市V1-V5.ipynb"
DATA_DIR.mkdir(parents=True, exist_ok=True)




_FINMIND_ASYNC_BROKEN = False


def _yfinance_taiwan_batch(stock_ids, start_date, end_date):
    """FinMind 額度不足時，以 Yahoo Finance 批次補台股 OHLCV。"""
    import yfinance as yf

    ids = [str(x).zfill(4) for x in stock_ids]
    tickers = [f"{x}.TW" for x in ids] + [f"{x}.TWO" for x in ids]

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    frames = []
    for stock_id in ids:
        chosen = None
        for ticker in (f"{stock_id}.TW", f"{stock_id}.TWO"):
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if ticker not in raw.columns.get_level_values(1):
                        continue
                    x = pd.DataFrame({
                        "open": raw[("Open", ticker)],
                        "max": raw[("High", ticker)],
                        "min": raw[("Low", ticker)],
                        "close": raw[("Close", ticker)],
                        "Trading_Volume": raw[("Volume", ticker)],
                    })
                else:
                    # 單一 ticker 的保險分支
                    x = pd.DataFrame({
                        "open": raw["Open"],
                        "max": raw["High"],
                        "min": raw["Low"],
                        "close": raw["Close"],
                        "Trading_Volume": raw["Volume"],
                    })
                x = x.dropna(subset=["close"])
                if not x.empty:
                    chosen = x.copy()
                    break
            except Exception:
                continue

        if chosen is None or chosen.empty:
            continue

        chosen = chosen.reset_index()
        date_col = "Date" if "Date" in chosen.columns else chosen.columns[0]
        chosen = chosen.rename(columns={date_col: "date"})
        chosen["date"] = pd.to_datetime(chosen["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        chosen["stock_id"] = stock_id
        chosen["Trading_money"] = (
            pd.to_numeric(chosen["close"], errors="coerce")
            * pd.to_numeric(chosen["Trading_Volume"], errors="coerce")
        )
        chosen["spread"] = pd.to_numeric(chosen["close"], errors="coerce").diff()
        chosen["Trading_turnover"] = 0

        frames.append(chosen[[
            "date","stock_id","Trading_Volume","Trading_money",
            "open","max","min","close","spread","Trading_turnover"
        ]])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def safe_finmind_batch(api, stock_ids, start_date, end_date):
    """
    先用 FinMind async；若回傳空資料、額度不足或其他錯誤，
    自動改用 Yahoo Finance 批次資料，不再逐檔消耗 FinMind 額度。
    """
    global _FINMIND_ASYNC_BROKEN

    if not _FINMIND_ASYNC_BROKEN:
        try:
            df = api.taiwan_stock_daily(
                stock_id_list=list(stock_ids),
                start_date=start_date,
                end_date=end_date,
                use_async=True,
            )
            if df is not None and not df.empty:
                # FinMind 有時只回傳部分股票；立即用 Yahoo 補齊這個 batch，
                # 避免 Notebook 後面再逐檔打 FinMind 耗盡額度。
                got = set(
                    df["stock_id"].astype(str).str.replace(".0","",regex=False).str.zfill(4)
                ) if "stock_id" in df.columns else set()
                requested = {str(x).zfill(4) for x in stock_ids}
                missing = sorted(requested - got)

                if missing:
                    print(f"⚠️ FinMind 此批缺 {len(missing)} 檔，改由 Yahoo 補齊")
                    backup = _yfinance_taiwan_batch(missing, start_date, end_date)
                    if backup is not None and not backup.empty:
                        df = pd.concat([df, backup], ignore_index=True)
                        print(f"✅ Yahoo 補齊 {backup['stock_id'].nunique()} 檔")

                return df

            print("⚠️ FinMind async 回傳空資料，切換 Yahoo Finance 備援")
            _FINMIND_ASYNC_BROKEN = True

        except Exception as e:
            print(f"⚠️ FinMind async 失敗，切換 Yahoo Finance 備援：{str(e)[:120]}")
            _FINMIND_ASYNC_BROKEN = True

    backup = _yfinance_taiwan_batch(stock_ids, start_date, end_date)
    if backup is not None and not backup.empty:
        print(f"✅ Yahoo Finance 備援成功：{backup['stock_id'].nunique()} 檔")
        return backup

    return pd.DataFrame()


def sanitize_code(source: str) -> str:
    source = "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("!pip ")
    )
    source = source.replace("from getpass import getpass\n", "")
    token_pattern = re.compile(
        r'FINMIND_TOKEN\s*=\s*getpass\(\s*"請輸入 FinMind Token："\s*\)\.strip\(\)',
        re.S,
    )
    source = token_pattern.sub(
        'FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()\n'
        'if not FINMIND_TOKEN:\n'
        '    raise RuntimeError("缺少 FINMIND_TOKEN GitHub Secret")',
        source,
    )

    # Notebook 原本使用 FinMind async 批次；GitHub Actions 偶爾會顯示下載完成
    # 卻回傳空 DataFrame。自動改用帶同步備援的 wrapper。
    batch_pattern = re.compile(
        r'df\s*=\s*api\.taiwan_stock_daily\(\s*'
        r'stock_id_list\s*=\s*batch\s*,\s*'
        r'start_date\s*=\s*開始日期\s*,\s*'
        r'end_date\s*=\s*結束日期\s*,\s*'
        r'use_async\s*=\s*True\s*\)',
        re.S,
    )
    source = batch_pattern.sub(
        'df = safe_finmind_batch(api, batch, 開始日期, 結束日期)',
        source,
    )
    return source




def build_global_market_environment():
    import yfinance as yf

    symbols = {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^NDX",
        "SOX 半導體": "^SOX",
        "VIX": "^VIX",
        "台灣加權": "^TWII",
        "TSM ADR": "TSM",
        "NVIDIA": "NVDA",
        "AMD": "AMD",
        "Broadcom": "AVGO",
        "Micron": "MU",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "美國10年債殖利率": "^TNX",
        "USD/TWD": "TWD=X",
    }

    rows = []

    for label, symbol in symbols.items():
        try:
            d = yf.download(
                symbol,
                period="6mo",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if d is None or d.empty:
                continue

            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)

            d = d.dropna(subset=["Close"]).copy()
            if d.empty:
                continue

            close = d["Close"].astype(float)
            last = float(close.iloc[-1])

            ret1 = (last / float(close.iloc[-2]) - 1) * 100 if len(close) >= 2 else float("nan")
            ret5 = (last / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else float("nan")
            ma20 = float(close.tail(20).mean()) if len(close) >= 20 else float("nan")
            ma60 = float(close.tail(60).mean()) if len(close) >= 60 else float("nan")

            rows.append({
                "項目": label,
                "代號": symbol,
                "日期": pd.Timestamp(d.index[-1]).strftime("%Y-%m-%d"),
                "最新值": last,
                "1日變動率": ret1,
                "5日變動率": ret5,
                "MA20": ma20,
                "MA60": ma60,
                "高於MA20": bool(last > ma20) if pd.notna(ma20) else False,
                "高於MA60": bool(last > ma60) if pd.notna(ma60) else False,
            })
        except Exception as e:
            print(f"⚠️ 海外市場資料抓取失敗 {label} ({symbol}): {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()

    score_parts = []

    def get_row(name):
        x = df[df["項目"].eq(name)]
        return None if x.empty else x.iloc[0]

    # 股市 / 科技 / 半導體：上漲、站上 MA20 視為順風
    for name, weight in [
        ("S&P 500", 15),
        ("Nasdaq 100", 20),
        ("SOX 半導體", 20),
        ("TSM ADR", 15),
        ("NVIDIA", 10),
    ]:
        r = get_row(name)
        if r is None:
            continue
        s = 0.0
        if pd.notna(r["1日變動率"]) and r["1日變動率"] > 0:
            s += weight * 0.35
        if pd.notna(r["5日變動率"]) and r["5日變動率"] > 0:
            s += weight * 0.35
        if bool(r["高於MA20"]):
            s += weight * 0.30
        score_parts.append((name, weight, s))

    # VIX：下降且低於 MA20 視為順風
    r = get_row("VIX")
    if r is not None:
        weight = 10
        s = 0.0
        if pd.notna(r["1日變動率"]) and r["1日變動率"] < 0:
            s += weight * 0.4
        if pd.notna(r["5日變動率"]) and r["5日變動率"] < 0:
            s += weight * 0.3
        if pd.notna(r["MA20"]) and r["最新值"] < r["MA20"]:
            s += weight * 0.3
        score_parts.append(("VIX", weight, s))

    # 利率：短線回落視為估值壓力降低
    r = get_row("美國10年債殖利率")
    if r is not None:
        weight = 5
        s = 0.0
        if pd.notna(r["5日變動率"]) and r["5日變動率"] <= 0:
            s += weight * 0.6
        if pd.notna(r["MA20"]) and r["最新值"] <= r["MA20"]:
            s += weight * 0.4
        score_parts.append(("美國10年債殖利率", weight, s))

    # USD/TWD：台幣未明顯走弱時視為較友善
    r = get_row("USD/TWD")
    if r is not None:
        weight = 5
        s = 0.0
        if pd.notna(r["5日變動率"]) and r["5日變動率"] <= 1.0:
            s += weight * 0.6
        if pd.notna(r["1日變動率"]) and r["1日變動率"] <= 0.5:
            s += weight * 0.4
        score_parts.append(("USD/TWD", weight, s))

    total_weight = sum(x[1] for x in score_parts)
    total_score = sum(x[2] for x in score_parts)
    env_score = (total_score / total_weight * 100) if total_weight else float("nan")

    if pd.isna(env_score):
        regime = "資料不足"
    elif env_score >= 75:
        regime = "偏多順風"
    elif env_score >= 60:
        regime = "中性偏多"
    elif env_score >= 40:
        regime = "震盪中性"
    elif env_score >= 25:
        regime = "中性偏空"
    else:
        regime = "偏空逆風"

    summary = pd.DataFrame([{
        "全球環境分數": env_score,
        "全球環境判定": regime,
        "資料日期": df["日期"].max(),
    }])

    return df, summary




def _safe_num(value, default=float("nan")):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _market_item_score(row):
    if row is None:
        return float("nan")
    score = 0.0
    ret1 = _safe_num(row.get("1日變動率"))
    ret5 = _safe_num(row.get("5日變動率"))
    if pd.notna(ret1) and ret1 > 0:
        score += 30
    if pd.notna(ret5) and ret5 > 0:
        score += 35
    if bool(row.get("高於MA20", False)):
        score += 35
    return score


def build_domestic_market_environment(history_df, global_df):
    if history_df is None or history_df.empty:
        return pd.DataFrame([{
            "台股環境分數": 50.0,
            "台股環境判定": "資料不足",
            "站上MA20比例": float("nan"),
            "5日上漲比例": float("nan"),
            "20日上漲比例": float("nan"),
        }])

    h = history_df.copy()
    h["stock_id"] = h["stock_id"].astype(str)
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h.sort_values(["stock_id", "date"])
    latest = h.groupby("stock_id", group_keys=False).tail(1).copy()

    def ratio(cond):
        x = pd.Series(cond).dropna()
        return float(x.mean()) if len(x) else float("nan")

    close = pd.to_numeric(latest.get("close"), errors="coerce")
    ma20 = pd.to_numeric(latest.get("20日均線"), errors="coerce")
    ret5 = pd.to_numeric(latest.get("5日報酬率"), errors="coerce")
    ret20 = pd.to_numeric(latest.get("20日報酬率"), errors="coerce")

    above_ma20 = ratio(close > ma20)
    up5 = ratio(ret5 > 0)
    up20 = ratio(ret20 > 0)

    twii_score = 50.0
    if global_df is not None and not global_df.empty:
        x = global_df[global_df["項目"].eq("台灣加權")]
        if not x.empty:
            r = x.iloc[0]
            twii_score = 0.0
            if _safe_num(r.get("5日變動率")) > 0:
                twii_score += 35
            if bool(r.get("高於MA20", False)):
                twii_score += 40
            if bool(r.get("高於MA60", False)):
                twii_score += 25

    breadth_score = 0.0
    breadth_weight = 0.0
    if pd.notna(above_ma20):
        breadth_score += above_ma20 * 100 * 0.45
        breadth_weight += 0.45
    if pd.notna(up5):
        breadth_score += up5 * 100 * 0.30
        breadth_weight += 0.30
    if pd.notna(up20):
        breadth_score += up20 * 100 * 0.25
        breadth_weight += 0.25
    if breadth_weight:
        breadth_score /= breadth_weight
    else:
        breadth_score = 50.0

    score = twii_score * 0.45 + breadth_score * 0.55

    if score >= 72:
        regime = "偏多順風"
    elif score >= 58:
        regime = "中性偏多"
    elif score >= 42:
        regime = "震盪中性"
    elif score >= 28:
        regime = "中性偏空"
    else:
        regime = "偏空逆風"

    return pd.DataFrame([{
        "台股環境分數": round(score, 2),
        "台股環境判定": regime,
        "加權指數技術分": round(twii_score, 2),
        "市場廣度分數": round(breadth_score, 2),
        "站上MA20比例": above_ma20 * 100 if pd.notna(above_ma20) else float("nan"),
        "5日上漲比例": up5 * 100 if pd.notna(up5) else float("nan"),
        "20日上漲比例": up20 * 100 if pd.notna(up20) else float("nan"),
    }])


def industry_tailwind_score(industry, global_df):
    if global_df is None or global_df.empty:
        return 50.0

    mapping = {
        "半導體業": ["SOX 半導體", "TSM ADR", "NVIDIA", "AMD", "Broadcom", "Micron"],
        "電腦及週邊設備業": ["Nasdaq 100", "NVIDIA", "Microsoft", "Broadcom"],
        "電子零組件業": ["Nasdaq 100", "Apple", "NVIDIA", "Broadcom"],
        "其他電子業": ["Nasdaq 100", "NVIDIA", "Microsoft", "Amazon"],
        "通信網路業": ["Nasdaq 100", "Broadcom", "Apple"],
        "資訊服務業": ["Nasdaq 100", "Microsoft", "Amazon"],
        "電子通路業": ["Nasdaq 100", "SOX 半導體", "Broadcom"],
        "光電業": ["Nasdaq 100", "Apple", "NVIDIA"],
        "金融保險": ["S&P 500", "美國10年債殖利率"],
        "航運業": ["S&P 500"],
        "建材營造": ["S&P 500"],
        "鋼鐵工業": ["S&P 500"],
    }

    names = mapping.get(str(industry), ["S&P 500"])
    scores = []
    for name in names:
        x = global_df[global_df["項目"].eq(name)]
        if x.empty:
            continue
        r = x.iloc[0]
        if name == "美國10年債殖利率":
            score = 50.0
            ret5 = _safe_num(r.get("5日變動率"))
            if pd.notna(ret5):
                score = 65.0 if ret5 <= 0 else 40.0
        else:
            score = _market_item_score(r)
        if pd.notna(score):
            scores.append(score)

    return float(sum(scores) / len(scores)) if scores else 50.0


def _recent_cross(series_a, series_b, lookback=3):
    a = pd.to_numeric(series_a, errors="coerce")
    b = pd.to_numeric(series_b, errors="coerce")
    cross = (a > b) & (a.shift(1) <= b.shift(1))
    return bool(cross.tail(lookback).fillna(False).any())


def _consecutive_positive(series):
    count = 0
    for x in reversed(pd.to_numeric(series, errors="coerce").fillna(0).tolist()):
        if x > 0:
            count += 1
        else:
            break
    return count




def build_historical_50_pattern_scores(history_df):
    """
    用現有歷史技術資料建立「未來40交易日內曾上漲50%」的歷史型態相似度。
    不使用未來資料產生當日特徵；未來價格只用來建立歷史標籤。
    回傳每檔目前型態的鄰近樣本命中率、PR與樣本統計。
    """
    if history_df is None or history_df.empty:
        return pd.DataFrame(), {
            "歷史樣本數": 0, "50%命中樣本數": 0, "50%基準命中率": float("nan")
        }

    h = history_df.copy()
    h["stock_id"] = h["stock_id"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h["close"] = pd.to_numeric(h.get("close"), errors="coerce")
    h = h.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # 50% 歷史型態模型需要的是「每一個歷史日期」的技術特徵。
    # namespace 裡的 歷史資料 原本只有 FinMind 原始 OHLCV，
    # 因此在這裡直接逐檔重算，不能只檢查欄位是否已存在。
    feature_frames = []
    for code, g in h.groupby("stock_id", sort=False):
        g = g.sort_values("date").copy()

        close = pd.to_numeric(g["close"], errors="coerce")
        high = pd.to_numeric(g["max"], errors="coerce") if "max" in g.columns else close
        low = pd.to_numeric(g["min"], errors="coerce") if "min" in g.columns else close
        volume = pd.to_numeric(g["Trading_Volume"], errors="coerce") if "Trading_Volume" in g.columns else pd.Series(index=g.index, dtype=float)

        g["5日報酬率"] = close.pct_change(5) * 100
        g["20日報酬率"] = close.pct_change(20) * 100
        g["60日報酬率"] = close.pct_change(60) * 100

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        g["14日RSI"] = 100 - (100 / (1 + rs))
        g.loc[(avg_loss == 0) & avg_gain.notna(), "14日RSI"] = 100

        ma20 = close.rolling(20, min_periods=20).mean()
        g["MA20乖離率"] = (close / ma20 - 1) * 100
        g["量比"] = volume / volume.rolling(20, min_periods=20).mean().replace(0, np.nan)

        daily_ret = close.pct_change()
        g["20日年化波動率"] = daily_ret.rolling(20, min_periods=20).std() * np.sqrt(252) * 100

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14, min_periods=14).mean()
        g["ATR百分比"] = atr14 / close.replace(0, np.nan) * 100

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        g["MACD柱狀體"] = macd - signal

        feature_frames.append(g)

    if feature_frames:
        h = pd.concat(feature_frames, ignore_index=True).sort_values(["stock_id", "date"]).reset_index(drop=True)

    feature_cols = [
        "5日報酬率", "20日報酬率", "60日報酬率",
        "14日RSI", "量比", "ATR百分比", "MA20乖離率",
        "20日年化波動率", "MACD柱狀體"
    ]
    feature_cols = [c for c in feature_cols if c in h.columns]
    if len(feature_cols) < 5:
        return pd.DataFrame(), {
            "歷史樣本數": 0, "50%命中樣本數": 0, "50%基準命中率": float("nan")
        }

    labeled_parts = []
    current_rows = []

    for code, g in h.groupby("stock_id", sort=False):
        g = g.sort_values("date").copy().reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue

        closes = pd.to_numeric(g["close"], errors="coerce").to_numpy(dtype=float)
        future_max_ret = [float("nan")] * n

        # 只對真的有完整40個後續交易日的歷史點建立標籤，避免把未成熟樣本當失敗。
        for i in range(n - 40):
            base = closes[i]
            future = closes[i+1:i+41]
            if pd.notna(base) and base > 0 and len(future) == 40:
                valid = future[pd.notna(future)]
                if len(valid) == 40:
                    future_max_ret[i] = (valid.max() / base - 1) * 100

        g["40日內最高報酬率_標籤"] = future_max_ret
        mature = g[g["40日內最高報酬率_標籤"].notna()].copy()
        if not mature.empty:
            mature["40日內達50_標籤"] = mature["40日內最高報酬率_標籤"] >= 50
            labeled_parts.append(mature)

        current_rows.append(g.iloc[-1].copy())

    if not labeled_parts or not current_rows:
        return pd.DataFrame(), {
            "歷史樣本數": 0, "50%命中樣本數": 0, "50%基準命中率": float("nan")
        }

    train = pd.concat(labeled_parts, ignore_index=True)
    current = pd.DataFrame(current_rows).reset_index(drop=True)

    for c in feature_cols:
        train[c] = pd.to_numeric(train[c], errors="coerce")
        current[c] = pd.to_numeric(current[c], errors="coerce")

    train = train.dropna(subset=feature_cols + ["40日內達50_標籤"]).copy()
    current = current.dropna(subset=feature_cols).copy()

    if len(train) < 200 or current.empty:
        return pd.DataFrame(), {
            "歷史樣本數": int(len(train)),
            "50%命中樣本數": int(train.get("40日內達50_標籤", pd.Series(dtype=bool)).sum()) if len(train) else 0,
            "50%基準命中率": float(train["40日內達50_標籤"].mean() * 100) if len(train) else float("nan")
        }

    means = train[feature_cols].mean()
    stds = train[feature_cols].std().replace(0, 1).fillna(1)
    ztrain = ((train[feature_cols] - means) / stds).clip(-5, 5)
    zcurrent = ((current[feature_cols] - means) / stds).clip(-5, 5)

    y = train["40日內達50_標籤"].astype(float).to_numpy()
    train_mat = ztrain.to_numpy(dtype=float)
    k = min(250, max(80, int(len(train) ** 0.5 * 3)))

    rows = []
    zcurrent = zcurrent.reset_index(drop=True)
    current = current.reset_index(drop=True)
    for pos in range(len(current)):
        r = current.iloc[pos]
        v = zcurrent.iloc[pos].to_numpy(dtype=float)
        if v.ndim != 1 or v.shape[0] != train_mat.shape[1]:
            raise RuntimeError(f"50%型態特徵維度異常 current={v.shape} train={train_mat.shape}")
        dist = ((train_mat - v) ** 2).mean(axis=1) ** 0.5
        order = dist.argsort()[:k]
        dsel = dist[order]
        ysel = y[order]

        # 越相似權重越大；加上極小值避免除零。
        w = 1.0 / (dsel + 0.20)
        hit_rate = float((ysel * w).sum() / w.sum() * 100)
        avg_future_max = float(
            (pd.to_numeric(train.iloc[order]["40日內最高報酬率_標籤"], errors="coerce").to_numpy() * w).sum()
            / w.sum()
        )
        positive_neighbors = int(ysel.sum())

        rows.append({
            "股票代號": str(r["stock_id"]).zfill(4),
            "50%歷史型態命中率": round(hit_rate, 3),
            "相似樣本40日最高報酬均值": round(avg_future_max, 2),
            "相似樣本數": int(k),
            "相似樣本50%命中數": positive_neighbors,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["50%歷史型態PR"] = out["50%歷史型態命中率"].rank(pct=True, method="average") * 100

    stats = {
        "歷史樣本數": int(len(train)),
        "50%命中樣本數": int(train["40日內達50_標籤"].sum()),
        "50%基準命中率": round(float(train["40日內達50_標籤"].mean() * 100), 3),
        "特徵數": len(feature_cols),
        "鄰近樣本K": int(k),
    }
    return out, stats


def _build_live_s2_test_frame(out, history_df):
    """Build current cross-sectional features aligned to the historical S2 research features."""
    rows = []
    hist_map = {}

    if history_df is not None and not history_df.empty:
        h = history_df.copy()
        h["stock_id"] = h["stock_id"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
        h["date"] = pd.to_datetime(h["date"], errors="coerce")
        h = h.sort_values(["stock_id", "date"])
        hist_map = {code: g.sort_values("date").copy() for code, g in h.groupby("stock_id")}

    for _, r in out.iterrows():
        code = str(r.get("股票代號", "")).zfill(4)
        g = hist_map.get(code)

        prior60_runup = np.nan
        dd20_high = np.nan
        dd60_high = np.nan
        if g is not None and len(g):
            close = pd.to_numeric(g.get("close"), errors="coerce").dropna()
            if len(close):
                last = float(close.iloc[-1])
                if len(close) >= 60:
                    low60 = float(close.tail(60).min())
                    high60 = float(close.tail(60).max())
                    if low60 > 0:
                        prior60_runup = (last / low60 - 1) * 100
                    if high60 > 0:
                        dd60_high = (last / high60 - 1) * 100
                if len(close) >= 20:
                    high20 = float(close.tail(20).max())
                    if high20 > 0:
                        dd20_high = (last / high20 - 1) * 100

        rows.append({
            "股票代號": code,
            "atr_pct": _safe_num(r.get("ATR百分比")),
            "vol20_ann": _safe_num(r.get("20日年化波動率")),
            "ret60": _safe_num(r.get("60日報酬率")),
            "ret20": _safe_num(r.get("20日報酬率")),
            "ret5": _safe_num(r.get("5日報酬率")),
            "rsi14": _safe_num(r.get("14日RSI")),
            "ma20_bias": _safe_num(r.get("MA20乖離率")),
            "vol_ratio": _safe_num(r.get("量比")),
            "prior60_runup": prior60_runup,
            "dd20_high": dd20_high,
            "dd60_high": dd60_high,
        })
    return pd.DataFrame(rows)


def apply_live_s2_balanced(out, history_df):
    """
    Promote the research-selected S2_E_BALANCED configuration to the daily live ranking.

    Historical evidence model:
      45% base historical +50 pattern percentile
      25% General explosion evidence
      15% Ignition evidence
      15% Second-leg evidence

    The supporting evidence bins are fit only on mature historical walk-forward rows
    already stored in data/walkforward_results.csv.
    """
    if out is None or out.empty:
        return out

    result = out.copy()
    result["舊主模型分數"] = pd.to_numeric(result.get("主模型分數"), errors="coerce")

    wf_path = DATA_DIR / "walkforward_results.csv"
    if not wf_path.exists():
        print("⚠️ 找不到 walkforward_results.csv，今日保留舊主模型排名")
        result["正式模型版本"] = "LEGACY_FALLBACK"
        result["S2狀態"] = "找不到walkforward_results.csv"
        return result

    try:
        train = pd.read_csv(wf_path, dtype={"stock_id": str})
        train["stock_id"] = train["stock_id"].astype(str).str.zfill(4)
        train["date"] = pd.to_datetime(train.get("date"), errors="coerce")
        train = train[pd.to_numeric(train.get("hit50"), errors="coerce").notna()].copy()

        # Keep research/live liquidity definition aligned.
        if "volume" in train.columns:
            train = train[pd.to_numeric(train["volume"], errors="coerce") >= 1_000_000]

        needed = {
            "hit50","atr_pct","vol20_ann","ret60","ret20","ret5",
            "rsi14","ma20_bias","vol_ratio","prior60_runup","dd20_high","dd60_high"
        }
        missing = needed - set(train.columns)
        if missing or len(train) < 3000:
            print(f"⚠️ S2歷史訓練資料不足/缺欄位 {sorted(missing)}，保留舊主模型")
            result["正式模型版本"] = "LEGACY_FALLBACK"
            result["S2狀態"] = f"歷史訓練資料不足或缺欄位:{sorted(missing)}"
            return result

        live = _build_live_s2_test_frame(result, history_df)
        live_scored = add_selection_v2_components(train, live)

        live_scored["S2 General PR"] = (
            pd.to_numeric(live_scored["s2_general_raw"], errors="coerce")
            .rank(pct=True, method="average") * 100
        )
        live_scored["S2 Ignition PR"] = (
            pd.to_numeric(live_scored["s2_ignition_raw"], errors="coerce")
            .rank(pct=True, method="average") * 100
        )
        live_scored["S2 SecondLeg PR"] = (
            pd.to_numeric(live_scored["s2_second_raw"], errors="coerce")
            .rank(pct=True, method="average") * 100
        )

        if "50%歷史型態PR" in result.columns:
            base_src = result["50%歷史型態PR"]
        elif "50%歷史型態PR_y" in result.columns:
            base_src = result["50%歷史型態PR_y"]
        elif "50%歷史型態PR_x" in result.columns:
            base_src = result["50%歷史型態PR_x"]
        else:
            base_src = pd.Series(50.0, index=result.index)
        base = pd.to_numeric(base_src, errors="coerce").fillna(50.0).reset_index(drop=True)
        gen = pd.to_numeric(live_scored["S2 General PR"], errors="coerce").fillna(50.0)
        ign = pd.to_numeric(live_scored["S2 Ignition PR"], errors="coerce").fillna(50.0)
        sec = pd.to_numeric(live_scored["S2 SecondLeg PR"], errors="coerce").fillna(50.0)

        s2_score = base * 0.45 + gen * 0.25 + ign * 0.15 + sec * 0.15

        result = result.reset_index(drop=True)
        result["S2 Base PR"] = base.round(2)
        result["S2 General PR"] = gen.round(2)
        result["S2 Ignition PR"] = ign.round(2)
        result["S2 SecondLeg PR"] = sec.round(2)
        result["S2主模型分數"] = s2_score.round(2)
        result["主模型分數"] = result["S2主模型分數"]
        result["正式模型版本"] = LIVE_S2_CONFIG
        result["S2狀態"] = "正常"

        def s2_label(x):
            if pd.isna(x):
                return "⚪ 資料不足"
            if x >= 85:
                return "🔥 S2高波段爆發潛力"
            if x >= 70:
                return "🟢 S2中高爆發潛力"
            if x >= 55:
                return "🟡 S2中等爆發潛力"
            return "⚪ S2爆發潛力不足"

        result["50%潛力判定"] = result["S2主模型分數"].apply(s2_label)

        print(
            f"✅ Live Selection Upgrade：{LIVE_S2_CONFIG} 已接管主排名；"
            f"歷史訓練樣本 {len(train):,}"
        )
        return result

    except Exception as e:
        print(f"⚠️ Live S2 升級失敗，保留舊主模型：{str(e)[:180]}")
        result["正式模型版本"] = "LEGACY_FALLBACK"
        result["S2狀態"] = f"例外:{str(e)[:160]}"
        return result


def build_one_week_model(result_df, history_df, inst_df, global_df, global_summary, domestic_summary, pattern_scores=None):
    if result_df is None or result_df.empty:
        return pd.DataFrame()

    out = result_df.copy()
    out["股票代號"] = (
        out["股票代號"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
    )

    if pattern_scores is not None and not pattern_scores.empty:
        out = out.merge(pattern_scores, on="股票代號", how="left")

    # Canonicalize duplicated historical-pattern columns.
    # Older daily outputs may already contain these fields, so pandas creates _x/_y.
    for base_col in [
        "50%歷史型態命中率",
        "50%歷史型態PR",
        "相似樣本40日最高報酬均值",
    ]:
        if base_col not in out.columns:
            preferred = f"{base_col}_y"
            fallback = f"{base_col}_x"
            if preferred in out.columns:
                out[base_col] = pd.to_numeric(out[preferred], errors="coerce")
            elif fallback in out.columns:
                out[base_col] = pd.to_numeric(out[fallback], errors="coerce")
        else:
            out[base_col] = pd.to_numeric(out[base_col], errors="coerce")

    global_score = 50.0
    global_regime = "資料不足"
    if global_summary is not None and not global_summary.empty:
        global_score = _safe_num(global_summary.iloc[0].get("全球環境分數"), 50.0)
        global_regime = str(global_summary.iloc[0].get("全球環境判定", "資料不足"))

    domestic_score = 50.0
    domestic_regime = "資料不足"
    if domestic_summary is not None and not domestic_summary.empty:
        domestic_score = _safe_num(domestic_summary.iloc[0].get("台股環境分數"), 50.0)
        domestic_regime = str(domestic_summary.iloc[0].get("台股環境判定", "資料不足"))

    hist_map = {}
    avg_volume_map = {}

    if history_df is not None and not history_df.empty:
        h = history_df.copy()
        h["stock_id"] = h["stock_id"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
        h["date"] = pd.to_datetime(h["date"], errors="coerce")
        h = h.sort_values(["stock_id", "date"])
        for code, g in h.groupby("stock_id"):
            g = g.sort_values("date").copy()
            hist_map[code] = g
            if "Trading_Volume" in g.columns:
                avg_volume_map[code] = pd.to_numeric(g["Trading_Volume"], errors="coerce").tail(20).mean()

    inst_map = {}
    if inst_df is not None and not inst_df.empty:
        inst = inst_df.copy()
        inst["股票代號"] = inst["股票代號"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
        inst["日期"] = pd.to_datetime(inst["日期"], errors="coerce")
        for code, g in inst.groupby("股票代號"):
            inst_map[code] = g.sort_values("日期").tail(5).copy()

    rows = []

    for _, row in out.iterrows():
        code = str(row["股票代號"])
        g = hist_map.get(code)

        tech_start = 0.0
        momentum = 0.0
        chip = 0.0
        risk_penalty = 0.0
        tech_reasons = []
        chip_reasons = []
        risk_reasons = []

        close = _safe_num(row.get("收盤價"))
        ma20 = _safe_num(row.get("20日均線"))
        ma5 = _safe_num(row.get("5日均線"))
        rsi = _safe_num(row.get("14日RSI"))
        macd = _safe_num(row.get("MACD"))
        signal = _safe_num(row.get("MACD訊號線"))
        hist = _safe_num(row.get("MACD柱狀體"))
        vol_ratio = _safe_num(row.get("量比"))
        ret5 = _safe_num(row.get("5日報酬率"))
        ret20 = _safe_num(row.get("20日報酬率"))
        ret60 = _safe_num(row.get("60日報酬率"))
        bias20 = _safe_num(row.get("MA20乖離率"))
        atr = _safe_num(row.get("ATR百分比"))
        annual_vol = _safe_num(row.get("20日年化波動率"))
        drawdown = _safe_num(row.get("20日最大回撤"))
        pattern_hit = _safe_num(row.get("50%歷史型態命中率"), 0.0)
        pattern_pr = _safe_num(row.get("50%歷史型態PR"), 50.0)
        pattern_future_max = _safe_num(row.get("相似樣本40日最高報酬均值"), 0.0)

        ma20_rising = False
        ma_cross = False
        macd_cross = False
        hist_accel = False
        near_breakout = False
        recent_breakout = False

        if g is not None and len(g):
            latest = g.iloc[-1]
            prev = g.iloc[-2] if len(g) >= 2 else latest

            if "20日均線" in g.columns:
                ma20_rising = _safe_num(latest.get("20日均線")) > _safe_num(prev.get("20日均線"))
            if "5日均線" in g.columns and "20日均線" in g.columns:
                ma_cross = _recent_cross(g["5日均線"], g["20日均線"], 3)
            if "MACD" in g.columns and "MACD訊號線" in g.columns:
                macd_cross = _recent_cross(g["MACD"], g["MACD訊號線"], 3)
            if "MACD柱狀體" in g.columns and len(g) >= 2:
                hv = pd.to_numeric(g["MACD柱狀體"], errors="coerce")
                hist_accel = bool(pd.notna(hv.iloc[-1]) and pd.notna(hv.iloc[-2]) and hv.iloc[-1] > 0 and hv.iloc[-1] > hv.iloc[-2])
            if "前20日最高價" in g.columns and "close" in g.columns:
                ph = _safe_num(latest.get("前20日最高價"))
                lc = _safe_num(latest.get("close"))
                if pd.notna(ph) and ph != 0 and pd.notna(lc):
                    dist = (lc / ph - 1) * 100
                    near_breakout = -3 <= dist <= 2
                gg = g.tail(3)
                c = pd.to_numeric(gg["close"], errors="coerce")
                hh = pd.to_numeric(gg["前20日最高價"], errors="coerce")
                recent_breakout = bool((c > hh).fillna(False).any())

        # V1-B：技術啟動分數（100）
        if pd.notna(close) and pd.notna(ma20) and close > ma20:
            if pd.notna(bias20) and 0 <= bias20 <= 6:
                tech_start += 15
                tech_reasons.append("站上MA20且乖離健康")
            else:
                tech_start += 8
        if ma20_rising:
            tech_start += 10
            tech_reasons.append("MA20向上")
        if ma_cross:
            tech_start += 15
            tech_reasons.append("MA5近期上穿MA20")
        elif pd.notna(ma5) and pd.notna(ma20) and ma5 > ma20:
            tech_start += 8
        if pd.notna(rsi):
            if 50 <= rsi <= 65:
                tech_start += 15
                tech_reasons.append("RSI位於起漲區")
            elif 45 <= rsi <= 70:
                tech_start += 8
        if macd_cross:
            tech_start += 15
            tech_reasons.append("MACD近期黃金交叉")
        elif pd.notna(macd) and pd.notna(signal) and macd > signal:
            tech_start += 8
        if hist_accel:
            tech_start += 10
            tech_reasons.append("MACD動能擴張")
        if pd.notna(vol_ratio):
            if 1.1 <= vol_ratio <= 2.5:
                tech_start += 10
                tech_reasons.append("成交量溫和放大")
            elif 1.0 <= vol_ratio < 1.1:
                tech_start += 5
        if recent_breakout:
            tech_start += 10
            tech_reasons.append("近期突破20日高")
        elif near_breakout:
            tech_start += 8
            tech_reasons.append("接近20日突破")
        tech_start = min(100.0, tech_start)

        # V4-B：純動能分數（100）
        if pd.notna(ret5):
            if 0 <= ret5 <= 8:
                momentum += 30
            elif 8 < ret5 <= 12:
                momentum += 20
            elif -3 <= ret5 < 0:
                momentum += 10
        if pd.notna(ret20):
            if 3 <= ret20 <= 15:
                momentum += 25
            elif 15 < ret20 <= 25:
                momentum += 15
            elif 0 <= ret20 < 3:
                momentum += 10
        if pd.notna(ret60):
            if 5 <= ret60 <= 30:
                momentum += 15
            elif 0 <= ret60 < 5:
                momentum += 8
        if hist_accel:
            momentum += 15
        if ma20_rising:
            momentum += 15
        momentum = min(100.0, momentum)

        # V2-B：法人籌碼動能（100）
        ig = inst_map.get(code)
        if ig is not None and not ig.empty:
            foreign = pd.to_numeric(ig["外資淨買超"], errors="coerce").fillna(0)
            trust = pd.to_numeric(ig["投信淨買超"], errors="coerce").fillna(0)
            dealer = pd.to_numeric(ig["自營商淨買超"], errors="coerce").fillna(0)

            if foreign.sum() > 0:
                chip += 10
                chip_reasons.append("外資5日買超")
            if trust.sum() > 0:
                chip += 15
                chip_reasons.append("投信5日買超")
            if dealer.sum() > 0:
                chip += 5

            if len(ig) >= 4:
                f_early = foreign.iloc[:-2].mean()
                f_late = foreign.iloc[-2:].mean()
                t_early = trust.iloc[:-2].mean()
                t_late = trust.iloc[-2:].mean()
                if f_late > f_early and f_late > 0:
                    chip += 10
                    chip_reasons.append("外資買盤加速")
                if t_late > t_early and t_late > 0:
                    chip += 15
                    chip_reasons.append("投信買盤加速")

            if len(ig) >= 2:
                if foreign.iloc[-1] > 0 and foreign.iloc[-2] <= 0:
                    chip += 10
                    chip_reasons.append("外資由賣轉買")
                if trust.iloc[-1] > 0 and trust.iloc[-2] <= 0:
                    chip += 10
                    chip_reasons.append("投信由賣轉買")

            if _consecutive_positive(foreign) >= 3:
                chip += 10
                chip_reasons.append("外資連買")
            if _consecutive_positive(trust) >= 3:
                chip += 10
                chip_reasons.append("投信連買")

            avg_vol = _safe_num(avg_volume_map.get(code))
            combined = foreign.sum() + trust.sum() + dealer.sum()
            if pd.notna(avg_vol) and avg_vol > 0:
                intensity = combined / (avg_vol * max(len(ig), 1))
                if intensity >= 0.01:
                    chip += 5
                    chip_reasons.append("法人買超占量提升")

        if ig is None or ig.empty:
            chip = _safe_num(row.get("籌碼標準分"), 0.0)
        chip = min(100.0, max(0.0, chip))

        quality = _safe_num(row.get("基本面分數"), 50.0)
        quality = min(100.0, max(0.0, quality))

        # 風險 Gate：只扣分，不再把低風險當作起漲理由
        if pd.notna(rsi):
            if rsi >= 75:
                risk_penalty += 12
                risk_reasons.append("RSI過熱")
            elif rsi >= 70:
                risk_penalty += 5
        if pd.notna(ret5):
            if ret5 >= 15:
                risk_penalty += 12
                risk_reasons.append("5日漲幅過大")
            elif ret5 >= 10:
                risk_penalty += 5
        if pd.notna(bias20):
            if bias20 >= 12:
                risk_penalty += 15
                risk_reasons.append("MA20乖離過大")
            elif bias20 >= 8:
                risk_penalty += 7
        if pd.notna(atr):
            if atr >= 6:
                risk_penalty += 10
                risk_reasons.append("ATR偏高")
            elif atr >= 4.5:
                risk_penalty += 5
        if pd.notna(annual_vol):
            if annual_vol >= 60:
                risk_penalty += 10
                risk_reasons.append("波動率偏高")
            elif annual_vol >= 45:
                risk_penalty += 5
        if pd.notna(drawdown) and drawdown <= -15:
            risk_penalty += 8
            risk_reasons.append("近期回撤偏大")
        risk_penalty = min(40.0, risk_penalty)

        # 啟動階段
        if (
            (pd.notna(close) and pd.notna(ma20) and close < ma20 and not ma20_rising)
            or (pd.notna(ret20) and ret20 < -10)
        ):
            stage = "⑥ 轉弱"
        elif (
            (pd.notna(rsi) and rsi >= 75)
            or (pd.notna(bias20) and bias20 >= 12)
            or (pd.notna(ret5) and ret5 >= 15)
        ):
            stage = "⑤ 過熱"
        elif (
            (ma_cross or macd_cross or recent_breakout)
            and ma20_rising
            and (pd.isna(rsi) or rsi < 70)
        ):
            stage = "③ 剛啟動"
        elif (
            ma20_rising
            and (near_breakout or (pd.notna(bias20) and -2 <= bias20 <= 5))
            and (pd.isna(rsi) or 45 <= rsi <= 65)
        ):
            stage = "② 蓄勢"
        elif (
            pd.notna(ret5) and 5 <= ret5 < 15
            and pd.notna(close) and pd.notna(ma20) and close > ma20
        ):
            stage = "④ 趨勢加速"
        else:
            stage = "① 尚未啟動"

        # 啟動時機分數：回答「現在是不是接近好買點」
        one_week = (
            tech_start * 0.40
            + chip * 0.30
            + quality * 0.15
            + momentum * 0.15
        )

        # 1～2 個月「50%波段」主目標：把爆發力納入主模型，而不是另外做獨立模型。
        # 重點不是低波動，而是「趨勢剛啟動 + 籌碼加速 + 中期動能 + 足夠波動空間」。
        volatility_potential = 0.0
        if pd.notna(annual_vol):
            if 35 <= annual_vol <= 75:
                volatility_potential += 45
            elif 25 <= annual_vol < 35 or 75 < annual_vol <= 90:
                volatility_potential += 28
            elif 15 <= annual_vol < 25:
                volatility_potential += 15
        if pd.notna(atr):
            if 2.0 <= atr <= 6.5:
                volatility_potential += 35
            elif 1.2 <= atr < 2.0 or 6.5 < atr <= 8.0:
                volatility_potential += 20
        if pd.notna(vol_ratio):
            if 1.2 <= vol_ratio <= 3.5:
                volatility_potential += 20
            elif 1.0 <= vol_ratio < 1.2:
                volatility_potential += 10
        volatility_potential = min(100.0, volatility_potential)

        breakout_bonus = 0.0
        if recent_breakout:
            breakout_bonus += 8
        elif near_breakout:
            breakout_bonus += 4
        if ma_cross:
            breakout_bonus += 4
        if macd_cross or hist_accel:
            breakout_bonus += 4
        if pd.notna(ret20) and 5 <= ret20 <= 25:
            breakout_bonus += 5
        if pd.notna(ret60) and 10 <= ret60 <= 50:
            breakout_bonus += 5

        # 40 交易日 +50% 候選主分數
        # 方向：突破強度、持續動能、法人流、波動空間、相對小型/活躍股特徵。
        # 基本面只作最低品質保護，不再給過高權重，以免大型成熟股壟斷排名。
        breakout_score = 0.0
        if recent_breakout:
            breakout_score += 35
        elif near_breakout:
            breakout_score += 20
        if ma_cross:
            breakout_score += 15
        if macd_cross:
            breakout_score += 15
        if hist_accel:
            breakout_score += 10
        if pd.notna(vol_ratio):
            if 1.5 <= vol_ratio <= 4.0:
                breakout_score += 25
            elif 1.15 <= vol_ratio < 1.5:
                breakout_score += 12
        breakout_score = min(100.0, breakout_score)

        persistence_score = 0.0
        if pd.notna(ret5):
            if 3 <= ret5 <= 12:
                persistence_score += 30
            elif 0 < ret5 < 3 or 12 < ret5 <= 18:
                persistence_score += 15
        if pd.notna(ret20):
            if 8 <= ret20 <= 30:
                persistence_score += 35
            elif 3 <= ret20 < 8 or 30 < ret20 <= 45:
                persistence_score += 18
        if pd.notna(ret60):
            if 12 <= ret60 <= 55:
                persistence_score += 25
            elif 5 <= ret60 < 12:
                persistence_score += 12
        if ma20_rising:
            persistence_score += 10
        persistence_score = min(100.0, persistence_score)

        # 活躍度 / 爆發性代理：成交額不宜太小，但極大型成熟股也不額外加分。
        turnover = _safe_num(row.get("Trading_turnover"))
        money = _safe_num(row.get("Trading_money"))
        activity_score = 0.0
        if pd.notna(money):
            if 80_000_000 <= money <= 3_000_000_000:
                activity_score += 45
            elif 30_000_000 <= money < 80_000_000 or 3_000_000_000 < money <= 8_000_000_000:
                activity_score += 25
        if pd.notna(turnover):
            if 800 <= turnover <= 12000:
                activity_score += 35
            elif 300 <= turnover < 800 or 12000 < turnover <= 25000:
                activity_score += 18
        if pd.notna(vol_ratio) and 1.3 <= vol_ratio <= 4.0:
            activity_score += 20
        activity_score = min(100.0, activity_score)

        # 基本品質只當保護層：太差扣分，普通以上即可。
        quality_gate = 100.0
        if quality < 35:
            quality_gate = 55.0
        elif quality < 45:
            quality_gate = 75.0

        # 主爆發分數直接對齊「歷史上40日內曾漲50%」的相似型態。
        # 歷史型態PR佔45%，其餘才是即時突破/動能/籌碼/波動。
        swing_power = (
            pattern_pr * 0.45
            + breakout_score * 0.18
            + persistence_score * 0.15
            + chip * 0.10
            + volatility_potential * 0.07
            + activity_score * 0.05
        )
        swing_power *= quality_gate / 100.0

        # 避免已經過度噴出才追：過熱扣分，但不是完全排除強勢股。
        if pd.notna(rsi) and rsi >= 78:
            swing_power -= 8
        if pd.notna(bias20) and bias20 >= 15:
            swing_power -= 10
        if pd.notna(ret5) and ret5 >= 20:
            swing_power -= 8

        swing_power = max(0.0, min(100.0, swing_power))

        industry_score = industry_tailwind_score(row.get("產業別", ""), global_df)

        market_adjustment = (
            (global_score - 50) * 0.08
            + (domestic_score - 50) * 0.12
            + (industry_score - 50) * 0.08
        )

        stage_adjustment = {
            "③ 剛啟動": 8,
            "② 蓄勢": 3,
            "④ 趨勢加速": 0,
            "① 尚未啟動": -8,
            "⑤ 過熱": -15,
            "⑥ 轉弱": -20,
        }.get(stage, 0)

        entry = one_week + market_adjustment + stage_adjustment - risk_penalty
        entry = max(0.0, min(100.0, entry))

        # 單一主模型：目標就是 40 交易日內的大波段，進場時機只佔 20%。
        main_score = swing_power * 0.80 + entry * 0.20
        if stage == "⑤ 過熱":
            main_score -= 8
        elif stage == "⑥ 轉弱":
            main_score -= 15
        main_score = max(0.0, min(100.0, main_score))

        if swing_power >= 82:
            swing_label = "🔥 高爆發潛力"
        elif swing_power >= 72:
            swing_label = "🟢 中高爆發潛力"
        elif swing_power >= 60:
            swing_label = "🟡 中等爆發潛力"
        else:
            swing_label = "⚪ 爆發潛力不足"

        if stage == "⑤ 過熱":
            entry_label = "🟠 短線過熱"
        elif stage == "⑥ 轉弱" or entry < 50:
            entry_label = "🔴 暫不考慮"
        elif stage == "② 蓄勢" and one_week >= 65:
            entry_label = "🔵 等待突破"
        elif stage == "③ 剛啟動" and entry >= 68 and global_score >= 35 and domestic_score >= 35:
            entry_label = "🟢 可觀察進場"
        elif stage == "④ 趨勢加速" or (pd.notna(bias20) and bias20 >= 7):
            entry_label = "🟡 等待回檔"
        elif entry >= 62:
            entry_label = "🔵 等待突破"
        else:
            entry_label = "🟡 等待回檔"

        rows.append({
            "股票代號": code,
            "技術啟動分數": round(tech_start, 2),
            "籌碼動能分數": round(chip, 2),
            "基本品質分數": round(quality, 2),
            "價格動能分數": round(momentum, 2),
            "一週起漲分數": round(one_week, 2),
            "波段爆發分數": round(swing_power, 2),
            "50%歷史型態命中率": round(pattern_hit, 3),
            "50%歷史型態PR": round(pattern_pr, 2),
            "相似樣本40日最高報酬均值": round(pattern_future_max, 2),
            "突破強度分數": round(breakout_score, 2),
            "動能持續分數": round(persistence_score, 2),
            "活躍爆發分數": round(activity_score, 2),
            "波動爆發潛力": round(volatility_potential, 2),
            "主模型分數": round(main_score, 2),
            "50%潛力判定": swing_label,
            "風險扣分": round(risk_penalty, 2),
            "全球環境分數": round(global_score, 2),
            "台股環境分數": round(domestic_score, 2),
            "產業海外順風分數": round(industry_score, 2),
            "啟動階段": stage,
            "進場時機分數": round(entry, 2),
            "進場判定": entry_label,
            "起漲原因": "、".join(tech_reasons + chip_reasons[:3]),
            "進場風險": "、".join(risk_reasons) if risk_reasons else "無明顯風險訊號",
            "全球環境判定": global_regime,
            "台股環境判定": domestic_regime,
        })

    weekly = pd.DataFrame(rows)
    out = out.merge(weekly, on="股票代號", how="left")

    # 正式 Live Selection Upgrade：
    # 研究期勝出的 S2_E_BALANCED 直接接管每日主排名；
    # 既有進場時機/風險/市場環境欄位仍保留作操作層。
    out = apply_live_s2_balanced(out, history_df)

    out["起漲潛力排名"] = (
        pd.to_numeric(out["一週起漲分數"], errors="coerce")
        .rank(ascending=False, method="min")
    )
    out["波段爆發排名"] = (
        pd.to_numeric(out["波段爆發分數"], errors="coerce")
        .rank(ascending=False, method="min")
    )
    out = out.sort_values(
        ["主模型分數", "波段爆發分數", "進場時機分數"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    out["一週模型排名"] = range(1, len(out) + 1)
    out["主模型排名"] = out["一週模型排名"]

    return out


def main() -> None:
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {NOTEBOOK_PATH.name}。請把 notebook 放在 GitHub repository 根目錄。"
        )

    if not os.environ.get("FINMIND_TOKEN", "").strip():
        raise RuntimeError("GitHub Actions 尚未設定 FINMIND_TOKEN Secret")

    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]

    if len(code_cells) < 2:
        raise RuntimeError("Notebook 找不到 V1～V5 主程式碼")

    namespace = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK_PATH),
        "os": os,
        "safe_finmind_batch": safe_finmind_batch,
    }

    for idx, raw in enumerate(code_cells):
        cleaned = sanitize_code(raw)
        meaningful = [
            line for line in cleaned.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not meaningful:
            continue

        print(f"\n===== 執行 Notebook code cell {idx + 1}/{len(code_cells)} =====")
        exec(
            compile(cleaned, f"{NOTEBOOK_PATH.name}:cell{idx+1}", "exec"),
            namespace,
        )

        if "結果" in namespace:
            result = namespace["結果"]
            if isinstance(result, pd.DataFrame) and len(result):
                if "技術分數" in result.columns and "風險動能分數" in result.columns:
                    tech_ok = int(result["技術分數"].notna().sum())
                    v4_ok = int(result["風險動能分數"].notna().sum())
                    coverage = min(tech_ok, v4_ok) / len(result)
                    print(
                        f"資料完整度檢查：V1={tech_ok}/{len(result)}, "
                        f"V4={v4_ok}/{len(result)}, coverage={coverage:.1%}"
                    )
                    if coverage < 0.95:
                        raise RuntimeError(
                            f"V1/V4 資料完整度只有 {coverage:.1%}，低於 95%，"
                            "取消今日更新，保留 App 原本資料。"
                        )

    latest_trade_date = namespace.get("最新交易日")
    if latest_trade_date is None:
        raise RuntimeError("程式執行完成，但找不到 最新交易日 變數")

    date_text = latest_trade_date.strftime("%Y%m%d")
    excel_name = f"台股V1V2V3V4V5最終選股_{date_text}.xlsx"
    excel_src = ROOT / excel_name
    history_src = ROOT / "history_latest.csv"

    if not excel_src.exists():
        raise FileNotFoundError(f"找不到今日輸出檔：{excel_src.name}")
    if not history_src.exists():
        raise FileNotFoundError("找不到 history_latest.csv")

    excel_dst = DATA_DIR / excel_name
    history_dst = DATA_DIR / "history_latest.csv"

    global_df = pd.DataFrame()
    global_summary = pd.DataFrame()
    domestic_summary = pd.DataFrame()

    # 全球市場 Gate
    try:
        global_df, global_summary = build_global_market_environment()
        if not global_df.empty:
            global_path = DATA_DIR / "global_market_latest.csv"
            global_df.to_csv(global_path, index=False, encoding="utf-8-sig")
            print(f"✅ 已更新 {global_path.relative_to(ROOT)}")
        if not global_summary.empty:
            global_summary_path = DATA_DIR / "global_market_summary.csv"
            global_summary.to_csv(global_summary_path, index=False, encoding="utf-8-sig")
            print(f"✅ 已更新 {global_summary_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"⚠️ 全球市場 Gate 更新失敗，先以中性環境處理：{e}")

    # 台股市場 Gate
    try:
        domestic_summary = build_domestic_market_environment(
            namespace.get("歷史資料", pd.DataFrame()),
            global_df,
        )
        domestic_path = DATA_DIR / "domestic_market_summary.csv"
        domestic_summary.to_csv(domestic_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已更新 {domestic_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"⚠️ 台股市場 Gate 更新失敗，先以中性環境處理：{e}")
        domestic_summary = pd.DataFrame([{
            "台股環境分數": 50.0,
            "台股環境判定": "資料不足",
        }])

    # 主模型：40交易日 +50% 目標 + 進場時機
    result = namespace.get("結果")
    weekly_result = pd.DataFrame()

    pattern_scores, pattern_stats = build_historical_50_pattern_scores(
        namespace.get("歷史資料", pd.DataFrame())
    )
    if pattern_stats:
        pd.DataFrame([pattern_stats]).to_csv(
            DATA_DIR / "pattern50_model_stats.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(f"✅ 50%歷史型態：樣本 {pattern_stats.get('歷史樣本數',0)}，"
              f"命中 {pattern_stats.get('50%命中樣本數',0)}，"
              f"基準率 {pattern_stats.get('50%基準命中率',float('nan'))}%")

        metadata = pd.DataFrame([{
            "模型版本": MODEL_VERSION,
            "模型目標": MODEL_TARGET,
            "正式選股模型": LIVE_S2_CONFIG,
            "正式選股權重": "Base45% + General25% + Ignition15% + SecondLeg15%",
            "歷史流動性定義": "volume >= 1,000,000 shares（1,000張）",
            "資料基準日": latest_trade_date.strftime("%Y-%m-%d"),
            "歷史視窗日數": 420,
            "歷史樣本數": pattern_stats.get("歷史樣本數", 0),
            "50%命中樣本數": pattern_stats.get("50%命中樣本數", 0),
            "50%歷史基準率": pattern_stats.get("50%基準命中率", float("nan")),
            "特徵數": pattern_stats.get("特徵數", float("nan")),
            "鄰近樣本K": pattern_stats.get("鄰近樣本K", float("nan")),
            "股票池檔數": len(result) if isinstance(result, pd.DataFrame) else 0,
            "GitHub_SHA": os.environ.get("GITHUB_SHA", ""),
            "產生時間": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S"),
        }])
        metadata.to_csv(DATA_DIR / "model_metadata.csv", index=False, encoding="utf-8-sig")
        print(f"✅ 已更新模型 metadata：{MODEL_VERSION}")

    if isinstance(result, pd.DataFrame) and not result.empty:
        weekly_result = build_one_week_model(
            result,
            namespace.get("歷史資料", pd.DataFrame()),
            namespace.get("法人資料", pd.DataFrame()),
            global_df,
            global_summary,
            domestic_summary,
            pattern_scores,
        )

        weekly_path = DATA_DIR / "weekly_model_latest.csv"
        weekly_result.to_csv(weekly_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已更新 {weekly_path.relative_to(ROOT)}")

        top20_path = DATA_DIR / "weekly_top20.csv"
        weekly_result.head(20).to_csv(top20_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已更新 {top20_path.relative_to(ROOT)}")

    # 原 Notebook 輸出照常保留
    shutil.move(str(excel_src), str(excel_dst))
    shutil.move(str(history_src), str(history_dst))

    # 長期價格資料庫：history_latest 每天覆蓋，但 V6 回測需要永久保留後續價格
    try:
        latest_price = pd.read_csv(history_dst, dtype={"股票代號": str})
        latest_price["股票代號"] = (
            latest_price["股票代號"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
        )
        price_archive_path = DATA_DIR / "price_history_archive.csv"

        if price_archive_path.exists():
            old_price = pd.read_csv(price_archive_path, dtype={"股票代號": str})
            price_archive = pd.concat([old_price, latest_price], ignore_index=True)
        else:
            price_archive = latest_price

        if "日期" in price_archive.columns:
            price_archive["日期"] = pd.to_datetime(price_archive["日期"], errors="coerce")
            price_archive = (
                price_archive
                .drop_duplicates(["股票代號", "日期"], keep="last")
                .sort_values(["股票代號", "日期"])
            )
            price_archive["日期"] = price_archive["日期"].dt.strftime("%Y-%m-%d")

        price_archive.to_csv(price_archive_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已累積長期價格資料：{price_archive_path.relative_to(ROOT)}")

        # App 日常圖表不需要把 420 天全部載入；保留最近約 180 個日曆日，
        # 長期資料仍完整存在 price_history_archive.csv 供 40 日回測 / 型態模型使用。
        try:
            latest_ui = latest_price.copy()
            latest_ui["日期"] = pd.to_datetime(latest_ui["日期"], errors="coerce")
            max_date = latest_ui["日期"].max()
            if pd.notna(max_date):
                latest_ui = latest_ui[latest_ui["日期"] >= max_date - pd.Timedelta(days=180)]
                latest_ui["日期"] = latest_ui["日期"].dt.strftime("%Y-%m-%d")
                latest_ui.to_csv(history_dst, index=False, encoding="utf-8-sig")
                print(f"✅ App 歷史圖保留最近 180 日：{history_dst.relative_to(ROOT)}")
        except Exception as e:
            print(f"⚠️ history_latest 精簡失敗，保留完整檔：{e}")
    except Exception as e:
        print(f"⚠️ 長期價格資料累積失敗：{e}")

    # V6：同時保存原始 V5-A 與一週模型 V5-B
    snapshot_source = weekly_result if not weekly_result.empty else result
    if isinstance(snapshot_source, pd.DataFrame) and not snapshot_source.empty:
        snapshot_cols = [
            "股票代號", "股票名稱", "市場", "產業別",
            "排名", "最終分數", "綜合PR", "候選等級", "目前狀態",
            "技術分數", "籌碼標準分", "基本面分數", "風險動能分數",
            "一週模型排名", "主模型排名", "起漲潛力排名", "波段爆發排名",
            "一週起漲分數", "波段爆發分數", "50%歷史型態命中率", "50%歷史型態PR",
            "相似樣本40日最高報酬均值", "突破強度分數", "動能持續分數", "活躍爆發分數",
            "波動爆發潛力", "舊主模型分數", "S2主模型分數",
            "S2 Base PR", "S2 General PR", "S2 Ignition PR", "S2 SecondLeg PR",
            "正式模型版本", "主模型分數", "50%潛力判定", "進場時機分數",
            "技術啟動分數", "籌碼動能分數", "基本品質分數", "價格動能分數",
            "風險扣分", "啟動階段", "進場判定",
            "全球環境分數", "台股環境分數", "產業海外順風分數",
            "收盤價", "V5綜合理由", "V5風險提示", "起漲原因", "進場風險"
        ]
        snapshot_cols = [c for c in snapshot_cols if c in snapshot_source.columns]

        snapshot = snapshot_source[snapshot_cols].copy()
        snapshot.insert(0, "快照日期", latest_trade_date.strftime("%Y-%m-%d"))
        snapshot.insert(1, "模型版本", MODEL_VERSION)
        snapshot["股票代號"] = (
            snapshot["股票代號"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
        )

        ranking_history_path = DATA_DIR / "ranking_history.csv"

        if ranking_history_path.exists():
            old_history = pd.read_csv(ranking_history_path, dtype={"股票代號": str})
            combined = pd.concat([old_history, snapshot], ignore_index=True)
        else:
            combined = snapshot

        combined["股票代號"] = (
            combined["股票代號"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
        )
        sort_col = "一週模型排名" if "一週模型排名" in combined.columns else "排名"
        combined = (
            combined
            .drop_duplicates(["快照日期", "股票代號"], keep="last")
            .sort_values(["快照日期", sort_col], ascending=[True, True])
        )
        combined.to_csv(ranking_history_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已累積 V6 排名歷史：{ranking_history_path.relative_to(ROOT)}")

    print(f"✅ 已更新 {excel_dst.relative_to(ROOT)}")
    print(f"✅ 已更新 {history_dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
