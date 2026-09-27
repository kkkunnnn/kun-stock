from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
NOTEBOOK_PATH = ROOT / "股市V1-V5.ipynb"
DATA_DIR.mkdir(parents=True, exist_ok=True)


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


def build_one_week_model(result_df, history_df, inst_df, global_df, global_summary, domestic_summary):
    if result_df is None or result_df.empty:
        return pd.DataFrame()

    out = result_df.copy()
    out["股票代號"] = (
        out["股票代號"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
    )

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

        one_week = (
            tech_start * 0.40
            + chip * 0.30
            + quality * 0.15
            + momentum * 0.15
        )

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

    out["起漲潛力排名"] = (
        pd.to_numeric(out["一週起漲分數"], errors="coerce")
        .rank(ascending=False, method="min")
    )
    sort_cols = ["進場時機分數", "一週起漲分數"]
    out = out.sort_values(sort_cols, ascending=[False, False], na_position="last").reset_index(drop=True)
    out["一週模型排名"] = range(1, len(out) + 1)

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

    shutil.move(str(excel_src), str(excel_dst))
    shutil.move(str(history_src), str(history_dst))

    # V6：累積每日排名快照，作為日後回測與模型驗證的基礎
    result = namespace.get("結果")
    if isinstance(result, pd.DataFrame) and not result.empty:
        snapshot_cols = [
            "股票代號", "股票名稱", "市場", "產業別",
            "排名", "最終分數", "綜合PR", "候選等級", "目前狀態",
            "技術分數", "籌碼標準分", "基本面分數", "風險動能分數",
            "收盤價", "V5綜合理由", "V5風險提示"
        ]
        snapshot_cols = [c for c in snapshot_cols if c in result.columns]

        snapshot = result[snapshot_cols].copy()
        snapshot.insert(0, "快照日期", latest_trade_date.strftime("%Y-%m-%d"))
        snapshot["股票代號"] = (
            snapshot["股票代號"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
        )

        ranking_history_path = DATA_DIR / "ranking_history.csv"

        if ranking_history_path.exists():
            old = pd.read_csv(ranking_history_path, dtype={"股票代號": str})
            combined = pd.concat([old, snapshot], ignore_index=True)
        else:
            combined = snapshot

        combined["股票代號"] = (
            combined["股票代號"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
        )
        combined = (
            combined
            .drop_duplicates(["快照日期", "股票代號"], keep="last")
            .sort_values(["快照日期", "排名"], ascending=[True, True])
        )
        combined.to_csv(ranking_history_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已累積 V6 排名歷史：{ranking_history_path.relative_to(ROOT)}")

    print(f"✅ 已更新 {excel_dst.relative_to(ROOT)}")
    # 全球市場 Gate：美股、半導體、波動、利率、匯率
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
        print(f"⚠️ 全球市場 Gate 更新失敗，不影響主選股結果：{e}")

    print(f"✅ 已更新 {history_dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
