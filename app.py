from __future__ import annotations

import io
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_validation import (
    attach_professional_forward_metrics,
    professional_validation_summary,
    regime_validation,
    validation_readiness,
)

APP_TITLE = "台股 V1～V5 選股"
DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide", initial_sidebar_state="collapsed")


def css():
    st.markdown("""
    <style>
    .block-container{padding-top:.7rem;padding-bottom:4rem;max-width:1450px}
    #MainMenu{visibility:hidden} footer{visibility:hidden}
    .hero{border:1px solid rgba(100,116,139,.18);border-radius:24px;padding:20px 22px;margin-bottom:14px;background:linear-gradient(135deg,rgba(37,99,235,.08),rgba(148,163,184,.04))}
    .hero h1{margin:0;font-size:2rem}.hero p{margin:.35rem 0 0;opacity:.68}
    div[data-testid="stMetric"]{background:rgba(148,163,184,.07);border:1px solid rgba(100,116,139,.16);padding:12px 14px;border-radius:16px}
    .card{border:1px solid rgba(100,116,139,.18);border-radius:18px;padding:15px;margin-bottom:8px;background:rgba(148,163,184,.045)}
    .rank{display:inline-block;background:rgba(37,99,235,.12);color:#2563eb;border-radius:999px;padding:3px 9px;font-size:.78rem;font-weight:800}
    .title{font-size:1.12rem;font-weight:800;margin-top:7px}.score{font-size:1.65rem;font-weight:850;margin-top:6px}.muted{opacity:.66;font-size:.86rem}
    .good{color:#159447;font-weight:750}.warn{color:#b7791f;font-weight:750}.bad{color:#d64545;font-weight:750}
    .news{padding:11px 2px;border-bottom:1px solid rgba(100,116,139,.15)}.news a{text-decoration:none;font-weight:700}.newsmeta{opacity:.6;font-size:.8rem;margin-top:4px}
    @media(max-width:700px){.block-container{padding-left:.7rem;padding-right:.7rem}.hero{padding:15px;border-radius:18px}.hero h1{font-size:1.5rem}.card{border-radius:15px}h2{font-size:1.25rem!important}}
    </style>
    """, unsafe_allow_html=True)


def fmt(v, d=1, suffix=""):
    if pd.isna(v): return "—"
    try: return f"{float(v):,.{d}f}{suffix}"
    except Exception: return str(v)


def goto_stock_detail(code):
    st.session_state.selected = code
    st.session_state.nav = "個股分析"


def status_css(s):
    if s == "趨勢健康": return "good"
    if s in ["等待回檔", "整理觀察", "一般觀察"]: return "warn"
    return "bad"



def excel_snapshots():
    files = list(DATA_DIR.glob("*.xlsx"))
    def date_key(p):
        m = re.search(r"(20\\d{6})", p.stem)
        return (m.group(1) if m else "00000000", p.stat().st_mtime)
    return sorted(files, key=date_key, reverse=True)


def load_previous_rank(current_source):
    for p in excel_snapshots():
        if p.name == current_source:
            continue
        try:
            sh = read_path(str(p), p.stat().st_mtime)
            d = normalize(sh.get("全部排名", pd.DataFrame()))
            if not d.empty:
                return d, p.name
        except Exception:
            continue
    return pd.DataFrame(), ""


def build_daily_change(current_rank, previous_rank):
    if previous_rank.empty:
        return pd.DataFrame()

    keep = ["股票代號", "排名", "最終分數", "候選等級", "目前狀態"]
    prev = previous_rank[[c for c in keep if c in previous_rank.columns]].copy()
    prev = prev.rename(columns={
        "排名": "昨日排名",
        "最終分數": "昨日最終分數",
        "候選等級": "昨日候選等級",
        "目前狀態": "昨日狀態",
    })

    cur = current_rank.copy()
    out = cur.merge(prev, on="股票代號", how="left")

    if "排名" in out.columns and "昨日排名" in out.columns:
        out["排名變化"] = pd.to_numeric(out["昨日排名"], errors="coerce") - pd.to_numeric(out["排名"], errors="coerce")

    if "最終分數" in out.columns and "昨日最終分數" in out.columns:
        out["分數變化"] = (
            pd.to_numeric(out["最終分數"], errors="coerce")
            - pd.to_numeric(out["昨日最終分數"], errors="coerce")
        )

    if "排名" in out.columns:
        prev_rank_num = pd.to_numeric(out.get("昨日排名"), errors="coerce")
        cur_rank_num = pd.to_numeric(out["排名"], errors="coerce")
        out["新進Top10"] = (cur_rank_num <= 10) & (prev_rank_num.isna() | (prev_rank_num > 10))

    cur_cand = out.get("候選等級", pd.Series("", index=out.index)).astype(str)
    prev_cand = out.get("昨日候選等級", pd.Series("", index=out.index)).astype(str)
    out["新進強勢"] = cur_cand.str.contains("強勢候選", na=False) & ~prev_cand.str.contains("強勢候選", na=False)

    return out


def latest_excel() -> Optional[Path]:
    files = list(DATA_DIR.glob("*.xlsx"))
    if not files: return None
    def key(p):
        m = re.search(r"(20\d{6})", p.stem)
        return (m.group(1) if m else "00000000", p.stat().st_mtime)
    return sorted(files, key=key, reverse=True)[0]


@st.cache_data(show_spinner=False)
def read_path(path, mtime):
    x = pd.ExcelFile(path)
    return {s: pd.read_excel(path, sheet_name=s) for s in x.sheet_names}


@st.cache_data(show_spinner=False)
def read_bytes(raw):
    b = io.BytesIO(raw); x = pd.ExcelFile(b); out = {}
    for s in x.sheet_names:
        b.seek(0); out[s] = pd.read_excel(b, sheet_name=s)
    return out


def load_data():
    up = st.sidebar.file_uploader("上傳今日 V1～V5 Excel", type=["xlsx"])
    if up is not None: return read_bytes(up.getvalue()), up.name
    p = latest_excel()
    if p is None: return {}, ""
    return read_path(str(p), p.stat().st_mtime), p.name


def normalize(df):
    d = df.copy()
    if "股票代號" in d.columns:
        d["股票代號"] = d["股票代號"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
    return d





def load_weekly_model():
    p = DATA_DIR / "weekly_model_latest.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(p, dtype={"股票代號": str})
        d["股票代號"] = d["股票代號"].astype(str).str.replace(".0","",regex=False).str.zfill(4)
        return d
    except Exception:
        return pd.DataFrame()


def load_domestic_market():
    p = DATA_DIR / "domestic_market_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def build_weekly_change(history_df):
    if history_df.empty or "快照日期" not in history_df.columns:
        return pd.DataFrame()

    d = history_df.copy()
    d["快照日期"] = pd.to_datetime(d["快照日期"], errors="coerce")
    dates = sorted(d["快照日期"].dropna().dt.normalize().unique())
    if len(dates) < 2:
        return pd.DataFrame()

    current_date, prev_date = dates[-1], dates[-2]
    cur = d[d["快照日期"].dt.normalize().eq(current_date)].copy()
    prev = d[d["快照日期"].dt.normalize().eq(prev_date)].copy()

    keep_prev = [
        "股票代號", "一週模型排名", "一週起漲分數",
        "進場時機分數", "啟動階段", "進場判定"
    ]
    prev = prev[[c for c in keep_prev if c in prev.columns]].copy()
    prev = prev.rename(columns={
        "一週模型排名": "昨日一週排名",
        "一週起漲分數": "昨日一週起漲分數",
        "進場時機分數": "昨日進場時機分數",
        "啟動階段": "昨日啟動階段",
        "進場判定": "昨日進場判定",
    })

    out = cur.merge(prev, on="股票代號", how="left")

    if "一週模型排名" in out.columns and "昨日一週排名" in out.columns:
        out["一週排名變化"] = (
            pd.to_numeric(out["昨日一週排名"], errors="coerce")
            - pd.to_numeric(out["一週模型排名"], errors="coerce")
        )
        cur_rank = pd.to_numeric(out["一週模型排名"], errors="coerce")
        prev_rank = pd.to_numeric(out["昨日一週排名"], errors="coerce")
        out["新進一週Top10"] = (cur_rank <= 10) & (prev_rank.isna() | (prev_rank > 10))

    if "進場判定" in out.columns:
        cur_entry = out["進場判定"].astype(str)
        prev_entry = out.get("昨日進場判定", pd.Series("", index=out.index)).astype(str)
        out["新進可觀察進場"] = cur_entry.str.contains("可觀察進場", na=False) & ~prev_entry.str.contains("可觀察進場", na=False)

    return out


def latest_history_csv() -> Optional[Path]:
    files = list(DATA_DIR.glob("history_*.csv"))
    if not files:
        p = DATA_DIR / "history_latest.csv"
        return p if p.exists() else None
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]


@st.cache_data(show_spinner=False)
def read_history(path, mtime):
    d = pd.read_csv(path, dtype={"股票代號": str})
    if "股票代號" in d.columns:
        d["股票代號"] = d["股票代號"].astype(str).str.replace(".0","",regex=False).str.zfill(4)
    if "日期" in d.columns:
        d["日期"] = pd.to_datetime(d["日期"], errors="coerce")
    return d


def load_history():
    p = latest_history_csv()
    if p is None:
        return pd.DataFrame()
    try:
        return read_history(str(p), p.stat().st_mtime)
    except Exception:
        return pd.DataFrame()




@st.cache_data(ttl=300, show_spinner=False)
def load_price_archive():
    p = DATA_DIR / "price_history_archive.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(p, dtype={"股票代號": str})
        if "股票代號" in d.columns:
            d["股票代號"] = d["股票代號"].astype(str).str.replace(".0","",regex=False).str.zfill(4)
        if "日期" in d.columns:
            d["日期"] = pd.to_datetime(d["日期"], errors="coerce")
        return d
    except Exception:
        return pd.DataFrame()




def _first_existing(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def build_ai_trade_plan(hist, row):
    """依價格結構、均線、近期高低點與 ATR 產生機械化交易觀察計畫。"""
    if hist is None or hist.empty:
        return {}

    d = hist.copy()
    if "日期" in d.columns:
        d["日期"] = pd.to_datetime(d["日期"], errors="coerce")
        d = d.sort_values("日期")

    close_col = _first_existing(d, ["收盤價", "close", "Close"])
    high_col = _first_existing(d, ["最高價", "max", "high", "High"])
    low_col = _first_existing(d, ["最低價", "min", "low", "Low"])

    if close_col is None:
        return {}

    close_s = pd.to_numeric(d[close_col], errors="coerce").dropna()
    if close_s.empty:
        return {}

    current = float(close_s.iloc[-1])

    def last_num(col):
        if col in d.columns:
            x = pd.to_numeric(d[col], errors="coerce").dropna()
            if not x.empty:
                return float(x.iloc[-1])
        return np.nan

    ma5 = last_num("5日均線")
    ma20 = last_num("20日均線")
    ma60 = last_num("60日均線")
    atr = row.get("ATR14", np.nan)
    try:
        atr = float(atr)
    except Exception:
        atr = np.nan
    if pd.isna(atr) or atr <= 0:
        atr_pct = row.get("ATR百分比", np.nan)
        try:
            atr = current * float(atr_pct) / 100
        except Exception:
            atr = current * 0.025

    highs = pd.to_numeric(d[high_col], errors="coerce") if high_col else pd.Series(index=d.index, dtype=float)
    lows = pd.to_numeric(d[low_col], errors="coerce") if low_col else pd.Series(index=d.index, dtype=float)

    h10 = float(highs.tail(10).max()) if high_col and highs.tail(10).notna().any() else np.nan
    h20 = float(highs.tail(20).max()) if high_col and highs.tail(20).notna().any() else np.nan
    h60 = float(highs.tail(60).max()) if high_col and highs.tail(60).notna().any() else np.nan
    l10 = float(lows.tail(10).min()) if low_col and lows.tail(10).notna().any() else np.nan
    l20 = float(lows.tail(20).min()) if low_col and lows.tail(20).notna().any() else np.nan
    l60 = float(lows.tail(60).min()) if low_col and lows.tail(60).notna().any() else np.nan

    # 短線支撐：選「現價下方且最接近現價」的技術位。
    support_candidates = [
        ("MA5", ma5), ("MA20", ma20), ("10日低點", l10), ("20日低點", l20)
    ]
    support_candidates = [(n,v) for n,v in support_candidates if pd.notna(v) and v < current]
    if support_candidates:
        short_support_name, short_support = max(support_candidates, key=lambda x: x[1])
    else:
        short_support_name, short_support = "ATR動態支撐", current - atr

    # 長線支撐：優先使用 MA60；若 MA60 不在現價下方則退回 60 日低點。
    if pd.notna(ma60) and ma60 < current:
        long_support_name, long_support = "MA60", ma60
    elif pd.notna(l60) and l60 < current:
        long_support_name, long_support = "60日低點", l60
    else:
        long_support_name, long_support = "ATR延伸支撐", current - 2 * atr

    # 壓力：找現價上方最近的近期高點；若已創波段新高則用 ATR 延伸作動態參考。
    resistance_candidates = [
        ("10日高點", h10), ("20日高點", h20), ("60日高點", h60)
    ]
    resistance_candidates = [(n,v) for n,v in resistance_candidates if pd.notna(v) and v > current * 1.002]
    if resistance_candidates:
        short_res_name, short_res = min(resistance_candidates, key=lambda x: x[1])
    else:
        short_res_name, short_res = "ATR動態壓力", current + atr

    if pd.notna(h60) and h60 > short_res * 1.002:
        long_res_name, long_res = "60日高點", h60
    else:
        long_res_name, long_res = "2ATR動態壓力", max(short_res + atr, current + 2 * atr)

    stage = str(row.get("啟動階段", ""))
    entry_label = str(row.get("進場判定", ""))
    entry_score = pd.to_numeric(pd.Series([row.get("進場時機分數")]), errors="coerce").iloc[0]

    # 買入觀察區：依啟動階段調整。
    if "剛啟動" in stage:
        entry_low = max(short_support, current - 0.9 * atr)
        entry_high = min(current, short_support + 0.55 * atr)
        if entry_high < entry_low:
            entry_high = min(current, entry_low + 0.4 * atr)
        plan_type = "回測支撐型"
    elif "蓄勢" in stage:
        entry_low = max(current, short_res - 0.25 * atr)
        entry_high = short_res + 0.35 * atr
        plan_type = "突破確認型"
    elif "趨勢加速" in stage:
        entry_low = max(short_support, current - 1.0 * atr)
        entry_high = max(entry_low, current - 0.35 * atr)
        plan_type = "等待回檔型"
    else:
        entry_low = max(short_support, current - 0.8 * atr)
        entry_high = min(current, short_support + 0.5 * atr)
        if entry_high < entry_low:
            entry_high = entry_low + 0.35 * atr
        plan_type = "保守觀察型"

    # 停損／失效：支撐下方留 ATR 緩衝，避免單純碰線就被洗出。
    stop = short_support - 0.55 * atr
    hard_floor = current * 0.92
    stop = max(stop, hard_floor)
    if stop >= entry_low:
        stop = entry_low - 0.6 * atr

    # 1～2 月波段目標：壓力位不再當作自動止盈，而是當「途中關卡」。
    # 真正的波段目標用進場區上緣計算 +15%、+30%、+50%，
    # 並用移動停利處理強勢股，避免一碰短壓就太早賣掉。
    take_profit1 = entry_high * 1.15
    take_profit2 = entry_high * 1.30
    take_profit3 = entry_high * 1.50

    risk_pct = (entry_high / stop - 1) * 100 if stop > 0 else np.nan
    reward1_pct = 15.0
    reward2_pct = 30.0
    reward3_pct = 50.0
    rr1 = reward1_pct / risk_pct if pd.notna(risk_pct) and risk_pct > 0 else np.nan
    rr2 = reward2_pct / risk_pct if pd.notna(risk_pct) and risk_pct > 0 else np.nan
    rr3 = reward3_pct / risk_pct if pd.notna(risk_pct) and risk_pct > 0 else np.nan

    # 1～2 月波段潛力分數：偏重中期動能、趨勢、量能與基本品質。
    ret20_v = pd.to_numeric(pd.Series([row.get("20日報酬率")]), errors="coerce").iloc[0]
    ret60_v = pd.to_numeric(pd.Series([row.get("60日報酬率")]), errors="coerce").iloc[0]
    vol_ratio_v = pd.to_numeric(pd.Series([row.get("量比")]), errors="coerce").iloc[0]
    tech_v = pd.to_numeric(pd.Series([row.get("技術啟動分數")]), errors="coerce").iloc[0]
    chip_v = pd.to_numeric(pd.Series([row.get("籌碼動能分數")]), errors="coerce").iloc[0]
    quality_v = pd.to_numeric(pd.Series([row.get("基本品質分數")]), errors="coerce").iloc[0]
    momentum_v = pd.to_numeric(pd.Series([row.get("價格動能分數")]), errors="coerce").iloc[0]

    swing_score = 0.0
    swing_score += (0 if pd.isna(tech_v) else tech_v) * 0.25
    swing_score += (0 if pd.isna(chip_v) else chip_v) * 0.20
    swing_score += (0 if pd.isna(quality_v) else quality_v) * 0.20
    swing_score += (0 if pd.isna(momentum_v) else momentum_v) * 0.20

    trend_bonus = 0.0
    if pd.notna(ret20_v) and 5 <= ret20_v <= 25:
        trend_bonus += 5
    if pd.notna(ret60_v) and 10 <= ret60_v <= 45:
        trend_bonus += 5
    if pd.notna(vol_ratio_v) and 1.2 <= vol_ratio_v <= 3.0:
        trend_bonus += 5
    swing_score = min(100.0, swing_score + trend_bonus)

    if swing_score >= 80:
        swing_label = "高波段潛力"
    elif swing_score >= 68:
        swing_label = "中高波段潛力"
    elif swing_score >= 55:
        swing_label = "中等波段潛力"
    else:
        swing_label = "波段潛力不足"

    if "可觀察進場" in entry_label:
        action = "可觀察分批進場"
        explanation = "模型處於可觀察進場狀態；較適合等價格落在觀察買入區，而不是直接追高。"
    elif "等待突破" in entry_label:
        action = "等待突破確認"
        explanation = "目前較適合等待短線壓力被有效突破，再觀察是否轉成可進場狀態。"
    elif "等待回檔" in entry_label:
        action = "等待回檔"
        explanation = "目前位置偏離支撐較遠，先等價格靠近短線支撐區較合理。"
    elif "短線過熱" in entry_label:
        action = "暫不追價"
        explanation = "短線過熱，模型不建議在壓力附近追價，優先等乖離收斂。"
    else:
        action = "暫不考慮"
        explanation = "目前進場條件不足，先等待模型重新轉強。"

    return {
        "現價": current,
        "策略": plan_type,
        "操作狀態": action,
        "說明": explanation,
        "觀察買入下緣": entry_low,
        "觀察買入上緣": entry_high,
        "停損失效價": stop,
        "第一波段目標": take_profit1,
        "第二波段目標": take_profit2,
        "50%挑戰價": take_profit3,
        "波段潛力分數": swing_score,
        "波段潛力判定": swing_label,
        "短期支撐": short_support,
        "短期支撐來源": short_support_name,
        "長期支撐": long_support,
        "長期支撐來源": long_support_name,
        "短期壓力": short_res,
        "短期壓力來源": short_res_name,
        "長期壓力": long_res,
        "長期壓力來源": long_res_name,
        "ATR": atr,
        "估計風險幅度%": risk_pct,
        "第一目標潛在空間%": reward1_pct,
        "第二目標潛在空間%": reward2_pct,
        "50%目標潛在空間%": reward3_pct,
        "第一目標風報比": rr1,
        "第二目標風報比": rr2,
        "50%目標風報比": rr3,
        "進場時機分數": entry_score,
    }


def ai_trade_plan_chart(hist, plan):
    if not plan or hist is None or hist.empty:
        return None

    d = hist.copy()
    if "日期" in d.columns:
        d["日期"] = pd.to_datetime(d["日期"], errors="coerce")
        d = d.sort_values("日期").tail(80)

    f = history_price_chart(d)
    f.update_layout(
        title="AI 操作教練：價格與均線",
        margin=dict(l=15,r=15,t=55,b=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.14,
            xanchor="left",
            x=0,
        ),
    )

    low = plan.get("觀察買入下緣")
    high = plan.get("觀察買入上緣")
    if pd.notna(low) and pd.notna(high):
        f.add_hrect(
            y0=min(low, high),
            y1=max(low, high),
            opacity=0.08,
            line_width=0,
        )

    return f

def history_price_chart(d):
    f = go.Figure()
    if "收盤價" in d.columns:
        f.add_trace(go.Scatter(x=d["日期"], y=d["收盤價"], mode="lines", name="收盤價"))
    for c, n in [("5日均線","MA5"),("20日均線","MA20"),("60日均線","MA60")]:
        if c in d.columns:
            f.add_trace(go.Scatter(x=d["日期"], y=d[c], mode="lines", name=n))
    f.update_layout(height=420, margin=dict(l=15,r=15,t=35,b=15), title="近 120 日股價與均線", hovermode="x unified")
    return f


def history_rsi_chart(d):
    f = go.Figure()
    if "14日RSI" in d.columns:
        f.add_trace(go.Scatter(x=d["日期"], y=d["14日RSI"], mode="lines", name="RSI14"))
    f.add_hline(y=70, line_dash="dash")
    f.add_hline(y=30, line_dash="dash")
    f.update_yaxes(range=[0,100])
    f.update_layout(height=280, margin=dict(l=15,r=15,t=35,b=15), title="RSI14", hovermode="x unified")
    return f


def history_macd_chart(d):
    f = go.Figure()
    if "MACD" in d.columns:
        f.add_trace(go.Scatter(x=d["日期"], y=d["MACD"], mode="lines", name="MACD"))
    if "MACD訊號線" in d.columns:
        f.add_trace(go.Scatter(x=d["日期"], y=d["MACD訊號線"], mode="lines", name="Signal"))
    if "MACD柱狀體" in d.columns:
        f.add_trace(go.Bar(x=d["日期"], y=d["MACD柱狀體"], name="Histogram"))
    f.update_layout(height=300, margin=dict(l=15,r=15,t=35,b=15), title="MACD", hovermode="x unified")
    return f



def load_ranking_history():
    p = DATA_DIR / "ranking_history.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(p, dtype={"股票代號": str})
        d["股票代號"] = d["股票代號"].astype(str).str.replace(".0","",regex=False).str.zfill(4)
        d["快照日期"] = pd.to_datetime(d["快照日期"], errors="coerce")
        return d
    except Exception:
        return pd.DataFrame()


def attach_forward_returns(snapshot_df, price_df):
    if snapshot_df.empty or price_df.empty:
        return pd.DataFrame()

    required = {"股票代號","快照日期"}
    if not required.issubset(snapshot_df.columns):
        return pd.DataFrame()

    p = price_df.copy()
    if "日期" not in p.columns or "收盤價" not in p.columns or "股票代號" not in p.columns:
        return pd.DataFrame()

    p["日期"] = pd.to_datetime(p["日期"], errors="coerce")
    p["收盤價"] = pd.to_numeric(p["收盤價"], errors="coerce")
    p = p.dropna(subset=["日期","收盤價"])

    price_map = {
        code: g.sort_values("日期")[["日期","收盤價"]].reset_index(drop=True)
        for code, g in p.groupby("股票代號")
    }

    records = []
    for _, r in snapshot_df.iterrows():
        code = str(r["股票代號"])
        snap_date = pd.to_datetime(r["快照日期"], errors="coerce")
        g = price_map.get(code)
        if g is None or pd.isna(snap_date):
            continue

        pos = g.index[g["日期"] >= snap_date]
        if len(pos) == 0:
            continue
        i = int(pos[0])
        base = g.iloc[i]["收盤價"]
        rec = r.to_dict()

        for horizon in [1,5,20,40]:
            j = i + horizon
            col = f"{horizon}日後報酬率"
            if j < len(g):
                future = g.iloc[j]["收盤價"]
                rec[col] = (future / base - 1) * 100
            else:
                rec[col] = np.nan

        for horizon in [20,40]:
            end = min(i + horizon, len(g) - 1)
            if end > i:
                future_slice = g.iloc[i+1:end+1]["收盤價"]
                max_ret = (future_slice.max() / base - 1) * 100 if len(future_slice) else np.nan
            else:
                max_ret = np.nan
            rec[f"{horizon}日內最高報酬率"] = max_ret

        rec["40日內達50%"] = (
            rec.get("40日內最高報酬率", np.nan) >= 50
            if pd.notna(rec.get("40日內最高報酬率", np.nan))
            else np.nan
        )

        records.append(rec)

    return pd.DataFrame(records)


def backtest_summary(bt):
    rows = []
    groups = [
        ("原始V5 Top 10", bt[pd.to_numeric(bt.get("排名"), errors="coerce") <= 10]),
        ("一週模型 Top 10", bt[pd.to_numeric(bt.get("一週模型排名"), errors="coerce") <= 10] if "一週模型排名" in bt.columns else bt.iloc[0:0]),
        ("可觀察進場", bt[bt.get("進場判定", pd.Series("",index=bt.index)).astype(str).str.contains("可觀察進場",na=False)] if "進場判定" in bt.columns else bt.iloc[0:0]),
        ("全部股票池", bt),
    ]

    for label, d in groups:
        for h in [1,5,20,40]:
            col=f"{h}日後報酬率"
            if col not in d.columns:
                continue
            x=pd.to_numeric(d[col],errors="coerce").dropna()
            if len(x)==0:
                continue
            rows.append({
                "群組":label,
                "期間":f"{h}交易日",
                "樣本數":len(x),
                "平均報酬率":x.mean(),
                "中位數報酬率":x.median(),
                "勝率":(x>0).mean()*100,
            })
    if "40日內達50%" in bt.columns:
        for label, d in groups:
            x = d["40日內達50%"].dropna()
            if len(x):
                rows.append({
                    "群組": label,
                    "期間": "40日內+50%",
                    "樣本數": len(x),
                    "平均報酬率": np.nan,
                    "中位數報酬率": np.nan,
                    "勝率": x.astype(float).mean() * 100,
                })
    return pd.DataFrame(rows)



def score_bucket_analysis(bt, score_col, horizon):
    ret_col = f"{horizon}日後報酬率"
    if score_col not in bt.columns or ret_col not in bt.columns:
        return pd.DataFrame()

    d = bt[[score_col, ret_col]].copy()
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d[ret_col] = pd.to_numeric(d[ret_col], errors="coerce")
    d = d.dropna()

    if d.empty:
        return pd.DataFrame()

    bins = [-np.inf, 50, 60, 70, 80, np.inf]
    labels = ["<50", "50–59", "60–69", "70–79", "80+"]
    d["分數區間"] = pd.cut(
        d[score_col],
        bins=bins,
        labels=labels,
        right=False,
    )

    out = (
        d.groupby("分數區間", observed=False)[ret_col]
        .agg(
            樣本數="count",
            平均報酬率="mean",
            中位數報酬率="median",
            標準差="std",
        )
        .reset_index()
    )

    win = (
        d.assign(_win=d[ret_col] > 0)
        .groupby("分數區間", observed=False)["_win"]
        .mean()
        .mul(100)
        .reset_index(name="勝率")
    )

    out = out.merge(win, on="分數區間", how="left")
    return out


def score_predictive_table(bt, horizon):
    ret_col = f"{horizon}日後報酬率"
    if ret_col not in bt.columns:
        return pd.DataFrame()

    score_cols = [
        ("一週起漲分數", "V5-B 一週起漲"),
        ("進場時機分數", "進場時機"),
        ("技術啟動分數", "V1-B 技術啟動"),
        ("籌碼動能分數", "V2-B 籌碼動能"),
        ("基本品質分數", "V3-B 基本品質"),
        ("價格動能分數", "V4-B 價格動能"),
        ("最終分數", "原始 V5-A"),
        ("技術分數", "原始 V1"),
        ("籌碼標準分", "原始 V2"),
        ("基本面分數", "原始 V3"),
        ("風險動能分數", "原始 V4"),
    ]

    rows = []
    for col, label in score_cols:
        if col not in bt.columns:
            continue

        x = pd.to_numeric(bt[col], errors="coerce")
        y = pd.to_numeric(bt[ret_col], errors="coerce")
        pair = pd.DataFrame({"x": x, "y": y}).dropna()

        if len(pair) < 5:
            continue

        # Spearman = 對兩欄先排名後，再算 Pearson correlation
        rho = pair["x"].rank().corr(pair["y"].rank())

        high = pair[pair["x"] >= 80]["y"]
        low = pair[pair["x"] < 60]["y"]

        rows.append({
            "構面": label,
            "有效樣本": len(pair),
            "Spearman相關": rho,
            "80分以上平均報酬": high.mean() if len(high) else np.nan,
            "60分以下平均報酬": low.mean() if len(low) else np.nan,
            "高低分報酬差": (
                high.mean() - low.mean()
                if len(high) and len(low)
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


def bucket_chart(bucket_df, title):
    if bucket_df.empty:
        return None

    f = go.Figure()
    f.add_trace(
        go.Bar(
            x=bucket_df["分數區間"].astype(str),
            y=bucket_df["平均報酬率"],
            name="平均報酬率",
            text=bucket_df["平均報酬率"].round(2),
            textposition="outside",
        )
    )
    f.update_layout(
        height=360,
        title=title,
        xaxis_title="分數區間",
        yaxis_title="後續平均報酬率 (%)",
        margin=dict(l=15, r=15, t=45, b=20),
    )
    return f






def load_walkforward_validation():
    summary_path = DATA_DIR / "walkforward_summary.csv"
    metadata_path = DATA_DIR / "walkforward_metadata.csv"
    results_path = DATA_DIR / "walkforward_results.csv"
    exit_summary_path = DATA_DIR / "walkforward_exit_summary.csv"
    exit_trades_path = DATA_DIR / "walkforward_exit_trades.csv"

    summary = pd.DataFrame()
    metadata = pd.DataFrame()
    results = pd.DataFrame()
    exit_summary = pd.DataFrame()
    exit_trades = pd.DataFrame()

    try:
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
    except Exception:
        summary = pd.DataFrame()

    try:
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
    except Exception:
        metadata = pd.DataFrame()

    try:
        if results_path.exists():
            results = pd.read_csv(results_path, dtype={"stock_id": str})
            if "date" in results.columns:
                results["date"] = pd.to_datetime(results["date"], errors="coerce")
    except Exception:
        results = pd.DataFrame()

    try:
        if exit_summary_path.exists():
            exit_summary = pd.read_csv(exit_summary_path)
    except Exception:
        exit_summary = pd.DataFrame()

    try:
        if exit_trades_path.exists():
            exit_trades = pd.read_csv(exit_trades_path, dtype={"stock_id": str})
            for c in ["signal_date","entry_date","exit_date"]:
                if c in exit_trades.columns:
                    exit_trades[c] = pd.to_datetime(exit_trades[c], errors="coerce")
    except Exception:
        exit_trades = pd.DataFrame()

    optimizer_path = DATA_DIR / "walkforward_exit_optimizer.csv"
    optimizer_shortlist_path = DATA_DIR / "walkforward_exit_optimizer_shortlist.csv"
    optimizer_yearly_path = DATA_DIR / "walkforward_exit_optimizer_yearly.csv"

    optimizer = pd.DataFrame()
    optimizer_shortlist = pd.DataFrame()
    optimizer_yearly = pd.DataFrame()

    try:
        if optimizer_path.exists():
            optimizer = pd.read_csv(optimizer_path)
    except Exception:
        optimizer = pd.DataFrame()

    try:
        if optimizer_shortlist_path.exists():
            optimizer_shortlist = pd.read_csv(optimizer_shortlist_path)
    except Exception:
        optimizer_shortlist = pd.DataFrame()

    try:
        if optimizer_yearly_path.exists():
            optimizer_yearly = pd.read_csv(optimizer_yearly_path)
    except Exception:
        optimizer_yearly = pd.DataFrame()

    robustness_stop_hold_path = DATA_DIR / "walkforward_robustness_stop_hold.csv"
    robustness_trailing_path = DATA_DIR / "walkforward_robustness_trailing.csv"
    robustness_cost_path = DATA_DIR / "walkforward_robustness_cost.csv"
    robustness_scorecard_path = DATA_DIR / "walkforward_robustness_scorecard.csv"

    robustness_stop_hold = pd.DataFrame()
    robustness_trailing = pd.DataFrame()
    robustness_cost = pd.DataFrame()
    robustness_scorecard = pd.DataFrame()

    try:
        if robustness_stop_hold_path.exists():
            robustness_stop_hold = pd.read_csv(robustness_stop_hold_path)
    except Exception:
        robustness_stop_hold = pd.DataFrame()

    try:
        if robustness_trailing_path.exists():
            robustness_trailing = pd.read_csv(robustness_trailing_path)
    except Exception:
        robustness_trailing = pd.DataFrame()

    try:
        if robustness_cost_path.exists():
            robustness_cost = pd.read_csv(robustness_cost_path)
    except Exception:
        robustness_cost = pd.DataFrame()

    try:
        if robustness_scorecard_path.exists():
            robustness_scorecard = pd.read_csv(robustness_scorecard_path)
    except Exception:
        robustness_scorecard = pd.DataFrame()

    regime_signal_path = DATA_DIR / "walkforward_regime_signal_summary.csv"
    regime_exit_path = DATA_DIR / "walkforward_regime_exit_summary.csv"
    regime_gate_path = DATA_DIR / "walkforward_regime_gate.csv"

    regime_signal = pd.DataFrame()
    regime_exit = pd.DataFrame()
    regime_gate = pd.DataFrame()

    try:
        if regime_signal_path.exists():
            regime_signal = pd.read_csv(regime_signal_path)
    except Exception:
        regime_signal = pd.DataFrame()

    try:
        if regime_exit_path.exists():
            regime_exit = pd.read_csv(regime_exit_path)
    except Exception:
        regime_exit = pd.DataFrame()

    try:
        if regime_gate_path.exists():
            regime_gate = pd.read_csv(regime_gate_path)
    except Exception:
        regime_gate = pd.DataFrame()

    selection_files = {
        "s2_pr": DATA_DIR / "selection_v2_precision_recall.csv",
        "s2_diag": DATA_DIR / "selection_v2_feature_diagnostics.csv",
        "s2_fn": DATA_DIR / "selection_v2_false_negatives.csv",
        "s2_runup": DATA_DIR / "selection_v2_runup_buckets.csv",
        "s2_ignition": DATA_DIR / "selection_v2_ignition_timing.csv",
        "s2_candidates": DATA_DIR / "selection_v2_model_candidates.csv",
        "s2_comparison": DATA_DIR / "selection_v2_model_comparison.csv",
        "s2_branches": DATA_DIR / "selection_v2_branch_summary.csv",
        "s2_selected": DATA_DIR / "selection_v2_selected_config.csv",
        "s21_candidates": DATA_DIR / "selection_v21_model_candidates.csv",
        "s21_comparison": DATA_DIR / "selection_v21_model_comparison.csv",
        "s21_attribution": DATA_DIR / "selection_v21_branch_attribution.csv",
        "s21_selected": DATA_DIR / "selection_v21_selected_config.csv",
        "ign2_candidates": DATA_DIR / "ignition_v2_candidates.csv",
        "ign2_comparison": DATA_DIR / "ignition_v2_comparison.csv",
        "ign2_recovery": DATA_DIR / "ignition_v2_false_negative_recovery.csv",
        "ign2_features": DATA_DIR / "ignition_v2_feature_diagnostics.csv",
        "ign2_selected": DATA_DIR / "ignition_v2_selected_config.csv",
    }
    selection_data = {}
    for key, path in selection_files.items():
        try:
            selection_data[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
        except Exception:
            selection_data[key] = pd.DataFrame()

    return (
        summary, metadata, results, exit_summary, exit_trades,
        optimizer, optimizer_shortlist, optimizer_yearly,
        robustness_stop_hold, robustness_trailing,
        robustness_cost, robustness_scorecard,
        regime_signal, regime_exit, regime_gate,
        selection_data
    )


def load_global_market():
    detail_path = DATA_DIR / "global_market_latest.csv"
    summary_path = DATA_DIR / "global_market_summary.csv"

    detail = pd.DataFrame()
    summary = pd.DataFrame()

    try:
        if detail_path.exists():
            detail = pd.read_csv(detail_path)
    except Exception:
        detail = pd.DataFrame()

    try:
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
    except Exception:
        summary = pd.DataFrame()

    return detail, summary


def global_regime_color(label):
    if label in ["偏多順風", "中性偏多"]:
        return "good"
    if label == "震盪中性":
        return "warn"
    return "bad"


def base_date(sheets, filename):
    d = sheets.get("系統說明")
    if d is not None and not d.empty and {"項目","說明"}.issubset(d.columns):
        x = d.loc[d["項目"].astype(str).eq("資料基準日"), "說明"]
        if not x.empty: return str(x.iloc[0])
    m = re.search(r"(20\d{6})", filename)
    if m:
        x = m.group(1); return f"{x[:4]}-{x[4:6]}-{x[6:]}"
    return "—"


def four_factor(row):
    labels = ["V1 技術","V2 籌碼","V3 基本面","V4 風險動能"]
    vals = [row.get("技術分數"), row.get("籌碼標準分"), row.get("基本面分數"), row.get("風險動能分數")]
    vals = [0 if pd.isna(v) else float(v) for v in vals]
    f = go.Figure(go.Bar(x=labels, y=vals, text=[f"{v:.0f}" for v in vals], textposition="outside"))
    f.update_yaxes(range=[0,105]); f.update_layout(height=320, margin=dict(l=15,r=15,t=35,b=15), title="四構面分數")
    return f


def radar(row):
    labels = ["技術","籌碼","基本面","風險動能"]
    vals = [row.get("技術分數"), row.get("籌碼標準分"), row.get("基本面分數"), row.get("風險動能分數")]
    vals = [0 if pd.isna(v) else float(v) for v in vals]
    f = go.Figure(go.Scatterpolar(r=vals+[vals[0]], theta=labels+[labels[0]], fill="toself"))
    f.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),showlegend=False,height=340,margin=dict(l=20,r=20,t=35,b=20),title="模型雷達圖")
    return f


def pr_chart(row):
    mp = {"EPS":"每股盈餘PR","ROE":"ROE PR","毛利率":"毛利率PR","營益率":"營業利益率PR","營收YoY":"月營收年增率PR","營收MoM":"月營收月增率PR","負債比":"負債比率PR","流動比":"流動比率PR","資產周轉":"總資產周轉率PR","估值":"估值PR","殖利率":"殖利率PR"}
    labs, vals = [], []
    for lab,col in mp.items():
        if col in row.index and pd.notna(row[col]): labs.append(lab); vals.append(float(row[col]))
    f = go.Figure(go.Bar(x=vals,y=labs,orientation="h",text=[f"{v:.0f}" for v in vals],textposition="auto"))
    f.update_xaxes(range=[0,100],title="同產業 PR"); f.update_layout(height=430,margin=dict(l=15,r=15,t=35,b=20),title="同產業基本面 PR")
    return f


def inst_chart(row):
    labs = ["外資","投信","自營商"]
    cols = ["外資近5日買賣超（張）","投信近5日買賣超（張）","自營商近5日買賣超（張）"]
    vals = [float(row.get(c)) if pd.notna(row.get(c,np.nan)) else 0 for c in cols]
    f = go.Figure(go.Bar(x=labs,y=vals,text=[f"{v:,.0f}" for v in vals],textposition="outside")); f.add_hline(y=0)
    f.update_layout(height=320,margin=dict(l=15,r=15,t=35,b=15),title="近 5 日法人買賣超（張）")
    return f


def ma_chart(row):
    labs = ["收盤","MA5","MA20","MA60"]
    cols = ["收盤價","5日均線","20日均線","60日均線"]
    vals = [float(row.get(c)) if pd.notna(row.get(c,np.nan)) else np.nan for c in cols]
    pairs = [(a,b) for a,b in zip(labs,vals) if not pd.isna(b)]
    if not pairs: return None
    labs,vals = zip(*pairs)
    f = go.Figure(go.Bar(x=list(labs),y=list(vals),text=[f"{v:,.2f}" for v in vals],textposition="outside"))
    f.update_layout(height=300,margin=dict(l=15,r=15,t=35,b=15),title="收盤價與均線")
    return f


@st.cache_data(ttl=900, show_spinner=False)
def news(name, code, limit=8):
    url = "https://news.google.com/rss/search?q=" + quote(f'"{name}" 股票 {code}') + "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            root = ET.fromstring(r.read())
        out = []
        for item in root.findall(".//item")[:limit]:
            src = item.find("source")
            out.append({"title":item.findtext("title",default="").strip(),"link":item.findtext("link",default="").strip(),"date":item.findtext("pubDate",default="").strip(),"source":src.text.strip() if src is not None and src.text else ""})
        return out
    except Exception:
        return []


def table(df, height=500):
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)


css()
sheets, source = load_data()
history_all = load_history()
price_archive_all = load_price_archive()
ranking_history_all = load_ranking_history()
weekly_model_all = load_weekly_model()
domestic_market_summary = load_domestic_market()
global_market_detail, global_market_summary = load_global_market()
(
    walkforward_summary,
    walkforward_metadata,
    walkforward_results,
    walkforward_exit_summary,
    walkforward_exit_trades,
    walkforward_exit_optimizer,
    walkforward_exit_optimizer_shortlist,
    walkforward_exit_optimizer_yearly,
    walkforward_robustness_stop_hold,
    walkforward_robustness_trailing,
    walkforward_robustness_cost,
    walkforward_robustness_scorecard,
    walkforward_regime_signal,
    walkforward_regime_exit,
    walkforward_regime_gate,
    walkforward_selection_v2,
) = load_walkforward_validation()
if not sheets:
    st.warning("找不到資料，請從側邊欄上傳 V1～V5 Excel。")
    st.stop()

rank = normalize(sheets.get("全部排名", pd.DataFrame()))

if not weekly_model_all.empty:
    weekly_cols = [
        c for c in [
            "股票代號","一週模型排名","主模型排名","起漲潛力排名","波段爆發排名",
            "一週起漲分數","波段爆發分數","50%歷史型態命中率","50%歷史型態PR",
            "相似樣本40日最高報酬均值","突破強度分數","動能持續分數","活躍爆發分數",
            "波動爆發潛力","主模型分數","50%潛力判定","進場時機分數",
            "技術啟動分數","籌碼動能分數","基本品質分數","價格動能分數",
            "風險扣分","啟動階段","進場判定","起漲原因","進場風險",
            "全球環境分數","台股環境分數","產業海外順風分數",
            "全球環境判定","台股環境判定"
        ] if c in weekly_model_all.columns
    ]
    # Excel 主表保留原始 V5-A；一週模型欄位另外合併進來
    rank = rank.merge(
        weekly_model_all[weekly_cols].drop_duplicates("股票代號"),
        on="股票代號",
        how="left",
    )


previous_rank, previous_source = load_previous_rank(source)
daily_change = build_daily_change(rank, previous_rank)
weekly_daily_change = build_weekly_change(ranking_history_all)

if rank.empty:
    st.error("Excel 找不到『全部排名』工作表。")
    st.stop()

bd = base_date(sheets, source)
tech_ok = int(rank.get("技術分數",pd.Series(index=rank.index,dtype=float)).notna().sum())
v4_ok = int(rank.get("風險動能分數",pd.Series(index=rank.index,dtype=float)).notna().sum())
coverage = min(tech_ok,v4_ok)/len(rank) if len(rank) else 0

st.markdown(f'<div class="hero"><h1>📈 台股 V1～V5 選股</h1><p>資料基準日：{bd} ｜ 依 V1～V5 模型分數與風險條件排序</p></div>', unsafe_allow_html=True)
if coverage < .95:
    st.error(f"⚠️ 今日 V1/V4 完整度只有 {min(tech_ok,v4_ok)}/{len(rank)}（{coverage:.0%}），排名不應視為完整市場比較。")

pages = ["今日 Top 10","專業驗證","一週模型","全球市場","每日變化","V6 回測","個股分析","完整排名","風險監控","產業分析"]
if "nav" not in st.session_state: st.session_state.nav = "今日 Top 10"
page = st.radio("導覽", pages, horizontal=True, label_visibility="collapsed", key="nav")

if page == "今日 Top 10":
    weekly_ready = "一週模型排名" in rank.columns and rank["一週模型排名"].notna().any()

    if weekly_ready:
        usable = rank.copy()
        can_enter = usable.get("進場判定", pd.Series("", index=usable.index)).astype(str).str.contains("可觀察進場", na=False).sum()
        just_started = usable.get("啟動階段", pd.Series("", index=usable.index)).astype(str).str.contains("剛啟動", na=False).sum()
        waiting_breakout = usable.get("進場判定", pd.Series("", index=usable.index)).astype(str).str.contains("等待突破", na=False).sum()

        a,b,c,d = st.columns(4)
        a.metric("股票池", f"{len(rank)} 檔")
        b.metric("可觀察進場", f"{int(can_enter)} 檔")
        c.metric("剛啟動", f"{int(just_started)} 檔")
        d.metric("等待突破", f"{int(waiting_breakout)} 檔")

        env1,env2 = st.columns(2)
        with env1:
            if not domestic_market_summary.empty:
                ds=domestic_market_summary.iloc[0]
                st.markdown(
                    f'<div class="card"><div class="title">台股環境：{fmt(ds.get("台股環境分數"),1)}</div>'
                    f'<div class="{global_regime_color(str(ds.get("台股環境判定","資料不足")))}">{ds.get("台股環境判定","資料不足")}</div>'
                    f'<div class="muted">站上MA20：{fmt(ds.get("站上MA20比例"),1,"%")} ｜ 5日上漲：{fmt(ds.get("5日上漲比例"),1,"%")}</div></div>',
                    unsafe_allow_html=True,
                )
        with env2:
            if not global_market_summary.empty:
                gs=global_market_summary.iloc[0]
                regime=str(gs.get("全球環境判定","資料不足"))
                st.markdown(
                    f'<div class="card"><div class="title">全球環境：{fmt(gs.get("全球環境分數"),1)}</div>'
                    f'<div class="{global_regime_color(regime)}">{regime}</div>'
                    f'<div class="muted">海外市場、半導體、VIX、利率與匯率 Gate</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("## 1～2 個月 50% 波段目標 Top 10")
        st.caption("主排名目標是找出未來 1～2 個月具大波段爆發潛力、且仍有操作意義的個股。首頁會排除『暫不考慮』與『短線過熱』；等待回檔 / 等待突破會直接顯示對應觀察價位。50% 是篩選與回測目標，不是保證報酬。")

        # 首頁 Top 10 只放「仍具操作意義」的標的。
        # 暫不考慮 / 短線過熱 不應佔用 Top10 名額。
        eligible_labels = ["可觀察進場", "等待回檔", "等待突破"]
        top_pool = rank.copy()
        if "進場判定" in top_pool.columns:
            entry_text = top_pool["進場判定"].astype(str)
            mask = False
            for label in eligible_labels:
                mask = mask | entry_text.str.contains(label, na=False)
            top_pool = top_pool[mask].copy()

        sort_col = "主模型排名" if "主模型排名" in top_pool.columns else "一週模型排名"
        top = top_pool.sort_values([sort_col]).head(10).copy()
        top["首頁排名"] = range(1, len(top) + 1)

        for i in range(0,len(top),2):
            cc=st.columns(2)
            for j in range(2):
                k=i+j
                if k>=len(top): break
                r=top.iloc[k]
                code=str(r.get("股票代號",""))
                name=str(r.get("股票名稱",""))
                stage=str(r.get("啟動階段","—"))
                entry_label=str(r.get("進場判定","—"))

                # 首頁直接給「等待回檔 / 等待突破」的具體價格區間。
                card_hist = (
                    history_all[history_all["股票代號"].eq(code)].copy()
                    if (not history_all.empty and "股票代號" in history_all.columns)
                    else pd.DataFrame()
                )
                card_plan = build_ai_trade_plan(card_hist, r) if not card_hist.empty else {}

                if "等待回檔" in entry_label and card_plan:
                    action_hint = (
                        f'回檔觀察區：<b>{fmt(card_plan.get("觀察買入下緣"),2)}'
                        f' ～ {fmt(card_plan.get("觀察買入上緣"),2)}</b>'
                    )
                elif "等待突破" in entry_label and card_plan:
                    action_hint = (
                        f'突破觀察價：<b>{fmt(card_plan.get("短期壓力"),2)}</b>'
                    )
                elif "可觀察進場" in entry_label and card_plan:
                    action_hint = (
                        f'觀察買入區：<b>{fmt(card_plan.get("觀察買入下緣"),2)}'
                        f' ～ {fmt(card_plan.get("觀察買入上緣"),2)}</b>'
                    )
                else:
                    action_hint = ""

                with cc[j]:
                    card_html = (
                        f'<div class="card">'
                        f'<span class="rank">#{int(r.get("首頁排名",k+1))}</span>'
                        f'<div class="title">{code}　{name}</div>'
                        f'<div class="muted">{r.get("產業別","—")} ｜ {stage}</div>'
                        f'<div class="muted">收盤價：<b>{fmt(r.get("收盤價"),2)}</b></div>'
                        f'<div class="score">{fmt(r.get("主模型分數"),2)}</div>'
                        f'<div class="muted">{r.get("50%潛力判定","—")} ｜ {entry_label}</div>'
                    )
                    if action_hint:
                        card_html += f'<div class="muted">{action_hint}</div>'
                    card_html += (
                        f'<div class="muted">波段爆發 {fmt(r.get("波段爆發分數"),1)} ｜ 進場時機 {fmt(r.get("進場時機分數"),1)}</div>'
                        f'<div class="muted">歷史50%型態PR {fmt(r.get("50%歷史型態PR"),0)} ｜ 相似型態命中率 {fmt(r.get("50%歷史型態命中率"),2,"%")}</div>'
                        f'<div class="muted">突破 {fmt(r.get("突破強度分數"),0)} ｜ 持續動能 {fmt(r.get("動能持續分數"),0)} ｜ 活躍爆發 {fmt(r.get("活躍爆發分數"),0)}</div>'
                        f'<div class="muted">技術啟動 {fmt(r.get("技術啟動分數"),0)} ｜ 籌碼動能 {fmt(r.get("籌碼動能分數"),0)} ｜ 基本品質 {fmt(r.get("基本品質分數"),0)} ｜ 價格動能 {fmt(r.get("價格動能分數"),0)}</div>'
                        f'<div class="muted">風險扣分 {fmt(r.get("風險扣分"),0)} ｜ 原始V5 {fmt(r.get("最終分數"),1)}</div>'
                        f'</div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)
                    st.button(
                        f"查看 {name} 詳細分析 →",
                        key=f"d_{code}",
                        use_container_width=True,
                        on_click=goto_stock_detail,
                        args=(code,),
                    )

        if not weekly_daily_change.empty:
            st.markdown("### 今日一週模型變化")
            new_top = weekly_daily_change[weekly_daily_change.get("新進一週Top10", False) == True].copy()
            new_entry = weekly_daily_change[weekly_daily_change.get("新進可觀察進場", False) == True].copy()
            c1,c2=st.columns(2)
            with c1:
                st.markdown("#### 🔥 新進一週 Top 10")
                if new_top.empty:
                    st.caption("今天沒有新進一週 Top 10。")
                else:
                    cols=[c for c in ["股票代號","股票名稱","一週模型排名","昨日一週排名","一週起漲分數","進場判定"] if c in new_top.columns]
                    table(new_top[cols].head(10),height=250)
            with c2:
                st.markdown("#### 🟢 新進可觀察進場")
                if new_entry.empty:
                    st.caption("今天沒有新進可觀察進場訊號。")
                else:
                    cols=[c for c in ["股票代號","股票名稱","一週模型排名","進場時機分數","啟動階段","進場判定"] if c in new_entry.columns]
                    table(new_entry[cols].head(10),height=250)

    else:
        st.warning("一週模型資料尚未產生。請手動跑一次 GitHub Actions；完成後首頁會自動切換成一週起漲 Top 10。")
        strong = rank.get("候選等級",pd.Series(dtype=str)).astype(str).str.contains("強勢候選",na=False).sum()
        a,b=st.columns(2)
        a.metric("股票池",f"{len(rank)} 檔")
        b.metric("原始強勢候選",f"{int(strong)} 檔")
        st.markdown("## 原始 V5-A Top 10（暫時）")
        top = rank.sort_values("排名").head(10) if "排名" in rank.columns else rank.head(10)
        cols=[c for c in ["排名","股票代號","股票名稱","最終分數","候選等級","目前狀態"] if c in top.columns]
        table(top[cols],height=390)


elif page == "專業驗證":
    st.markdown("## 專業投資人驗證面板")
    st.caption(
        "這一頁只使用每天實際留下的模型快照，等未來價格真的發生後再驗證。"
        "這是 forward / live OOS 驗證，不用今天的模型回頭改寫昨天的訊號。"
    )

    if ranking_history_all.empty or price_archive_all.empty:
        st.info("目前還沒有足夠的排名歷史或長期價格資料。系統會每天自動累積。")
    else:
        pro_bt = attach_professional_forward_metrics(ranking_history_all, price_archive_all)
        readiness = validation_readiness(pro_bt)

        c1,c2,c3 = st.columns(3)
        c1.metric("40日成熟樣本", readiness.get("成熟樣本數", 0))
        c2.metric("成熟訊號日", readiness.get("成熟交易日數", 0))
        c3.metric("驗證狀態", readiness.get("狀態", "—"))
        st.caption(readiness.get("說明", ""))

        summary = professional_validation_summary(pro_bt)
        if summary.empty:
            st.warning("目前還沒有 40 個交易日都走完的成熟樣本。先持續累積，不應過度解讀短期結果。")
        else:
            st.markdown("### Top-K 效果")
            show = summary.copy()
            for c in [
                "40日+20%命中率","40日+30%命中率","40日+50%命中率",
                "40日MFE中位數","40日MAE中位數","40日報酬中位數","40日正報酬率"
            ]:
                if c in show.columns:
                    show[c] = pd.to_numeric(show[c], errors="coerce").round(2)
            if "+50% Lift" in show.columns:
                show["+50% Lift"] = pd.to_numeric(show["+50% Lift"], errors="coerce").round(2)
            table(show, height=330)

            top10_row = summary[summary["群組"].eq("Top 10")]
            base_row = summary[summary["群組"].eq("全部股票池")]
            if not top10_row.empty and not base_row.empty:
                t = top10_row.iloc[0]
                b = base_row.iloc[0]
                a1,a2,a3,a4 = st.columns(4)
                a1.metric("Top10 +50%命中", fmt(t.get("40日+50%命中率"),2,"%"))
                a2.metric("市場基準 +50%", fmt(b.get("40日+50%命中率"),2,"%"))
                a3.metric("Top10 Lift", fmt(t.get("+50% Lift"),2,"x"))
                a4.metric("Top10 40日MFE中位數", fmt(t.get("40日MFE中位數"),2,"%"))

                if pd.notna(t.get("+50% Lift")) and float(t.get("+50% Lift")) > 1:
                    st.success("目前 Top10 的 +50% 命中率高於全部股票池基準。仍需確認樣本數與不同市場 regime 下是否穩定。")
                else:
                    st.warning("目前尚未證明 Top10 對 +50% 目標具有穩定 Lift；不要只看單一批次結果。")

            st.markdown("### MFE / MAE")
            st.caption(
                "MFE = 訊號後最大有利變動；MAE = 訊號後最大不利變動。"
                "專業版本用它檢查停損是否過緊、以及成功案例通常要承受多少逆向波動。"
            )

            mature = pro_bt[pro_bt.get("40日成熟", False) == True].copy()
            cols = [c for c in [
                "快照日期","股票代號","股票名稱","主模型排名","主模型分數","進場判定",
                "40日MFE","40日MAE","40日後報酬率","40日達20%","40日達30%","40日達50%"
            ] if c in mature.columns]
            if cols:
                table(
                    mature.sort_values(["快照日期","主模型排名"], ascending=[False, True])[cols].head(200),
                    height=430,
                )

            regimes = regime_validation(pro_bt)
            st.markdown("### 市場 Regime 穩定度")
            if regimes.empty:
                st.caption("尚無足夠成熟樣本做台股 / 全球環境分組。")
            else:
                show_reg = regimes.copy()
                for c in ["40日+50%命中率","40日MFE中位數","40日MAE中位數"]:
                    if c in show_reg.columns:
                        show_reg[c] = pd.to_numeric(show_reg[c], errors="coerce").round(2)
                table(show_reg, height=360)

        st.markdown("### 專業審查原則")
        st.info(
            "目前的 50% 歷史型態命中率是 historical analog 指標，不等於機率保證。"
            "真正對外報告應以 forward OOS 的 Top5 / Top10 / Top20 命中率、Lift、MFE/MAE、"
            "不同市場 regime 的穩定度與樣本數作主要證據。"
        )


        st.markdown("### Historical Walk-forward")
        st.caption(
            "這是另一條獨立驗證線：用較長歷史資料做時間序列 walk-forward，"
            "每一個測試日只允許使用當時以前、且 40 日標籤已完全成熟的資料訓練。"
        )

        if walkforward_summary.empty:
            st.info(
                "尚未執行歷史 Walk-forward。到 GitHub Actions 手動執行 "
                "『Historical Walk-Forward Validation』後，結果會出現在這裡。"
            )
        else:
            wf_show = walkforward_summary.copy()
            for c in [
                "40日+20%命中率","40日+30%命中率","40日+50%命中率",
                "40日MFE中位數","40日MAE中位數","40日報酬中位數",
                "40日正報酬率","+50% Lift"
            ]:
                if c in wf_show.columns:
                    wf_show[c] = pd.to_numeric(wf_show[c], errors="coerce").round(2)
            table(wf_show, height=300)

            if not walkforward_metadata.empty:
                md = walkforward_metadata.iloc[0]
                st.caption(
                    f"版本：{md.get('模型版本','—')} ｜ "
                    f"開始日期：{md.get('開始日期','—')} ｜ "
                    f"測試頻率：{md.get('測試頻率','—')} ｜ "
                    f"成熟測試樣本：{md.get('成熟測試樣本','—')}"
                )
                st.warning(
                    "目前 Walk-forward 使用『現存股票池』回測，因此仍有 survivorship bias；"
                    "另外這一版只驗證核心價格型態模型，尚未完整重建歷史法人與基本面因子。"
                    "這些限制必須在對外專業報告中揭露。"
                )


        st.markdown("### Exit Model / 完整交易模擬")
        st.caption(
            "這一層把 Walk-forward 訊號轉成真正的交易：訊號後下一交易日開盤進場，"
            "比較不同停損 / 停利 / 移動停利規則，並扣除設定的往返交易摩擦。"
        )

        if walkforward_exit_summary.empty:
            st.info(
                "目前尚未產生 Exit Model 結果。請重新執行一次 "
                "『Historical Walk-Forward Validation』，新版流程會一起產生出場策略比較。"
            )
        else:
            exit_show = walkforward_exit_summary.copy()
            for c in [
                "平均淨報酬","中位數淨報酬","勝率","Profit Factor",
                "中位持有天數","最大單筆虧損","10分位淨報酬",
                "+20%實現率","+30%實現率","+50%實現率","MFE捕捉率中位數"
            ]:
                if c in exit_show.columns:
                    exit_show[c] = pd.to_numeric(exit_show[c], errors="coerce").round(2)

            top10_exit = exit_show[exit_show["群組"].eq("Top 10")].copy()
            if not top10_exit.empty:
                st.markdown("#### Top 10：不同出場規則")
                sort_pf = pd.to_numeric(top10_exit["Profit Factor"], errors="coerce")
                top10_exit = top10_exit.assign(_pf=sort_pf).sort_values(
                    ["_pf","中位數淨報酬"], ascending=[False,False]
                ).drop(columns="_pf")
                table(top10_exit, height=360)

                valid_pf = pd.to_numeric(top10_exit["Profit Factor"], errors="coerce")
                if valid_pf.notna().any():
                    best = top10_exit.iloc[0]
                    b1,b2,b3,b4 = st.columns(4)
                    b1.metric("目前較佳策略", str(best.get("策略","—")))
                    b2.metric("Profit Factor", fmt(best.get("Profit Factor"),2))
                    b3.metric("中位數淨報酬", fmt(best.get("中位數淨報酬"),2,"%"))
                    b4.metric("勝率", fmt(best.get("勝率"),1,"%"))

            st.markdown("#### 全部 Top-K / Exit Strategy")
            table(exit_show, height=520)

            if not walkforward_metadata.empty:
                md = walkforward_metadata.iloc[0]
                st.caption(
                    f"進場假設：{md.get('出場回測進場','—')} ｜ "
                    f"交易摩擦：{md.get('出場回測交易摩擦假設','—')} ｜ "
                    f"同日同時觸發：{md.get('同日停損與停利皆觸發','—')}"
                )

            st.warning(
                "這裡的『目前較佳策略』只代表這批 walk-forward 歷史樣本中的相對結果，"
                "不能直接當成未來保證。專業版本後續還要做不同年代 / regime、參數敏感度與成本敏感度測試。"
            )


        st.markdown("### Exit Optimization 2.0")
        st.caption(
            "這一層不再只比較少數人工規則，而是系統化測試停損、持有天數、"
            "啟動移動停利門檻與 trailing distance。最重要的是："
            "策略排名只看 2025 年以前的開發期，2025 年以後只當 OOS 驗證，不參與挑選。"
        )

        if walkforward_exit_optimizer_shortlist.empty:
            st.info(
                "尚未產生 Exit Optimization 2.0 結果。請重新執行一次 "
                "Historical Walk-Forward Validation。"
            )
        else:
            opt = walkforward_exit_optimizer_shortlist.copy()

            display_cols = [c for c in [
                "開發期排名","strategy_id","期間","交易數","平均淨報酬","中位數淨報酬",
                "勝率","Profit Factor","10分位淨報酬","最大單筆虧損",
                "毛+30%實現率","毛+50%實現率","淨+30%實現率","淨+50%實現率",
                "中位持有天數","開發期綜合分數"
            ] if c in opt.columns]

            show_opt = opt[display_cols].copy()
            for c in [
                "平均淨報酬","中位數淨報酬","勝率","Profit Factor",
                "10分位淨報酬","最大單筆虧損","毛+30%實現率","毛+50%實現率",
                "淨+30%實現率","淨+50%實現率","中位持有天數","開發期綜合分數"
            ]:
                if c in show_opt.columns:
                    show_opt[c] = pd.to_numeric(show_opt[c], errors="coerce").round(2)

            table(show_opt.sort_values(["開發期排名","期間"]), height=520)

            # The first-ranked development strategy is frozen before reading OOS.
            dev_top = opt[
                (pd.to_numeric(opt.get("開發期排名"), errors="coerce") == 1)
            ].copy()

            if not dev_top.empty:
                dev_row = dev_top[dev_top["期間"].eq("開發期")]
                oos_row = dev_top[dev_top["期間"].eq("OOS期")]

                st.markdown("#### 開發期第 1 名策略：OOS 驗證")
                if not dev_row.empty:
                    d = dev_row.iloc[0]
                    strategy_name = str(d.get("strategy_id","—"))
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("策略", strategy_name)
                    c2.metric("開發期 PF", fmt(d.get("Profit Factor"),2))
                    c3.metric("開發期平均淨報酬", fmt(d.get("平均淨報酬"),2,"%"))
                    c4.metric("開發期10%尾端", fmt(d.get("10分位淨報酬"),2,"%"))

                if not oos_row.empty:
                    o = oos_row.iloc[0]
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("OOS Profit Factor", fmt(o.get("Profit Factor"),2))
                    c2.metric("OOS 平均淨報酬", fmt(o.get("平均淨報酬"),2,"%"))
                    c3.metric("OOS 中位數", fmt(o.get("中位數淨報酬"),2,"%"))
                    c4.metric("OOS 勝率", fmt(o.get("勝率"),1,"%"))

                    dev_pf = pd.to_numeric(pd.Series([d.get("Profit Factor")]), errors="coerce").iloc[0] if not dev_row.empty else np.nan
                    oos_pf = pd.to_numeric(pd.Series([o.get("Profit Factor")]), errors="coerce").iloc[0]
                    if pd.notna(oos_pf) and oos_pf > 1:
                        st.success(
                            "開發期選出的策略在 OOS 仍維持 Profit Factor > 1。"
                            "這比直接用全樣本挑最佳參數更有可信度。"
                        )
                    else:
                        st.warning(
                            "開發期第 1 名策略在 OOS 未維持有效 edge。"
                            "這種情況不能把開發期最佳參數直接接到實盤。"
                        )

            st.markdown("#### 年度穩定度")
            if walkforward_exit_optimizer_yearly.empty:
                st.caption("尚無年度穩定度資料。")
            else:
                yr = walkforward_exit_optimizer_yearly.copy()
                yr = yr[pd.to_numeric(yr.get("開發期排名"), errors="coerce") <= 5].copy()
                yr_cols = [c for c in [
                    "開發期排名","strategy_id","年份","交易數","平均淨報酬",
                    "中位數淨報酬","勝率","Profit Factor","10分位淨報酬",
                    "毛+30%實現率","毛+50%實現率"
                ] if c in yr.columns]
                for c in [
                    "平均淨報酬","中位數淨報酬","勝率","Profit Factor",
                    "10分位淨報酬","毛+30%實現率","毛+50%實現率"
                ]:
                    if c in yr.columns:
                        yr[c] = pd.to_numeric(yr[c], errors="coerce").round(2)
                table(
                    yr.sort_values(["開發期排名","年份"])[yr_cols],
                    height=500
                )

            st.warning(
                "Exit Optimizer 仍屬研究工具。現階段的目標是找『參數區域是否穩健』，"
                "不是只挑某一組數字最高。下一步還要做參數敏感度熱圖與不同交易成本情境。"
            )


        st.markdown("### Robustness / 參數穩健性")
        st.caption(
            "專業研究不是找單一最佳參數，而是確認附近參數、不同成本與不同年份是否仍保有 edge。"
            "這裡只做描述性穩健性檢查，不會用 OOS 結果重新挑策略。"
        )

        if walkforward_robustness_scorecard.empty:
            st.info(
                "尚未產生 Robustness 結果。請重新執行一次 Historical Walk-Forward Validation。"
            )
        else:
            scorecard = walkforward_robustness_scorecard.copy()
            sc_cols = [c for c in [
                "開發期排名","strategy_id","開發期PF","OOS PF","PF OOS/開發",
                "開發期平均淨報酬","OOS平均淨報酬","OOS報酬差",
                "PF>1年份數","年度樣本數","年度穩定率","OOS仍有效"
            ] if c in scorecard.columns]
            for c in [
                "開發期PF","OOS PF","PF OOS/開發","開發期平均淨報酬",
                "OOS平均淨報酬","OOS報酬差","年度穩定率"
            ]:
                if c in scorecard.columns:
                    scorecard[c] = pd.to_numeric(scorecard[c], errors="coerce").round(2)
            table(
                scorecard.sort_values("開發期排名")[sc_cols].head(20),
                height=430,
            )

            top1 = scorecard[pd.to_numeric(scorecard.get("開發期排名"), errors="coerce") == 1]
            if not top1.empty:
                r = top1.iloc[0]
                q1,q2,q3,q4 = st.columns(4)
                q1.metric("基準策略", str(r.get("strategy_id","—")))
                q2.metric("OOS PF", fmt(r.get("OOS PF"),2))
                q3.metric("年度穩定率", fmt(r.get("年度穩定率"),1,"%"))
                q4.metric("OOS仍有效", "是" if bool(r.get("OOS仍有效")) else "否")

        st.markdown("#### 停損 × 最長持有日")
        if walkforward_robustness_stop_hold.empty:
            st.caption("尚無停損 / 持有期敏感度資料。")
        else:
            sh = walkforward_robustness_stop_hold.copy()
            period = st.radio(
                "查看期間",
                ["開發期","OOS期"],
                horizontal=True,
                key="robust_stop_period",
            )
            sh = sh[sh["期間"].eq(period)].copy()

            metric = st.selectbox(
                "熱圖指標",
                ["Profit Factor","平均淨報酬","10分位淨報酬","毛+50%實現率"],
                key="robust_stop_metric",
            )
            if not sh.empty and metric in sh.columns:
                pivot = sh.pivot_table(
                    index="停損",
                    columns="最長持有日",
                    values=metric,
                    aggfunc="mean",
                )
                fig = go.Figure(
                    data=go.Heatmap(
                        z=pivot.values,
                        x=[str(x) for x in pivot.columns],
                        y=[str(y) for y in pivot.index],
                        text=np.round(pivot.values,2),
                        texttemplate="%{text}",
                        colorbar=dict(title=metric),
                    )
                )
                fig.update_layout(
                    height=360,
                    xaxis_title="最長持有交易日",
                    yaxis_title="停損",
                    margin=dict(l=20,r=20,t=25,b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Trailing 參數敏感度")
        if walkforward_robustness_trailing.empty:
            st.caption("尚無 trailing 敏感度資料。")
        else:
            tr = walkforward_robustness_trailing.copy()
            tr_period = st.radio(
                "Trailing 查看期間",
                ["開發期","OOS期"],
                horizontal=True,
                key="robust_trail_period",
            )
            tr_hold = st.selectbox(
                "最長持有日",
                sorted(pd.to_numeric(tr["最長持有日"], errors="coerce").dropna().astype(int).unique()),
                key="robust_trail_hold",
            )
            tr_metric = st.selectbox(
                "Trailing 熱圖指標",
                ["Profit Factor","平均淨報酬","10分位淨報酬","毛+50%實現率"],
                key="robust_trail_metric",
            )
            tr = tr[
                tr["期間"].eq(tr_period)
                & (pd.to_numeric(tr["最長持有日"], errors="coerce") == int(tr_hold))
            ].copy()
            if not tr.empty and tr_metric in tr.columns:
                pivot = tr.pivot_table(
                    index="啟動門檻",
                    columns="Trailing幅度",
                    values=tr_metric,
                    aggfunc="mean",
                )
                fig = go.Figure(
                    data=go.Heatmap(
                        z=pivot.values,
                        x=[f"{x:.0f}%" for x in pivot.columns],
                        y=[f"{y:.0f}%" for y in pivot.index],
                        text=np.round(pivot.values,2),
                        texttemplate="%{text}",
                        colorbar=dict(title=tr_metric),
                    )
                )
                fig.update_layout(
                    height=350,
                    xaxis_title="Trailing 幅度",
                    yaxis_title="啟動門檻",
                    margin=dict(l=20,r=20,t=25,b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 交易成本敏感度")
        if walkforward_robustness_cost.empty:
            st.caption("尚無成本敏感度資料。")
        else:
            cost = walkforward_robustness_cost.copy()
            cost = cost[
                pd.to_numeric(cost.get("開發期排名"), errors="coerce") <= 5
            ].copy()
            cost_cols = [c for c in [
                "開發期排名","strategy_id","期間","往返成本%",
                "平均淨報酬","中位數淨報酬","勝率","Profit Factor","10分位淨報酬"
            ] if c in cost.columns]
            for c in [
                "平均淨報酬","中位數淨報酬","勝率","Profit Factor","10分位淨報酬"
            ]:
                if c in cost.columns:
                    cost[c] = pd.to_numeric(cost[c], errors="coerce").round(2)
            table(
                cost.sort_values(["開發期排名","strategy_id","期間","往返成本%"])[cost_cols],
                height=480,
            )

        if not walkforward_metadata.empty:
            md = walkforward_metadata.iloc[0]
            st.caption(
                f"Robustness：{md.get('Robustness版本','—')} ｜ "
                f"成本情境：{md.get('成本敏感度情境','—')} ｜ "
                f"檢查內容：{md.get('參數穩健性','—')}"
            )

        st.warning(
            "看到一個參數點特別高，不代表它可靠。真正值得保留的是："
            "附近參數也有效、成本提高後仍有效、OOS 不崩壞、且多個年份 PF 仍大於 1。"
        )


        st.markdown("### Selection Model 2.0 / 找出真正會噴的股票")
        st.caption(
            "研究目標：對完整流動性股票池逐檔預測未來 40 個交易日是否可能出現 +50%，"
            "同時兼顧 Precision、Recall、爆發前啟動與第二段再加速。S2 設定只用 2025 年前開發期挑選，"
            "2025 年起 OOS 只做驗證。"
        )

        s2_cmp = walkforward_selection_v2.get("s2_comparison", pd.DataFrame())
        s2_sel = walkforward_selection_v2.get("s2_selected", pd.DataFrame())
        s2_cand = walkforward_selection_v2.get("s2_candidates", pd.DataFrame())
        s2_fn = walkforward_selection_v2.get("s2_fn", pd.DataFrame())
        s2_runup = walkforward_selection_v2.get("s2_runup", pd.DataFrame())
        s2_ign = walkforward_selection_v2.get("s2_ignition", pd.DataFrame())
        s2_diag = walkforward_selection_v2.get("s2_diag", pd.DataFrame())

        if s2_cmp.empty:
            st.info("Selection Model 2.0 候選引擎尚未完成回測。執行一次 Historical Walk-Forward Validation 即可一次產生全部結果。")
        else:
            selected_name = (
                str(s2_sel.iloc[0].get("selected_config","—"))
                if not s2_sel.empty else "—"
            )
            st.markdown(f"#### 開發期選定候選設定：{selected_name}")

            oos = s2_cmp[
                (s2_cmp["期間"].astype(str) == "OOS期")
                & (pd.to_numeric(s2_cmp["K"], errors="coerce") == 10)
            ].copy()
            if not oos.empty:
                s1 = oos[oos["模型"].astype(str) == "S1原模型"]
                s2m = oos[oos["模型"].astype(str) == "S2候選模型"]
                a,b,c,d = st.columns(4)
                if not s1.empty and not s2m.empty:
                    a.metric("S1 OOS Top10 Precision", fmt(s1.iloc[0].get("Precision"),2,"%"))
                    b.metric("S2 OOS Top10 Precision", fmt(s2m.iloc[0].get("Precision"),2,"%"))
                    c.metric("S1 OOS Top10 Recall", fmt(s1.iloc[0].get("Recall"),2,"%"))
                    d.metric("S2 OOS Top10 Recall", fmt(s2m.iloc[0].get("Recall"),2,"%"))

            st.markdown("#### S1 vs S2：Precision / Recall / Lift")
            show_cmp = s2_cmp.copy()
            for c in ["Precision","Recall","Lift"]:
                if c in show_cmp.columns:
                    show_cmp[c] = pd.to_numeric(show_cmp[c], errors="coerce").round(3)
            table(show_cmp, height=390)

            if not s2_cand.empty:
                st.markdown("#### 候選模型設定（只看開發期排名）")
                cand = s2_cand.copy()
                for c in cand.columns:
                    if c not in ["config"] and c in cand.columns:
                        try:
                            cand[c] = pd.to_numeric(cand[c], errors="ignore")
                        except Exception:
                            pass
                table(cand, height=300)

            if not s2_fn.empty:
                st.markdown("#### +50% 成功股：目前抓到與漏掉的型態")
                table(s2_fn, height=260)

            if not s2_runup.empty:
                st.markdown("#### 第一段 / 第二段：前 60 日已漲幅")
                table(s2_runup, height=260)

            if not s2_ign.empty:
                st.markdown("#### Ignition Timing：距離 +50% 還有多久")
                table(s2_ign, height=230)

            if not s2_diag.empty:
                st.markdown("#### 核心特徵區分力")
                table(s2_diag, height=330)

            s21_cmp = walkforward_selection_v2.get("s21_comparison", pd.DataFrame())
            s21_sel = walkforward_selection_v2.get("s21_selected", pd.DataFrame())
            s21_cand = walkforward_selection_v2.get("s21_candidates", pd.DataFrame())
            s21_attr = walkforward_selection_v2.get("s21_attribution", pd.DataFrame())

            st.markdown("### Selection Model 2.1 / 分支獨立模型")
            st.caption(
                "S2.1 不再把 General、Ignition、Second-Leg 用固定權重平均。"
                "三個分支各自獨立排名，再依開發期選出的輪替規則合併；2025 起 OOS 只驗證，不參與挑選。"
            )

            if s21_cmp.empty:
                st.info("S2.1 尚未完成回測。重新執行一次 Historical Walk-Forward Validation 即可產生。")
            else:
                s21_name = str(s21_sel.iloc[0].get("selected_config","—")) if not s21_sel.empty else "—"
                st.markdown(f"#### S2.1 開發期選定設定：{s21_name}")

                oos21 = s21_cmp[
                    (s21_cmp["期間"].astype(str) == "OOS期")
                    & (pd.to_numeric(s21_cmp["K"], errors="coerce") == 10)
                ].copy()
                if not oos21.empty:
                    q1,q2,q3 = st.columns(3)
                    for label, col in [("S1原模型","q1"),("S2加權候選","q2"),("S2.1分支模型","q3")]:
                        row = oos21[oos21["模型"].astype(str) == label]
                        if row.empty:
                            continue
                        val = row.iloc[0]
                        target = {"q1":q1,"q2":q2,"q3":q3}[col]
                        target.metric(
                            f"{label} OOS Top10",
                            fmt(val.get("Precision"),2,"%"),
                            f"Recall {fmt(val.get('Recall'),2,'%')}"
                        )

                st.markdown("#### S1 vs S2 vs S2.1")
                show21 = s21_cmp.copy()
                for c in ["Precision","Recall","Lift"]:
                    if c in show21.columns:
                        show21[c] = pd.to_numeric(show21[c], errors="coerce").round(3)
                table(show21, height=430)

                if not s21_cand.empty:
                    st.markdown("#### 分支輪替候選（只用開發期排序）")
                    table(s21_cand, height=300)

                if not s21_attr.empty:
                    st.markdown("#### S2.1 Top20 主要來源分支")
                    table(s21_attr, height=260)

            st.warning(
                "S2 / S2.1 現階段仍是候選研究模型，不會因為單次 OOS 表現較高就自動取代正式排行榜。"
                "只有在 Precision、Recall、Lift、不同年份與市場環境都保持穩定後，才會升級成正式排名引擎。"
            )

            ign2_cmp = walkforward_selection_v2.get("ign2_comparison", pd.DataFrame())
            ign2_sel = walkforward_selection_v2.get("ign2_selected", pd.DataFrame())
            ign2_cand = walkforward_selection_v2.get("ign2_candidates", pd.DataFrame())
            ign2_rec = walkforward_selection_v2.get("ign2_recovery", pd.DataFrame())
            ign2_feat = walkforward_selection_v2.get("ign2_features", pd.DataFrame())

            st.markdown("### Ignition Model 2.0 / 爆發前加速度")
            st.caption(
                "專門研究原本容易漏掉、尚未明顯上漲的 +50% 股票。"
                "新增波動加速度、量能加速度、區間壓縮、布林帶寬度變化、均線斜率、RSI/MACD 加速度與前高距離；"
                "訓練目標偏向 +50% 於 30 日內或 +30% 於 20 日內的快速爆發。"
            )

            if ign2_cmp.empty:
                st.info("Ignition 2.0 尚未完成回測。重新執行一次 Historical Walk-Forward Validation 即可產生。")
            else:
                ign_name = str(ign2_sel.iloc[0].get("selected_config","—")) if not ign2_sel.empty else "—"
                st.markdown(f"#### 開發期選定 Ignition 融合設定：{ign_name}")

                oos_ign = ign2_cmp[
                    (ign2_cmp["期間"].astype(str) == "OOS期")
                    & (pd.to_numeric(ign2_cmp["K"], errors="coerce") == 10)
                ].copy()
                if not oos_ign.empty:
                    cols = st.columns(3)
                    for i,(label,key) in enumerate([
                        ("S1原模型","S1原模型"),
                        ("S2 leading","S2 leading"),
                        ("S2+Ignition2","S2+Ignition2"),
                    ]):
                        row = oos_ign[oos_ign["模型"].astype(str) == key]
                        if not row.empty:
                            v=row.iloc[0]
                            cols[i].metric(
                                f"{label} OOS Top10",
                                fmt(v.get("Precision"),2,"%"),
                                f"Recall {fmt(v.get('Recall'),2,'%')}"
                            )

                st.markdown("#### S1 vs S2 vs Ignition 2.0")
                z=ign2_cmp.copy()
                for c in ["Precision","Recall","Lift"]:
                    if c in z.columns:
                        z[c]=pd.to_numeric(z[c],errors="coerce").round(3)
                table(z,height=430)

                if not ign2_rec.empty:
                    st.markdown("#### False Negative 專項：原本 Top50 外的 +50% 股票救回多少")
                    table(ign2_rec,height=280)

                if not ign2_cand.empty:
                    st.markdown("#### Ignition 融合候選（只用開發期挑選）")
                    table(ign2_cand,height=260)

                if not ign2_feat.empty:
                    st.markdown("#### 爆發前特徵診斷")
                    table(ign2_feat,height=360)

                st.warning(
                    "Ignition 2.0 目前仍是研究層。重點不是只看 Top10 是否變漂亮，"
                    "而是它能不能在 OOS 真正救回原本 Top50 外、之後卻 +50% 的股票，同時不明顯破壞 Precision。"
                )


        st.markdown("### Regime Filter / 市場環境驗證")
        st.caption(
            "這一層檢查 Top10 在不同市場環境下是否仍有 edge。"
            "Regime 使用當時可得的 TWII / Nasdaq100 / SOX / VIX / USD/TWD 與市場 breadth，"
            "不使用未來資料。"
        )

        if walkforward_regime_gate.empty:
            st.info(
                "尚未產生 Regime 結果。請重新執行一次 Historical Walk-Forward Validation。"
            )
        else:
            gate = walkforward_regime_gate.copy()
            for c in [
                "40日+50%命中率","40日MFE中位數","40日MAE中位數",
                "Regime分數中位數","平均淨報酬","Profit Factor","勝率","10分位淨報酬"
            ]:
                if c in gate.columns:
                    gate[c] = pd.to_numeric(gate[c], errors="coerce").round(2)

            st.markdown("#### Top10 × 市場環境")
            table(gate, height=330)

            if "研究判定" in gate.columns:
                counts = gate["研究判定"].value_counts()
                g1,g2,g3 = st.columns(3)
                g1.metric("歷史順風 Regime", int(counts.get("歷史順風",0)))
                g2.metric("中性觀察 Regime", int(counts.get("中性觀察",0)))
                g3.metric("歷史逆風 Regime", int(counts.get("歷史逆風",0)))

            st.markdown("#### Top5 / Top10 / Top20 Regime 命中率")
            if not walkforward_regime_signal.empty:
                rs = walkforward_regime_signal.copy()
                for c in [
                    "40日+20%命中率","40日+30%命中率","40日+50%命中率",
                    "40日MFE中位數","40日MAE中位數","40日報酬中位數",
                    "國內分數中位數","全球分數中位數","Regime分數中位數"
                ]:
                    if c in rs.columns:
                        rs[c] = pd.to_numeric(rs[c], errors="coerce").round(2)
                table(rs, height=430)

            st.markdown("#### 基準 Exit Strategy × Regime")
            if not walkforward_regime_exit.empty:
                re = walkforward_regime_exit.copy()
                for c in [
                    "平均淨報酬","中位數淨報酬","勝率",
                    "Profit Factor","10分位淨報酬","毛+30%實現率","毛+50%實現率"
                ]:
                    if c in re.columns:
                        re[c] = pd.to_numeric(re[c], errors="coerce").round(2)
                table(re, height=300)

            st.warning(
                "Regime 結果目前只用來判斷『何時應提高或降低信心』，"
                "不會直接用 OOS 結果重新訓練或改排行榜。"
                "等確認不同市場環境下的差異穩定，再考慮把 Regime Gate 正式接回主模型。"
            )

elif page == "一週模型":
    st.markdown("## 1～2 個月波段爆發＋進場時機模型")
    st.caption("這是目前的主模型：目標不是短線小幅上漲，而是篩選 1～2 個月內具大波段、甚至挑戰 +50% 潛力的個股；短期啟動訊號只負責判斷何時進場。")

    if "一週模型排名" not in rank.columns or rank["一週模型排名"].isna().all():
        st.info("尚未產生一週模型資料。請手動跑一次 GitHub Actions。")
    else:
        a,b,c,d=st.columns(4)
        a.metric("V1-B 技術啟動權重","40%")
        b.metric("V2-B 籌碼動能權重","30%")
        c.metric("V3-B 基本品質權重","15%")
        d.metric("V4-B 價格動能權重","15%")

        f1,f2,f3=st.columns(3)
        stages=sorted(rank["啟動階段"].dropna().astype(str).unique()) if "啟動階段" in rank.columns else []
        entries=sorted(rank["進場判定"].dropna().astype(str).unique()) if "進場判定" in rank.columns else []
        selected_stages=f1.multiselect("啟動階段",stages)
        selected_entries=f2.multiselect("進場判定",entries)
        min_week=f3.slider("最低一週起漲分數",0,100,0)

        d=rank.copy()
        if selected_stages:
            d=d[d["啟動階段"].isin(selected_stages)]
        if selected_entries:
            d=d[d["進場判定"].isin(selected_entries)]
        if "一週起漲分數" in d.columns:
            d=d[pd.to_numeric(d["一週起漲分數"],errors="coerce").fillna(-1)>=min_week]
        d=d.sort_values("一週模型排名")

        cols=[c for c in [
            "主模型排名","股票代號","股票名稱","產業別",
            "主模型分數","波段爆發分數","50%潛力判定","50%歷史型態命中率","50%歷史型態PR",
            "相似樣本40日最高報酬均值","突破強度分數","動能持續分數","活躍爆發分數",
            "一週起漲分數","進場時機分數","啟動階段","進場判定",
            "技術啟動分數","籌碼動能分數","基本品質分數","價格動能分數",
            "風險扣分","台股環境分數","全球環境分數","產業海外順風分數",
            "起漲原因","進場風險","最終分數"
        ] if c in d.columns]
        table(d[cols],height=620)

elif page == "全球市場":
    st.markdown("## 市場環境 Gate")
    st.caption("這一頁不是替個股加基本面分，而是判斷未來一週的市場順風程度。台股 Gate、全球 Gate、產業海外龍頭 Gate 都會影響『進場時機分數』。")

    g1,g2=st.columns(2)
    with g1:
        st.markdown("### 台股市場")
        if domestic_market_summary.empty:
            st.info("尚未產生台股市場 Gate。請手動跑一次 GitHub Actions。")
        else:
            ds=domestic_market_summary.iloc[0]
            dscore=ds.get("台股環境分數",np.nan)
            dreg=str(ds.get("台股環境判定","資料不足"))
            c1,c2=st.columns(2)
            c1.metric("台股環境分數",fmt(dscore,1))
            c2.metric("判定",dreg)
            st.write({
                "加權指數技術分": fmt(ds.get("加權指數技術分"),1),
                "市場廣度分數": fmt(ds.get("市場廣度分數"),1),
                "站上MA20比例": fmt(ds.get("站上MA20比例"),1,"%"),
                "5日上漲比例": fmt(ds.get("5日上漲比例"),1,"%"),
                "20日上漲比例": fmt(ds.get("20日上漲比例"),1,"%"),
            })

    with g2:
        st.markdown("### 全球市場")
        if global_market_summary.empty:
            st.info("尚未產生全球市場 Gate。請手動跑一次 GitHub Actions。")
        else:
            gs=global_market_summary.iloc[0]
            gscore=gs.get("全球環境分數",np.nan)
            greg=str(gs.get("全球環境判定","資料不足"))
            c1,c2=st.columns(2)
            c1.metric("全球環境分數",fmt(gscore,1))
            c2.metric("判定",greg)
            st.caption(f"資料日期：{gs.get('資料日期','—')}")

    st.markdown("### 海外大盤與龍頭")
    if global_market_detail.empty:
        st.info("尚未產生海外市場明細。")
    else:
        display_cols=[c for c in [
            "項目","最新值","1日變動率","5日變動率",
            "MA20","MA60","高於MA20","高於MA60"
        ] if c in global_market_detail.columns]
        show=global_market_detail[display_cols].copy()
        for c in ["最新值","1日變動率","5日變動率","MA20","MA60"]:
            if c in show.columns:
                show[c]=pd.to_numeric(show[c],errors="coerce").round(2)
        table(show,height=560)

    st.markdown("### 目前追蹤邏輯")
    st.write("大盤：S&P 500、Nasdaq 100、SOX、台灣加權；風險：VIX、美國10年債、USD/TWD；龍頭：TSM ADR、NVIDIA、AMD、Broadcom、Micron、Apple、Microsoft、Amazon。")
    st.caption("產業海外順風分數會依台股產業映射不同海外龍頭，例如半導體會看 SOX、TSM、NVDA、AMD、AVGO、MU；蘋果鏈與電子零組件則會提高 AAPL 的參考權重。")

elif page == "每日變化":
    st.markdown("## 每日變化")

    st.markdown("### 一週模型變化")
    if weekly_daily_change.empty:
        st.info("一週模型目前還沒有兩個交易日的歷史。等下一個交易日自動累積後，這裡就會開始比較。")
    else:
        new_top=weekly_daily_change[weekly_daily_change.get("新進一週Top10",False)==True].copy()
        new_entry=weekly_daily_change[weekly_daily_change.get("新進可觀察進場",False)==True].copy()
        risers=weekly_daily_change.copy()
        if "一週排名變化" in risers.columns:
            risers=risers[pd.to_numeric(risers["一週排名變化"],errors="coerce")>0]
            risers=risers.sort_values("一週排名變化",ascending=False).head(15)

        c1,c2,c3=st.columns(3)
        c1.metric("新進一週 Top10",f"{len(new_top)} 檔")
        c2.metric("新進可觀察進場",f"{len(new_entry)} 檔")
        c3.metric("排名上升",f"{len(risers)} 檔")

        st.markdown("#### 一週排名上升最多")
        cols=[c for c in [
            "股票代號","股票名稱","一週模型排名","昨日一週排名","一週排名變化",
            "一週起漲分數","進場時機分數","啟動階段","進場判定"
        ] if c in risers.columns]
        if cols and not risers.empty:
            table(risers[cols],height=420)
        else:
            st.caption("目前沒有排名上升資料。")

        c1,c2=st.columns(2)
        with c1:
            st.markdown("#### 🔥 新進一週 Top10")
            if new_top.empty:
                st.caption("今天沒有。")
            else:
                cols=[c for c in [
                    "股票代號","股票名稱","一週模型排名","昨日一週排名",
                    "一週起漲分數","進場時機分數","進場判定"
                ] if c in new_top.columns]
                table(new_top[cols],height=300)
        with c2:
            st.markdown("#### 🟢 新進可觀察進場")
            if new_entry.empty:
                st.caption("今天沒有。")
            else:
                cols=[c for c in [
                    "股票代號","股票名稱","一週模型排名",
                    "一週起漲分數","進場時機分數","啟動階段"
                ] if c in new_entry.columns]
                table(new_entry[cols],height=300)

    with st.expander("查看原始 V5-A 的昨日 vs 今日"):
        if daily_change.empty:
            st.caption("原始模型目前也沒有兩天可比較的資料。")
        else:
            risers=daily_change.copy()
            if "排名變化" in risers.columns:
                risers=risers[pd.to_numeric(risers["排名變化"],errors="coerce")>0].sort_values("排名變化",ascending=False).head(15)
            cols=[c for c in [
                "股票代號","股票名稱","排名","昨日排名","排名變化",
                "最終分數","昨日最終分數","分數變化","目前狀態"
            ] if c in risers.columns]
            if cols:
                table(risers[cols],height=380)

elif page == "V6 回測":
    st.markdown("## V6 回測與模型驗證")
    st.caption("V6 不新增選股分數，而是驗證 V1～V5 高分股票之後的實際表現。資料會從啟用 ranking_history.csv 後逐日累積。")

    if ranking_history_all.empty:
        st.info("目前尚未有 ranking_history.csv。今晚 GitHub Actions 下一次成功執行後，系統會自動開始累積每日排名快照。")
    else:
        dates = ranking_history_all["快照日期"].dropna().dt.date.nunique()
        st.metric("已累積交易日", f"{dates} 日")

        backtest_prices = price_archive_all if not price_archive_all.empty else history_all
        bt = attach_forward_returns(ranking_history_all, backtest_prices)

        if bt.empty:
            st.info("已有排名歷史，但尚未累積足夠後續價格來計算報酬。")
        else:
            summary = backtest_summary(bt)

            if summary.empty:
                st.info("回測資料仍在累積。至少經過下一個交易日後才會開始出現 1 日報酬；5 日、20 日指標會依序解鎖。")
            else:
                st.markdown("### 模型績效摘要")
                show = summary.copy()
                for c in ["平均報酬率","中位數報酬率","勝率"]:
                    if c in show.columns:
                        show[c] = show[c].round(2)
                table(show, height=340)

                st.markdown("### 原始 V5-A vs 一週模型 V5-B")
                chart_df = summary[summary["群組"].isin(["原始V5 Top 10","一週模型 Top 10","全部股票池"])].copy()
                if not chart_df.empty:
                    f=go.Figure()
                    for grp in ["原始V5 Top 10","一週模型 Top 10","全部股票池"]:
                        x=chart_df[chart_df["群組"].eq(grp)]
                        if not x.empty:
                            f.add_trace(go.Bar(name=grp,x=x["期間"],y=x["平均報酬率"]))
                    f.update_layout(
                        barmode="group",
                        height=380,
                        title="兩套 Top10 與全市場：不同持有期間平均報酬",
                        yaxis_title="平均報酬率 (%)",
                        margin=dict(l=15,r=15,t=45,b=20),
                    )
                    st.plotly_chart(f,use_container_width=True)

                st.markdown("### 最近已成熟的一週模型 Top 10 樣本")
                if "一週模型排名" in bt.columns:
                    top_bt = bt[pd.to_numeric(bt.get("一週模型排名"),errors="coerce")<=10].copy()
                else:
                    top_bt = pd.DataFrame()
                cols=[c for c in [
                    "快照日期","股票代號","股票名稱","一週模型排名","一週起漲分數","進場時機分數",
                    "啟動階段","進場判定","1日後報酬率","5日後報酬率","20日後報酬率","40日後報酬率",
                    "20日內最高報酬率","40日內最高報酬率","40日內達50%"
                ] if c in top_bt.columns]
                if cols and not top_bt.empty:
                    table(top_bt.sort_values(["快照日期","一週模型排名"],ascending=[False,True])[cols].head(100),height=460)

                st.markdown("### V6.1 分數區間績效分析")
                st.caption("用實際後續報酬檢查：高分區間是否真的比低分區間表現更好。樣本數太少時先不要解讀。")

                v61a,v61b = st.columns(2)
                horizon = v61a.selectbox(
                    "驗證持有期間",
                    [1,5,20],
                    format_func=lambda x: f"{x} 交易日",
                    key="v61_horizon",
                )
                component_map = {
                    "主模型分數":"主模型分數",
                    "波段爆發分數":"波段爆發分數",
                    "V5-B 一週起漲分數":"一週起漲分數",
                    "進場時機分數":"進場時機分數",
                    "V1-B 技術啟動":"技術啟動分數",
                    "V2-B 籌碼動能":"籌碼動能分數",
                    "V3-B 基本品質":"基本品質分數",
                    "V4-B 價格動能":"價格動能分數",
                    "原始 V5-A 最終分數":"最終分數",
                    "原始 V1 技術":"技術分數",
                    "原始 V2 籌碼":"籌碼標準分",
                    "原始 V3 基本面":"基本面分數",
                    "原始 V4 風險動能":"風險動能分數",
                }
                component_label = v61b.selectbox(
                    "分析哪個分數",
                    list(component_map.keys()),
                    key="v61_component",
                )
                component_col = component_map[component_label]

                bucket = score_bucket_analysis(bt, component_col, horizon)

                if bucket.empty:
                    st.info("目前還沒有足夠資料進行分數區間分析。")
                else:
                    show_bucket = bucket.copy()
                    for c in ["平均報酬率","中位數報酬率","標準差","勝率"]:
                        if c in show_bucket.columns:
                            show_bucket[c] = show_bucket[c].round(2)

                    c1,c2 = st.columns([1.05,1])
                    with c1:
                        fig_bucket = bucket_chart(
                            bucket,
                            f"{component_label}：{horizon}日後平均報酬",
                        )
                        if fig_bucket is not None:
                            st.plotly_chart(fig_bucket,use_container_width=True)

                    with c2:
                        table(show_bucket,height=360)

                    small_groups = bucket[bucket["樣本數"] < 20]
                    if not small_groups.empty:
                        st.warning("部分分數區間樣本少於 20 筆，目前只能視為初步觀察，不能據此調整權重。")

                st.markdown("### 哪個分數與未來報酬關聯較高？")
                predictive = score_predictive_table(bt, horizon)

                if predictive.empty:
                    st.info("目前樣本不足，尚不能比較各構面的預測關聯。")
                else:
                    show_pred = predictive.copy()
                    for c in ["Spearman相關","80分以上平均報酬","60分以下平均報酬","高低分報酬差"]:
                        if c in show_pred.columns:
                            show_pred[c] = show_pred[c].round(3 if c=="Spearman相關" else 2)
                    table(show_pred,height=300)

                    st.caption(
                        "Spearman相關 > 0 代表分數越高時，後續報酬傾向越高；"
                        "接近 0 代表關聯弱。這不是因果關係，也不應在少量樣本下直接改權重。"
                    )

                st.warning("目前 V6 是『前瞻式驗證』：從系統開始每天保存排名後累積樣本。這可避免用今天知道的資料回填過去，降低前視偏誤。")

elif page == "個股分析":
    opts=rank.apply(lambda r:f"{r['股票代號']} {r['股票名稱']}",axis=1).tolist()
    selcode=st.session_state.get("selected",str(rank.iloc[0]["股票代號"]))
    idx=next((i for i,x in enumerate(opts) if x.startswith(selcode+" ")),0)
    selected=st.selectbox("選擇股票",opts,index=idx); code=selected.split(" ",1)[0]; st.session_state.selected=code
    row=rank.loc[rank["股票代號"].eq(code)].iloc[0]; name=str(row.get("股票名稱",""))

    change_row = None
    if not daily_change.empty:
        xchg = daily_change[daily_change["股票代號"].eq(code)]
        if not xchg.empty:
            change_row = xchg.iloc[0]

    st.markdown(f"## {code}　{name}"); st.caption(f"{row.get('市場','—')} ｜ {row.get('產業別','—')} ｜ {row.get('財報類型','—')}")

    if pd.notna(row.get("一週模型排名", np.nan)):
        st.markdown("### 一週模型")
        a,b,c,d=st.columns(4)
        a.metric("主模型排名",fmt(row.get("主模型排名",row.get("一週模型排名")),0))
        b.metric("主模型分數",fmt(row.get("主模型分數"),1))
        c.metric("波段爆發分數",fmt(row.get("波段爆發分數"),1))
        d.metric("進場時機",fmt(row.get("進場時機分數"),1))
        st.caption(f"{row.get('50%潛力判定','—')} ｜ 啟動階段：{row.get('啟動階段','—')}")

        x1,x2,x3,x4=st.columns(4)
        x1.metric("歷史50%型態PR",fmt(row.get("50%歷史型態PR"),0))
        x2.metric("相似型態命中率",fmt(row.get("50%歷史型態命中率"),2,"%"))
        x3.metric("相似樣本平均最高漲幅",fmt(row.get("相似樣本40日最高報酬均值"),1,"%"))
        x4.metric("突破強度",fmt(row.get("突破強度分數"),0))

        y1,y2,y3,y4=st.columns(4)
        y1.metric("持續動能",fmt(row.get("動能持續分數"),0))
        y2.metric("籌碼動能",fmt(row.get("籌碼動能分數"),0))
        y3.metric("活躍爆發",fmt(row.get("活躍爆發分數"),0))
        y4.metric("波動爆發",fmt(row.get("波動爆發潛力"),0))

        st.markdown(f"**進場判定：{row.get('進場判定','—')}**")
        c1,c2=st.columns(2)
        with c1:
            st.info(f"起漲原因：{row.get('起漲原因','—')}")
        with c2:
            risk_text=str(row.get("進場風險","—"))
            if risk_text=="無明顯風險訊號":
                st.success(f"進場風險：{risk_text}")
            else:
                st.warning(f"進場風險：{risk_text}")

        g1,g2,g3=st.columns(3)
        g1.metric("台股環境",fmt(row.get("台股環境分數"),1))
        g2.metric("全球環境",fmt(row.get("全球環境分數"),1))
        g3.metric("產業海外順風",fmt(row.get("產業海外順風分數"),1))
        st.caption("主模型目標：40 個交易日內的大波段／挑戰 +50%。現在直接用歷史資料標記『40日內曾漲50%』，把目前型態與這些歷史成功樣本做相似度比較；歷史50%型態PR是波段爆發分數的最大權重，再搭配突破、持續動能、籌碼、波動與活躍度。")
    else:
        st.info("這檔目前還沒有一週模型資料，請先跑一次 GitHub Actions。")

    st.markdown("### AI 操作教練")
    coach_hist = history_all[history_all["股票代號"].eq(code)].copy() if (not history_all.empty and "股票代號" in history_all.columns) else pd.DataFrame()
    plan = build_ai_trade_plan(coach_hist, row)

    if not plan:
        st.info("歷史價格資料不足，暫時無法計算支撐、壓力與操作區間。")
    else:
        st.markdown(f"**模型操作狀態：{plan['操作狀態']}**　｜　{plan['策略']}")
        st.caption(plan["說明"])

        chart_col, info_col = st.columns([2.35, 1])

        with chart_col:
            coach_fig = ai_trade_plan_chart(coach_hist, plan)
            if coach_fig is not None:
                st.plotly_chart(coach_fig, use_container_width=True)

        with info_col:
            st.markdown("#### 關鍵價位")
            st.metric("觀察買入區", f"{fmt(plan['觀察買入下緣'],2)} ～ {fmt(plan['觀察買入上緣'],2)}")
            st.metric("停損 / 失效", fmt(plan["停損失效價"],2))
            st.metric("波段潛力", fmt(plan["波段潛力分數"],1))
            st.caption(plan["波段潛力判定"])

            st.markdown("**1～2 月波段目標**")
            st.write(f"第一目標 +15%：**{fmt(plan['第一波段目標'],2)}**")
            st.write(f"第二目標 +30%：**{fmt(plan['第二波段目標'],2)}**")
            st.write(f"50% 挑戰價：**{fmt(plan['50%挑戰價'],2)}**")

            st.markdown("**支撐**")
            st.write(f"短期：**{fmt(plan['短期支撐'],2)}**　{plan['短期支撐來源']}")
            st.write(f"長期：**{fmt(plan['長期支撐'],2)}**　{plan['長期支撐來源']}")

            st.markdown("**壓力**")
            st.write(f"短期：**{fmt(plan['短期壓力'],2)}**　{plan['短期壓力來源']}")
            st.write(f"長期：**{fmt(plan['長期壓力'],2)}**　{plan['長期壓力來源']}")

            st.markdown("**風險 / 報酬**")
            st.write(f"風險幅度：**{fmt(plan['估計風險幅度%'],1,'%')}**")
            st.write(f"+15% 目標風報比：**{fmt(plan['第一目標風報比'],2)}**")
            st.write(f"+30% 目標風報比：**{fmt(plan['第二目標風報比'],2)}**")
            st.write(f"+50% 目標風報比：**{fmt(plan['50%目標風報比'],2)}**")

        st.info(
            "操作方式：若模型為『可觀察進場』，優先等價格進入觀察買入區再分批評估；"
            "跌破停損/失效價代表原本的結構被破壞。短期/長期壓力現在只當作『途中關卡』，"
            "不再碰到就自動止盈；若趨勢延續，可用 +15%、+30%、+50% 波段目標搭配移動停利。"
            "所有價位會隨每日行情重新計算。"
        )

    st.markdown("### 原始 V5-A")
    a,b,c,d,e=st.columns(5); a.metric("最終分數",fmt(row.get("最終分數"),2)); b.metric("綜合 PR",fmt(row.get("綜合PR"),1)); c.metric("收盤價",fmt(row.get("收盤價"),2)); d.metric("候選等級",str(row.get("候選等級","—"))); e.metric("目前狀態",str(row.get("目前狀態","—")))
    if change_row is not None:
        p1,p2,p3=st.columns(3)
        p1.metric("昨日排名", fmt(change_row.get("昨日排名"),0))
        rank_delta = change_row.get("排名變化", np.nan)
        p2.metric("排名變化", ("↑ " + fmt(rank_delta,0)) if pd.notna(rank_delta) and float(rank_delta)>0 else (("↓ " + fmt(abs(float(rank_delta)),0)) if pd.notna(rank_delta) and float(rank_delta)<0 else "—"))
        score_delta = change_row.get("分數變化", np.nan)
        p3.metric("分數變化", fmt(score_delta,2))

    st.markdown("### 基本資訊")
    a,b,c,d,e,f=st.columns(6); a.metric("EPS",fmt(row.get("每股盈餘"),2)); b.metric("ROE",fmt(row.get("ROE"),1,"%")); c.metric("本益比",fmt(row.get("本益比"),1)); d.metric("P/B",fmt(row.get("股價淨值比"),2)); e.metric("殖利率",fmt(row.get("殖利率"),2,"%")); f.metric("營收 YoY",fmt(row.get("月營收年增率"),1,"%"))
    l,r=st.columns(2)
    with l: st.plotly_chart(four_factor(row),use_container_width=True)
    with r: st.plotly_chart(radar(row),use_container_width=True)
    c1,c2=st.columns(2)
    with c1: st.info(str(row.get("V5綜合理由","—")))
    with c2:
        x=str(row.get("V5風險提示","—")); st.success(x) if x=="無明顯風險訊號" else st.warning(x)
    t1,t2,t3,t4=st.tabs(["技術面","法人籌碼","基本面","風險動能"])
    with t1:
        hist = history_all[history_all["股票代號"].eq(code)].copy() if (not history_all.empty and "股票代號" in history_all.columns) else pd.DataFrame()
        if not hist.empty and "日期" in hist.columns:
            hist = hist.sort_values("日期").tail(120)
            st.plotly_chart(history_price_chart(hist), use_container_width=True)
            c1,c2=st.columns(2)
            with c1: st.plotly_chart(history_rsi_chart(hist), use_container_width=True)
            with c2: st.plotly_chart(history_macd_chart(hist), use_container_width=True)
        else:
            st.info("尚未放入歷史技術資料；目前先顯示當日快照。")
            f=ma_chart(row)
            if f: st.plotly_chart(f,use_container_width=True)
        a,b,c,d=st.columns(4); a.metric("RSI14",fmt(row.get("14日RSI"),1)); b.metric("量比",fmt(row.get("量比"),2)); c.metric("20日報酬",fmt(row.get("20日報酬率"),1,"%")); d.metric("60日報酬",fmt(row.get("60日報酬率"),1,"%"))
        st.caption(str(row.get("技術面原因","")))
    with t2:
        st.plotly_chart(inst_chart(row),use_container_width=True)
        a,b,c=st.columns(3); a.metric("外資連買",fmt(row.get("外資連續買超天數"),0," 日")); b.metric("投信連買",fmt(row.get("投信連續買超天數"),0," 日")); c.metric("籌碼標準分",fmt(row.get("籌碼標準分"),1))
    with t3:
        st.plotly_chart(pr_chart(row),use_container_width=True)
        a,b,c,d=st.columns(4); a.metric("毛利率",fmt(row.get("毛利率"),1,"%")); b.metric("營益率",fmt(row.get("營業利益率"),1,"%")); c.metric("負債比",fmt(row.get("負債比率"),1,"%")); d.metric("流動比",fmt(row.get("流動比率"),1,"%"))
    with t4:
        a,b,c,d=st.columns(4); a.metric("ATR%",fmt(row.get("ATR百分比"),2,"%")); b.metric("最大回撤",fmt(row.get("20日最大回撤"),1,"%")); c.metric("MA20乖離",fmt(row.get("MA20乖離率"),1,"%")); d.metric("年化波動",fmt(row.get("20日年化波動率"),1,"%"))
    st.markdown("### 最新個股新聞")
    st.caption("新聞即時抓取，與模型評分分開顯示。")
    items=news(name,code)
    if not items: st.info("目前暫時抓不到新聞，稍後重新整理即可。")
    for item in items:
        st.markdown(f'<div class="news"><a href="{item["link"]}" target="_blank">{item["title"]}</a><div class="newsmeta">{item["source"]}　{item["date"]}</div></div>',unsafe_allow_html=True)

elif page == "完整排名":
    st.markdown("## 完整排名")
    a,b,c=st.columns(3)
    q=a.text_input("搜尋代號 / 名稱")
    inds=sorted(rank["產業別"].dropna().astype(str).unique()) if "產業別" in rank.columns else []
    ind=b.multiselect("產業",inds)
    entry_opts=sorted(rank["進場判定"].dropna().astype(str).unique()) if "進場判定" in rank.columns else []
    entry_sel=c.multiselect("進場判定",entry_opts)

    d=rank.copy()
    if q:
        d=d[d["股票代號"].astype(str).str.contains(q,na=False)|d["股票名稱"].astype(str).str.contains(q,na=False)]
    if ind:
        d=d[d["產業別"].isin(ind)]
    if entry_sel and "進場判定" in d.columns:
        d=d[d["進場判定"].isin(entry_sel)]
    if "一週模型排名" in d.columns:
        d=d.sort_values("一週模型排名")

    cols=[x for x in [
        "一週模型排名","排名","股票代號","股票名稱","產業別",
        "一週起漲分數","進場時機分數","啟動階段","進場判定",
        "技術啟動分數","籌碼動能分數","基本品質分數","價格動能分數",
        "風險扣分","最終分數","候選等級","目前狀態","收盤價"
    ] if x in d.columns]
    table(d[cols],height=620)

elif page == "風險監控":
    st.markdown("## 風險監控")
    if "進場判定" in rank.columns:
        options=sorted(rank["進場判定"].dropna().astype(str).unique())
        defaults=[x for x in options if ("過熱" in x or "暫不" in x or "等待回檔" in x)]
        sel=st.multiselect("一週模型進場狀態",options,default=defaults)
        d=rank.copy()
        if sel:
            d=d[d["進場判定"].isin(sel)]
        if "一週模型排名" in d.columns:
            d=d.sort_values("一週模型排名")
        cols=[x for x in [
            "一週模型排名","股票代號","股票名稱","產業別",
            "一週起漲分數","進場時機分數","啟動階段","進場判定",
            "風險扣分","進場風險","ATR百分比","20日年化波動率",
            "20日最大回撤","MA20乖離率","5日報酬率","20日報酬率"
        ] if x in d.columns]
        table(d[cols],height=600)
    else:
        st.info("一週模型尚未產生，請先跑一次 GitHub Actions。")

else:
    st.markdown("## 產業分析")
    d=normalize(sheets.get("全市場產業PR",pd.DataFrame()))
    if d.empty: st.info("沒有全市場產業PR工作表。")
    else:
        inds=sorted(d["產業別"].dropna().astype(str).unique()); ind=st.selectbox("選擇產業",inds); x=d[d["產業別"].eq(ind)].copy()
        if "基本面分數" in x.columns: x=x.sort_values("基本面分數",ascending=False)
        cols=[c for c in ["股票代號","公司名稱","基本面分數","每股盈餘PR","ROE PR","毛利率PR","營業利益率PR","月營收年增率PR","負債比率PR","估值PR","殖利率PR"] if c in x.columns]
        table(x[cols])

st.sidebar.caption("每日把最新 V1～V5 Excel 放進 data 資料夾即可更新。")
