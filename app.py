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





@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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




@st.cache_data(show_spinner=False)
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



@st.cache_data(show_spinner=False)
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

        for horizon in [1,5,20]:
            j = i + horizon
            col = f"{horizon}日後報酬率"
            if j < len(g):
                future = g.iloc[j]["收盤價"]
                rec[col] = (future / base - 1) * 100
            else:
                rec[col] = np.nan

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
        for h in [1,5,20]:
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




@st.cache_data(show_spinner=False)
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
if not sheets:
    st.warning("找不到資料，請從側邊欄上傳 V1～V5 Excel。")
    st.stop()

rank = normalize(sheets.get("全部排名", pd.DataFrame()))

if not weekly_model_all.empty:
    weekly_cols = [
        c for c in [
            "股票代號","一週模型排名","起漲潛力排名","一週起漲分數","進場時機分數",
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

pages = ["今日 Top 10","一週模型","全球市場","每日變化","V6 回測","個股分析","完整排名","風險監控","產業分析"]
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

        st.markdown("## 未來一週優先觀察 Top 10")
        st.caption("依『進場時機分數』排序；同時考慮起漲潛力、台股/全球環境、產業海外順風與風險扣分。它是模型訊號，不是保證報酬。")

        top = rank.sort_values(["一週模型排名"]).head(10).copy()

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
                with cc[j]:
                    st.markdown(
                        f'<div class="card">'
                        f'<span class="rank">#{int(r.get("一週模型排名",k+1))}</span>'
                        f'<div class="title">{code}　{name}</div>'
                        f'<div class="muted">{r.get("產業別","—")} ｜ {stage}</div>'
                        f'<div class="score">{fmt(r.get("進場時機分數"),2)}</div>'
                        f'<div class="muted">一週起漲 {fmt(r.get("一週起漲分數"),1)} ｜ {entry_label}</div>'
                        f'<div class="muted">技術啟動 {fmt(r.get("技術啟動分數"),0)} ｜ 籌碼動能 {fmt(r.get("籌碼動能分數"),0)} ｜ 基本品質 {fmt(r.get("基本品質分數"),0)} ｜ 價格動能 {fmt(r.get("價格動能分數"),0)}</div>'
                        f'<div class="muted">風險扣分 {fmt(r.get("風險扣分"),0)} ｜ 原始V5 {fmt(r.get("最終分數"),1)}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
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


elif page == "一週模型":
    st.markdown("## 一週起漲＋進場時機模型")
    st.caption("V5-A 原始模型完整保留；這裡是針對『未來約 5 個交易日可能啟動、以及現在是否適合介入』另外建立的 V5-B。")

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
            "一週模型排名","股票代號","股票名稱","產業別",
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
                    "啟動階段","進場判定","1日後報酬率","5日後報酬率","20日後報酬率"
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
        a.metric("一週模型排名",fmt(row.get("一週模型排名"),0))
        b.metric("一週起漲分數",fmt(row.get("一週起漲分數"),1))
        c.metric("進場時機分數",fmt(row.get("進場時機分數"),1))
        d.metric("啟動階段",str(row.get("啟動階段","—")))

        x1,x2,x3,x4=st.columns(4)
        x1.metric("技術啟動",fmt(row.get("技術啟動分數"),0))
        x2.metric("籌碼動能",fmt(row.get("籌碼動能分數"),0))
        x3.metric("基本品質",fmt(row.get("基本品質分數"),0))
        x4.metric("價格動能",fmt(row.get("價格動能分數"),0))

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
        st.caption("一週模型權重：技術啟動 40%｜籌碼動能 30%｜基本品質 15%｜價格動能 15%；風險與市場環境另外作 Gate。")
    else:
        st.info("這檔目前還沒有一週模型資料，請先跑一次 GitHub Actions。")

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
