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


def status_css(s):
    if s == "趨勢健康": return "good"
    if s in ["等待回檔", "整理觀察", "一般觀察"]: return "warn"
    return "bad"


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
if not sheets:
    st.warning("找不到資料，請從側邊欄上傳 V1～V5 Excel。")
    st.stop()

rank = normalize(sheets.get("全部排名", pd.DataFrame()))
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

pages = ["今日 Top 10","個股分析","完整排名","風險監控","產業分析"]
if "nav" not in st.session_state: st.session_state.nav = "今日 Top 10"
page = st.radio("導覽", pages, horizontal=True, label_visibility="collapsed", key="nav")

if page == "今日 Top 10":
    strong = rank.get("候選等級",pd.Series(dtype=str)).astype(str).str.contains("強勢候選",na=False).sum()
    healthy = rank.get("目前狀態",pd.Series(dtype=str)).astype(str).eq("趨勢健康").sum()
    overheat = rank.get("目前狀態",pd.Series(dtype=str)).astype(str).eq("短線過熱").sum()
    a,b,c,d = st.columns(4); a.metric("股票池",f"{len(rank)} 檔"); b.metric("強勢候選",f"{int(strong)} 檔"); c.metric("趨勢健康",f"{int(healthy)} 檔"); d.metric("短線過熱",f"{int(overheat)} 檔")
    st.markdown("## 今日模型 Top 10")
    st.caption("依 V1～V5 最終分數排序，作為優先研究清單，不代表保證報酬。")
    top = rank.sort_values("排名").head(10) if "排名" in rank.columns else rank.head(10)
    for i in range(0,len(top),2):
        cc = st.columns(2)
        for j in range(2):
            k=i+j
            if k>=len(top): break
            r=top.iloc[k]; code=str(r.get("股票代號","")); name=str(r.get("股票名稱","")); status=str(r.get("目前狀態","—"))
            with cc[j]:
                st.markdown(f'<div class="card"><span class="rank">#{int(r.get("排名",k+1))}</span><div class="title">{code}　{name}</div><div class="muted">{r.get("產業別","—")} ｜ {r.get("候選等級","—")}</div><div class="score">{fmt(r.get("最終分數"),2)}</div><div class="muted">綜合 PR {fmt(r.get("綜合PR"),1)} ｜ <span class="{status_css(status)}">{status}</span></div><div class="muted">風險：{r.get("V5風險提示","—")}</div></div>',unsafe_allow_html=True)
                if st.button(f"查看 {name} 詳細分析 →",key=f"d_{code}",use_container_width=True):
                    st.session_state.selected=code; st.session_state.nav="個股分析"; st.rerun()

elif page == "個股分析":
    opts=rank.apply(lambda r:f"{r['股票代號']} {r['股票名稱']}",axis=1).tolist()
    selcode=st.session_state.get("selected",str(rank.iloc[0]["股票代號"]))
    idx=next((i for i,x in enumerate(opts) if x.startswith(selcode+" ")),0)
    selected=st.selectbox("選擇股票",opts,index=idx); code=selected.split(" ",1)[0]; st.session_state.selected=code
    row=rank.loc[rank["股票代號"].eq(code)].iloc[0]; name=str(row.get("股票名稱",""))
    st.markdown(f"## {code}　{name}"); st.caption(f"{row.get('市場','—')} ｜ {row.get('產業別','—')} ｜ {row.get('財報類型','—')}")
    a,b,c,d,e=st.columns(5); a.metric("最終分數",fmt(row.get("最終分數"),2)); b.metric("綜合 PR",fmt(row.get("綜合PR"),1)); c.metric("收盤價",fmt(row.get("收盤價"),2)); d.metric("候選等級",str(row.get("候選等級","—"))); e.metric("目前狀態",str(row.get("目前狀態","—")))
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
    a,b,c=st.columns(3); q=a.text_input("搜尋代號 / 名稱"); inds=sorted(rank["產業別"].dropna().astype(str).unique()) if "產業別" in rank.columns else []; ind=b.multiselect("產業",inds); states=sorted(rank["目前狀態"].dropna().astype(str).unique()) if "目前狀態" in rank.columns else []; state=c.multiselect("狀態",states)
    d=rank.copy()
    if q:
        d=d[d["股票代號"].astype(str).str.contains(q,na=False)|d["股票名稱"].astype(str).str.contains(q,na=False)]
    if ind: d=d[d["產業別"].isin(ind)]
    if state: d=d[d["目前狀態"].isin(state)]
    cols=[x for x in ["排名","股票代號","股票名稱","產業別","最終分數","綜合PR","候選等級","目前狀態","技術分數","籌碼標準分","基本面分數","風險動能分數","收盤價","V5風險提示"] if x in d.columns]
    table(d[cols])

elif page == "風險監控":
    st.markdown("## 風險監控")
    statuses=["短線過熱","風險偏高","等待回檔","趨勢健康","整理觀察","一般觀察"]
    sel=st.multiselect("狀態",statuses,default=["短線過熱","風險偏高","等待回檔"]); d=rank.copy()
    if sel: d=d[d["目前狀態"].isin(sel)]
    cols=[x for x in ["排名","股票代號","股票名稱","產業別","最終分數","目前狀態","風險動能分數","5日報酬率","20日報酬率","20日年化波動率","ATR百分比","20日最大回撤","MA20乖離率","V5風險提示"] if x in d.columns]
    table(d[cols])

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
