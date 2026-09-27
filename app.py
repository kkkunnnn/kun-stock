from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


APP_TITLE = "台股 V1～V5 選股 Dashboard"
DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------
# UI helpers
# -----------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 12px 14px;
            border-radius: 16px;
        }
        .stock-card {
            border: 1px solid rgba(255,255,255,.10);
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 10px;
            background: rgba(255,255,255,.025);
        }
        .muted {opacity:.68; font-size:.9rem;}
        .good {color:#67d391; font-weight:700;}
        .warn {color:#ffcc66; font-weight:700;}
        .bad {color:#ff7b72; font-weight:700;}
        @media (max-width: 700px) {
            .block-container {padding-left: .8rem; padding-right: .8rem;}
            h1 {font-size: 1.65rem !important;}
            h2 {font-size: 1.35rem !important;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fmt_num(v, digits=1, suffix=""):
    if pd.isna(v):
        return "—"
    try:
        return f"{float(v):,.{digits}f}{suffix}"
    except Exception:
        return str(v)


def latest_local_excel() -> Optional[Path]:
    files = list(DATA_DIR.glob("*.xlsx"))
    if not files:
        return None

    def key(p: Path):
        m = re.search(r"(20\d{6})", p.stem)
        return (m.group(1) if m else "00000000", p.stat().st_mtime)

    return sorted(files, key=key, reverse=True)[0]


@st.cache_data(show_spinner=False)
def read_excel_bytes(raw: bytes):
    bio = io.BytesIO(raw)
    xls = pd.ExcelFile(bio)
    sheets = {}
    for s in xls.sheet_names:
        sheets[s] = pd.read_excel(bio, sheet_name=s)
        bio.seek(0)
    return sheets


@st.cache_data(show_spinner=False)
def read_excel_path(path: str, mtime: float):
    xls = pd.ExcelFile(path)
    return {s: pd.read_excel(path, sheet_name=s) for s in xls.sheet_names}


def load_workbook():
    uploaded = st.sidebar.file_uploader(
        "上傳今日 V1～V5 Excel",
        type=["xlsx"],
        help="第一版可每天把最新選股 Excel 上傳到 App；之後可再改成全自動更新。",
    )
    if uploaded is not None:
        return read_excel_bytes(uploaded.getvalue()), uploaded.name, True

    local = latest_local_excel()
    if local is None:
        return {}, "", False
    return read_excel_path(str(local), local.stat().st_mtime), local.name, False


def get_date_text(sheets, filename: str) -> str:
    sysdf = sheets.get("系統說明")
    if sysdf is not None and not sysdf.empty and {"項目", "說明"}.issubset(sysdf.columns):
        rows = sysdf.loc[sysdf["項目"].astype(str).eq("資料基準日"), "說明"]
        if not rows.empty:
            return str(rows.iloc[0])
    m = re.search(r"(20\d{6})", filename)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return "—"


def normalize_code(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "股票代號" in out.columns:
        out["股票代號"] = (
            out["股票代號"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
        )
    return out


def score_color(v):
    if pd.isna(v):
        return ""
    v = float(v)
    if v >= 75:
        return "background-color: rgba(46,160,67,.22)"
    if v >= 60:
        return "background-color: rgba(210,153,34,.18)"
    return ""


def display_table(df: pd.DataFrame, height=520):
    if df.empty:
        st.info("目前沒有符合條件的股票。")
        return
    score_cols = [c for c in ["最終分數", "技術分數", "籌碼標準分", "基本面分數", "風險動能分數"] if c in df.columns]
    styler = df.style
    if score_cols:
        styler = styler.map(score_color, subset=score_cols)
    st.dataframe(styler, use_container_width=True, hide_index=True, height=height)


def radar_chart(row):
    vals = [
        row.get("技術分數", np.nan),
        row.get("籌碼標準分", np.nan),
        row.get("基本面分數", np.nan),
        row.get("風險動能分數", np.nan),
    ]
    vals = [0 if pd.isna(x) else float(x) for x in vals]
    cats = ["V1 技術", "V2 籌碼", "V3 基本面", "V4 風險動能"]
    fig = go.Figure(
        go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself")
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        margin=dict(l=30, r=30, t=30, b=30),
        height=360,
    )
    return fig


def pr_chart(row):
    mapping = {
        "EPS": "每股盈餘PR",
        "ROE": "ROE PR",
        "毛利率": "毛利率PR",
        "營益率": "營業利益率PR",
        "營收YoY": "月營收年增率PR",
        "營收MoM": "月營收月增率PR",
        "負債比": "負債比率PR",
        "流動比率": "流動比率PR",
        "資產周轉": "總資產周轉率PR",
        "估值": "估值PR",
        "殖利率": "殖利率PR",
    }
    labels, vals = [], []
    for lab, col in mapping.items():
        if col in row.index and pd.notna(row[col]):
            labels.append(lab)
            vals.append(float(row[col]))
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h"))
    fig.update_xaxes(range=[0, 100], title="同產業 PR")
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=30), height=430)
    return fig


inject_css()
sheets, source_name, is_upload = load_workbook()

if not sheets:
    st.title(APP_TITLE)
    st.warning("找不到資料。請從側邊欄上傳 V1～V5 Excel。")
    st.stop()

rank = normalize_code(sheets.get("全部排名", pd.DataFrame()))
if rank.empty:
    st.error("Excel 找不到「全部排名」工作表。")
    st.stop()

base_date = get_date_text(sheets, source_name)

st.title("📈 台股 V1～V5 選股")
st.caption(f"資料基準日：{base_date} ｜ 資料來源：{source_name}")

# Data-quality guard
tech_ok = int(rank.get("技術分數", pd.Series(index=rank.index, dtype=float)).notna().sum())
v4_ok = int(rank.get("風險動能分數", pd.Series(index=rank.index, dtype=float)).notna().sum())
total_n = len(rank)
coverage = min(tech_ok, v4_ok) / total_n if total_n else 0
if coverage < 0.95:
    st.error(
        f"⚠️ 今日資料不完整：股票池 {total_n} 檔，但 V1/V4 完整資料只有 {min(tech_ok, v4_ok)} 檔（{coverage:.0%}）。"
        "目前排名只能視為測試結果，不建議當成完整市場排名。"
    )

# Navigation
page = st.sidebar.radio(
    "頁面",
    ["今日總覽", "完整排名", "個股分析", "風險監控", "產業分析"],
)

# Shared counts
strong = int(rank.get("候選等級", pd.Series(dtype=str)).astype(str).str.contains("強勢候選", na=False).sum())
watch = int(rank.get("候選等級", pd.Series(dtype=str)).astype(str).str.contains("觀察候選", na=False).sum())
healthy = int(rank.get("目前狀態", pd.Series(dtype=str)).astype(str).eq("趨勢健康").sum())
overheat = int(rank.get("目前狀態", pd.Series(dtype=str)).astype(str).eq("短線過熱").sum())

if page == "今日總覽":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("股票池", f"{total_n} 檔")
    c2.metric("強勢候選", f"{strong} 檔")
    c3.metric("觀察候選", f"{watch} 檔")
    c4.metric("趨勢健康", f"{healthy} 檔")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("短線過熱", f"{overheat} 檔")
    c6.metric("V1資料完整", f"{tech_ok}/{total_n}")
    c7.metric("V4資料完整", f"{v4_ok}/{total_n}")
    top_score = rank["最終分數"].max() if "最終分數" in rank.columns else np.nan
    c8.metric("最高最終分數", fmt_num(top_score, 2))

    st.subheader("今日 Top 10")
    cols = [c for c in [
        "排名", "股票代號", "股票名稱", "產業別", "最終分數", "綜合PR",
        "候選等級", "目前狀態", "技術分數", "籌碼標準分", "基本面分數", "風險動能分數",
        "V5風險提示"
    ] if c in rank.columns]
    display_table(rank[cols].head(10), height=390)

    st.subheader("今日值得先看的股票")
    focus = rank.copy()
    if "目前狀態" in focus.columns:
        focus = focus[focus["目前狀態"].isin(["趨勢健康", "等待回檔", "短線過熱"])]
    if "最終分數" in focus.columns:
        focus = focus.sort_values("最終分數", ascending=False)
    for _, r in focus.head(6).iterrows():
        status = str(r.get("目前狀態", "—"))
        css = "good" if status == "趨勢健康" else ("bad" if status == "短線過熱" else "warn")
        st.markdown(
            f"""<div class="stock-card"><b>{r.get('股票代號','')} {r.get('股票名稱','')}</b>　"
            f"<span class='{css}'>{status}</span><br>"
            f"<span class='muted'>最終 {fmt_num(r.get('最終分數'),2)} ｜ PR {fmt_num(r.get('綜合PR'),1)} ｜ {r.get('產業別','')}</span><br>"
            f"{r.get('V5綜合理由','')}<br><span class='muted'>風險：{r.get('V5風險提示','')}</span></div>""",
            unsafe_allow_html=True,
        )

elif page == "完整排名":
    st.subheader("完整排名與篩選")
    f1, f2, f3 = st.columns(3)
    q = f1.text_input("搜尋代號 / 名稱")
    industries = sorted(rank["產業別"].dropna().astype(str).unique()) if "產業別" in rank.columns else []
    industry_sel = f2.multiselect("產業", industries)
    states = sorted(rank["目前狀態"].dropna().astype(str).unique()) if "目前狀態" in rank.columns else []
    state_sel = f3.multiselect("目前狀態", states)

    f4, f5, f6 = st.columns(3)
    min_score = f4.slider("最低最終分數", 0, 100, 0)
    min_fund = f5.slider("最低基本面分數", 0, 100, 0)
    candidate_only = f6.checkbox("只看強勢／觀察候選")

    filt = rank.copy()
    if q:
        qq = q.strip().lower()
        filt = filt[
            filt["股票代號"].astype(str).str.lower().str.contains(qq, na=False)
            | filt["股票名稱"].astype(str).str.lower().str.contains(qq, na=False)
        ]
    if industry_sel:
        filt = filt[filt["產業別"].isin(industry_sel)]
    if state_sel:
        filt = filt[filt["目前狀態"].isin(state_sel)]
    if "最終分數" in filt.columns:
        filt = filt[pd.to_numeric(filt["最終分數"], errors="coerce").fillna(-1) >= min_score]
    if "基本面分數" in filt.columns:
        filt = filt[pd.to_numeric(filt["基本面分數"], errors="coerce").fillna(-1) >= min_fund]
    if candidate_only and "候選等級" in filt.columns:
        filt = filt[filt["候選等級"].astype(str).str.contains("強勢候選|觀察候選", regex=True, na=False)]

    show_cols = [c for c in [
        "排名", "股票代號", "股票名稱", "產業別", "最終分數", "綜合PR", "候選等級", "目前狀態",
        "技術分數", "籌碼標準分", "基本面分數", "風險動能分數", "收盤價", "V5風險提示"
    ] if c in filt.columns]
    st.caption(f"符合條件：{len(filt)} 檔")
    display_table(filt[show_cols])

elif page == "個股分析":
    st.subheader("個股 V1～V5 分析")
    opts = rank.apply(lambda r: f"{r['股票代號']} {r['股票名稱']}", axis=1).tolist()
    default_index = 0
    selected = st.selectbox("選擇股票", opts, index=default_index)
    code = selected.split(" ", 1)[0]
    row = rank.loc[rank["股票代號"].eq(code)].iloc[0]

    st.markdown(f"### {code}　{row.get('股票名稱','')}　｜　{row.get('產業別','')}")
    a, b, c, d = st.columns(4)
    a.metric("最終分數", fmt_num(row.get("最終分數"), 2))
    b.metric("綜合 PR", fmt_num(row.get("綜合PR"), 1))
    c.metric("候選等級", str(row.get("候選等級", "—")))
    d.metric("目前狀態", str(row.get("目前狀態", "—")))

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(radar_chart(row), use_container_width=True)
    with right:
        st.plotly_chart(pr_chart(row), use_container_width=True)

    st.markdown("#### V5摘要")
    st.write(row.get("V5綜合理由", "—"))
    risk_text = row.get("V5風險提示", "—")
    if risk_text == "無明顯風險訊號":
        st.success(risk_text)
    else:
        st.warning(risk_text)

    t1, t2, t3, t4 = st.tabs(["V1 技術", "V2 籌碼", "V3 基本面", "V4 風險動能"])
    with t1:
        cols = ["收盤價", "5日均線", "20日均線", "60日均線", "14日RSI", "MACD", "MACD訊號線", "MACD柱狀體", "量比", "技術面原因"]
        st.dataframe(pd.DataFrame({"項目": cols, "數值": [row.get(x, np.nan) for x in cols]}), hide_index=True, use_container_width=True)
    with t2:
        cols = ["籌碼分數", "籌碼標準分", "外資近5日買賣超（張）", "投信近5日買賣超（張）", "自營商近5日買賣超（張）", "三大法人近5日合計（張）", "外資連續買超天數", "投信連續買超天數", "籌碼面原因"]
        st.dataframe(pd.DataFrame({"項目": cols, "數值": [row.get(x, np.nan) for x in cols]}), hide_index=True, use_container_width=True)
    with t3:
        cols = ["每股盈餘", "ROE", "毛利率", "營業利益率", "月營收年增率", "月營收月增率", "負債比率", "流動比率", "總資產周轉率", "本益比", "股價淨值比", "殖利率", "基本面分數", "基本面原因"]
        st.dataframe(pd.DataFrame({"項目": cols, "數值": [row.get(x, np.nan) for x in cols]}), hide_index=True, use_container_width=True)
    with t4:
        cols = ["5日報酬率", "20日報酬率", "60日報酬率", "20日年化波動率", "ATR百分比", "20日最大回撤", "MA20乖離率", "MA60乖離率", "20日成交量變異係數", "風險動能分數", "V4原因"]
        st.dataframe(pd.DataFrame({"項目": cols, "數值": [row.get(x, np.nan) for x in cols]}), hide_index=True, use_container_width=True)

elif page == "風險監控":
    st.subheader("V4 / V5 風險監控")
    risk = rank.copy()
    statuses = ["短線過熱", "風險偏高", "等待回檔", "趨勢健康", "整理觀察", "一般觀察"]
    sel = st.multiselect("狀態", statuses, default=["短線過熱", "風險偏高", "等待回檔"])
    if sel and "目前狀態" in risk.columns:
        risk = risk[risk["目前狀態"].isin(sel)]
    risk_cols = [c for c in [
        "排名", "股票代號", "股票名稱", "產業別", "最終分數", "目前狀態", "風險動能分數",
        "5日報酬率", "20日報酬率", "20日年化波動率", "ATR百分比", "20日最大回撤",
        "MA20乖離率", "MA60乖離率", "V5風險提示"
    ] if c in risk.columns]
    display_table(risk[risk_cols])

elif page == "產業分析":
    st.subheader("同產業基本面比較")
    industry_df = sheets.get("全市場產業PR", pd.DataFrame())
    industry_df = normalize_code(industry_df)
    if industry_df.empty:
        st.info("Excel 沒有「全市場產業PR」工作表。")
    else:
        industries = sorted(industry_df["產業別"].dropna().astype(str).unique())
        ind = st.selectbox("選擇產業", industries)
        d = industry_df[industry_df["產業別"].eq(ind)].copy()
        if "基本面分數" in d.columns:
            d = d.sort_values("基本面分數", ascending=False)
        cols = [c for c in [
            "股票代號", "公司名稱", "基本面分數", "每股盈餘PR", "ROE PR", "毛利率PR",
            "營業利益率PR", "月營收年增率PR", "負債比率PR", "估值PR", "殖利率PR"
        ] if c in d.columns]
        st.caption(f"{ind}：{len(d)} 家")
        display_table(d[cols])

st.sidebar.divider()
st.sidebar.caption("第一版：每日 Excel 上傳／data 資料夾讀取。下一版可做每日自動更新與歷史排名追蹤。")
