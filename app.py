        return out
    except Exception:
        return []


def table(df, height=500):
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)


css()
sheets, source = load_data()
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
