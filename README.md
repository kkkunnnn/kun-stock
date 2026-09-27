# 台股 V1～V5 Dashboard

這是把每日 V1～V5 選股 Excel 轉成手機可看的 Streamlit Dashboard。

## 本機啟動

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 每日更新方式（第一版）

1. 執行 V1～V5，產生新的 `.xlsx`。
2. 打開 App。
3. 在左側「上傳今日 V1～V5 Excel」選擇最新檔案。
4. App 會即時顯示首頁、完整排名、個股分析、風險監控與產業分析。

也可把每天的 Excel 放到 `data/` 資料夾，App 會自動選檔名日期最新的一份。

## iPhone

部署到 Streamlit Community Cloud 後，用 Safari 開啟網址，再選「分享」→「加入主畫面」。
這會像 App 圖示一樣從 iPhone 主畫面啟動，但第一版仍是 Web App，不是 App Store 原生程式。
