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
        "TSM ADR": "TSM",
        "NVIDIA": "NVDA",
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
