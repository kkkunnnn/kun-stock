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
    print(f"✅ 已更新 {history_dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
