from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

TWSE_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"


def parse_date(value: str):
    x = str(value or "").strip().replace("/", "").replace("-", "")
    if len(x) == 7 and x.isdigit():
        return datetime(
            int(x[:3]) + 1911,
            int(x[3:5]),
            int(x[5:7]),
            tzinfo=ZoneInfo("Asia/Taipei"),
        ).date()
    if len(x) == 8 and x.isdigit():
        return datetime(
            int(x[:4]),
            int(x[4:6]),
            int(x[6:8]),
            tzinfo=ZoneInfo("Asia/Taipei"),
        ).date()
    return None


def latest_twse_trade_date():
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
    }
    last_error = None
    for attempt in range(3):
        try:
            r = requests.get(TWSE_URL, headers=headers, timeout=30)
            r.raise_for_status()
            rows = r.json()
            if not isinstance(rows, list) or not rows:
                raise RuntimeError("TWSE OpenAPI returned no rows")

            for key in ("Date", "date", "日期"):
                if key in rows[0]:
                    d = parse_date(rows[0].get(key))
                    if d is not None:
                        return d
            raise RuntimeError("TWSE OpenAPI date field not found")
        except Exception as e:
            last_error = str(e)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Unable to determine TWSE latest trade date: {last_error}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    today = datetime.now(ZoneInfo("Asia/Taipei")).date()

    if args.force:
        print(f"FORCE_RUN=1; bypass market-day check for {today}")
        print("market_open=true")
        return 0

    latest = latest_twse_trade_date()
    is_open = latest == today

    print(f"Taiwan today: {today}")
    print(f"TWSE latest trade date: {latest}")

    if is_open:
        print("market_open=true")
        print("✅ 今日有台股成交資料，執行完整選股。")
    else:
        print("market_open=false")
        print("⏭️ 今日不是台股交易日，略過模型與資料更新。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
