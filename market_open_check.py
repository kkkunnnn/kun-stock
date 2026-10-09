from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

TWSE_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TWSE_HOLIDAY_URL = "https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule"


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


def official_market_open_for_date(target_date):
    if target_date.weekday() >= 5:
        return False, "weekend"
    headers = {"User-Agent":"Mozilla/5.0","Accept":"application/json,text/plain,*/*"}
    r = requests.get(TWSE_HOLIDAY_URL, headers=headers, timeout=20)
    r.raise_for_status()
    rows = r.json()
    roc = f"{target_date.year - 1911:03d}{target_date.month:02d}{target_date.day:02d}"
    matches = [x for x in rows if str(x.get("Date","")).strip() == roc]
    if not matches:
        return True, "regular weekday"
    desc = " ".join(f"{x.get('Name','')} {x.get('Description','')}" for x in matches)
    if "開始交易" in desc or "最後交易" in desc:
        return True, desc
    return False, desc or "official holiday"


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
    p.add_argument("--scheduled", action="store_true")
    args = p.parse_args()

    now_tw = datetime.now(ZoneInfo("Asia/Taipei"))
    today = now_tw.date()

    if args.force:
        print(f"FORCE_RUN=1; bypass market-day check for {today}")
        print("market_open=true")
        return 0

    # GitHub Actions scheduled jobs can start hours late. The nominal schedule is
    # 21:00 Asia/Taipei. If a scheduled job finally starts after midnight but
    # before the next 21:00, it still belongs to the previous trading session.
    if args.scheduled and now_tw.hour < 21:
        target_date = today - timedelta(days=1)
    else:
        target_date = today

    try:
        is_open, reason = official_market_open_for_date(target_date)
    except Exception as e:
        is_open, reason = True, f"holiday calendar unavailable: {e}"

    print(f"Taiwan now: {now_tw.isoformat(timespec='seconds')}")
    print(f"Target trading session: {target_date}")
    print(f"TWSE calendar: {'OPEN' if is_open else 'CLOSED'}")
    print(f"TWSE calendar reason: {reason}")

    try:
        latest = latest_twse_trade_date()
        print(f"TWSE latest trade date (diagnostic only): {latest}")
    except Exception as e:
        print(f"TWSE latest trade date unavailable: {e}")

    if is_open:
        print("market_open=true")
        print("✅ 依官方開休市日曆執行完整選股。")
    else:
        print("market_open=false")
        print("⏭️ 依官方開休市日曆略過模型與資料更新。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
