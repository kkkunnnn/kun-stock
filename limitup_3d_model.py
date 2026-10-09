from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_VERSION = "LIMITUP-3D-V1"
MODEL_TARGET = "未來3個交易日內出現漲停啟動訊號"


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _rsi(close: pd.Series, period: int = 14) -> float:
    c = _num(close).dropna()
    if len(c) < period + 1:
        return np.nan
    d = c.diff()
    up = d.clip(lower=0).rolling(period).mean()
    dn = (-d.clip(upper=0)).rolling(period).mean()
    rs = up / dn.replace(0, np.nan)
    val = 100 - 100 / (1 + rs.iloc[-1])
    return float(val) if pd.notna(val) else np.nan


def _build_feature_row(code: str, g: pd.DataFrame) -> dict | None:
    g = g.sort_values("date").copy()
    if len(g) < 25:
        return None

    close = _num(g["close"])
    high = _num(g["max"])
    low = _num(g["min"])
    open_ = _num(g["open"])
    vol = _num(g["Trading_Volume"]).fillna(0)

    if close.dropna().empty:
        return None

    c = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) >= 2 and pd.notna(close.iloc[-2]) else np.nan
    o = float(open_.iloc[-1]) if pd.notna(open_.iloc[-1]) else np.nan
    h = float(high.iloc[-1]) if pd.notna(high.iloc[-1]) else np.nan
    l = float(low.iloc[-1]) if pd.notna(low.iloc[-1]) else np.nan
    v = float(vol.iloc[-1]) if pd.notna(vol.iloc[-1]) else 0.0

    ma5 = float(close.tail(5).mean())
    ma10 = float(close.tail(10).mean())
    ma20 = float(close.tail(20).mean())

    prev20_high = float(high.iloc[-21:-1].max()) if len(high) >= 21 else np.nan
    ret1 = (c / prev - 1) * 100 if pd.notna(prev) and prev > 0 else np.nan
    ret3 = (c / float(close.iloc[-4]) - 1) * 100 if len(close) >= 4 and close.iloc[-4] > 0 else np.nan
    ret5 = (c / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 and close.iloc[-6] > 0 else np.nan
    ret10 = (c / float(close.iloc[-11]) - 1) * 100 if len(close) >= 11 and close.iloc[-11] > 0 else np.nan

    vol20 = float(vol.tail(20).mean())
    vol5 = float(vol.tail(5).mean())
    volume_ratio = v / vol20 if vol20 > 0 else np.nan
    volume_5v20 = vol5 / vol20 if vol20 > 0 else np.nan

    day_range = h - l if pd.notna(h) and pd.notna(l) else np.nan
    close_pos = (c - l) / day_range if pd.notna(day_range) and day_range > 0 else 0.5
    body_pct = abs(c - o) / o * 100 if pd.notna(o) and o > 0 else np.nan
    upper_shadow_pct = (h - max(c, o)) / o * 100 if pd.notna(o) and o > 0 and pd.notna(h) else np.nan

    breakout20 = (c / prev20_high - 1) * 100 if pd.notna(prev20_high) and prev20_high > 0 else np.nan
    bias20 = (c / ma20 - 1) * 100 if ma20 > 0 else np.nan

    ma5_prev = float(close.iloc[-10:-5].mean()) if len(close) >= 10 else np.nan
    ma20_prev = float(close.iloc[-25:-5].mean()) if len(close) >= 25 else np.nan
    ma5_slope = (ma5 / ma5_prev - 1) * 100 if pd.notna(ma5_prev) and ma5_prev > 0 else np.nan
    ma20_slope = (ma20 / ma20_prev - 1) * 100 if pd.notna(ma20_prev) and ma20_prev > 0 else np.nan

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr5 = float(tr.tail(5).mean())
    atr20 = float(tr.tail(20).mean())
    atr_pct = atr5 / c * 100 if c > 0 else np.nan
    atr_accel = atr5 / atr20 if atr20 > 0 else np.nan

    std5 = float(close.tail(5).pct_change().std())
    std20 = float(close.tail(20).pct_change().std())
    vol_compression = std5 / std20 if std20 > 0 else np.nan

    turnover_value = c * v
    rsi14 = _rsi(close)

    return {
        "股票代號": code,
        "收盤價_3D": c,
        "3D_ret1": ret1,
        "3D_ret3": ret3,
        "3D_ret5": ret5,
        "3D_ret10": ret10,
        "3D_volume_ratio": volume_ratio,
        "3D_volume_5v20": volume_5v20,
        "3D_breakout20": breakout20,
        "3D_bias20": bias20,
        "3D_ma5_slope": ma5_slope,
        "3D_ma20_slope": ma20_slope,
        "3D_close_pos": close_pos * 100,
        "3D_body_pct": body_pct,
        "3D_upper_shadow_pct": upper_shadow_pct,
        "3D_atr_pct": atr_pct,
        "3D_atr_accel": atr_accel,
        "3D_vol_compression": vol_compression,
        "3D_rsi14": rsi14,
        "3D_turnover_value": turnover_value,
        "3D_ma5_gt_ma10": float(ma5 > ma10),
        "3D_ma10_gt_ma20": float(ma10 > ma20),
    }


def _score_row(r: pd.Series) -> tuple[float, dict, list[str], list[str]]:
    ret1 = r.get("3D_ret1", np.nan)
    ret3 = r.get("3D_ret3", np.nan)
    ret5 = r.get("3D_ret5", np.nan)
    volume_ratio = r.get("3D_volume_ratio", np.nan)
    volume_5v20 = r.get("3D_volume_5v20", np.nan)
    breakout = r.get("3D_breakout20", np.nan)
    bias20 = r.get("3D_bias20", np.nan)
    ma5s = r.get("3D_ma5_slope", np.nan)
    ma20s = r.get("3D_ma20_slope", np.nan)
    close_pos = r.get("3D_close_pos", np.nan)
    body = r.get("3D_body_pct", np.nan)
    upper = r.get("3D_upper_shadow_pct", np.nan)
    atr_accel = r.get("3D_atr_accel", np.nan)
    compression = r.get("3D_vol_compression", np.nan)
    rsi = r.get("3D_rsi14", np.nan)
    turnover = r.get("3D_turnover_value", np.nan)

    # 量能 25
    volume_score = 0.0
    if pd.notna(volume_ratio):
        volume_score += 15 * _clip01((volume_ratio - 0.8) / 1.7)
    if pd.notna(volume_5v20):
        volume_score += 10 * _clip01((volume_5v20 - 0.8) / 1.0)

    # 突破位置 20：最偏好接近前20日高點、剛突破，不偏好已遠離
    breakout_score = 0.0
    if pd.notna(breakout):
        if -3 <= breakout <= 3:
            breakout_score = 20 - abs(breakout) * 2.0
        elif -7 <= breakout < -3:
            breakout_score = 10 * _clip01((breakout + 7) / 4)
        elif 3 < breakout <= 7:
            breakout_score = 14 * _clip01((7 - breakout) / 4)

    # 趨勢結構 15
    trend_score = 0.0
    trend_score += 5 * float(r.get("3D_ma5_gt_ma10", 0) == 1)
    trend_score += 5 * float(r.get("3D_ma10_gt_ma20", 0) == 1)
    if pd.notna(ma5s):
        trend_score += 3 * _clip01(ma5s / 3)
    if pd.notna(ma20s):
        trend_score += 2 * _clip01(ma20s / 2)

    # K棒/收盤位置 10
    candle_score = 0.0
    if pd.notna(close_pos):
        candle_score += 6 * _clip01((close_pos - 50) / 45)
    if pd.notna(body):
        candle_score += 4 * _clip01(body / 5)
    if pd.notna(upper) and upper > 4:
        candle_score -= min(3.0, (upper - 4) * 0.5)

    # 波動由壓縮轉擴張 10
    volatility_score = 0.0
    if pd.notna(compression):
        volatility_score += 5 * _clip01((1.15 - compression) / 0.7)
    if pd.notna(atr_accel):
        volatility_score += 5 * _clip01((atr_accel - 0.8) / 0.8)

    # 動能 10：需要轉強，但避免已經噴太多
    momentum_score = 0.0
    if pd.notna(ret1):
        momentum_score += 4 * _clip01((ret1 + 1) / 6)
    if pd.notna(ret3):
        momentum_score += 3 * _clip01((ret3 + 2) / 10)
    if pd.notna(ret5):
        momentum_score += 3 * _clip01((ret5 + 3) / 15)

    # 未過熱 + 流動性 10
    setup_score = 10.0
    if pd.notna(bias20):
        if bias20 > 12:
            setup_score -= min(6, (bias20 - 12) * 0.5)
        elif bias20 < -8:
            setup_score -= min(4, (-8 - bias20) * 0.4)
    if pd.notna(rsi) and rsi > 78:
        setup_score -= min(4, (rsi - 78) * 0.25)
    if pd.notna(ret5) and ret5 > 18:
        setup_score -= min(5, (ret5 - 18) * 0.4)
    if pd.notna(turnover) and turnover < 50_000_000:
        setup_score -= 3

    parts = {
        "量能分": max(0.0, volume_score),
        "突破分": max(0.0, breakout_score),
        "趨勢分": max(0.0, trend_score),
        "K棒分": max(0.0, candle_score),
        "波動分": max(0.0, volatility_score),
        "動能分": max(0.0, momentum_score),
        "未過熱分": max(0.0, setup_score),
    }
    raw = float(sum(parts.values()))

    reasons = []
    risks = []
    if pd.notna(volume_ratio) and volume_ratio >= 1.5:
        reasons.append(f"量比{volume_ratio:.1f}倍")
    if pd.notna(breakout) and -3 <= breakout <= 3:
        reasons.append("接近20日突破位")
    if r.get("3D_ma5_gt_ma10", 0) == 1 and r.get("3D_ma10_gt_ma20", 0) == 1:
        reasons.append("短中期均線多頭")
    if pd.notna(close_pos) and close_pos >= 80:
        reasons.append("收盤接近當日高點")
    if pd.notna(atr_accel) and atr_accel >= 1.15:
        reasons.append("波動開始擴張")
    if pd.notna(compression) and compression <= 0.8:
        reasons.append("短期波動壓縮")
    if pd.notna(ret1) and 1 <= ret1 <= 7:
        reasons.append("單日動能轉強")

    if pd.notna(ret5) and ret5 > 18:
        risks.append("近5日漲幅偏大")
    if pd.notna(bias20) and bias20 > 12:
        risks.append("乖離MA20偏高")
    if pd.notna(rsi) and rsi > 78:
        risks.append("RSI偏熱")
    if pd.notna(upper) and upper > 4:
        risks.append("上影線偏長")
    if pd.notna(turnover) and turnover < 50_000_000:
        risks.append("成交額偏低")

    return raw, parts, reasons, risks


def apply_limitup_3d_model(result_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily cross-sectional ranking for stocks that look ready to launch within 3 trading days.

    This is a signal score, not a calibrated probability and not a guarantee of limit-up.
    It intentionally emphasizes pre-breakout structure, volume expansion and non-overheated setups.
    """
    if result_df is None or result_df.empty:
        return result_df
    if history_df is None or history_df.empty:
        out = result_df.copy()
        out["3日漲停啟動分數"] = np.nan
        out["3日漲停啟動排名"] = np.nan
        out["3日漲停啟動判定"] = "資料不足"
        return out

    h = history_df.copy()
    h["stock_id"] = h["stock_id"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h.dropna(subset=["date"]).sort_values(["stock_id", "date"])

    feature_rows = []
    for code, g in h.groupby("stock_id", sort=False):
        row = _build_feature_row(code, g.tail(80))
        if row:
            feature_rows.append(row)

    feat = pd.DataFrame(feature_rows)
    if feat.empty:
        return result_df

    scored = []
    for _, r in feat.iterrows():
        raw, parts, reasons, risks = _score_row(r)
        x = r.to_dict()
        x.update(parts)
        x["3日漲停啟動原始分數"] = raw
        x["3日啟動理由"] = "、".join(reasons[:5]) if reasons else "尚無明顯啟動訊號"
        x["3日啟動風險"] = "、".join(risks[:4]) if risks else "無明顯短線過熱訊號"
        scored.append(x)

    s = pd.DataFrame(scored)
    s["3日漲停啟動PR"] = s["3日漲停啟動原始分數"].rank(pct=True, method="average") * 100
    s["3日漲停啟動分數"] = (
        0.70 * s["3日漲停啟動原始分數"] + 0.30 * s["3日漲停啟動PR"]
    ).clip(0, 100)
    s["3日漲停啟動排名"] = s["3日漲停啟動分數"].rank(ascending=False, method="first")

    def label(x):
        if pd.isna(x):
            return "⚪ 資料不足"
        if x >= 85:
            return "🔥 極高啟動"
        if x >= 75:
            return "🟢 高啟動"
        if x >= 65:
            return "🟡 中高啟動"
        if x >= 55:
            return "🔵 觀察"
        return "⚪ 尚未啟動"

    s["3日漲停啟動判定"] = s["3日漲停啟動分數"].apply(label)
    s["正式模型版本_3D"] = MODEL_VERSION

    out = result_df.copy()
    out["股票代號"] = out["股票代號"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
    out = out.merge(s, on="股票代號", how="left")

    # 正式接管每日主排名
    out["舊40日主模型分數"] = pd.to_numeric(out.get("主模型分數"), errors="coerce")
    out["主模型分數"] = pd.to_numeric(out["3日漲停啟動分數"], errors="coerce")
    out = out.sort_values(
        ["主模型分數", "3D_volume_ratio", "3D_breakout20"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    out["主模型排名"] = np.arange(1, len(out) + 1)
    out["一週模型排名"] = out["主模型排名"]
    out["正式模型版本"] = MODEL_VERSION
    out["50%潛力判定"] = out["3日漲停啟動判定"]
    out["起漲原因"] = out["3日啟動理由"]
    out["進場風險"] = out["3日啟動風險"]

    return out
