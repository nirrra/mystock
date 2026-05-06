from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .indicators import add_indicators


FEATURE_COLUMNS = [
    "pattern_id_numeric",
    "pattern_recent_frequency_20d",
    "same_symbol_pattern_count_20d",
    "multi_pattern_hit_count",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "distance_to_ma20",
    "distance_to_ma60",
    "ma20_slope",
    "ma60_slope",
    "new_high_20d",
    "new_high_60d",
    "atr_pct",
    "realized_vol_20d",
    "realized_vol_60d",
    "max_drawdown_20d",
    "gap_pct",
    "intraday_range_pct",
    "down_day_count_10d",
    "amount_ma20",
    "avg_amount_20d",
    "amount_ratio_5d_20d",
    "volume_ratio_1d_20d",
    "price_volume_corr_20d",
    "up_volume_share_20d",
    "accumulation_days_20d",
    "distribution_days_20d",
    "limit_up_recent_count",
    "limit_down_recent_count",
    "candidate_count_today",
    "market_breadth_ma20",
]


def build_event_features(labels: pd.DataFrame, daily_history_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()

    prepared_history = {
        str(symbol).zfill(6): _prepare_history(history)
        for symbol, history in daily_history_by_symbol.items()
        if not history.empty
    }
    labels_prepared = labels.copy()
    labels_prepared["symbol"] = labels_prepared["symbol"].astype(str).str.zfill(6)
    labels_prepared["signal_date"] = pd.to_datetime(labels_prepared["signal_date"], errors="coerce")
    candidate_count = labels_prepared.groupby(labels_prepared["signal_date"].dt.date)["event_id"].transform("count")
    labels_prepared["candidate_count_today"] = candidate_count
    rows: list[dict[str, Any]] = []

    for label in labels_prepared.to_dict("records"):
        symbol = str(label["symbol"]).zfill(6)
        signal_day = pd.Timestamp(label["signal_date"]).date()
        history = prepared_history.get(symbol)
        if history is None:
            continue
        signal_index = history.attrs["index_by_date"].get(signal_day)
        if signal_index is None:
            continue
        rows.append({**label, **_feature_row(history, signal_index), "candidate_count_today": label["candidate_count_today"]})

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result = _append_pattern_context(result)
    result = _append_market_context(result)
    return result


def add_rule_baseline_scores(frame: pd.DataFrame, *, min_avg_amount_20d: float) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    atr_p80 = _daily_quantile(result, "atr_pct", 0.80)
    vol_p80 = _daily_quantile(result, "realized_vol_20d", 0.80)

    close_above_ma20 = pd.to_numeric(result["distance_to_ma20"], errors="coerce").gt(0).astype(float)
    ma20_above_ma60 = (
        pd.to_numeric(result["distance_to_ma20"], errors="coerce")
        > pd.to_numeric(result["distance_to_ma60"], errors="coerce")
    ).astype(float)
    ma20_slope = pd.to_numeric(result["ma20_slope"], errors="coerce").gt(0).astype(float)
    trend_score = 0.5 * close_above_ma20 + 0.5 * ma20_above_ma60 + 0.5 * ma20_slope

    momentum_score = _clip(pd.to_numeric(result["return_20d"], errors="coerce") / 0.15, -1, 1) + 0.5 * _clip(
        pd.to_numeric(result["return_60d"], errors="coerce") / 0.30, -1, 1
    )
    volume_score = 0.5 * pd.to_numeric(result["amount_ratio_5d_20d"], errors="coerce").ge(1.1).astype(float) + 0.5 * pd.to_numeric(
        result["up_volume_share_20d"], errors="coerce"
    ).ge(0.55).astype(float)
    liquidity_score = pd.to_numeric(result["avg_amount_20d"], errors="coerce").ge(min_avg_amount_20d).astype(float)
    market_score = 0.5 * pd.to_numeric(result["market_breadth_ma20"], errors="coerce").ge(0.45).astype(float)
    volatility_penalty = 0.5 * pd.to_numeric(result["atr_pct"], errors="coerce").gt(atr_p80).astype(float) + 0.5 * pd.to_numeric(
        result["realized_vol_20d"], errors="coerce"
    ).gt(vol_p80).astype(float)
    overheat_penalty = 0.5 * pd.to_numeric(result["return_5d"], errors="coerce").gt(0.18).astype(float) + 0.5 * pd.to_numeric(
        result["distance_to_ma20"], errors="coerce"
    ).gt(0.18).astype(float)
    gap_penalty = pd.to_numeric(result["gap_pct"], errors="coerce").abs().gt(0.06).astype(float)

    result["rule_score"] = (
        trend_score + momentum_score + volume_score + liquidity_score + market_score - volatility_penalty - overheat_penalty - gap_penalty
    )
    result["rule_risk_pass"] = result["rule_score"] > result.groupby("signal_date")["rule_score"].transform("median")
    result["rule_reason"] = result["rule_risk_pass"].map({True: "rule_pass", False: "rule_block"})
    return result


def _feature_row(history: pd.DataFrame, index: int) -> dict[str, float]:
    row = history.iloc[index]
    close = pd.to_numeric(history["close"], errors="coerce")
    volume = pd.to_numeric(history["volume"], errors="coerce")
    amount = pd.to_numeric(history["amount"], errors="coerce")
    high = pd.to_numeric(history["high"], errors="coerce")
    low = pd.to_numeric(history["low"], errors="coerce")
    returns = close.pct_change()
    start20 = max(0, index - 19)
    window20 = history.iloc[start20 : index + 1]

    previous_close = _safe_float(close.iloc[index - 1]) if index > 0 else math.nan
    open_price = _safe_float(row.get("open"))
    close_price = _safe_float(row.get("close"))
    high_price = _safe_float(row.get("high"))
    low_price = _safe_float(row.get("low"))
    ma20 = _safe_float(row.get("ma_20"))
    ma60 = _safe_float(row.get("ma_60"))

    return {
        "return_5d": _period_return(close, index, 5),
        "return_10d": _period_return(close, index, 10),
        "return_20d": _period_return(close, index, 20),
        "return_60d": _period_return(close, index, 60),
        "distance_to_ma20": close_price / ma20 - 1 if ma20 and ma20 > 0 else math.nan,
        "distance_to_ma60": close_price / ma60 - 1 if ma60 and ma60 > 0 else math.nan,
        "ma20_slope": _slope(history, index, "ma_20", 5),
        "ma60_slope": _slope(history, index, "ma_60", 10),
        "new_high_20d": float(close_price >= _safe_float(close.iloc[max(0, index - 19) : index + 1].max())),
        "new_high_60d": float(close_price >= _safe_float(close.iloc[max(0, index - 59) : index + 1].max())),
        "atr_pct": _safe_float(row.get("atr_pct_14")),
        "realized_vol_20d": _safe_float(returns.iloc[max(0, index - 19) : index + 1].std()) * math.sqrt(20),
        "realized_vol_60d": _safe_float(returns.iloc[max(0, index - 59) : index + 1].std()) * math.sqrt(60),
        "max_drawdown_20d": _max_drawdown(close.iloc[start20 : index + 1]),
        "gap_pct": open_price / previous_close - 1 if previous_close and previous_close > 0 else math.nan,
        "intraday_range_pct": (high_price - low_price) / close_price if close_price and close_price > 0 else math.nan,
        "down_day_count_10d": float((returns.iloc[max(0, index - 9) : index + 1] < 0).sum()),
        "amount_ma20": _safe_float(row.get("amount_ma_20")),
        "avg_amount_20d": _safe_float(amount.iloc[start20 : index + 1].mean()),
        "amount_ratio_5d_20d": _safe_float(amount.iloc[max(0, index - 4) : index + 1].mean())
        / _safe_float(amount.iloc[start20 : index + 1].mean()),
        "volume_ratio_1d_20d": _safe_float(volume.iloc[index]) / _safe_float(volume.iloc[start20 : index + 1].mean()),
        "price_volume_corr_20d": _safe_float(close.iloc[start20 : index + 1].pct_change().corr(volume.iloc[start20 : index + 1].pct_change())),
        "up_volume_share_20d": _up_volume_share(window20),
        "accumulation_days_20d": float(((window20["close"] > window20["open"]) & (window20["volume"] >= window20["volume"].rolling(5).mean())).sum()),
        "distribution_days_20d": float(((window20["close"] < window20["open"]) & (window20["volume"] >= window20["volume"].rolling(5).mean())).sum()),
        "limit_up_recent_count": _limit_count(close, index, direction="up"),
        "limit_down_recent_count": _limit_count(close, index, direction="down"),
    }


def _append_pattern_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["pattern_id_numeric"] = pd.to_numeric(result["pattern_id"], errors="coerce")
    result["multi_pattern_hit_count"] = result.groupby(["signal_date", "symbol"])["event_id"].transform("count")
    result["same_symbol_pattern_count_20d"] = 1.0
    result["pattern_recent_frequency_20d"] = 1.0
    for index, row in result.iterrows():
        signal_date = pd.Timestamp(row["signal_date"])
        lookback_start = signal_date - pd.Timedelta(days=32)
        same_symbol = (
            (result["symbol"] == row["symbol"])
            & (result["pattern_id"].astype(str) == str(row["pattern_id"]))
            & (pd.to_datetime(result["signal_date"]) <= signal_date)
            & (pd.to_datetime(result["signal_date"]) >= lookback_start)
        )
        same_pattern = (
            (result["pattern_id"].astype(str) == str(row["pattern_id"]))
            & (pd.to_datetime(result["signal_date"]) <= signal_date)
            & (pd.to_datetime(result["signal_date"]) >= lookback_start)
        )
        result.at[index, "same_symbol_pattern_count_20d"] = float(same_symbol.sum())
        result.at[index, "pattern_recent_frequency_20d"] = float(same_pattern.sum())
    result["sample_weight"] = 1.0 / result["multi_pattern_hit_count"].clip(lower=1)
    return result


def _append_market_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    breadth = result.groupby("signal_date")["distance_to_ma20"].apply(lambda value: float(pd.to_numeric(value, errors="coerce").gt(0).mean()))
    result["market_breadth_ma20"] = result["signal_date"].map(breadth)
    return result


def _prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    prepared = add_indicators(history) if "ma_20" not in history.columns or "atr_14" not in history.columns else history.copy()
    prepared = prepared.sort_values("trade_date").reset_index(drop=True)
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"])
    prepared.attrs["index_by_date"] = {item.date(): index for index, item in enumerate(prepared["trade_date"])}
    return prepared


def _period_return(close: pd.Series, index: int, days: int) -> float:
    if index - days < 0:
        return math.nan
    previous = _safe_float(close.iloc[index - days])
    current = _safe_float(close.iloc[index])
    return current / previous - 1 if previous and previous > 0 else math.nan


def _slope(history: pd.DataFrame, index: int, column: str, lookback: int) -> float:
    if index - lookback < 0 or column not in history.columns:
        return math.nan
    previous = _safe_float(history[column].iloc[index - lookback])
    current = _safe_float(history[column].iloc[index])
    return current / previous - 1 if previous and previous > 0 else math.nan


def _max_drawdown(close: pd.Series) -> float:
    values = pd.to_numeric(close, errors="coerce").dropna()
    if values.empty:
        return math.nan
    drawdown = 1 - values / values.cummax()
    return _safe_float(drawdown.max())


def _limit_count(close: pd.Series, index: int, *, direction: str) -> float:
    returns = pd.to_numeric(close, errors="coerce").pct_change().iloc[max(0, index - 19) : index + 1]
    if direction == "up":
        return float((returns >= 0.095).sum())
    return float((returns <= -0.095).sum())


def _up_volume_share(window: pd.DataFrame) -> float:
    if window.empty:
        return math.nan
    volume = pd.to_numeric(window["volume"], errors="coerce")
    up_volume = volume[window["close"] > window["open"]].sum()
    total = volume.sum()
    return float(up_volume / total) if total else math.nan


def _daily_quantile(frame: pd.DataFrame, column: str, quantile: float) -> pd.Series:
    return frame.groupby("signal_date")[column].transform(lambda value: pd.to_numeric(value, errors="coerce").quantile(quantile))


def _clip(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return series.clip(lower=lower, upper=upper).fillna(0.0)


def _safe_float(value: object) -> float:
    if value is None or pd.isna(value):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan
