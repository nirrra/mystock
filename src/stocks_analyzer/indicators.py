from __future__ import annotations

import math

import pandas as pd


def add_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy().sort_values("trade_date").reset_index(drop=True)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    amount = df["amount"]

    df["ma_5"] = close.rolling(5).mean()
    df["ma_10"] = close.rolling(10).mean()
    df["ma_20"] = close.rolling(20).mean()
    df["ma_60"] = close.rolling(60).mean()

    df["close_max_20"] = close.rolling(20).max()
    df["is_20d_high"] = close >= df["close_max_20"]
    df["volume_ma_20"] = volume.rolling(20).mean()
    df["amount_ma_20"] = amount.rolling(20).mean()

    df["return_1d"] = close.pct_change()
    df["return_15d"] = close.pct_change(15)
    df["drawdown_10d"] = 1 - close.div(close.rolling(10).max())
    df["volatility_10d"] = df["return_1d"].rolling(10).std() * math.sqrt(10)

    df["distance_to_ma20"] = close.div(df["ma_20"]) - 1
    df["volume_ratio_20"] = volume.div(df["volume_ma_20"])
    df["volume_ratio_3d"] = volume.rolling(3).mean().div(df["volume_ma_20"])
    df["consolidation_range_3d"] = high.rolling(3).max().sub(low.rolling(3).min()).div(close)
    df["ema_12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["macd_dif"] = df["ema_12"] - df["ema_26"]
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd_hist"] = (df["macd_dif"] - df["macd_dea"]) * 2
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high.sub(low),
            high.sub(prev_close).abs(),
            low.sub(prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["true_range"] = true_range
    df["atr_14"] = _wilder_average(true_range, period=14)
    df["atr_pct_14"] = df["atr_14"].div(close.replace(0, pd.NA))
    df["atr_stop_loss_1x"] = close.sub(df["atr_14"])
    df["atr_stop_loss_2x"] = close.sub(df["atr_14"] * 2.0)
    df["atr_take_profit_2x"] = close.add(df["atr_14"] * 2.0)
    df["atr_take_profit_3x"] = close.add(df["atr_14"] * 3.0)
    df["atr_volatility_regime"] = df["atr_pct_14"].map(_classify_atr_regime)

    return df


def _wilder_average(series: pd.Series, *, period: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").tolist()
    result: list[float] = [math.nan] * len(values)
    previous: float | None = None
    window: list[float] = []

    for index, value in enumerate(values):
        if pd.isna(value):
            continue
        numeric = float(value)
        window.append(numeric)
        if len(window) < period:
            continue
        if previous is None:
            previous = sum(window[-period:]) / period
            result[index] = previous
            continue
        previous = ((previous * (period - 1)) + numeric) / period
        result[index] = previous

    return pd.Series(result, index=series.index, dtype="float64")


def _classify_atr_regime(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    if numeric < 0.03:
        return "低波动"
    if numeric < 0.06:
        return "中等波动"
    return "高波动"
