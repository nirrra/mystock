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

    return df
