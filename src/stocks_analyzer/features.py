from __future__ import annotations

import math

import pandas as pd

from .indicators import add_indicators
from .technical_ratings import add_technical_ratings


IDENTIFIER_COLUMNS = ("trade_date", "symbol", "name")
TARGET_COLUMNS = (
    "future_20d_return",
    "future_20d_max_drawdown",
    "future_20d_max_upside",
    "future_20d_min_return",
    "label_stable_up",
)


def build_feature_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(dataframe)
    df = _add_extended_features(df)
    df = add_technical_ratings(df)
    return df


def numeric_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    excluded = set(IDENTIFIER_COLUMNS) | set(TARGET_COLUMNS)
    columns: list[str] = []
    for column in dataframe.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            columns.append(column)
    return columns


def _add_extended_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy().sort_values("trade_date").reset_index(drop=True)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    amount = df["amount"].astype(float)

    for period in (3, 5, 10, 20, 30, 50, 60, 100, 200):
        if f"ma_{period}" not in df.columns:
            df[f"ma_{period}"] = close.rolling(period).mean()
        df[f"return_{period}d"] = close.pct_change(period)

    for period in (5, 10, 20, 60):
        df[f"volatility_{period}d"] = df["return_1d"].rolling(period).std() * math.sqrt(period)
        df[f"close_max_{period}"] = close.rolling(period).max()
        df[f"close_min_{period}"] = close.rolling(period).min()
        df[f"drawdown_{period}d"] = 1 - close.div(df[f"close_max_{period}"])
        df[f"position_in_range_{period}d"] = (close - df[f"close_min_{period}"]).div(
            (df[f"close_max_{period}"] - df[f"close_min_{period}"]).replace(0.0, pd.NA)
        )

    for period in (5, 10, 20):
        df[f"volume_ma_{period}"] = volume.rolling(period).mean()
        df[f"amount_ma_{period}"] = amount.rolling(period).mean()
        df[f"volume_ratio_{period}"] = volume.div(df[f"volume_ma_{period}"])
        df[f"amount_ratio_{period}"] = amount.div(df[f"amount_ma_{period}"])

    for period in (5, 10, 20, 60, 100, 200):
        df[f"distance_to_ma{period}"] = close.div(df[f"ma_{period}"]) - 1

    candle_range = (high - low).replace(0.0, pd.NA)
    df["intraday_range_pct"] = candle_range.div(close.replace(0.0, pd.NA))
    df["body_pct"] = (close - df["open"].astype(float)).abs().div(candle_range)
    df["upper_shadow_pct"] = (high - df[["open", "close"]].max(axis=1)).div(candle_range)
    df["lower_shadow_pct"] = (df[["open", "close"]].min(axis=1) - low).div(candle_range)

    up = close > close.shift(1)
    down = close < close.shift(1)
    df["up_day"] = up.astype(float)
    df["down_day"] = down.astype(float)
    df["up_days_5"] = up.rolling(5).sum()
    df["down_days_5"] = down.rolling(5).sum()
    df["up_days_10"] = up.rolling(10).sum()
    df["down_days_10"] = down.rolling(10).sum()

    return df
