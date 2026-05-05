from __future__ import annotations

import math
import re

import pandas as pd

from .indicators import add_indicators
from .technical_ratings import add_technical_ratings


IDENTIFIER_COLUMNS = ("trade_date", "symbol", "name")
TARGET_COLUMNS = (
    "entry_open",
    "outcome_days",
    "future_return",
    "future_max_drawdown",
    "future_max_upside",
    "future_min_return",
    "horizon_weight",
    "future_20d_return",
    "future_20d_max_drawdown",
    "future_20d_max_upside",
    "future_20d_min_return",
    "label_stable_up",
    "label_tp10_sl8_20d",
    "outcome_class",
    "label_conflict",
)
TRADINGVIEW_AGGREGATE_COLUMNS = (
    "ma_rating",
    "osc_rating",
    "all_rating",
    "avg_ma_rating_5d",
    "avg_osc_rating_5d",
    "avg_all_rating_5d",
)


def build_feature_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(dataframe)
    df = _add_extended_features(df)
    df = add_technical_ratings(df)
    df = _add_higher_timeframe_features(df)
    return df


def numeric_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    excluded = set(IDENTIFIER_COLUMNS) | set(TARGET_COLUMNS) | set(TRADINGVIEW_AGGREGATE_COLUMNS)
    columns: list[str] = []
    for column in dataframe.columns:
        if column in excluded:
            continue
        if _is_generated_target_column(column):
            continue
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            columns.append(column)
    return columns


def _is_generated_target_column(column: str) -> bool:
    return bool(re.fullmatch(r"future_\d+d_(return|max_drawdown|max_upside|min_return)", column))


def _add_extended_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy().sort_values("trade_date").reset_index(drop=True)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    amount = df["amount"].astype(float)

    for period in (3, 5, 10, 20, 30, 50, 60, 100, 120, 200):
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

    for period in (120, 250):
        df[f"close_max_{period}"] = close.rolling(period).max()
        df[f"close_min_{period}"] = close.rolling(period).min()
        df[f"distance_to_{period}d_high"] = close.div(df[f"close_max_{period}"]) - 1
        df[f"distance_to_{period}d_low"] = close.div(df[f"close_min_{period}"]) - 1

    for period in (5, 10, 20):
        df[f"volume_ma_{period}"] = volume.rolling(period).mean()
        df[f"amount_ma_{period}"] = amount.rolling(period).mean()
        df[f"volume_ratio_{period}"] = volume.div(df[f"volume_ma_{period}"])
        df[f"amount_ratio_{period}"] = amount.div(df[f"amount_ma_{period}"])

    for period in (5, 10, 20, 60, 100, 120, 200):
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
    df["atr_14_pct"] = df["atr_pct_14"] if "atr_pct_14" in df.columns else pd.NA
    df["long_downtrend_repair_120d"] = close.div(df["ma_120"].replace(0.0, pd.NA)) - 1 if "ma_120" in df.columns else pd.NA
    df["ma60_to_ma120"] = df["ma_60"].div(df["ma_120"].replace(0.0, pd.NA)) - 1 if "ma_120" in df.columns else pd.NA

    return df


def _add_higher_timeframe_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy().sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = _merge_period_features(
        df,
        freq="W-FRI",
        prefix="weekly",
        suffix="w",
        return_windows=(1, 4, 8, 12),
        ma_windows=(5, 10, 20),
        volume_windows=(4, 8, 12),
        range_windows=(12, 24),
        include_adx=True,
    )
    df = _merge_period_features(
        df,
        freq="M",
        prefix="monthly",
        suffix="m",
        return_windows=(1, 3, 6, 12),
        ma_windows=(3, 6, 12),
        volume_windows=(3, 6, 12),
        range_windows=(6, 12),
        include_adx=False,
    )
    return df


def _merge_period_features(
    dataframe: pd.DataFrame,
    *,
    freq: str,
    prefix: str,
    suffix: str,
    return_windows: tuple[int, ...],
    ma_windows: tuple[int, ...],
    volume_windows: tuple[int, ...],
    range_windows: tuple[int, ...],
    include_adx: bool,
) -> pd.DataFrame:
    period_bars = _build_period_bars(dataframe, freq=freq)
    if period_bars.empty:
        return dataframe

    period_features = _build_period_feature_frame(
        period_bars,
        prefix=prefix,
        suffix=suffix,
        return_windows=return_windows,
        ma_windows=ma_windows,
        volume_windows=volume_windows,
        range_windows=range_windows,
        include_adx=include_adx,
    )
    merge_columns = ["trade_date", *[column for column in period_features.columns if column.startswith(f"{prefix}_")]]
    merged = pd.merge_asof(
        dataframe.sort_values("trade_date"),
        period_features.loc[:, merge_columns].sort_values("trade_date"),
        on="trade_date",
        direction="backward",
    )
    return merged.sort_values("trade_date").reset_index(drop=True)


def _build_period_bars(dataframe: pd.DataFrame, *, freq: str) -> pd.DataFrame:
    frame = dataframe.copy().sort_values("trade_date").reset_index(drop=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["_period"] = frame["trade_date"].dt.to_period(freq)
    grouped = frame.groupby("_period", sort=True)
    bars = grouped.agg(
        trade_date=("trade_date", "max"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    ).reset_index(drop=True)
    return bars.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)


def _build_period_feature_frame(
    period_bars: pd.DataFrame,
    *,
    prefix: str,
    suffix: str,
    return_windows: tuple[int, ...],
    ma_windows: tuple[int, ...],
    volume_windows: tuple[int, ...],
    range_windows: tuple[int, ...],
    include_adx: bool,
) -> pd.DataFrame:
    frame = add_technical_ratings(period_bars).copy()
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    amount = pd.to_numeric(frame["amount"], errors="coerce")

    frame[f"{prefix}_close"] = close
    frame[f"{prefix}_volume"] = volume
    frame[f"{prefix}_amount"] = amount

    for window in return_windows:
        frame[f"{prefix}_return_{window}{suffix}"] = close.pct_change(window)

    for window in ma_windows:
        ma = close.rolling(window).mean()
        frame[f"{prefix}_ma_{window}"] = ma
        frame[f"{prefix}_distance_to_ma{window}"] = close.div(ma.replace(0.0, pd.NA)) - 1
        frame[f"{prefix}_ma{window}_slope"] = ma.pct_change(fill_method=None)

    for window in volume_windows:
        volume_ma = volume.rolling(window).mean()
        amount_ma = amount.rolling(window).mean()
        frame[f"{prefix}_volume_ma_{window}{suffix}"] = volume_ma
        frame[f"{prefix}_amount_ma_{window}{suffix}"] = amount_ma
        frame[f"{prefix}_volume_ratio_{window}{suffix}"] = volume.div(volume_ma.replace(0.0, pd.NA))
        frame[f"{prefix}_amount_ratio_{window}{suffix}"] = amount.div(amount_ma.replace(0.0, pd.NA))

    for window in range_windows:
        high = close.rolling(window).max()
        low = close.rolling(window).min()
        frame[f"{prefix}_position_{window}{suffix}"] = (close - low).div((high - low).replace(0.0, pd.NA))
        frame[f"{prefix}_distance_to_{window}{suffix}_high"] = close.div(high.replace(0.0, pd.NA)) - 1
        frame[f"{prefix}_drawdown_{window}{suffix}"] = 1 - close.div(high.replace(0.0, pd.NA))

    frame[f"{prefix}_rsi_14"] = frame["rsi_14"]
    frame[f"{prefix}_macd"] = frame["macd"]
    frame[f"{prefix}_macd_signal"] = frame["macd_signal_line"]
    frame[f"{prefix}_macd_hist"] = frame["macd_hist"]
    if include_adx:
        frame[f"{prefix}_adx_14"] = frame["adx_14"]

    selected = ["trade_date", *[column for column in frame.columns if column.startswith(f"{prefix}_")]]
    return frame.loc[:, selected]
