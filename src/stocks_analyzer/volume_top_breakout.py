from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .indicators import add_indicators


@dataclass(slots=True)
class VolumeTopBreakoutConfig:
    min_old_high_gap_days: int
    min_drawdown_pct: float
    peak_window_days: int
    breakout_volume_lookback_days: int
    breakout_volume_multiplier: float
    require_break_below_ma60: bool = True


@dataclass(slots=True)
class VolumeTopBreakoutEvent:
    old_high_index: int
    old_high_date: str
    old_high_price: float
    days_since_old_high: int
    max_drawdown_since_old_high: float
    breakout_index: int | None
    breakout_date: str | None
    breakout_volume_ratio: float | None


def detect_volume_top_breakout(
    dataframe: pd.DataFrame,
    config: VolumeTopBreakoutConfig,
) -> tuple[pd.DataFrame, VolumeTopBreakoutEvent | None]:
    df = _prepare_frame(dataframe)
    latest_index = len(df) - 1
    minimum_length = max(
        config.min_old_high_gap_days + 1,
        2 * config.peak_window_days + 1,
        config.breakout_volume_lookback_days + 1,
        60,
    )
    if len(df) < minimum_length:
        return df, None

    for old_high in _iter_recent_old_high_candidates(df, config):
        old_high_index = int(old_high["index"])
        old_high_price = float(old_high["old_high_price"])
        breakout = _find_first_breakout_day(
            df,
            old_high_index=old_high_index,
            old_high_price=old_high_price,
            config=config,
        )
        if breakout is None:
            if _has_strictly_higher_high_between(
                df,
                start_index=old_high_index + 1,
                end_index=latest_index,
                threshold=old_high_price,
            ):
                continue
        else:
            if _has_strictly_higher_high_between(
                df,
                start_index=old_high_index + 1,
                end_index=int(breakout["index"]) - 1,
                threshold=old_high_price,
            ):
                continue

        event = VolumeTopBreakoutEvent(
            old_high_index=old_high_index,
            old_high_date=str(old_high["old_high_date"]),
            old_high_price=old_high_price,
            days_since_old_high=latest_index - old_high_index,
            max_drawdown_since_old_high=float(old_high["max_drawdown_since_old_high"]),
            breakout_index=None if breakout is None else int(breakout["index"]),
            breakout_date=None if breakout is None else str(breakout["breakout_date"]),
            breakout_volume_ratio=None if breakout is None else float(breakout["breakout_volume_ratio"]),
        )
        return df, event

    return df, None


def _prepare_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    required = {"ma_20", "ma_60"}
    if required.issubset(dataframe.columns):
        return dataframe.copy().sort_values("trade_date").reset_index(drop=True)
    return add_indicators(dataframe).sort_values("trade_date").reset_index(drop=True)


def _iter_recent_old_high_candidates(df: pd.DataFrame, config: VolumeTopBreakoutConfig):
    latest_index = len(df) - 1
    for index in range(latest_index - config.min_old_high_gap_days, config.peak_window_days - 1, -1):
        if index + config.peak_window_days >= len(df):
            continue
        if not _is_local_peak(df, index, config.peak_window_days):
            continue

        old_high_price = float(df.iloc[index]["high"])
        if old_high_price <= 0:
            continue

        subsequent = df.iloc[index + 1 :].reset_index(drop=True)
        if subsequent.empty:
            continue

        subsequent_min_low = float(subsequent["low"].astype(float).min())
        drawdown = (old_high_price - subsequent_min_low) / old_high_price
        if drawdown < config.min_drawdown_pct:
            continue

        if config.require_break_below_ma60 and not _fell_below_ma60(df.iloc[index + 1 :]):
            continue

        yield {
            "index": index,
            "old_high_date": pd.Timestamp(df.iloc[index]["trade_date"]).date().isoformat(),
            "old_high_price": old_high_price,
            "max_drawdown_since_old_high": drawdown,
        }


def _is_local_peak(df: pd.DataFrame, index: int, peak_window_days: int) -> bool:
    left = index - peak_window_days
    right = index + peak_window_days
    if left < 0 or right >= len(df):
        return False
    highs = df.iloc[left : right + 1]["high"].astype(float)
    current_high = float(df.iloc[index]["high"])
    return current_high >= float(highs.max())


def _fell_below_ma60(df: pd.DataFrame) -> bool:
    window = df.loc[:, ["low", "ma_60"]].copy()
    window["low"] = pd.to_numeric(window["low"], errors="coerce")
    window["ma_60"] = pd.to_numeric(window["ma_60"], errors="coerce")
    window = window.dropna(subset=["low", "ma_60"])
    if window.empty:
        return False
    return bool((window["low"] < window["ma_60"]).any())


def _has_strictly_higher_high_between(
    df: pd.DataFrame,
    *,
    start_index: int,
    end_index: int,
    threshold: float,
) -> bool:
    if end_index < start_index:
        return False

    highs = df.iloc[start_index : end_index + 1]["high"].astype(float)
    if highs.empty:
        return False
    return bool((highs > threshold).any())


def _find_first_breakout_day(
    df: pd.DataFrame,
    *,
    old_high_index: int,
    old_high_price: float,
    config: VolumeTopBreakoutConfig,
) -> dict[str, object] | None:
    for index in range(old_high_index + 1, len(df)):
        row = df.iloc[index]
        open_price = float(row["open"])
        close_price = float(row["close"])
        high_price = float(row["high"])
        if close_price <= open_price:
            continue
        if high_price <= old_high_price:
            continue

        baseline_start = index - config.breakout_volume_lookback_days
        if baseline_start < 0:
            continue
        baseline = df.iloc[baseline_start:index]["volume"].astype(float)
        baseline_average = float(baseline.mean()) if not baseline.empty else 0.0
        if baseline_average <= 0:
            continue

        volume = float(row["volume"])
        volume_ratio = volume / baseline_average
        if volume_ratio < config.breakout_volume_multiplier:
            continue

        return {
            "index": index,
            "breakout_date": pd.Timestamp(row["trade_date"]).date().isoformat(),
            "breakout_volume_ratio": volume_ratio,
        }

    return None
