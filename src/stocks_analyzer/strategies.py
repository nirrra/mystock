from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .models import AppConfig, Type1Config, Type2Config, Type3Config, Type4Config, Type5Config, Type6Config
from .volume_top_breakout import VolumeTopBreakoutConfig, VolumeTopBreakoutEvent, detect_volume_top_breakout


VOLUME_TOP_PRE_BREAKOUT = "volume_top_pre_breakout"
VOLUME_TOP_BREAKOUT = "volume_top_breakout"
VOLUME_TOP_FOLLOW_THROUGH = "volume_top_follow_through"
PLATFORM_BREAKOUT = "platform_breakout"
TREND_PULLBACK = "trend_pullback"
SECOND_WAVE = "second_wave"

STRATEGY_NAMES = (
    VOLUME_TOP_PRE_BREAKOUT,
    VOLUME_TOP_BREAKOUT,
    VOLUME_TOP_FOLLOW_THROUGH,
    PLATFORM_BREAKOUT,
    TREND_PULLBACK,
    SECOND_WAVE,
)


def evaluate_strategies(
    history_df: pd.DataFrame,
    instrument: dict[str, object],
    config: AppConfig,
    selected: Sequence[str],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    symbol = str(instrument["symbol"])
    name = str(instrument["name"])

    if not _has_recent_short_term_momentum_history(history_df, config):
        return results

    volume_top_context: tuple[pd.DataFrame, VolumeTopBreakoutEvent | None] | None = None
    if any(item in selected for item in (VOLUME_TOP_PRE_BREAKOUT, VOLUME_TOP_BREAKOUT, VOLUME_TOP_FOLLOW_THROUGH)):
        volume_top_context = detect_volume_top_breakout(history_df, _to_volume_top_config(config.type1))

    if VOLUME_TOP_PRE_BREAKOUT in selected:
        match = _apply_volume_top_pre_breakout(history_df, symbol, name, config.type1, volume_top_context)
        if match is not None:
            results.append(match)
    if VOLUME_TOP_BREAKOUT in selected:
        match = _apply_volume_top_breakout(history_df, symbol, name, config.type2, volume_top_context)
        if match is not None:
            results.append(match)
    if VOLUME_TOP_FOLLOW_THROUGH in selected:
        match = _apply_volume_top_follow_through(history_df, symbol, name, config.type3, volume_top_context)
        if match is not None:
            results.append(match)
    if PLATFORM_BREAKOUT in selected:
        match = _apply_type4(history_df, symbol, name, config.type4)
        if match is not None:
            results.append(match)
    if TREND_PULLBACK in selected:
        match = _apply_type5(history_df, symbol, name, config.type5)
        if match is not None:
            results.append(match)
    if SECOND_WAVE in selected:
        match = _apply_type6(history_df, symbol, name, config.type6)
        if match is not None:
            results.append(match)

    return results


def required_history_days(config: AppConfig, selected: Sequence[str]) -> int:
    requirements = [config.universe.min_history_days, config.history_momentum_filter.lookback_days]
    if any(item in selected for item in (VOLUME_TOP_PRE_BREAKOUT, VOLUME_TOP_BREAKOUT, VOLUME_TOP_FOLLOW_THROUGH)):
        volume_top = _to_volume_top_config(config.type1)
        requirements.append(max(60, volume_top.min_old_high_gap_days + 2 * volume_top.peak_window_days + 1))
        requirements.append(volume_top.breakout_volume_lookback_days + 1)
    if PLATFORM_BREAKOUT in selected:
        requirements.append(
            max(
                60,
                config.type4.trend_lookback_days + 1,
                config.type4.ma60_rising_lookback + 1,
                config.type4.platform_window_days + 1,
            )
        )
    if TREND_PULLBACK in selected:
        requirements.append(
            max(
                60,
                config.type5.trend_lookback_days + 1,
                config.type5.ma_rising_lookback + 1,
                20,
            )
        )
    if SECOND_WAVE in selected:
        requirements.append(max(20, config.type6.strong_lookback_days + 1, 2 * config.type6.consolidation_max_days + 1))
    return max(requirements)


def _has_recent_short_term_momentum_history(history_df: pd.DataFrame, config: AppConfig) -> bool:
    filter_config = config.history_momentum_filter
    lookback_days = int(filter_config.lookback_days)
    window_days = int(filter_config.window_days)
    min_return = float(filter_config.min_return)

    if lookback_days <= 0 or window_days <= 1 or min_return <= 0:
        return True
    if len(history_df) < lookback_days:
        return False

    recent = history_df["close"].astype(float).tail(lookback_days)
    window_returns = recent / recent.shift(window_days - 1) - 1
    window_returns = window_returns.dropna()
    if window_returns.empty:
        return False
    return bool((window_returns >= min_return).any())


def _apply_volume_top_pre_breakout(
    history_df: pd.DataFrame,
    symbol: str,
    name: str,
    config: Type1Config,
    context: tuple[pd.DataFrame, VolumeTopBreakoutEvent | None] | None,
) -> dict[str, object] | None:
    _, event = context or detect_volume_top_breakout(history_df, _to_volume_top_config(config))
    if event is None or event.breakout_index is not None:
        return None

    latest = history_df.iloc[-1]
    close_price = float(latest["close"])
    if close_price > event.old_high_price:
        return None

    distance_pct = (event.old_high_price - close_price) / event.old_high_price
    if distance_pct > config.near_high_threshold_pct:
        return None

    reason = (
        f"near old high {event.old_high_price:.2f} from {event.old_high_date}, "
        f"distance={distance_pct:.2%}, drawdown={event.max_drawdown_since_old_high:.2%}"
    )
    return _build_result(
        history_df=history_df,
        symbol=symbol,
        name=name,
        strategy_name=VOLUME_TOP_PRE_BREAKOUT,
        reason=reason,
        old_high_date=event.old_high_date,
        old_high_price=event.old_high_price,
        days_since_old_high=event.days_since_old_high,
        max_drawdown_since_old_high=event.max_drawdown_since_old_high,
        distance_to_old_high_pct=distance_pct,
        breakout_date=None,
        breakout_volume_ratio=None,
        extension_above_old_high_pct=0.0,
    )


def _apply_volume_top_breakout(
    history_df: pd.DataFrame,
    symbol: str,
    name: str,
    config: Type2Config,
    context: tuple[pd.DataFrame, VolumeTopBreakoutEvent | None] | None,
) -> dict[str, object] | None:
    _, event = context or detect_volume_top_breakout(history_df, _to_volume_top_config(config))
    if event is None or event.breakout_index is None or event.breakout_index != len(history_df) - 1:
        return None

    reason = (
        f"volume breakout above old high {event.old_high_price:.2f} from {event.old_high_date}, "
        f"vol_ratio={float(event.breakout_volume_ratio or 0.0):.2f}"
    )
    return _build_result(
        history_df=history_df,
        symbol=symbol,
        name=name,
        strategy_name=VOLUME_TOP_BREAKOUT,
        reason=reason,
        old_high_date=event.old_high_date,
        old_high_price=event.old_high_price,
        days_since_old_high=event.days_since_old_high,
        max_drawdown_since_old_high=event.max_drawdown_since_old_high,
        distance_to_old_high_pct=(event.old_high_price - float(history_df.iloc[-1]["close"])) / event.old_high_price,
        breakout_date=event.breakout_date,
        breakout_volume_ratio=event.breakout_volume_ratio,
        extension_above_old_high_pct=max(0.0, float(history_df.iloc[-1]["close"]) / event.old_high_price - 1.0),
    )


def _apply_volume_top_follow_through(
    history_df: pd.DataFrame,
    symbol: str,
    name: str,
    config: Type3Config,
    context: tuple[pd.DataFrame, VolumeTopBreakoutEvent | None] | None,
) -> dict[str, object] | None:
    _, event = context or detect_volume_top_breakout(history_df, _to_volume_top_config(config))
    if event is None or event.breakout_index is None:
        return None

    current_index = len(history_df) - 1
    days_after_breakout = current_index - event.breakout_index
    if days_after_breakout < 1 or days_after_breakout > config.post_breakout_max_days:
        return None

    latest = history_df.iloc[-1]
    close_price = float(latest["close"])
    extension_pct = close_price / event.old_high_price - 1.0
    if extension_pct > config.post_breakout_max_extension_pct:
        return None

    ma20 = float(latest["ma_20"]) if pd.notna(latest.get("ma_20")) else float("nan")
    low_price = float(latest["low"])
    if pd.notna(ma20) and low_price < ma20 and close_price <= ma20:
        return None

    reason = (
        f"post breakout {days_after_breakout}d from {event.breakout_date}, "
        f"extension={extension_pct:.2%}, vol_ratio={float(event.breakout_volume_ratio or 0.0):.2f}"
    )
    return _build_result(
        history_df=history_df,
        symbol=symbol,
        name=name,
        strategy_name=VOLUME_TOP_FOLLOW_THROUGH,
        reason=reason,
        old_high_date=event.old_high_date,
        old_high_price=event.old_high_price,
        days_since_old_high=event.days_since_old_high,
        max_drawdown_since_old_high=event.max_drawdown_since_old_high,
        distance_to_old_high_pct=(event.old_high_price - close_price) / event.old_high_price,
        breakout_date=event.breakout_date,
        breakout_volume_ratio=event.breakout_volume_ratio,
        extension_above_old_high_pct=max(0.0, extension_pct),
        days_after_breakout=days_after_breakout,
    )


def _apply_type4(history_df: pd.DataFrame, symbol: str, name: str, config: Type4Config) -> dict[str, object] | None:
    minimum_length = max(60, config.trend_lookback_days + 1, config.platform_window_days + config.breakout_lookback_days + 1)
    if len(history_df) < minimum_length:
        return None

    latest = history_df.iloc[-1]
    if pd.isna(latest["ma_20"]) or pd.isna(latest["ma_60"]) or pd.isna(latest["volume_ratio_20"]):
        return None

    if pd.isna(latest["ma_5"]):
        return None
    if not (float(latest["ma_5"]) > float(latest["ma_20"]) > float(latest["ma_60"])):
        return None

    trend_return = _lookback_return(history_df, config.trend_lookback_days)
    if trend_return is None or trend_return < config.min_return_trend_lookback:
        return None

    breakout_event = _find_recent_platform_breakout(history_df, config)
    if breakout_event is None:
        return None

    platform_high = breakout_event["platform_high"]
    distance_pct = 0.0 if platform_high <= 0 else (float(latest["close"]) - platform_high) / platform_high
    if distance_pct < config.breakout_min_distance_pct or distance_pct > config.breakout_max_distance_pct:
        return None

    reason = (
        f"recent platform breakout on {breakout_event['breakout_date']}, "
        f"platform_range={breakout_event['platform_range_pct']:.2%}, distance={distance_pct:.2%}"
    )
    return _build_result(
        history_df=history_df,
        symbol=symbol,
        name=name,
        strategy_name=PLATFORM_BREAKOUT,
        reason=reason,
        platform_window_days=config.platform_window_days,
        platform_range_pct=breakout_event["platform_range_pct"],
        breakout_date=breakout_event["breakout_date"],
        distance_to_platform_high_pct=distance_pct,
    )


def _apply_type5(history_df: pd.DataFrame, symbol: str, name: str, config: Type5Config) -> dict[str, object] | None:
    if len(history_df) < max(60, config.trend_lookback_days + 1, config.ma_rising_lookback + 1, 20):
        return None

    latest = history_df.iloc[-1]
    if pd.isna(latest["ma_20"]) or pd.isna(latest["ma_60"]) or pd.isna(latest["distance_to_ma20"]):
        return None

    ma20_prev = history_df["ma_20"].shift(config.ma_rising_lookback).iloc[-1]
    ma60_prev = history_df["ma_60"].shift(config.ma_rising_lookback).iloc[-1]
    if pd.isna(ma20_prev) or pd.isna(ma60_prev):
        return None
    if float(latest["ma_20"]) <= float(latest["ma_60"]):
        return None
    if float(latest["ma_20"]) < float(ma20_prev) or float(latest["ma_60"]) < float(ma60_prev):
        return None

    trend_return = _lookback_return(history_df, config.trend_lookback_days)
    if trend_return is None or trend_return < config.min_return_trend_lookback:
        return None

    distance_to_ma20 = float(latest["distance_to_ma20"])
    if abs(distance_to_ma20) > config.proximity_to_ma20:
        return None
    if float(latest["close"]) < float(latest["ma_20"]) or float(latest["close"]) < float(latest["ma_60"]):
        return None

    recent_peak = float(history_df["close"].tail(15).max())
    drawdown_15d = 0.0 if recent_peak <= 0 else 1 - float(latest["close"]) / recent_peak
    if drawdown_15d > config.max_drawdown_15d:
        return None

    recent_volume_5d = float(history_df["volume"].tail(5).mean())
    recent_volume_20d = float(history_df["volume"].tail(20).mean())
    if recent_volume_20d <= 0:
        return None
    volume_contraction_ratio = recent_volume_5d / recent_volume_20d
    if volume_contraction_ratio > config.volume_contraction_max:
        return None

    reason = (
        f"trend pullback near MA20: distance={distance_to_ma20:.2%}, "
        f"drawdown_15d={drawdown_15d:.2%}, vol_5d/20d={volume_contraction_ratio:.2f}"
    )
    return _build_result(
        history_df=history_df,
        symbol=symbol,
        name=name,
        strategy_name=TREND_PULLBACK,
        reason=reason,
        distance_to_ma20=distance_to_ma20,
        drawdown_15d=drawdown_15d,
        volume_contraction_ratio=volume_contraction_ratio,
    )


def _apply_type6(history_df: pd.DataFrame, symbol: str, name: str, config: Type6Config) -> dict[str, object] | None:
    if len(history_df) < max(20, config.strong_lookback_days + 1, 2 * config.consolidation_max_days + 1):
        return None

    latest = history_df.iloc[-1]
    if pd.isna(latest["ma_10"]) or pd.isna(latest["ma_20"]):
        return None
    if float(latest["close"]) < float(latest["ma_10"]) or float(latest["ma_10"]) < float(latest["ma_20"]):
        return None

    strong_return = _lookback_return(history_df, config.strong_lookback_days)
    if strong_return is None or strong_return < config.min_return_strong_lookback:
        return None

    strong_day_return = float(history_df["return_1d"].tail(config.strong_lookback_days).max())
    if pd.isna(strong_day_return) or strong_day_return < config.strong_day_return_min:
        return None

    breakout_window = history_df.iloc[-(config.restart_breakout_days + 1) : -1]
    if len(breakout_window) < config.restart_breakout_days:
        return None
    if float(latest["close"]) <= float(breakout_window["close"].max()) and float(latest["high"]) <= float(
        breakout_window["high"].max()
    ):
        return None

    consolidation_days, consolidation_range_pct, volume_ratio = _find_second_wave_window(history_df, config)
    if consolidation_days is None:
        return None

    reason = (
        f"second wave after {strong_return:.2%} in {config.strong_lookback_days}d, "
        f"consolidation={consolidation_days}d range={consolidation_range_pct:.2%}, vol_ratio={volume_ratio:.2f}"
    )
    return _build_result(
        history_df=history_df,
        symbol=symbol,
        name=name,
        strategy_name=SECOND_WAVE,
        reason=reason,
        consolidation_days=consolidation_days,
        consolidation_range_pct=consolidation_range_pct,
        consolidation_volume_ratio=volume_ratio,
    )


def _find_second_wave_window(history_df: pd.DataFrame, config: Type6Config) -> tuple[int | None, float | None, float | None]:
    for window in range(config.consolidation_max_days, config.consolidation_min_days - 1, -1):
        if len(history_df) < 2 * window + 1:
            continue

        consolidation = history_df.iloc[-(window + 1) : -1]
        launch = history_df.iloc[-(2 * window + 1) : -(window + 1)]
        if consolidation.empty or launch.empty:
            continue

        consolidation_low = float(consolidation["low"].min())
        consolidation_high = float(consolidation["high"].max())
        if consolidation_low <= 0:
            continue

        consolidation_range_pct = (consolidation_high - consolidation_low) / consolidation_low
        if consolidation_range_pct > config.consolidation_range_max:
            continue

        if float(consolidation["low"].min()) < float(launch["low"].min()):
            continue

        launch_volume = float(launch["volume"].mean())
        if launch_volume <= 0:
            continue
        consolidation_volume = float(consolidation["volume"].mean())
        volume_ratio = consolidation_volume / launch_volume
        if volume_ratio >= 1:
            continue

        return window, consolidation_range_pct, volume_ratio

    return None, None, None


def _find_recent_platform_breakout(history_df: pd.DataFrame, config: Type4Config) -> dict[str, object] | None:
    latest_index = len(history_df) - 1
    start_index = max(config.platform_window_days, latest_index - config.breakout_lookback_days + 1)

    best_match: dict[str, object] | None = None
    for breakout_index in range(start_index, latest_index + 1):
        breakout_row = history_df.iloc[breakout_index]
        if pd.isna(breakout_row["volume_ratio_20"]) or float(breakout_row["volume_ratio_20"]) < config.breakout_volume_ratio_min:
            continue

        platform_start = breakout_index - config.platform_window_days
        platform = history_df.iloc[platform_start:breakout_index]
        if len(platform) < config.platform_window_days:
            continue

        platform_high = float(platform["high"].max())
        platform_low = float(platform["low"].min())
        if platform_low <= 0:
            continue
        platform_range_pct = (platform_high - platform_low) / platform_low
        if platform_range_pct > config.platform_range_max:
            continue

        breakout_close = float(breakout_row["close"])
        if breakout_close <= platform_high:
            continue

        best_match = {
            "breakout_index": breakout_index,
            "breakout_date": pd.Timestamp(breakout_row["trade_date"]).date().isoformat(),
            "platform_high": platform_high,
            "platform_range_pct": platform_range_pct,
        }

    return best_match


def _lookback_return(history_df: pd.DataFrame, lookback_days: int) -> float | None:
    if len(history_df) <= lookback_days:
        return None
    start_price = float(history_df.iloc[-(lookback_days + 1)]["close"])
    end_price = float(history_df.iloc[-1]["close"])
    if start_price <= 0:
        return None
    return end_price / start_price - 1


def _build_result(
    history_df: pd.DataFrame,
    symbol: str,
    name: str,
    strategy_name: str,
    reason: str,
    **extra: object,
) -> dict[str, object]:
    latest = history_df.iloc[-1]
    result = {
        "trade_date": pd.Timestamp(latest["trade_date"]).date().isoformat(),
        "symbol": symbol,
        "name": name,
        "strategy_name": strategy_name,
        "close": round(float(latest["close"]), 4),
        "ma_10": round(float(latest["ma_10"]), 4) if pd.notna(latest.get("ma_10")) else None,
        "ma_20": round(float(latest["ma_20"]), 4) if pd.notna(latest.get("ma_20")) else None,
        "ma_60": round(float(latest["ma_60"]), 4) if pd.notna(latest.get("ma_60")) else None,
        "return_15d": round(float(latest["return_15d"]), 4) if pd.notna(latest.get("return_15d")) else None,
        "volume_ratio_20": round(float(latest["volume_ratio_20"]), 4) if pd.notna(latest.get("volume_ratio_20")) else None,
        "reason": reason,
    }
    result.update(extra)
    return result


def _to_volume_top_config(config: Type1Config | Type2Config | Type3Config) -> VolumeTopBreakoutConfig:
    return VolumeTopBreakoutConfig(
        min_old_high_gap_days=config.min_old_high_gap_days,
        min_drawdown_pct=config.min_drawdown_pct,
        peak_window_days=config.peak_window_days,
        breakout_volume_lookback_days=config.breakout_volume_lookback_days,
        breakout_volume_multiplier=config.breakout_volume_multiplier,
        require_break_below_ma60=getattr(config, "require_break_below_ma60", True),
    )
