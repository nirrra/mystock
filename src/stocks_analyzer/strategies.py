from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .models import AppConfig, Type1Config, Type2Config, Type3Config, Type4Config, Type5Config
from .volume_top_breakout import VolumeTopBreakoutConfig, VolumeTopBreakoutEvent, detect_volume_top_breakout


VOLUME_TOP_PRE_BREAKOUT = "volume_top_pre_breakout"
VOLUME_TOP_BREAKOUT = "volume_top_breakout"
VOLUME_TOP_FOLLOW_THROUGH = "volume_top_follow_through"
PLATFORM_BREAKOUT = "platform_breakout"
TREND_PULLBACK = "trend_pullback"

STRATEGY_NAMES = (
    VOLUME_TOP_PRE_BREAKOUT,
    VOLUME_TOP_BREAKOUT,
    VOLUME_TOP_FOLLOW_THROUGH,
    PLATFORM_BREAKOUT,
    TREND_PULLBACK,
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
                config.type4.main_rise_window_days
                + config.type4.transition_max_days
                + config.type4.platform_max_days
                + 1,
            )
        )
    if TREND_PULLBACK in selected:
        requirements.append(
            max(
                60,
                config.type5.recent_high_lookback_days + config.type5.high_pre_lookback_days + 1,
                config.type5.ma20_touch_lookback_days + 1,
                20,
            )
        )
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
    minimum_length = max(
        60,
        config.main_rise_window_days + config.transition_max_days + config.platform_min_days + 1,
    )
    if len(history_df) < minimum_length:
        return None

    latest = history_df.iloc[-1]
    if pd.isna(latest["ma_10"]):
        return None
    if float(latest["close"]) <= float(latest["ma_10"]):
        return None

    breakout_event = _find_recent_platform_breakout(history_df, config)
    if breakout_event is None:
        return None

    platform_high = float(breakout_event["platform_high"])
    distance_pct = 0.0 if platform_high <= 0 else float(latest["close"]) / platform_high - 1.0
    if distance_pct > config.post_breakout_max_distance_pct:
        return None

    reason = (
        f"first rise {breakout_event['main_rise_return_pct']:.2%} in {config.main_rise_window_days}d, "
        f"platform={breakout_event['platform_window_days']}d range={breakout_event['platform_range_pct']:.2%}, "
        f"breakout on {breakout_event['breakout_date']}, distance={distance_pct:.2%}"
    )
    return _build_result(
        history_df=history_df,
        symbol=symbol,
        name=name,
        strategy_name=PLATFORM_BREAKOUT,
        reason=reason,
        main_rise_start_date=breakout_event["main_rise_start_date"],
        main_rise_end_date=breakout_event["main_rise_end_date"],
        main_rise_return_pct=breakout_event["main_rise_return_pct"],
        transition_days=breakout_event["transition_days"],
        platform_start_date=breakout_event["platform_start_date"],
        platform_end_date=breakout_event["platform_end_date"],
        platform_high=platform_high,
        breakout_volume_ratio=breakout_event["breakout_volume_ratio"],
        days_after_breakout=breakout_event["days_after_breakout"],
        platform_range_pct=breakout_event["platform_range_pct"],
        platform_window_days=breakout_event["platform_window_days"],
        breakout_date=breakout_event["breakout_date"],
        distance_to_platform_high_pct=distance_pct,
    )


def _apply_type5(history_df: pd.DataFrame, symbol: str, name: str, config: Type5Config) -> dict[str, object] | None:
    if len(history_df) < max(60, config.recent_high_lookback_days + config.high_pre_lookback_days + 1):
        return None

    latest = history_df.iloc[-1]
    if pd.isna(latest["ma_20"]) or pd.isna(latest["ma_60"]):
        return None
    if float(latest["ma_20"]) <= float(latest["ma_60"]):
        return None
    if float(latest["close"]) <= float(latest["ma_20"]):
        return None

    recent_high = _find_recent_pattern5_high(history_df, config)
    if recent_high is None:
        return None

    ma20_touch = _find_recent_ma20_touch(history_df, config)
    if ma20_touch is None:
        return None

    recent_high_price = float(recent_high["recent_high_price"])
    distance_from_recent_high_pct = 0.0 if recent_high_price <= 0 else float(latest["close"]) / recent_high_price - 1.0

    reason = (
        f"recent 10d high on {recent_high['recent_high_date']} at {recent_high_price:.2f}, "
        f"MA20 touch on {ma20_touch['ma20_touch_date']}, latest distance={distance_from_recent_high_pct:.2%}"
    )
    return _build_result(
        history_df=history_df,
        symbol=symbol,
        name=name,
        strategy_name=TREND_PULLBACK,
        reason=reason,
        recent_high_date=recent_high["recent_high_date"],
        recent_high_price=recent_high_price,
        days_since_recent_high=recent_high["days_since_recent_high"],
        distance_from_recent_high_pct=distance_from_recent_high_pct,
        ma20_touch_date=ma20_touch["ma20_touch_date"],
        ma20_touch_distance=ma20_touch["ma20_touch_distance"],
        distance_to_ma20=(float(latest["close"]) - float(latest["ma_20"])) / float(latest["ma_20"]),
    )


def _find_recent_platform_breakout(history_df: pd.DataFrame, config: Type4Config) -> dict[str, object] | None:
    latest_index = len(history_df) - 1
    start_index = max(
        config.main_rise_window_days + config.transition_max_days + config.platform_min_days,
        latest_index - config.post_breakout_max_days,
    )

    for breakout_index in range(latest_index, start_index - 1, -1):
        breakout_row = history_df.iloc[breakout_index]
        if pd.isna(breakout_row["volume_ratio_20"]) or float(breakout_row["volume_ratio_20"]) < config.breakout_volume_ratio_min:
            continue

        breakout_close = float(breakout_row["close"])
        for platform_window_days in range(config.platform_max_days, config.platform_min_days - 1, -1):
            platform_start = breakout_index - platform_window_days
            if platform_start <= 0:
                continue

            platform = history_df.iloc[platform_start:breakout_index]
            if len(platform) < platform_window_days:
                continue

            platform_high = float(platform["high"].max())
            platform_low = float(platform["low"].min())
            if platform_low <= 0:
                continue
            platform_range_pct = (platform_high - platform_low) / platform_low
            if platform_range_pct > config.platform_range_max:
                continue
            if breakout_close <= platform_high:
                continue

            for transition_days in range(config.transition_min_days, config.transition_max_days + 1):
                main_rise_end = platform_start - transition_days - 1
                main_rise_start = main_rise_end - config.main_rise_window_days + 1
                if main_rise_start < 0:
                    continue

                main_rise = history_df.iloc[main_rise_start : main_rise_end + 1]
                if len(main_rise) < config.main_rise_window_days:
                    continue

                main_rise_start_close = float(main_rise.iloc[0]["close"])
                main_rise_end_close = float(main_rise.iloc[-1]["close"])
                if main_rise_start_close <= 0:
                    continue
                main_rise_return_pct = main_rise_end_close / main_rise_start_close - 1.0
                if main_rise_return_pct < config.main_rise_return_min:
                    continue

                return {
                    "breakout_index": breakout_index,
                    "breakout_date": pd.Timestamp(breakout_row["trade_date"]).date().isoformat(),
                    "breakout_volume_ratio": float(breakout_row["volume_ratio_20"]),
                    "days_after_breakout": latest_index - breakout_index,
                    "platform_high": platform_high,
                    "platform_range_pct": platform_range_pct,
                    "platform_window_days": platform_window_days,
                    "platform_start_date": pd.Timestamp(platform.iloc[0]["trade_date"]).date().isoformat(),
                    "platform_end_date": pd.Timestamp(platform.iloc[-1]["trade_date"]).date().isoformat(),
                    "main_rise_start_date": pd.Timestamp(main_rise.iloc[0]["trade_date"]).date().isoformat(),
                    "main_rise_end_date": pd.Timestamp(main_rise.iloc[-1]["trade_date"]).date().isoformat(),
                    "main_rise_return_pct": main_rise_return_pct,
                    "transition_days": transition_days,
                }

    return None


def _find_recent_pattern5_high(history_df: pd.DataFrame, config: Type5Config) -> dict[str, object] | None:
    latest_index = len(history_df) - 1
    start_index = max(config.high_pre_lookback_days, latest_index - config.recent_high_lookback_days + 1)
    for index in range(latest_index, start_index - 1, -1):
        current_high = float(history_df.iloc[index]["high"])
        previous_window = history_df.iloc[index - config.high_pre_lookback_days : index]
        if previous_window.empty:
            continue
        previous_max_high = float(previous_window["high"].max())
        if previous_max_high > current_high:
            continue
        return {
            "recent_high_date": pd.Timestamp(history_df.iloc[index]["trade_date"]).date().isoformat(),
            "recent_high_price": current_high,
            "days_since_recent_high": latest_index - index,
        }
    return None


def _find_recent_ma20_touch(history_df: pd.DataFrame, config: Type5Config) -> dict[str, object] | None:
    recent = history_df.tail(config.ma20_touch_lookback_days).reset_index(drop=True)
    for offset in range(len(recent) - 1, -1, -1):
        row = recent.iloc[offset]
        if pd.isna(row.get("ma_20")):
            continue
        ma20 = float(row["ma_20"])
        low_price = float(row["low"])
        close_price = float(row["close"])
        if close_price <= ma20:
            continue
        if abs(low_price - ma20) <= config.ma20_touch_abs_tolerance or low_price < ma20:
            return {
                "ma20_touch_date": pd.Timestamp(row["trade_date"]).date().isoformat(),
                "ma20_touch_distance": low_price - ma20,
            }
    return None


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
