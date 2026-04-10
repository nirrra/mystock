from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .models import AppConfig, Type1Config, Type2Config, Type3Config, Type4Config
from .pattern_scan import PatternScanConfig, find_near_old_high_match


STRATEGY_NAMES = ("type1", "type2", "type3", "type4")


def evaluate_strategies(
    history_df: pd.DataFrame,
    instrument: dict[str, object],
    config: AppConfig,
    selected: Sequence[str],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    symbol = str(instrument["symbol"])
    name = str(instrument["name"])

    if "type1" in selected:
        match = _apply_type1(history_df, symbol, name, config.type1)
        if match is not None:
            results.append(match)
    if "type2" in selected:
        match = _apply_type2(history_df, symbol, name, config.type2)
        if match is not None:
            results.append(match)
    if "type3" in selected:
        match = _apply_type3(history_df, symbol, name, config.type3)
        if match is not None:
            results.append(match)
    if "type4" in selected:
        match = _apply_type4(history_df, symbol, name, config.type4)
        if match is not None:
            results.append(match)

    return results


def required_history_days(config: AppConfig, selected: Sequence[str]) -> int:
    requirements = [config.universe.min_history_days]
    if "type1" in selected:
        requirements.append(config.type1.volume_window_max_days + config.type1.min_old_high_gap_days + 1)
    if "type2" in selected:
        requirements.append(
            max(
                60,
                config.type2.trend_lookback_days + 1,
                config.type2.ma60_rising_lookback + 1,
                config.type2.platform_window_days + 1,
            )
        )
    if "type3" in selected:
        requirements.append(
            max(
                60,
                config.type3.trend_lookback_days + 1,
                config.type3.ma_rising_lookback + 1,
                20,
            )
        )
    if "type4" in selected:
        requirements.append(max(20, config.type4.strong_lookback_days + 1, 2 * config.type4.consolidation_max_days + 1))
    return max(requirements)


def _apply_type1(history_df: pd.DataFrame, symbol: str, name: str, config: Type1Config) -> dict[str, object] | None:
    match = find_near_old_high_match(history_df, symbol, _to_pattern_scan_config(config))
    if match is None:
        return None

    latest = history_df.iloc[-1]
    reason = (
        f"near old high {float(match['old_high_price']):.2f} from {match['old_high_date']}, "
        f"distance={float(match['distance_to_old_high_pct']):.2%}, "
        f"drawdown={float(match['max_drawdown_since_old_high']):.2%}"
    )
    return _build_result(
        history_df=history_df,
        symbol=symbol,
        name=name,
        strategy_name="type1",
        reason=reason,
        old_high_date=match["old_high_date"],
        old_high_price=float(match["old_high_price"]),
        days_since_old_high=int(match["days_since_old_high"]),
        max_drawdown_since_old_high=float(match["max_drawdown_since_old_high"]),
        distance_to_old_high_pct=float(match["distance_to_old_high_pct"]),
        breakout_date=None,
    )


def _apply_type2(history_df: pd.DataFrame, symbol: str, name: str, config: Type2Config) -> dict[str, object] | None:
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
        f"platform_range={breakout_event['platform_range_pct']:.2%}, "
        f"distance={distance_pct:.2%}"
    )
    return _build_result(
        history_df=history_df,
        symbol=symbol,
        name=name,
        strategy_name="type2",
        reason=reason,
        platform_window_days=config.platform_window_days,
        platform_range_pct=breakout_event["platform_range_pct"],
        breakout_date=breakout_event["breakout_date"],
        distance_to_platform_high_pct=distance_pct,
    )


def _apply_type3(history_df: pd.DataFrame, symbol: str, name: str, config: Type3Config) -> dict[str, object] | None:
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
        f"drawdown_15d={drawdown_15d:.2%}, "
        f"vol_5d/20d={volume_contraction_ratio:.2f}"
    )
    return _build_result(
        history_df=history_df,
        symbol=symbol,
        name=name,
        strategy_name="type3",
        reason=reason,
        distance_to_ma20=distance_to_ma20,
        drawdown_15d=drawdown_15d,
        volume_contraction_ratio=volume_contraction_ratio,
    )


def _apply_type4(history_df: pd.DataFrame, symbol: str, name: str, config: Type4Config) -> dict[str, object] | None:
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
        f"consolidation={consolidation_days}d range={consolidation_range_pct:.2%}, "
        f"vol_ratio={volume_ratio:.2f}"
    )
    return _build_result(
        history_df=history_df,
        symbol=symbol,
        name=name,
        strategy_name="type4",
        reason=reason,
        consolidation_days=consolidation_days,
        consolidation_range_pct=consolidation_range_pct,
        consolidation_volume_ratio=volume_ratio,
    )


def _find_second_wave_window(history_df: pd.DataFrame, config: Type4Config) -> tuple[int | None, float | None, float | None]:
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


def _find_recent_platform_breakout(history_df: pd.DataFrame, config: Type2Config) -> dict[str, object] | None:
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


def _to_pattern_scan_config(config: Type1Config) -> PatternScanConfig:
    return PatternScanConfig(
        min_old_high_gap_days=config.min_old_high_gap_days,
        max_old_high_gap_days=config.max_old_high_gap_days,
        min_drawdown_pct=config.min_drawdown_pct,
        near_high_threshold_pct=config.near_high_threshold_pct,
        breakout_lookback_days=config.breakout_lookback_days,
        breakout_pullback_min_distance_pct=config.breakout_pullback_min_distance_pct,
        breakout_pullback_max_distance_pct=config.breakout_pullback_max_distance_pct,
        peak_window_days=config.peak_window_days,
        volume_window_min_days=config.volume_window_min_days,
        volume_window_max_days=config.volume_window_max_days,
        volume_median_multiplier=config.volume_median_multiplier,
    )
