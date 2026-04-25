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
DOUBLE_VOLUME_SUPPORT_REBOUND = "double_volume_support_rebound"

STRATEGY_NAMES = (
    VOLUME_TOP_PRE_BREAKOUT,
    VOLUME_TOP_BREAKOUT,
    VOLUME_TOP_FOLLOW_THROUGH,
    PLATFORM_BREAKOUT,
    TREND_PULLBACK,
    DOUBLE_VOLUME_SUPPORT_REBOUND,
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
    if DOUBLE_VOLUME_SUPPORT_REBOUND in selected:
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
    if DOUBLE_VOLUME_SUPPORT_REBOUND in selected:
        requirements.append(
            max(
                60,
                config.type6.max_anchor_scan_days,
                config.type6.min_anchor_age_days + config.type6.launch_confirm_days + config.type6.pullback_volume_split_min_days,
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


def _apply_type6(history_df: pd.DataFrame, symbol: str, name: str, config: Type6Config) -> dict[str, object] | None:
    if len(history_df) < max(60, config.min_anchor_age_days + config.launch_confirm_days + config.pullback_volume_split_min_days):
        return None

    latest_index = len(history_df) - 1
    first_anchor_index = max(1, latest_index - config.max_anchor_scan_days + 1)
    last_anchor_index = latest_index - config.min_anchor_age_days
    if last_anchor_index < first_anchor_index:
        return None

    for anchor_index in range(last_anchor_index, first_anchor_index - 1, -1):
        context = _build_type6_context(history_df, anchor_index, config)
        if context is None:
            continue

        branch = _match_type6_break_reclaim(history_df, context, config)
        if branch is None:
            branch = _match_type6_support_hold(history_df, context, config)
        if branch is None:
            continue

        return _build_type6_result(history_df, symbol, name, context, branch)

    return None


def _build_type6_context(history_df: pd.DataFrame, anchor_index: int, config: Type6Config) -> dict[str, object] | None:
    if not _is_type6_anchor(history_df, anchor_index, config):
        return None

    latest_index = len(history_df) - 1
    anchor = history_df.iloc[anchor_index]
    anchor_close = float(anchor["close"])
    anchor_volume = float(anchor["volume"])
    if anchor_close <= 0 or anchor_volume <= 0:
        return None

    launch_end = min(latest_index, anchor_index + config.launch_confirm_days - 1)
    if launch_end <= anchor_index:
        return None
    launch_window = history_df.iloc[anchor_index : launch_end + 1]
    launch_high_offset = int(launch_window["high"].astype(float).to_numpy().argmax())
    launch_high_index = anchor_index + launch_high_offset
    launch_high_price = float(history_df.iloc[launch_high_index]["high"])
    launch_return = launch_high_price / anchor_close - 1.0
    if launch_return < config.launch_min_high_return:
        return None

    limit_up_like_count = _count_limit_up_like_days(history_df, anchor_index, launch_high_index, config.launch_limit_up_return)
    if limit_up_like_count < config.launch_limit_up_min_count:
        return None

    peak_window = history_df.iloc[anchor_index : latest_index + 1]
    peak_offset = int(peak_window["high"].astype(float).to_numpy().argmax())
    peak_index = anchor_index + peak_offset
    if peak_index <= anchor_index or peak_index >= latest_index:
        return None
    peak_price = float(history_df.iloc[peak_index]["high"])

    pullback = history_df.iloc[peak_index + 1 : latest_index + 1].reset_index(drop=False)
    if len(pullback) < config.pullback_volume_split_min_days:
        return None

    pullback_low_offset = int(pullback["low"].astype(float).to_numpy().argmin())
    pullback_low_row = pullback.iloc[pullback_low_offset]
    pullback_low_index = int(pullback_low_row["index"])
    pullback_low_price = float(pullback_low_row["low"])
    peak_to_pullback_drawdown = 1.0 - pullback_low_price / peak_price
    if peak_to_pullback_drawdown < config.peak_to_pullback_min_drawdown_pct:
        return None
    if pullback_low_price > anchor_close * (1.0 + config.support_tolerance_pct):
        return None
    if pullback_low_index - peak_index > config.pullback_max_days:
        return None

    pullback_avg_volume = float(pullback["volume"].astype(float).mean())
    pullback_volume_ratio_to_anchor = pullback_avg_volume / anchor_volume
    if pullback_volume_ratio_to_anchor > config.pullback_volume_max_anchor_ratio:
        return None

    split_index = len(pullback) // 2
    front_half = pullback.iloc[:split_index]
    back_half = pullback.iloc[split_index:]
    if front_half.empty or back_half.empty:
        return None
    front_half_avg_volume = float(front_half["volume"].astype(float).mean())
    back_half_avg_volume = float(back_half["volume"].astype(float).mean())
    if front_half_avg_volume <= 0:
        return None
    back_half_volume_ratio = back_half_avg_volume / front_half_avg_volume
    if back_half_volume_ratio > config.pullback_back_half_volume_ratio:
        return None

    prev_volume = float(history_df.iloc[anchor_index - 1]["volume"])
    anchor_volume_ratio_prev = anchor_volume / prev_volume if prev_volume > 0 else None
    volume_ma20 = anchor.get("volume_ma_20")
    anchor_volume_ratio_ma20 = None
    if volume_ma20 is not None and pd.notna(volume_ma20) and float(volume_ma20) > 0:
        anchor_volume_ratio_ma20 = anchor_volume / float(volume_ma20)

    return {
        "anchor_index": anchor_index,
        "anchor_date": _row_date(anchor),
        "anchor_close": anchor_close,
        "anchor_volume": anchor_volume,
        "support_price": anchor_close,
        "anchor_volume_ratio_prev": anchor_volume_ratio_prev,
        "anchor_volume_ratio_ma20": anchor_volume_ratio_ma20,
        "launch_confirm_high_index": launch_high_index,
        "launch_confirm_high_date": _row_date(history_df.iloc[launch_high_index]),
        "launch_confirm_high_price": launch_high_price,
        "launch_confirm_return_pct": launch_return,
        "limit_up_like_count": limit_up_like_count,
        "peak_index": peak_index,
        "peak_date": _row_date(history_df.iloc[peak_index]),
        "peak_price": peak_price,
        "anchor_to_peak_return_pct": peak_price / anchor_close - 1.0,
        "pullback_low_index": pullback_low_index,
        "pullback_low_date": _row_date(history_df.iloc[pullback_low_index]),
        "pullback_low_price": pullback_low_price,
        "peak_to_pullback_drawdown_pct": peak_to_pullback_drawdown,
        "pullback_volume_ratio_to_anchor": pullback_volume_ratio_to_anchor,
        "pullback_front_half_avg_volume": front_half_avg_volume,
        "pullback_back_half_avg_volume": back_half_avg_volume,
        "pullback_back_half_volume_ratio": back_half_volume_ratio,
    }


def _is_type6_anchor(history_df: pd.DataFrame, index: int, config: Type6Config) -> bool:
    if index <= 0:
        return False
    row = history_df.iloc[index]
    previous = history_df.iloc[index - 1]
    close_price = float(row["close"])
    open_price = float(row["open"])
    previous_close = float(previous["close"])
    if close_price <= open_price or previous_close <= 0:
        return False
    if close_price / previous_close - 1.0 < config.anchor_min_return:
        return False

    volume = float(row["volume"])
    previous_volume = float(previous["volume"])
    prev_volume_ok = previous_volume > 0 and volume / previous_volume >= config.anchor_prev_volume_multiplier
    ma_volume_ok = False
    volume_ma20 = row.get("volume_ma_20")
    if volume_ma20 is not None and pd.notna(volume_ma20) and float(volume_ma20) > 0:
        ma_volume_ok = volume / float(volume_ma20) >= config.anchor_ma_volume_multiplier
    return prev_volume_ok or ma_volume_ok


def _count_limit_up_like_days(history_df: pd.DataFrame, start_index: int, end_index: int, min_return: float) -> int:
    count = 0
    for index in range(max(1, start_index), end_index + 1):
        previous_close = float(history_df.iloc[index - 1]["close"])
        if previous_close <= 0:
            continue
        close_price = float(history_df.iloc[index]["close"])
        if close_price / previous_close - 1.0 >= min_return:
            count += 1
    return count


def _match_type6_break_reclaim(
    history_df: pd.DataFrame,
    context: dict[str, object],
    config: Type6Config,
) -> dict[str, object] | None:
    latest_index = len(history_df) - 1
    latest = history_df.iloc[latest_index]
    support_price = float(context["support_price"])
    anchor_volume = float(context["anchor_volume"])
    start_index = max(int(context["peak_index"]) + 1, latest_index - config.break_reclaim_lookback_days + 1)

    for breakdown_index in range(latest_index, start_index - 1, -1):
        breakdown = history_df.iloc[breakdown_index]
        if float(breakdown["close"]) >= support_price * (1.0 - config.break_below_pct):
            continue
        breakdown_volume = float(breakdown["volume"])
        breakdown_volume_ratio = breakdown_volume / anchor_volume if anchor_volume > 0 else float("inf")
        if breakdown_volume_ratio > config.breakdown_volume_max_anchor_ratio:
            continue

        reclaim_deadline = min(latest_index, breakdown_index + config.max_reclaim_days)
        for reclaim_index in range(breakdown_index + 1, reclaim_deadline + 1):
            reclaim = history_df.iloc[reclaim_index]
            if float(reclaim["close"]) < support_price:
                continue
            if not _is_bullish_or_positive(history_df, reclaim_index):
                continue

            post_reclaim_days = latest_index - reclaim_index
            if post_reclaim_days > config.post_reclaim_max_sideways_days:
                continue
            post = history_df.iloc[reclaim_index : latest_index + 1]
            if float(post["close"].astype(float).min()) < support_price * (1.0 - config.break_below_pct):
                continue
            post_low = float(post["low"].astype(float).min())
            if post_low <= 0:
                continue
            post_range = float(post["high"].astype(float).max()) / post_low - 1.0
            if post_range > config.post_reclaim_range_max:
                continue
            if float(latest["close"]) < support_price or not _is_bullish_or_positive(history_df, latest_index):
                continue

            return {
                "pattern6_branch": "break_reclaim",
                "breakdown_date": _row_date(breakdown),
                "breakdown_volume_ratio_to_anchor": breakdown_volume_ratio,
                "reclaim_date": _row_date(reclaim),
                "days_to_reclaim": reclaim_index - breakdown_index,
                "post_reclaim_days": post_reclaim_days,
            }
    return None


def _match_type6_support_hold(
    history_df: pd.DataFrame,
    context: dict[str, object],
    config: Type6Config,
) -> dict[str, object] | None:
    latest_index = len(history_df) - 1
    latest = history_df.iloc[latest_index]
    support_price = float(context["support_price"])
    start_index = max(int(context["peak_index"]) + 1, latest_index - config.support_touch_lookback_days + 1)
    recent = history_df.iloc[start_index : latest_index + 1].reset_index(drop=False)
    if recent.empty:
        return None

    touched = recent[recent["low"].astype(float) <= support_price * (1.0 + config.support_tolerance_pct)]
    if touched.empty:
        return None
    if float(recent["close"].astype(float).min()) < support_price * (1.0 - config.support_break_tolerance_pct):
        return None
    if float(latest["close"]) < support_price:
        return None
    if float(latest["close"]) <= float(latest["open"]):
        return None
    if latest_index <= 0:
        return None
    previous_close = float(history_df.iloc[latest_index - 1]["close"])
    if previous_close <= 0 or float(latest["close"]) / previous_close - 1.0 < config.stable_min_return:
        return None

    touch_row = touched.iloc[-1]
    return {
        "pattern6_branch": "support_hold",
        "support_touch_date": _row_date(history_df.iloc[int(touch_row["index"])]),
    }


def _build_type6_result(
    history_df: pd.DataFrame,
    symbol: str,
    name: str,
    context: dict[str, object],
    branch: dict[str, object],
) -> dict[str, object]:
    if branch["pattern6_branch"] == "break_reclaim":
        reason = (
            f"double-volume break reclaim: anchor={context['anchor_date']} support={float(context['support_price']):.2f}, "
            f"peak={float(context['peak_price']):.2f} on {context['peak_date']}, "
            f"launch={float(context['launch_confirm_return_pct']):.2%}, limit_up_like={context['limit_up_like_count']}, "
            f"broke={branch['breakdown_date']} vol_ratio={float(branch['breakdown_volume_ratio_to_anchor']):.2f}, "
            f"reclaimed={branch['reclaim_date']} in {branch['days_to_reclaim']}d"
        )
    else:
        reason = (
            f"double-volume support hold: anchor={context['anchor_date']} close={float(context['anchor_close']):.2f}, "
            f"peak={float(context['peak_price']):.2f} on {context['peak_date']}, "
            f"launch={float(context['launch_confirm_return_pct']):.2%}, limit_up_like={context['limit_up_like_count']}, "
            f"pullback={float(context['peak_to_pullback_drawdown_pct']):.2%}, "
            f"vol_back/front={float(context['pullback_back_half_volume_ratio']):.2f}, "
            f"touch={branch['support_touch_date']}"
        )

    extra = {key: value for key, value in context.items() if not key.endswith("_index") and key != "anchor_volume"}
    extra.update(branch)
    return _build_result(
        history_df=history_df,
        symbol=symbol,
        name=name,
        strategy_name=DOUBLE_VOLUME_SUPPORT_REBOUND,
        reason=reason,
        **extra,
    )


def _is_bullish_or_positive(history_df: pd.DataFrame, index: int) -> bool:
    row = history_df.iloc[index]
    if float(row["close"]) > float(row["open"]):
        return True
    if index <= 0:
        return False
    previous_close = float(history_df.iloc[index - 1]["close"])
    return previous_close > 0 and float(row["close"]) > previous_close


def _row_date(row: pd.Series) -> str:
    return pd.Timestamp(row["trade_date"]).date().isoformat()


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
        left = index - config.high_peak_window_days
        right = index + config.high_peak_window_days
        if left < 0 or right >= len(history_df):
            continue
        current_high = float(history_df.iloc[index]["high"])
        peak_window = history_df.iloc[left : right + 1]["high"].astype(float)
        if float(peak_window.max()) > current_high:
            continue
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
        reclaim_min_price = ma20 * (1.0 + config.ma20_reclaim_min_pct)
        if close_price < reclaim_min_price:
            continue
        touch_tolerance = min(config.ma20_touch_abs_tolerance, ma20 * config.ma20_touch_pct_tolerance)
        if abs(low_price - ma20) <= touch_tolerance or low_price < ma20:
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
