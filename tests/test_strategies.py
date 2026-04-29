from pathlib import Path

import pandas as pd

from stocks_analyzer.config import load_config
from stocks_analyzer.indicators import add_indicators
from stocks_analyzer.strategies import (
    DOUBLE_VOLUME_SUPPORT_REBOUND,
    DUCK_NOSTRIL_CROSS,
    TREND_PULLBACK,
    VOLUME_TOP_BREAKOUT,
    VOLUME_TOP_FOLLOW_THROUGH,
    VOLUME_TOP_PRE_BREAKOUT,
    evaluate_strategies,
)


def test_evaluate_volume_top_pre_breakout_returns_match() -> None:
    config = _load_test_config()

    dataframe = _build_volume_top_frame(final_closes=[96.0, 97.0, 98.0])
    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [VOLUME_TOP_PRE_BREAKOUT])

    assert [row["strategy_name"] for row in result] == [VOLUME_TOP_PRE_BREAKOUT]


def test_evaluate_volume_top_pre_breakout_allows_equal_high_after_old_high() -> None:
    config = _load_test_config()

    dataframe = _build_volume_top_frame(final_closes=[96.0, 97.0, 98.0])
    equal_high_index = len(dataframe) - 10
    dataframe.loc[equal_high_index, "high"] = 101.0

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [VOLUME_TOP_PRE_BREAKOUT])

    assert [row["strategy_name"] for row in result] == [VOLUME_TOP_PRE_BREAKOUT]


def test_evaluate_volume_top_pre_breakout_rejects_large_volume_below_old_high() -> None:
    config = _load_test_config()

    dataframe = _build_volume_top_frame(final_closes=[96.0, 97.0, 98.0])
    latest_index = len(dataframe) - 1
    dataframe.loc[latest_index, "volume"] = 1_000_000

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [VOLUME_TOP_PRE_BREAKOUT])

    assert result == []


def test_evaluate_volume_top_breakout_rejects_breakout_day_before_followup_window() -> None:
    config = _load_test_config()

    dataframe = _build_volume_top_frame(final_closes=[92.0, 94.0, 95.0, 102.0], breakout_offset=0)
    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [VOLUME_TOP_BREAKOUT])

    assert result == []


def test_evaluate_volume_top_breakout_returns_match() -> None:
    config = _load_test_config()

    dataframe = _build_volume_top_frame(final_closes=[92.0, 94.0, 95.0, 102.0, 104.0], breakout_offset=1)
    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [VOLUME_TOP_BREAKOUT])

    assert [row["strategy_name"] for row in result] == [VOLUME_TOP_BREAKOUT]
    assert result[0]["breakout_date"] == pd.Timestamp(dataframe.iloc[-2]["trade_date"]).date().isoformat()
    assert result[0]["days_after_breakout"] == 1
    assert result[0]["breakout_close_position"] >= 0.60
    assert result[0]["breakout_upper_shadow_pct"] <= 0.35
    assert result[0]["breakout_body_pct"] >= 0.25
    assert result[0]["breakout_turnover"] == 1.0
    assert result[0]["breakout_turnover_state"] == "normal"
    assert result[0]["post_breakout_max_high_extension_pct"] <= config.type2.post_breakout_max_high_extension_pct


def test_evaluate_volume_top_breakout_rejects_after_time_window() -> None:
    config = _load_test_config()
    config.type2.post_breakout_max_days = 10

    dataframe = _build_volume_top_frame(
        final_closes=[92.0, 94.0, 95.0, 102.0, 103.0, 103.5, 104.0, 104.5, 105.0, 105.5, 106.0, 106.5, 107.0, 107.5],
        breakout_offset=11,
    )
    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [VOLUME_TOP_BREAKOUT])

    assert result == []


def test_evaluate_volume_top_breakout_rejects_if_post_breakout_high_is_overextended() -> None:
    config = _load_test_config()

    dataframe = _build_volume_top_frame(final_closes=[92.0, 94.0, 95.0, 102.0, 104.0, 105.0], breakout_offset=2)
    breakout_index = len(dataframe) - 1 - 2
    dataframe.loc[breakout_index + 1, "high"] = float(dataframe.loc[breakout_index, "close"]) * 1.11

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [VOLUME_TOP_BREAKOUT])

    assert result == []


def test_evaluate_volume_top_breakout_rejects_if_volume_does_not_make_lookback_high() -> None:
    config = _load_test_config()

    dataframe = _build_volume_top_frame(final_closes=[92.0, 94.0, 95.0, 102.0], breakout_offset=0)
    breakout_index = len(dataframe) - 1
    prior_high = float(dataframe.iloc[breakout_index - config.type2.breakout_volume_high_lookback_days : breakout_index]["volume"].max())
    dataframe.loc[breakout_index, "volume"] = prior_high

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [VOLUME_TOP_BREAKOUT])

    assert result == []


def test_evaluate_volume_top_breakout_rejects_poor_breakout_candle_quality() -> None:
    config = _load_test_config()

    dataframe = _build_volume_top_frame(final_closes=[92.0, 94.0, 95.0, 102.0], breakout_offset=0)
    breakout_index = len(dataframe) - 1
    dataframe.loc[breakout_index, "open"] = 99.0
    dataframe.loc[breakout_index, "close"] = 102.0
    dataframe.loc[breakout_index, "high"] = 120.0
    dataframe.loc[breakout_index, "low"] = 98.0

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [VOLUME_TOP_BREAKOUT])

    assert result == []


def test_evaluate_volume_top_breakout_rejects_if_earlier_higher_high_exists_before_breakout() -> None:
    config = _load_test_config()

    dataframe = _build_volume_top_frame(final_closes=[92.0, 94.0, 95.0, 102.0], breakout_offset=0)
    invalid_index = len(dataframe) - 3
    dataframe.loc[invalid_index, "open"] = 100.0
    dataframe.loc[invalid_index, "close"] = 99.0
    dataframe.loc[invalid_index, "high"] = 101.6
    dataframe.loc[invalid_index, "low"] = 98.5
    dataframe.loc[invalid_index, "volume"] = 400_000

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [VOLUME_TOP_BREAKOUT])

    assert result == []


def test_evaluate_volume_top_follow_through_returns_match() -> None:
    config = _load_test_config()

    dataframe = _build_volume_top_frame(final_closes=[92.0, 94.0, 95.0, 102.0, 104.0, 103.5, 105.0], breakout_offset=3)
    enriched = add_indicators(dataframe)
    enriched.loc[len(enriched) - 1, "low"] = float(enriched.iloc[-1]["ma_20"]) - 0.2
    enriched.loc[len(enriched) - 1, "close"] = float(enriched.iloc[-1]["ma_20"]) + 0.3
    enriched.loc[len(enriched) - 1, "high"] = max(float(enriched.iloc[-1]["high"]), float(enriched.iloc[-1]["close"]) * 1.01)
    enriched.loc[len(enriched) - 1, "volume"] = float(enriched.iloc[-1]["volume_ma_5"]) * 0.8

    result = evaluate_strategies(enriched, _instrument(), config, [VOLUME_TOP_FOLLOW_THROUGH])

    assert [row["strategy_name"] for row in result] == [VOLUME_TOP_FOLLOW_THROUGH]
    assert result[0]["days_after_breakout"] == 3
    assert result[0]["post_breakout_max_high_extension_pct"] <= config.type3.post_breakout_max_high_extension_pct


def test_evaluate_volume_top_follow_through_rejects_after_time_window() -> None:
    config = _load_test_config()
    config.type3.post_breakout_max_days = 10

    dataframe = _build_volume_top_frame(
        final_closes=[
            92.0,
            94.0,
            95.0,
            102.0,
            99.2,
            99.1,
            99.0,
            98.9,
            98.8,
            98.7,
            98.6,
            98.5,
            98.4,
            98.3,
            98.2,
        ],
        breakout_offset=11,
    )
    enriched = add_indicators(dataframe)
    enriched.loc[len(enriched) - 1, "volume"] = float(enriched.iloc[-1]["volume_ma_5"]) * 0.8

    result = evaluate_strategies(enriched, _instrument(), config, [VOLUME_TOP_FOLLOW_THROUGH])

    assert result == []


def test_evaluate_volume_top_follow_through_rejects_if_post_breakout_high_is_overextended() -> None:
    config = _load_test_config()

    dataframe = _build_volume_top_frame(final_closes=[92.0, 94.0, 95.0, 102.0, 99.0, 98.5, 99.0], breakout_offset=3)
    enriched = add_indicators(dataframe)
    breakout_index = len(enriched) - 1 - 3
    enriched.loc[breakout_index + 1, "high"] = float(enriched.loc[breakout_index, "close"]) * 1.11
    enriched.loc[len(enriched) - 1, "low"] = float(enriched.iloc[-1]["ma_20"]) - 0.2
    enriched.loc[len(enriched) - 1, "close"] = float(enriched.iloc[-1]["ma_20"]) + 0.3
    enriched.loc[len(enriched) - 1, "volume"] = float(enriched.iloc[-1]["volume_ma_5"]) * 0.8

    result = evaluate_strategies(enriched, _instrument(), config, [VOLUME_TOP_FOLLOW_THROUGH])

    assert result == []


def test_evaluate_volume_top_follow_through_rejects_if_close_breaks_ma20_floor_after_breakout() -> None:
    config = _load_test_config()

    dataframe = _build_volume_top_frame(final_closes=[92.0, 94.0, 95.0, 102.0, 99.0, 98.5, 99.0], breakout_offset=3)
    enriched = add_indicators(dataframe)
    breakout_index = len(enriched) - 1 - 3
    break_index = breakout_index + 1
    enriched.loc[break_index, "close"] = float(enriched.loc[break_index, "ma_20"]) * 0.97
    enriched.loc[len(enriched) - 1, "close"] = float(enriched.iloc[-1]["ma_20"]) * 1.01

    result = evaluate_strategies(enriched, _instrument(), config, [VOLUME_TOP_FOLLOW_THROUGH])

    assert result == []


def test_evaluate_volume_top_follow_through_rejects_if_pullback_volume_is_not_contracting() -> None:
    config = _load_test_config()

    dataframe = _build_volume_top_frame(final_closes=[92.0, 94.0, 95.0, 102.0, 104.0, 103.5, 105.0], breakout_offset=3)
    enriched = add_indicators(dataframe)
    enriched.loc[len(enriched) - 1, "low"] = float(enriched.iloc[-1]["ma_20"]) - 0.2
    enriched.loc[len(enriched) - 1, "close"] = float(enriched.iloc[-1]["ma_20"]) + 0.3
    enriched.loc[len(enriched) - 1, "high"] = max(float(enriched.iloc[-1]["high"]), float(enriched.iloc[-1]["close"]) * 1.01)
    enriched.loc[len(enriched) - 1, "volume"] = float(enriched.iloc[-1]["volume_ma_5"]) * 1.05

    result = evaluate_strategies(enriched, _instrument(), config, [VOLUME_TOP_FOLLOW_THROUGH])

    assert result == []


def test_evaluate_volume_top_follow_through_rejects_if_earlier_higher_high_exists_before_breakout() -> None:
    config = _load_test_config()

    dataframe = _build_volume_top_frame(final_closes=[92.0, 94.0, 95.0, 102.0, 104.0, 103.5, 105.0], breakout_offset=3)
    breakout_index = len(dataframe) - 1 - 3
    invalid_index = breakout_index - 2
    dataframe.loc[invalid_index, "open"] = 100.0
    dataframe.loc[invalid_index, "close"] = 99.0
    dataframe.loc[invalid_index, "high"] = 101.6
    dataframe.loc[invalid_index, "low"] = 98.5
    dataframe.loc[invalid_index, "volume"] = 400_000

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [VOLUME_TOP_FOLLOW_THROUGH])

    assert result == []


def test_evaluate_duck_nostril_cross_returns_match() -> None:
    config = _load_test_config()

    dataframe = _build_duck_nostril_cross_frame()

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [DUCK_NOSTRIL_CROSS])

    assert [row["strategy_name"] for row in result] == [DUCK_NOSTRIL_CROSS]
    assert result[0]["neck_return_pct"] >= config.type4.neck_min_return
    assert result[0]["peak_to_pullback_drawdown_pct"] >= config.type4.peak_to_pullback_min_drawdown_pct
    assert result[0]["pullback_volume_peak_tail_ratio"] <= config.type4.pullback_volume_max_peak_tail_ratio
    assert result[0]["pullback_back_half_volume_ratio"] <= config.type4.pullback_back_half_volume_ratio
    assert 0 <= result[0]["days_since_nostril_cross"] <= config.type4.cross_lookback_days
    assert result[0]["nostril_volume_ma20_ratio"] <= config.type4.nostril_day_volume_ma20_max_ratio


def test_evaluate_duck_nostril_cross_rejects_without_pullback_volume_contraction() -> None:
    config = _load_test_config()

    dataframe = _build_duck_nostril_cross_frame()
    peak_index = 109
    dataframe.loc[peak_index + 1 :, "volume"] = 1_300_000

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [DUCK_NOSTRIL_CROSS])

    assert result == []


def test_evaluate_duck_nostril_cross_rejects_without_prior_ma5_below_ma10() -> None:
    config = _load_test_config()
    config.type4.prior_ma5_below_ma10_min_days = 20

    dataframe = _build_duck_nostril_cross_frame()

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [DUCK_NOSTRIL_CROSS])

    assert result == []


def test_evaluate_duck_nostril_cross_rejects_ma60_break() -> None:
    config = _load_test_config()

    dataframe = _build_duck_nostril_cross_frame()
    dataframe.loc[116, "low"] = 50.0
    dataframe.loc[116, "close"] = 52.0

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [DUCK_NOSTRIL_CROSS])

    assert result == []


def test_evaluate_duck_nostril_cross_rejects_large_bearish_volume_candle() -> None:
    config = _load_test_config()

    dataframe = _build_duck_nostril_cross_frame()
    large_bearish_index = 114
    dataframe.loc[large_bearish_index, "open"] = 76.0
    dataframe.loc[large_bearish_index, "close"] = 72.5
    dataframe.loc[large_bearish_index, "high"] = 76.2
    dataframe.loc[large_bearish_index, "low"] = 72.3
    dataframe.loc[large_bearish_index, "volume"] = 2_000_000

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [DUCK_NOSTRIL_CROSS])

    assert result == []


def test_evaluate_duck_nostril_cross_rejects_failed_cross() -> None:
    config = _load_test_config()

    dataframe = _build_duck_nostril_cross_frame()
    for index, close_price in enumerate([65.0, 64.5, 64.0, 63.5], start=len(dataframe) - 4):
        dataframe.loc[index, "open"] = close_price + 0.3
        dataframe.loc[index, "close"] = close_price
        dataframe.loc[index, "high"] = close_price + 0.4
        dataframe.loc[index, "low"] = close_price - 0.3

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [DUCK_NOSTRIL_CROSS])

    assert result == []


def test_evaluate_duck_nostril_cross_rejects_cross_before_pullback_low() -> None:
    config = _load_test_config()

    dataframe = _build_duck_nostril_cross_frame()
    dataframe.loc[128, "low"] = 62.6

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [DUCK_NOSTRIL_CROSS])

    assert result == []


def test_evaluate_duck_nostril_cross_rejects_peak_without_20d_right_high() -> None:
    config = _load_test_config()

    dataframe = _build_duck_nostril_cross_frame()
    dataframe.loc[115, "high"] = 83.0

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [DUCK_NOSTRIL_CROSS])

    assert result == []


def test_evaluate_trend_pullback_returns_match() -> None:
    config = _load_test_config()

    dataframe = _build_trend_pullback_frame()

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [TREND_PULLBACK])

    assert [row["strategy_name"] for row in result] == [TREND_PULLBACK]
    assert result[0]["recent_high_date"] == pd.Timestamp(dataframe.iloc[-6]["trade_date"]).date().isoformat()
    assert result[0]["ma20_slope_short_pct"] > 0
    assert result[0]["ma20_slope_long_pct"] > 0
    assert result[0]["ma60_slope_short_pct"] > 0
    assert result[0]["ma60_slope_long_pct"] > 0
    assert result[0]["pullback_volume_contraction_ratio"] <= config.type5.pullback_volume_contraction_max


def test_evaluate_trend_pullback_rejects_if_ma20_not_above_short_and_long_slope_refs() -> None:
    config = _load_test_config()

    enriched = add_indicators(_build_trend_pullback_frame())
    latest_index = len(enriched) - 1
    enriched.loc[latest_index, "ma_20"] = min(
        float(enriched.loc[latest_index - 1, "ma_20"]),
        float(enriched.loc[latest_index - 10, "ma_20"]),
    )

    result = evaluate_strategies(enriched, _instrument(), config, [TREND_PULLBACK])

    assert result == []


def test_evaluate_trend_pullback_rejects_if_ma60_not_above_short_and_long_slope_refs() -> None:
    config = _load_test_config()

    enriched = add_indicators(_build_trend_pullback_frame())
    latest_index = len(enriched) - 1
    enriched.loc[latest_index, "ma_60"] = min(
        float(enriched.loc[latest_index - 1, "ma_60"]),
        float(enriched.loc[latest_index - 10, "ma_60"]),
    )

    result = evaluate_strategies(enriched, _instrument(), config, [TREND_PULLBACK])

    assert result == []


def test_evaluate_trend_pullback_rejects_without_volume_contraction() -> None:
    config = _load_test_config()

    dataframe = _build_trend_pullback_frame()
    start_index = len(dataframe) - 5
    dataframe.loc[start_index:, "volume"] = 1_200_000

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [TREND_PULLBACK])

    assert result == []


def test_evaluate_trend_pullback_allows_rebound_volume_after_contracting_touch() -> None:
    config = _load_test_config()

    dataframe = _build_trend_pullback_frame()
    latest_index = len(dataframe) - 1
    dataframe.loc[latest_index, "volume"] = 1_200_000

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [TREND_PULLBACK])

    assert [row["strategy_name"] for row in result] == [TREND_PULLBACK]


def test_evaluate_double_volume_support_rebound_support_hold_returns_match() -> None:
    config = _load_test_config()

    dataframe = _build_pattern6_support_hold_frame()
    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [DOUBLE_VOLUME_SUPPORT_REBOUND])

    assert [row["strategy_name"] for row in result] == [DOUBLE_VOLUME_SUPPORT_REBOUND]
    assert result[0]["pattern6_branch"] == "support_hold"
    assert result[0]["limit_up_like_count"] >= 2
    assert result[0]["anchor_to_peak_return_pct"] >= 0.18
    assert result[0]["pullback_back_half_volume_ratio"] <= 0.8
    assert result[0]["pullback_max_rise_tail_volume_ratio"] <= config.type6.pullback_max_rise_tail_volume_ratio


def test_evaluate_double_volume_support_rebound_break_reclaim_returns_match() -> None:
    config = _load_test_config()

    dataframe = _build_pattern6_break_reclaim_frame()
    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [DOUBLE_VOLUME_SUPPORT_REBOUND])

    assert [row["strategy_name"] for row in result] == [DOUBLE_VOLUME_SUPPORT_REBOUND]
    assert result[0]["pattern6_branch"] == "break_reclaim"
    assert result[0]["breakdown_volume_ratio_to_anchor"] <= 0.6
    assert result[0]["days_to_reclaim"] == 1


def test_evaluate_double_volume_support_rebound_rejects_weak_launch_after_anchor() -> None:
    config = _load_test_config()

    dataframe = _build_pattern6_support_hold_frame()
    anchor_index = 55
    for index in range(anchor_index + 1, anchor_index + 4):
        dataframe.loc[index, "close"] = min(float(dataframe.loc[index, "close"]), 11.2)
        dataframe.loc[index, "high"] = min(float(dataframe.loc[index, "high"]), 11.3)

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [DOUBLE_VOLUME_SUPPORT_REBOUND])

    assert result == []


def test_evaluate_double_volume_support_rebound_rejects_high_volume_breakdown() -> None:
    config = _load_test_config()

    dataframe = _build_pattern6_break_reclaim_frame()
    breakdown_index = len(dataframe) - 4
    dataframe.loc[breakdown_index, "volume"] = 700_000

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [DOUBLE_VOLUME_SUPPORT_REBOUND])

    assert result == []


def test_evaluate_double_volume_support_rebound_rejects_if_pullback_volume_exceeds_rise_tail() -> None:
    config = _load_test_config()

    dataframe = _build_pattern6_support_hold_frame()
    peak_index = 58
    pullback_index = peak_index + 1
    rise_tail_avg_volume = float(dataframe.loc[peak_index - 2 : peak_index, "volume"].mean())
    dataframe.loc[pullback_index, "volume"] = int(rise_tail_avg_volume * 1.21)

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [DOUBLE_VOLUME_SUPPORT_REBOUND])

    assert result == []


def test_evaluate_double_volume_support_rebound_allows_non_bullish_query_day() -> None:
    config = _load_test_config()

    dataframe = _build_pattern6_support_hold_frame()
    latest_index = len(dataframe) - 1
    previous_index = latest_index - 1
    dataframe.loc[previous_index, "close"] = 10.78
    dataframe.loc[previous_index, "high"] = 10.85
    dataframe.loc[latest_index, "open"] = 10.80
    dataframe.loc[latest_index, "close"] = 10.74
    dataframe.loc[latest_index, "low"] = 10.48
    dataframe.loc[latest_index, "high"] = 10.82

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [DOUBLE_VOLUME_SUPPORT_REBOUND])

    assert [row["strategy_name"] for row in result] == [DOUBLE_VOLUME_SUPPORT_REBOUND]
    assert result[0]["pattern6_branch"] == "support_hold"


def test_evaluate_double_volume_support_rebound_rejects_close_outside_support_range_after_pullback_low() -> None:
    config = _load_test_config()

    dataframe = _build_pattern6_support_hold_frame()
    latest_index = len(dataframe) - 1
    dataframe.loc[latest_index, "close"] = 11.10
    dataframe.loc[latest_index, "high"] = 11.15

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [DOUBLE_VOLUME_SUPPORT_REBOUND])

    assert result == []


def test_evaluate_strategies_filters_out_stock_without_recent_5d_plus_10pct_history() -> None:
    config = _load_test_config()
    config.history_momentum_filter.lookback_days = 50

    dataframe = _build_slow_volume_top_frame()
    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [VOLUME_TOP_PRE_BREAKOUT])

    assert result == []


def test_evaluate_strategies_allows_stock_with_recent_5d_plus_10pct_history() -> None:
    config = _load_test_config()

    dataframe = _build_volume_top_frame(final_closes=[90.0, 94.0, 96.0])
    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [VOLUME_TOP_PRE_BREAKOUT])

    assert [row["strategy_name"] for row in result] == [VOLUME_TOP_PRE_BREAKOUT]


def test_evaluate_strategies_excludes_old_momentum_outside_recent_lookback() -> None:
    config = _load_test_config()
    config.history_momentum_filter.lookback_days = 80

    old_momentum = [50.0, 56.0, 58.0, 60.0, 61.0, 62.0]
    filler = [62.0] * 80
    setup = _build_slow_volume_top_frame()["close"].tolist()
    dataframe = _build_dataframe(old_momentum + filler + setup)

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [VOLUME_TOP_PRE_BREAKOUT])

    assert result == []


def _load_test_config():
    config = load_config(Path("config/default.yaml"))
    config.history_momentum_filter.lookback_days = 60
    config.history_momentum_filter.min_return = 0.05
    config.type1.min_old_high_gap_days = 20
    config.type1.peak_window_days = 2
    config.type1.breakout_volume_high_lookback_days = 5
    config.type1.breakout_min_close_position = 0.60
    config.type1.breakout_max_upper_shadow_pct = 0.35
    config.type1.breakout_min_body_pct = 0.25
    config.type1.near_high_threshold_pct = 0.05
    config.type1.pre_breakout_volume_ratio_max = 1.5
    config.type2.min_old_high_gap_days = 20
    config.type2.peak_window_days = 2
    config.type2.breakout_volume_high_lookback_days = 5
    config.type2.breakout_min_close_position = 0.60
    config.type2.breakout_max_upper_shadow_pct = 0.35
    config.type2.breakout_min_body_pct = 0.25
    config.type2.post_breakout_max_days = 10
    config.type2.post_breakout_max_high_extension_pct = 0.10
    config.type2.post_breakout_ma20_break_tolerance_pct = 0.02
    config.type3.min_old_high_gap_days = 20
    config.type3.peak_window_days = 2
    config.type3.breakout_volume_high_lookback_days = 5
    config.type3.breakout_min_close_position = 0.60
    config.type3.breakout_max_upper_shadow_pct = 0.35
    config.type3.breakout_min_body_pct = 0.25
    config.type3.post_breakout_max_days = 10
    config.type3.post_breakout_max_high_extension_pct = 0.10
    config.type3.post_breakout_ma20_break_tolerance_pct = 0.02
    config.type4.max_peak_scan_days = 80
    config.type4.min_peak_age_days = 20
    config.type4.max_peak_age_days = 35
    config.type4.peak_left_window_days = 20
    config.type4.peak_right_window_days = 20
    config.type4.neck_lookback_days = 30
    config.type4.neck_min_return = 0.18
    config.type4.neck_low_to_peak_min_return = 0.20
    config.type4.require_peak_above_ma60 = True
    config.type4.peak_close_above_ma20_min_pct = 0.0
    config.type4.pullback_min_days = 5
    config.type4.pullback_max_days = 25
    config.type4.peak_to_pullback_min_drawdown_pct = 0.05
    config.type4.pullback_low_ma60_tolerance_pct = 0.05
    config.type4.forbid_effective_break_ma60 = True
    config.type4.pullback_volume_max_peak_tail_ratio = 0.75
    config.type4.pullback_back_half_volume_ratio = 0.85
    config.type4.nostril_day_volume_ma20_max_ratio = 0.90
    config.type4.pullback_max_single_day_volume_peak_tail_ratio = 1.20
    config.type4.require_prior_ma5_below_ma10 = True
    config.type4.prior_ma5_below_ma10_min_days = 2
    config.type4.require_cross_after_pullback_low = True
    config.type4.cross_lookback_days = 8
    config.type4.cross_confirm_gap_min_pct = 0.0
    config.type4.cross_confirm_gap_max_pct = 0.035
    config.type4.require_post_cross_ma5_above_ma10 = True
    config.type4.latest_close_ma10_tolerance_pct = 0.01
    config.type4.current_below_peak_min_pct = -0.03
    config.type4.current_above_ma20_min_pct = -0.02
    config.type4.current_above_ma60_min_pct = -0.05
    config.type4.max_today_return_pct = 0.07
    config.type4.large_bearish_body_min_pct = 0.04
    config.type4.large_bearish_volume_ratio_min = 1.5
    config.type4.max_large_bearish_count = 0
    config.type5.recent_high_lookback_days = 10
    config.type5.high_pre_lookback_days = 20
    config.type5.high_peak_window_days = 5
    config.type5.ma20_touch_lookback_days = 2
    config.type5.ma20_touch_abs_tolerance = 0.5
    config.type5.ma20_touch_pct_tolerance = 0.01
    config.type5.ma20_reclaim_min_pct = 0.01
    config.type5.ma_slope_short_lookback_days = 1
    config.type5.ma_slope_long_lookback_days = 10
    config.type5.pullback_volume_contraction_max = 0.95
    config.type6.pullback_max_rise_tail_volume_ratio = 1.2
    return config


def _build_volume_top_frame(*, final_closes: list[float], breakout_offset: int | None = None) -> pd.DataFrame:
    base = (
        [50.0] * 20
        + [100.0]
        + [95.0, 90.0, 84.0, 78.0, 72.0, 66.0, 60.0, 58.0, 56.0]
        + [55.0] * 35
        + [58.0, 60.0, 63.0, 66.0, 69.0, 72.0, 75.0, 78.0, 81.0, 84.0, 87.0, 90.0]
        + final_closes
    )
    dataframe = _build_dataframe(base)
    dataframe.loc[20, "volume"] = 1_400_000

    if breakout_offset is not None:
        breakout_index = len(dataframe) - 1 - breakout_offset
        baseline_high = float(dataframe.iloc[breakout_index - 5 : breakout_index]["volume"].max())
        close_price = float(dataframe.iloc[breakout_index]["close"])
        dataframe.loc[breakout_index, "open"] = min(close_price - 4.0, 99.0)
        dataframe.loc[breakout_index, "high"] = 103.0
        dataframe.loc[breakout_index, "close"] = max(close_price, 101.5)
        dataframe.loc[breakout_index, "low"] = float(dataframe.loc[breakout_index, "open"]) * 0.99
        dataframe.loc[breakout_index, "volume"] = baseline_high * 1.2

        for index in range(22, breakout_index):
            dataframe.loc[index, "high"] = min(float(dataframe.loc[index, "high"]), 99.5)

    return dataframe


def _build_slow_volume_top_frame() -> pd.DataFrame:
    recovery = [55.0 + 0.5 * index for index in range(1, 81)]
    base = [50.0] * 20 + [100.0] + [95.0, 90.0, 84.0, 78.0, 72.0, 66.0, 60.0, 58.0, 56.0] + [55.0] * 35 + recovery
    dataframe = _build_dataframe(base)
    dataframe.loc[20, "volume"] = 1_400_000
    for index in range(22, len(dataframe)):
        dataframe.loc[index, "high"] = min(float(dataframe.loc[index, "high"]), 99.5)
    return dataframe


def _build_duck_nostril_cross_frame() -> pd.DataFrame:
    prelude = [50.0] * 80
    neck = [52.0 + index * (28.0 / 29.0) for index in range(30)]
    pullback = [
        78.0,
        76.0,
        74.0,
        72.0,
        70.0,
        68.0,
        66.0,
        64.0,
        63.0,
        63.2,
        63.5,
        64.0,
        64.5,
        65.0,
        65.5,
        66.0,
        67.0,
        68.0,
        69.0,
        70.0,
        71.0,
        72.0,
        73.0,
    ]
    dataframe = _build_dataframe(prelude + neck + pullback)

    peak_index = len(prelude) + len(neck) - 1
    dataframe.loc[peak_index - 2 : peak_index, "volume"] = 1_500_000
    dataframe.loc[peak_index, "high"] = 82.0
    dataframe.loc[peak_index, "close"] = 80.0

    for offset, index in enumerate(range(peak_index + 1, len(dataframe))):
        volume = 650_000 if offset < 7 else 500_000
        dataframe.loc[index, "volume"] = volume
        dataframe.loc[index, "amount"] = volume * float(dataframe.loc[index, "close"])
        dataframe.loc[index, "low"] = min(float(dataframe.loc[index, "open"]), float(dataframe.loc[index, "close"])) - 0.2
        dataframe.loc[index, "high"] = max(float(dataframe.loc[index, "open"]), float(dataframe.loc[index, "close"])) + 0.3
    return dataframe


def _build_trend_pullback_frame() -> pd.DataFrame:
    prelude = [40.0] * 30
    rising = list(pd.Series(range(50, 111)).astype(float))
    recent = [110.5, 111.0, 111.5, 112.0, 113.0, 112.5, 112.0, 108.0, 107.8, 108.6]
    dataframe = _build_dataframe(prelude + rising + recent)
    touch_index = len(dataframe) - 2
    enriched = add_indicators(dataframe)
    ma20 = float(enriched.iloc[touch_index]["ma_20"])
    reclaim_close = ma20 * 1.012
    latest_close = ma20 * 1.02
    dataframe.loc[touch_index, "low"] = ma20 - 0.2
    dataframe.loc[touch_index, "close"] = reclaim_close
    dataframe.loc[touch_index, "high"] = max(float(dataframe.loc[touch_index, "high"]), reclaim_close + 0.4)
    dataframe.loc[len(dataframe) - 1, "close"] = latest_close
    dataframe.loc[len(dataframe) - 1, "low"] = min(float(dataframe.loc[len(dataframe) - 1, "low"]), ma20 + 0.2)
    dataframe.loc[len(dataframe) - 1, "high"] = max(float(dataframe.loc[len(dataframe) - 1, "high"]), latest_close + 0.3)
    dataframe.loc[len(dataframe) - 20 : len(dataframe) - 6, "volume"] = 900_000
    dataframe.loc[len(dataframe) - 5 :, "volume"] = 500_000
    return dataframe


def _build_pattern6_support_hold_frame() -> pd.DataFrame:
    closes = [10.0] * 55 + [10.5, 11.5, 12.6, 12.8, 12.0, 11.3, 10.9, 10.65, 10.7, 10.85]
    dataframe = _build_dataframe(closes)
    anchor_index = 55
    dataframe.loc[anchor_index, "volume"] = 1_000_000
    dataframe.loc[anchor_index, "open"] = 10.0
    dataframe.loc[anchor_index, "close"] = 10.5
    dataframe.loc[anchor_index, "high"] = 10.65
    dataframe.loc[anchor_index, "low"] = 9.95
    _shape_pattern6_after_anchor(dataframe, anchor_index)
    latest_index = len(dataframe) - 1
    dataframe.loc[latest_index, "open"] = 10.65
    dataframe.loc[latest_index, "close"] = 10.78
    dataframe.loc[latest_index, "low"] = 10.48
    dataframe.loc[latest_index, "high"] = 10.85
    return dataframe


def _build_pattern6_break_reclaim_frame() -> pd.DataFrame:
    closes = [10.0] * 55 + [10.5, 11.5, 12.6, 12.8, 12.0, 11.0, 10.2, 10.65, 10.7, 10.9]
    dataframe = _build_dataframe(closes)
    anchor_index = 55
    dataframe.loc[anchor_index, "volume"] = 1_000_000
    dataframe.loc[anchor_index, "open"] = 10.0
    dataframe.loc[anchor_index, "close"] = 10.5
    dataframe.loc[anchor_index, "high"] = 10.65
    dataframe.loc[anchor_index, "low"] = 9.95
    _shape_pattern6_after_anchor(dataframe, anchor_index)
    breakdown_index = len(dataframe) - 4
    reclaim_index = len(dataframe) - 3
    dataframe.loc[breakdown_index, "open"] = 10.6
    dataframe.loc[breakdown_index, "close"] = 10.2
    dataframe.loc[breakdown_index, "low"] = 10.1
    dataframe.loc[breakdown_index, "high"] = 10.75
    dataframe.loc[breakdown_index, "volume"] = 550_000
    dataframe.loc[reclaim_index, "open"] = 10.3
    dataframe.loc[reclaim_index, "close"] = 10.65
    dataframe.loc[reclaim_index, "low"] = 10.3
    dataframe.loc[reclaim_index, "high"] = 10.8
    dataframe.loc[reclaim_index, "volume"] = 450_000
    latest_index = len(dataframe) - 1
    dataframe.loc[latest_index, "open"] = 10.65
    dataframe.loc[latest_index, "close"] = 10.78
    dataframe.loc[latest_index, "low"] = 10.62
    dataframe.loc[latest_index, "high"] = 10.9
    dataframe.loc[latest_index, "volume"] = 300_000
    return dataframe


def _shape_pattern6_after_anchor(dataframe: pd.DataFrame, anchor_index: int) -> None:
    overrides = {
        anchor_index + 1: {"open": 10.5, "close": 11.5, "high": 11.6, "low": 10.45, "volume": 900_000},
        anchor_index + 2: {"open": 11.5, "close": 12.6, "high": 12.75, "low": 11.45, "volume": 850_000},
        anchor_index + 3: {"open": 12.6, "close": 12.8, "high": 13.0, "low": 12.35, "volume": 800_000},
        anchor_index + 4: {"open": 12.8, "close": 12.0, "high": 12.9, "low": 11.85, "volume": 700_000},
        anchor_index + 5: {"open": 12.0, "close": 11.3, "high": 12.05, "low": 11.2, "volume": 650_000},
        anchor_index + 6: {"open": 11.3, "close": 10.9, "high": 11.4, "low": 10.7, "volume": 500_000},
        anchor_index + 7: {"open": 10.9, "close": 10.65, "high": 10.95, "low": 10.48, "volume": 420_000},
        anchor_index + 8: {"open": 10.65, "close": 10.7, "high": 10.8, "low": 10.52, "volume": 350_000},
        anchor_index + 9: {"open": 10.7, "close": 10.85, "high": 10.95, "low": 10.48, "volume": 300_000},
    }
    for index, values in overrides.items():
        for column, value in values.items():
            dataframe.loc[index, column] = value


def _build_dataframe(closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    data = []
    previous = closes[0]
    for idx, close in enumerate(closes):
        open_price = previous
        high = max(open_price, close) * 1.01
        low = min(open_price, close) * 0.99
        data.append(
            {
                "trade_date": dates[idx],
                "symbol": "600000",
                "open": open_price,
                "close": close,
                "high": high,
                "low": low,
                "volume": 300_000 + idx * 2_000,
                "amount": (300_000 + idx * 2_000) * close,
                "pct_change": 0.0,
                "change": close - previous,
                "amplitude": 0.0,
                "turnover": 1.0,
            }
        )
        previous = close
    return pd.DataFrame(data)


def _instrument() -> dict[str, object]:
    return {"symbol": "600000", "name": "测试股份"}
