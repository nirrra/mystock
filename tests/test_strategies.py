from pathlib import Path

import pandas as pd

from stocks_analyzer.config import load_config
from stocks_analyzer.indicators import add_indicators
from stocks_analyzer.strategies import (
    PLATFORM_BREAKOUT,
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


def test_evaluate_volume_top_breakout_returns_match() -> None:
    config = _load_test_config()

    dataframe = _build_volume_top_frame(final_closes=[92.0, 94.0, 95.0, 102.0], breakout_offset=0)
    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [VOLUME_TOP_BREAKOUT])

    assert [row["strategy_name"] for row in result] == [VOLUME_TOP_BREAKOUT]
    assert result[0]["breakout_date"] == pd.Timestamp(dataframe.iloc[-1]["trade_date"]).date().isoformat()


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

    result = evaluate_strategies(enriched, _instrument(), config, [VOLUME_TOP_FOLLOW_THROUGH])

    assert [row["strategy_name"] for row in result] == [VOLUME_TOP_FOLLOW_THROUGH]
    assert result[0]["days_after_breakout"] == 3


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


def test_evaluate_platform_breakout_returns_match() -> None:
    config = _load_test_config()

    prelude = [40.0] * 40
    first_rise = [50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0, 66.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0]
    transition = [72.5, 72.0]
    platform = [
        71.2,
        70.8,
        71.5,
        71.0,
        70.9,
        71.4,
        71.1,
        70.7,
        71.3,
        71.0,
        70.8,
        71.2,
        71.0,
        70.9,
        71.4,
        71.1,
        70.8,
        71.2,
        71.0,
        71.3,
    ]
    breakout_and_follow = [74.0, 75.0, 74.6]
    closes = prelude + first_rise + transition + platform + breakout_and_follow
    dataframe = _build_dataframe(closes)
    breakout_index = len(prelude) + len(first_rise) + len(transition) + len(platform)
    dataframe.loc[breakout_index, "high"] = 74.8
    dataframe.loc[breakout_index, "close"] = 74.0
    dataframe.loc[breakout_index, "volume"] = 3_200_000

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [PLATFORM_BREAKOUT])

    assert [row["strategy_name"] for row in result] == [PLATFORM_BREAKOUT]
    assert 20 <= result[0]["platform_window_days"] <= 30
    assert 1 <= result[0]["transition_days"] <= 3


def test_evaluate_trend_pullback_returns_match() -> None:
    config = _load_test_config()

    prelude = [40.0] * 30
    rising = list(pd.Series(range(50, 111)).astype(float))
    recent = [110.5, 111.0, 111.5, 112.0, 113.0, 112.5, 112.0, 108.0, 107.8, 108.6]
    closes = prelude + rising + recent
    dataframe = _build_dataframe(closes)
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

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [TREND_PULLBACK])

    assert [row["strategy_name"] for row in result] == [TREND_PULLBACK]
    assert result[0]["recent_high_date"] == pd.Timestamp(dataframe.iloc[len(prelude) + len(rising) + 4]["trade_date"]).date().isoformat()


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
    config.type1.breakout_volume_lookback_days = 5
    config.type1.breakout_volume_multiplier = 2.0
    config.type1.near_high_threshold_pct = 0.05
    config.type2.min_old_high_gap_days = 20
    config.type2.peak_window_days = 2
    config.type2.breakout_volume_lookback_days = 5
    config.type2.breakout_volume_multiplier = 2.0
    config.type3.min_old_high_gap_days = 20
    config.type3.peak_window_days = 2
    config.type3.breakout_volume_lookback_days = 5
    config.type3.breakout_volume_multiplier = 2.0
    config.type3.post_breakout_max_days = 8
    config.type3.post_breakout_max_extension_pct = 0.10
    config.type4.main_rise_window_days = 15
    config.type4.main_rise_return_min = 0.20
    config.type4.transition_min_days = 1
    config.type4.transition_max_days = 5
    config.type4.platform_min_days = 15
    config.type4.platform_max_days = 30
    config.type4.platform_range_max = 0.15
    config.type4.breakout_volume_ratio_min = 1.3
    config.type4.post_breakout_max_days = 8
    config.type4.post_breakout_max_distance_pct = 0.10
    config.type5.recent_high_lookback_days = 10
    config.type5.high_pre_lookback_days = 20
    config.type5.high_peak_window_days = 5
    config.type5.ma20_touch_lookback_days = 2
    config.type5.ma20_touch_abs_tolerance = 0.5
    config.type5.ma20_touch_pct_tolerance = 0.01
    config.type5.ma20_reclaim_min_pct = 0.01
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
        baseline_volume = float(dataframe.iloc[breakout_index - 5 : breakout_index]["volume"].mean())
        close_price = float(dataframe.iloc[breakout_index]["close"])
        dataframe.loc[breakout_index, "open"] = min(close_price - 4.0, 99.0)
        dataframe.loc[breakout_index, "high"] = 103.0
        dataframe.loc[breakout_index, "close"] = max(close_price, 101.5)
        dataframe.loc[breakout_index, "low"] = float(dataframe.loc[breakout_index, "open"]) * 0.99
        dataframe.loc[breakout_index, "volume"] = baseline_volume * 3.5

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
