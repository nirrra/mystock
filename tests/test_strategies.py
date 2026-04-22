from pathlib import Path

import pandas as pd

from stocks_analyzer.config import load_config
from stocks_analyzer.indicators import add_indicators
from stocks_analyzer.strategies import (
    PLATFORM_BREAKOUT,
    SECOND_WAVE,
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

    prelude = [45.0] * 20
    rising = list(pd.Series(range(50, 76)).astype(float))
    platform = [
        74.8,
        75.2,
        75.5,
        75.1,
        75.6,
        75.3,
        75.8,
        75.4,
        75.7,
        75.2,
        75.9,
        75.5,
        75.8,
        75.6,
        75.4,
        75.7,
        75.9,
        75.5,
        75.8,
        75.6,
    ]
    breakout_and_follow = [78.5, 79.0, 78.8]
    closes = prelude + rising + platform + breakout_and_follow
    dataframe = _build_dataframe(closes)
    breakout_index = len(prelude) + len(rising) + len(platform)
    dataframe.loc[breakout_index, "high"] = 79.2
    dataframe.loc[breakout_index, "close"] = 78.5
    dataframe.loc[breakout_index, "volume"] = 3_000_000
    dataframe.loc[len(dataframe) - 1, "high"] = 79.1

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [PLATFORM_BREAKOUT])

    assert [row["strategy_name"] for row in result] == [PLATFORM_BREAKOUT]


def test_evaluate_trend_pullback_returns_match() -> None:
    config = _load_test_config()

    prelude = [40.0] * 20
    rising = list(pd.Series(range(50, 101)).astype(float))
    pullback = [100.5, 100.0, 99.5, 99.0, 98.8, 98.7, 98.9, 99.1, 99.3, 99.5]
    closes = prelude + rising + pullback
    dataframe = _build_dataframe(closes)
    recent_start = len(dataframe) - 5
    dataframe.loc[: recent_start - 1, "volume"] = 1_600_000
    dataframe.loc[recent_start:, "volume"] = 600_000

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [TREND_PULLBACK])

    assert [row["strategy_name"] for row in result] == [TREND_PULLBACK]


def test_evaluate_second_wave_returns_match() -> None:
    config = _load_test_config()
    config.history_momentum_filter.lookback_days = 30

    launch = list(pd.Series([50, 52, 54, 56, 60, 64, 68, 72, 77, 82, 88, 94]).astype(float))
    trend = list(pd.Series([96, 98, 100, 102, 104, 106, 108, 109, 110, 111, 112, 113]).astype(float))
    consolidation = [111.0, 110.5, 110.8, 111.2, 110.9, 111.1, 111.0, 111.3]
    closes = launch + trend + consolidation + [114.5]
    dataframe = _build_dataframe(closes)
    dataframe.loc[4, "close"] = 64.8
    dataframe.loc[4, "high"] = 65.4
    dataframe.loc[4, "low"] = 59.8
    dataframe.loc[4, "open"] = 60.0
    dataframe.loc[len(launch) : len(launch) + len(trend) - 1, "volume"] = 1_600_000
    dataframe.loc[len(launch) + len(trend) : len(dataframe) - 2, "volume"] = 850_000
    dataframe.loc[len(dataframe) - 1, "high"] = 115.0

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, [SECOND_WAVE])

    assert [row["strategy_name"] for row in result] == [SECOND_WAVE]


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
