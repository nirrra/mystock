from pathlib import Path

import pandas as pd

from stocks_analyzer.config import load_config
from stocks_analyzer.indicators import add_indicators
from stocks_analyzer.strategies import evaluate_strategies


def test_evaluate_type1_returns_match() -> None:
    config = load_config(Path("config/default.yaml"))
    config.history_momentum_filter.lookback_days = 60
    config.history_momentum_filter.min_return = 0.05
    config.type1.min_old_high_gap_days = 20
    config.type1.max_old_high_gap_days = 80
    config.type1.peak_window_days = 2
    config.type1.volume_window_min_days = 5
    config.type1.volume_window_max_days = 15

    closes = (
        [50.0] * 20
        + [100.0]
        + [95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 68.0, 66.0, 64.0, 62.0, 60.0]
        + list(pd.Series(range(61, 96)).astype(float))
        + [96.0, 97.0, 96.0, 97.0, 97.0, 98.0]
    )
    dataframe = _build_dataframe(closes)
    dataframe.loc[20, "volume"] = 2_500_000

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, ["type1"])

    assert [row["strategy_name"] for row in result] == ["type1"]


def test_evaluate_type2_returns_match() -> None:
    config = load_config(Path("config/default.yaml"))
    config.history_momentum_filter.lookback_days = 60
    config.history_momentum_filter.min_return = 0.05

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

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, ["type2"])

    assert [row["strategy_name"] for row in result] == ["type2"]


def test_evaluate_type3_returns_match() -> None:
    config = load_config(Path("config/default.yaml"))
    config.history_momentum_filter.lookback_days = 60
    config.history_momentum_filter.min_return = 0.05

    prelude = [40.0] * 20
    rising = list(pd.Series(range(50, 101)).astype(float))
    pullback = [100.5, 100.0, 99.5, 99.0, 98.8, 98.7, 98.9, 99.1, 99.3, 99.5]
    closes = prelude + rising + pullback
    dataframe = _build_dataframe(closes)
    recent_start = len(dataframe) - 5
    dataframe.loc[recent_start:, "volume"] = 600_000

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, ["type3"])

    assert [row["strategy_name"] for row in result] == ["type3"]


def test_evaluate_type4_returns_match() -> None:
    config = load_config(Path("config/default.yaml"))
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

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, ["type4"])

    assert [row["strategy_name"] for row in result] == ["type4"]


def test_evaluate_strategies_filters_out_stock_without_recent_5d_plus_10pct_history() -> None:
    config = load_config(Path("config/default.yaml"))
    config.history_momentum_filter.lookback_days = 50
    config.type1.min_old_high_gap_days = 20
    config.type1.max_old_high_gap_days = 80
    config.type1.peak_window_days = 2
    config.type1.volume_window_min_days = 5
    config.type1.volume_window_max_days = 15

    closes = (
        [50.0] * 20
        + [100.0]
        + [95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 68.0, 66.0, 64.0, 62.0, 60.0]
        + list(pd.Series(range(61, 96)).astype(float))
        + [96.0, 97.0, 96.0, 97.0, 97.0, 98.0]
    )
    dataframe = _build_dataframe(closes)
    dataframe.loc[20, "volume"] = 2_500_000

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, ["type1"])

    assert result == []


def test_evaluate_strategies_allows_stock_with_recent_5d_plus_10pct_history() -> None:
    config = load_config(Path("config/default.yaml"))
    config.history_momentum_filter.lookback_days = 60
    config.type1.min_old_high_gap_days = 20
    config.type1.max_old_high_gap_days = 80
    config.type1.peak_window_days = 2
    config.type1.volume_window_min_days = 5
    config.type1.volume_window_max_days = 15

    closes = (
        [50.0] * 20
        + [100.0]
        + [95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 68.0, 66.0, 64.0, 62.0]
        + [60.0, 66.5, 69.0, 71.0, 73.0, 74.0]
        + list(pd.Series(range(75, 96)).astype(float))
        + [96.0, 97.0, 96.0, 97.0, 97.0, 98.0]
    )
    dataframe = _build_dataframe(closes)
    dataframe.loc[20, "volume"] = 2_500_000

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, ["type1"])

    assert [row["strategy_name"] for row in result] == ["type1"]


def test_evaluate_strategies_excludes_old_momentum_outside_recent_lookback() -> None:
    config = load_config(Path("config/default.yaml"))
    config.history_momentum_filter.lookback_days = 200
    config.type1.min_old_high_gap_days = 20
    config.type1.max_old_high_gap_days = 80
    config.type1.peak_window_days = 2
    config.type1.volume_window_min_days = 5
    config.type1.volume_window_max_days = 15

    old_momentum = [50.0, 56.0, 58.0, 60.0, 61.0, 62.0]
    filler = [62.0] * 150
    setup = (
        [50.0] * 20
        + [100.0]
        + [95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 68.0, 66.0, 64.0, 62.0, 60.0]
        + list(pd.Series(range(61, 96)).astype(float))
        + [96.0, 97.0, 96.0, 97.0, 97.0, 98.0]
    )
    dataframe = _build_dataframe(old_momentum + filler + setup)
    setup_start = len(old_momentum) + len(filler)
    dataframe.loc[setup_start + 20, "volume"] = 2_500_000

    result = evaluate_strategies(add_indicators(dataframe), _instrument(), config, ["type1"])

    assert result == []


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
                "volume": 1_200_000 + idx * 5_000,
                "amount": (1_200_000 + idx * 5_000) * close,
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
