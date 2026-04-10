import pandas as pd

from stocks_analyzer.pattern_scan import PatternScanConfig, analyze_symbol


def test_analyze_symbol_detects_near_old_high_pattern() -> None:
    config = PatternScanConfig(
        min_old_high_gap_days=20,
        max_old_high_gap_days=80,
        peak_window_days=2,
        volume_window_min_days=5,
        volume_window_max_days=15,
    )
    closes = (
        [50.0] * 20
        + [100.0]
        + [95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 68.0, 66.0, 64.0, 62.0, 60.0]
        + list(pd.Series(range(61, 96)).astype(float))
        + [96.0, 97.0, 96.0, 97.0, 97.0, 98.0]
    )
    dataframe = _build_dataframe(closes)
    dataframe.loc[20, "volume"] = 2_500_000

    results = analyze_symbol(dataframe, "600000", config)

    pattern_types = {row["pattern_type"] for row in results}
    assert "near_old_high" in pattern_types


def test_analyze_symbol_detects_breakout_pullback_watch_pattern() -> None:
    config = PatternScanConfig(
        min_old_high_gap_days=20,
        max_old_high_gap_days=80,
        peak_window_days=2,
        volume_window_min_days=5,
        volume_window_max_days=15,
    )
    closes = (
        [50.0] * 20
        + [100.0]
        + [95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 68.0, 66.0, 64.0, 62.0, 60.0]
        + list(pd.Series(range(61, 103)).astype(float))
        + [101.0, 99.0, 97.0, 98.0, 99.0]
    )
    dataframe = _build_dataframe(closes)
    dataframe.loc[20, "volume"] = 2_500_000
    breakout_index = len(dataframe) - 6
    dataframe.loc[breakout_index, "high"] = 103.0
    dataframe.loc[len(dataframe) - 1, "close"] = 98.0
    dataframe.loc[len(dataframe) - 1, "open"] = 98.5
    dataframe.loc[len(dataframe) - 1, "high"] = 99.0
    dataframe.loc[len(dataframe) - 1, "low"] = 97.5

    results = analyze_symbol(dataframe, "600001", config)

    pattern_types = {row["pattern_type"] for row in results}
    assert "breakout_pullback_watch" in pattern_types


def test_analyze_symbol_excludes_old_high_if_later_higher_high_exists_before_now() -> None:
    config = PatternScanConfig(
        min_old_high_gap_days=20,
        max_old_high_gap_days=80,
        peak_window_days=2,
        volume_window_min_days=5,
        volume_window_max_days=15,
    )
    closes = [50.0] * 80 + [100.0] + [95.0] * 10 + [108.0] + [85.0] * 10 + [95.0] * 30
    dataframe = _build_dataframe(closes)
    dataframe.loc[80, "volume"] = 2_500_000
    dataframe.loc[91, "volume"] = 2_500_000

    results = analyze_symbol(dataframe, "600002", config)
    near_rows = [row for row in results if row["pattern_type"] == "near_old_high"]

    assert all(row["old_high_date"] != dataframe.loc[80, "trade_date"].date().isoformat() for row in near_rows)


def test_breakout_pullback_watch_excludes_if_close_too_far_below_old_high() -> None:
    config = PatternScanConfig(
        min_old_high_gap_days=20,
        max_old_high_gap_days=80,
        breakout_pullback_min_distance_pct=-0.05,
        breakout_pullback_max_distance_pct=0.15,
        peak_window_days=2,
        volume_window_min_days=5,
        volume_window_max_days=15,
    )
    closes = (
        [50.0] * 20
        + [100.0]
        + [95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 68.0, 66.0, 64.0, 62.0, 60.0]
        + list(pd.Series(range(61, 103)).astype(float))
        + [101.0, 99.0, 97.0, 96.0, 94.0]
    )
    dataframe = _build_dataframe(closes)
    dataframe.loc[20, "volume"] = 2_500_000
    breakout_index = len(dataframe) - 6
    dataframe.loc[breakout_index, "high"] = 103.0
    dataframe.loc[len(dataframe) - 1, "close"] = 94.0
    dataframe.loc[len(dataframe) - 1, "open"] = 95.0
    dataframe.loc[len(dataframe) - 1, "high"] = 95.5
    dataframe.loc[len(dataframe) - 1, "low"] = 93.5

    results = analyze_symbol(dataframe, "600003", config)

    pattern_types = {row["pattern_type"] for row in results}
    assert "breakout_pullback_watch" not in pattern_types


def test_breakout_pullback_watch_excludes_if_close_too_far_above_old_high() -> None:
    config = PatternScanConfig(
        min_old_high_gap_days=20,
        max_old_high_gap_days=80,
        breakout_pullback_min_distance_pct=-0.05,
        breakout_pullback_max_distance_pct=0.15,
        peak_window_days=2,
        volume_window_min_days=5,
        volume_window_max_days=15,
    )
    closes = (
        [50.0] * 20
        + [100.0]
        + [95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 68.0, 66.0, 64.0, 62.0, 60.0]
        + list(pd.Series(range(61, 113)).astype(float))
        + [118.0]
    )
    dataframe = _build_dataframe(closes)
    dataframe.loc[20, "volume"] = 2_500_000
    breakout_index = len(dataframe) - 2
    dataframe.loc[breakout_index, "high"] = 103.0
    dataframe.loc[len(dataframe) - 1, "close"] = 118.0
    dataframe.loc[len(dataframe) - 1, "open"] = 117.0
    dataframe.loc[len(dataframe) - 1, "high"] = 118.5
    dataframe.loc[len(dataframe) - 1, "low"] = 116.5

    results = analyze_symbol(dataframe, "600004", config)

    pattern_types = {row["pattern_type"] for row in results}
    assert "breakout_pullback_watch" not in pattern_types


def _build_dataframe(closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
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
                "volume": 1_000_000 + idx * 1_000,
                "amount": (1_000_000 + idx * 1_000) * close,
                "pct_change": 0.0,
                "change": close - previous,
                "amplitude": 0.0,
                "turnover": 1.0,
            }
        )
        previous = close
    return pd.DataFrame(data)
