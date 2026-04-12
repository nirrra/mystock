from __future__ import annotations

import pandas as pd

from stocks_analyzer.macd_divergence import MacdDivergenceConfig, summarize_recent_macd_divergence


def test_summarize_recent_macd_divergence_detects_top_divergence_within_recent_window() -> None:
    dataframe = _build_frame(
        highs=[10, 11, 13, 16, 14, 12, 13, 18, 14, 12, 11, 10],
        lows=[9, 10, 11, 13, 12, 11, 12, 14, 12, 10, 9, 8],
        closes=[9.5, 10.5, 12.0, 15.0, 13.0, 11.5, 12.5, 16.5, 13.0, 11.0, 10.0, 9.0],
        dif_values=[0.2, 0.5, 0.9, 1.5, 1.2, 0.8, 1.0, 1.1, 0.7, 0.4, 0.2, 0.1],
    )

    result = summarize_recent_macd_divergence(
        dataframe,
        MacdDivergenceConfig(lookback_days=5, pivot_left_bars=2, pivot_right_bars=2),
    )

    assert result["macd_top_divergence_15d"] is True
    assert result["macd_bottom_divergence_15d"] is False
    assert result["macd_top_divergence_signal_date"] == "2026-01-10"


def test_summarize_recent_macd_divergence_detects_bottom_divergence_within_recent_window() -> None:
    dataframe = _build_frame(
        highs=[11, 10, 9, 8, 9, 10, 8, 7, 8, 9, 10, 11],
        lows=[9, 8, 7, 4, 6, 7, 6, 3, 6, 7, 8, 9],
        closes=[10, 9, 8, 5, 7, 8, 7, 4, 7, 8, 9, 10],
        dif_values=[-0.2, -0.5, -0.8, -1.5, -1.2, -0.9, -1.1, -1.0, -0.6, -0.3, -0.1, 0.1],
    )

    result = summarize_recent_macd_divergence(
        dataframe,
        MacdDivergenceConfig(lookback_days=5, pivot_left_bars=2, pivot_right_bars=2),
    )

    assert result["macd_top_divergence_15d"] is False
    assert result["macd_bottom_divergence_15d"] is True
    assert result["macd_bottom_divergence_signal_date"] == "2026-01-10"


def _build_frame(
    *,
    highs: list[float],
    lows: list[float],
    closes: list[float],
    dif_values: list[float],
) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": ["600000"] * len(closes),
            "open": closes,
            "close": closes,
            "high": highs,
            "low": lows,
            "volume": [1_000_000] * len(closes),
            "amount": [10_000_000] * len(closes),
            "pct_change": [0.0] * len(closes),
            "change": [0.0] * len(closes),
            "amplitude": [0.0] * len(closes),
            "turnover": [1.0] * len(closes),
            "macd_dif": dif_values,
            "macd_dea": dif_values,
            "macd_hist": [0.0] * len(closes),
        }
    )
