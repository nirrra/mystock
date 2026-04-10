import pandas as pd

from stocks_analyzer.indicators import add_indicators


def test_add_indicators_creates_expected_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-01-01", periods=70, freq="D"),
            "symbol": ["600000"] * 70,
            "open": range(1, 71),
            "close": range(1, 71),
            "high": range(2, 72),
            "low": range(0, 70),
            "volume": [1000 + idx for idx in range(70)],
            "amount": [1_000_000 + idx * 1000 for idx in range(70)],
            "pct_change": [0.0] * 70,
            "change": [0.0] * 70,
            "amplitude": [0.0] * 70,
            "turnover": [1.0] * 70,
        }
    )

    result = add_indicators(dataframe)

    latest = result.iloc[-1]
    assert latest["ma_5"] == 68
    assert latest["ma_20"] == 60.5
    assert "consolidation_range_3d" in result.columns
