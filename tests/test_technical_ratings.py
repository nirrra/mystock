import pandas as pd

from stocks_analyzer.technical_ratings import MA_SIGNAL_COLUMNS, OSCILLATOR_SIGNAL_COLUMNS, add_technical_ratings, rating_status


def _build_trend_frame(length: int = 260) -> pd.DataFrame:
    close = [10 + idx * 0.3 for idx in range(length)]
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=length, freq="D"),
            "symbol": ["600000"] * length,
            "open": [value * 0.995 for value in close],
            "close": close,
            "high": [value * 1.01 for value in close],
            "low": [value * 0.99 for value in close],
            "volume": [1_000_000 + idx * 5000 for idx in range(length)],
            "amount": [1_000_000_000 + idx * 1_000_000 for idx in range(length)],
            "pct_change": [0.0] * length,
            "change": [0.0] * length,
            "amplitude": [0.0] * length,
            "turnover": [1.0] * length,
        }
    )


def test_rating_status_uses_expected_thresholds() -> None:
    assert rating_status(-0.6) == "strong_sell"
    assert rating_status(-0.2) == "sell"
    assert rating_status(0.0) == "neutral"
    assert rating_status(0.2) == "buy"
    assert rating_status(0.6) == "strong_buy"


def test_add_technical_ratings_creates_signal_groups_and_positive_ma_rating_on_uptrend() -> None:
    dataframe = _build_trend_frame()

    result = add_technical_ratings(dataframe)

    latest = result.iloc[-1]
    for column in MA_SIGNAL_COLUMNS + OSCILLATOR_SIGNAL_COLUMNS:
        assert column in result.columns
    assert latest["ma_rating"] > 0.8
    assert latest["ma_rating_label"] in {"buy", "strong_buy"}
