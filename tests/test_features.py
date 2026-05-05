import pandas as pd

from stocks_analyzer.features import build_feature_frame


def _make_daily_frame(length: int = 340) -> pd.DataFrame:
    close = [20.0 + idx * 0.2 for idx in range(length)]
    high = [value * 1.02 for value in close]
    low = [value * 0.98 for value in close]
    volume = [1_000_000 + idx * 1_000 for idx in range(length)]
    amount = [close[idx] * volume[idx] * 100 for idx in range(length)]
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=length, freq="B"),
            "open": [value * 0.995 for value in close],
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "amount": amount,
            "pct_change": pd.Series(close).pct_change().fillna(0.0),
            "change": pd.Series(close).diff().fillna(0.0),
            "amplitude": [(high[idx] - low[idx]) / close[idx] for idx in range(length)],
            "turnover": [1.0] * length,
        }
    )


def test_build_feature_frame_adds_weekly_and_monthly_volume_and_rsi_features() -> None:
    result = build_feature_frame(_make_daily_frame())

    expected_columns = {
        "weekly_volume",
        "weekly_volume_ratio_8w",
        "weekly_position_24w",
        "monthly_volume",
        "monthly_volume_ratio_6m",
        "monthly_rsi_14",
    }
    assert expected_columns.issubset(result.columns)


def test_weekly_features_do_not_use_future_days_inside_same_week() -> None:
    dataframe = _make_daily_frame(length=15)
    result = build_feature_frame(dataframe)

    monday = pd.Timestamp("2024-01-08")
    monday_row = result[result["trade_date"] == monday].iloc[0]
    previous_friday_close = dataframe[dataframe["trade_date"] == pd.Timestamp("2024-01-05")].iloc[0]["close"]

    assert monday_row["weekly_close"] == previous_friday_close
