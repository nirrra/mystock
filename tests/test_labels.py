import pandas as pd

from stocks_analyzer.labels import add_forward_labels


def test_add_forward_labels_marks_success_when_return_and_drawdown_pass() -> None:
    length = 25
    dataframe = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=length, freq="D"),
            "close": [100.0] + [101.0] * 19 + [104.0] * 5,
            "low": [100.0] + [95.0] * 19 + [103.0] * 5,
            "high": [101.0] * length,
        }
    )

    result = add_forward_labels(dataframe, horizon_days=20, min_future_return=0.03, max_future_drawdown=0.08)

    assert round(result.iloc[0]["future_20d_return"], 4) == 0.04
    assert round(result.iloc[0]["future_20d_max_drawdown"], 4) == 0.05
    assert result.iloc[0]["label_stable_up"] == 1.0


def test_add_forward_labels_rejects_large_drawdown() -> None:
    length = 25
    dataframe = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=length, freq="D"),
            "close": [100.0] + [101.0] * 19 + [104.0] * 5,
            "low": [100.0] + [89.0] * 19 + [103.0] * 5,
            "high": [101.0] * length,
        }
    )

    result = add_forward_labels(dataframe, horizon_days=20, min_future_return=0.03, max_future_drawdown=0.08)

    assert round(result.iloc[0]["future_20d_max_drawdown"], 4) == 0.11
    assert result.iloc[0]["label_stable_up"] == 0.0
