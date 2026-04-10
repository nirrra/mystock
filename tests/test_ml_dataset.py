from datetime import date

import pandas as pd

from stocks_analyzer.ml_dataset import infer_split_dates, split_probability_dataset


def test_split_probability_dataset_respects_time_boundaries() -> None:
    dataframe = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=15, freq="D"),
            "symbol": ["600000"] * 15,
            "feature_a": range(15),
            "feature_b": range(100, 115),
            "label_stable_up": [0, 1] * 7 + [0],
        }
    )

    split = split_probability_dataset(
        dataset=dataframe,
        train_end=date(2025, 1, 6),
        valid_end=date(2025, 1, 10),
        test_end=date(2025, 1, 15),
    )

    assert split.train["trade_date"].max().date() <= date(2025, 1, 6)
    assert split.valid["trade_date"].min().date() > date(2025, 1, 6)
    assert split.valid["trade_date"].max().date() <= date(2025, 1, 10)
    assert split.test["trade_date"].min().date() > date(2025, 1, 10)
    assert "feature_a" in split.feature_columns
    assert "label_stable_up" not in split.feature_columns


def test_infer_split_dates_returns_ordered_dates() -> None:
    dataframe = pd.DataFrame({"trade_date": pd.date_range("2025-01-01", periods=20, freq="D")})

    train_end, valid_end, test_end = infer_split_dates(dataframe)

    assert train_end < valid_end < test_end
