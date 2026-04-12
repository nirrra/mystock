import pandas as pd

from stocks_analyzer.universe import build_main_board_universe


def test_build_main_board_universe_filters_non_main_boards_and_st() -> None:
    instruments = pd.DataFrame(
        [
            {"symbol": "600000", "name": "浦发银行", "latest_price": 10.0, "volume": 1000, "amount": 1e8, "turnover_rate": 1.0},
            {"symbol": "300750", "name": "宁德时代", "latest_price": 200.0, "volume": 1000, "amount": 1e9, "turnover_rate": 1.0},
            {"symbol": "000001", "name": "ST平安", "latest_price": 11.0, "volume": 1000, "amount": 1e8, "turnover_rate": 1.0},
        ]
    )

    result = build_main_board_universe(instruments, exclude_st=True)

    assert result["symbol"].tolist() == ["600000"]


def test_build_main_board_universe_handles_empty_input() -> None:
    instruments = pd.DataFrame(columns=["symbol", "name", "trade_status"])

    result = build_main_board_universe(instruments, exclude_st=True)

    assert result.empty
    assert {"exchange", "board", "is_st", "is_suspended"}.issubset(result.columns)
