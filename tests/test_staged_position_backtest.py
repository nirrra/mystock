from __future__ import annotations

import pandas as pd

from stocks_analyzer.staged_position_backtest import simulate_staged_position_path


def test_staged_position_uses_strict_two_percent_r_cap() -> None:
    path = pd.DataFrame(
        [
            {
                "day_offset": 1,
                "trade_date": "2026-01-05",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "entry_open": 100.0,
                "atr_14": 9.0,
                "ma_20": 95.0,
            }
        ]
    )

    outcome, orders = simulate_staged_position_path(path, signal_atr_pct=0.09, horizon=1)

    assert outcome is not None
    assert round(float(outcome["r_pct"]), 6) == round(0.02 / (0.40 * (0.30 + 0.30 / 2.0)), 6)
    assert round(float(outcome["initial_stop_price"]), 6) == round(100.0 * (1.0 - float(outcome["r_pct"])), 6)
    assert orders[0]["action"] == "buy"
    assert orders[-1]["reason"] == "timeout"


def test_staged_position_second_batch_requires_half_r_pullback() -> None:
    path = pd.DataFrame(
        [
            {
                "day_offset": 1,
                "trade_date": "2026-01-05",
                "open": 100.0,
                "high": 101.0,
                "low": 95.5,
                "close": 98.0,
                "entry_open": 100.0,
                "atr_14": 4.0,
                "ma_20": 95.0,
            },
            {
                "day_offset": 2,
                "trade_date": "2026-01-06",
                "open": 98.0,
                "high": 99.0,
                "low": 97.0,
                "close": 98.0,
                "entry_open": 100.0,
                "atr_14": 4.0,
                "ma_20": 95.0,
            },
        ]
    )

    outcome, orders = simulate_staged_position_path(path, signal_atr_pct=0.04, horizon=2)

    assert outcome is not None
    assert outcome["second_batch_filled"] is True
    buys = [order for order in orders if order["action"] == "buy"]
    assert len(buys) == 2
    assert round(float(buys[1]["price"]), 6) == 96.0


def test_staged_position_moves_stop_and_clears_position() -> None:
    path = pd.DataFrame(
        [
            {
                "day_offset": 1,
                "trade_date": "2026-01-05",
                "open": 100.0,
                "high": 108.0,
                "low": 101.0,
                "close": 107.0,
                "entry_open": 100.0,
                "atr_14": 4.0,
                "ma_20": 95.0,
            },
            {
                "day_offset": 2,
                "trade_date": "2026-01-06",
                "open": 106.0,
                "high": 107.0,
                "low": 99.0,
                "close": 100.0,
                "entry_open": 100.0,
                "atr_14": 4.0,
                "ma_20": 95.0,
            },
        ]
    )

    outcome, orders = simulate_staged_position_path(path, signal_atr_pct=0.04, horizon=2)

    assert outcome is not None
    assert outcome["third_batch_filled"] is True
    assert outcome["exit_reason"] == "moved_stop"
    assert orders[-1]["reason"] == "moved_stop"
    assert round(float(orders[-1]["position_qty_after"]), 8) == 0.0
