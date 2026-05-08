from __future__ import annotations

from datetime import date

import pandas as pd

from stocks_analyzer.daily_screening_backtest import simulate_forward_trade


def test_simulate_forward_trade_uses_next_open_and_take_profit() -> None:
    bars = pd.DataFrame(
        [
            {"trade_date": "2026-01-02", "open": 90.0, "high": 95.0, "low": 89.0, "close": 94.0},
            {"trade_date": "2026-01-05", "open": 100.0, "high": 108.0, "low": 99.0, "close": 105.0},
            {"trade_date": "2026-01-06", "open": 106.0, "high": 116.0, "low": 103.0, "close": 115.0},
            {"trade_date": "2026-01-07", "open": 114.0, "high": 117.0, "low": 111.0, "close": 112.0},
        ]
    )

    outcome = simulate_forward_trade(
        bars,
        signal_date=date(2026, 1, 2),
        horizon_days=3,
        stop_loss_pct=0.08,
        take_profit_pct=0.15,
    )

    assert outcome is not None
    assert outcome["entry_date"] == "2026-01-05"
    assert outcome["entry_price"] == 100.0
    assert outcome["exit_reason"] == "take_profit"
    assert outcome["exit_date"] == "2026-01-06"
    assert outcome["holding_days"] == 2
    assert outcome["take_profit_hit"] is True
    assert outcome["stop_loss_hit"] is False
    assert round(float(outcome["barrier_return"]), 6) == 0.15
    assert round(float(outcome["max_drawdown"]), 6) == -0.01
    assert round(float(outcome["raw_return"]), 6) == 0.12


def test_simulate_forward_trade_is_conservative_when_same_day_touches_both_barriers() -> None:
    bars = pd.DataFrame(
        [
            {"trade_date": "2026-01-02", "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0},
            {"trade_date": "2026-01-05", "open": 100.0, "high": 116.0, "low": 91.0, "close": 110.0},
            {"trade_date": "2026-01-06", "open": 110.0, "high": 112.0, "low": 105.0, "close": 108.0},
        ]
    )

    outcome = simulate_forward_trade(
        bars,
        signal_date=date(2026, 1, 2),
        horizon_days=2,
        stop_loss_pct=0.08,
        take_profit_pct=0.15,
    )

    assert outcome is not None
    assert outcome["exit_reason"] == "stop_loss_first"
    assert outcome["exit_date"] == "2026-01-05"
    assert outcome["holding_days"] == 1
    assert outcome["stop_loss_hit"] is True
    assert outcome["take_profit_hit"] is False
    assert round(float(outcome["barrier_return"]), 6) == -0.08
