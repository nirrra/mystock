from __future__ import annotations

from datetime import date

import pandas as pd

from stocks_analyzer.daily_screening_backtest import (
    _DailySnapshots,
    _build_benchmark_comparison,
    _select_strategy_candidates,
    _simulate_market_benchmark_from_frame,
    simulate_forward_trade,
)


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


def test_mixed_score_strategy_filters_low_phase_scores_and_sorts_by_weighted_score() -> None:
    universe = pd.DataFrame(
        [
            {"symbol": "600001", "name": "低P4高风险分"},
            {"symbol": "600002", "name": "高P4低P1"},
            {"symbol": "600003", "name": "中P4高风险分"},
        ]
    )
    snapshots = _DailySnapshots(
        phase1=pd.DataFrame(
            [
                {"symbol": "600001", "phase1_score_100": 95.0, "phase1_pass": True},
                {"symbol": "600002", "phase1_score_100": 39.0, "phase1_pass": True},
                {"symbol": "600003", "phase1_score_100": 80.0, "phase1_pass": True},
            ]
        ),
        phase2=pd.DataFrame(
            [
                {"symbol": "600001", "phase2_score_100": 95.0, "phase2_pass": True},
                {"symbol": "600002", "phase2_score_100": 90.0, "phase2_pass": True},
                {"symbol": "600003", "phase2_score_100": 80.0, "phase2_pass": True},
            ]
        ),
        phase4=pd.DataFrame(
            [
                {"symbol": "600001", "phase4_score_100": 80.0, "phase4_return_score": 0.08, "phase4_rank": 2},
                {"symbol": "600002", "phase4_score_100": 100.0, "phase4_return_score": 0.10, "phase4_rank": 1},
                {"symbol": "600003", "phase4_score_100": 85.0, "phase4_return_score": 0.085, "phase4_rank": 3},
            ]
        ),
        phase5=pd.DataFrame(),
        patterns=pd.DataFrame(),
        phase7_permission="allow",
    )

    selected = _select_strategy_candidates(
        strategy="phase1_phase2_phase4_mixed_top20",
        signal_date=date(2026, 1, 2),
        universe=universe,
        snapshots=snapshots,
        top_n=2,
        phase4_top_n=20,
    )

    assert selected["symbol"].tolist() == ["600001", "600003"]
    assert selected["mixed_score"].tolist() == [118.0, 117.0]
    assert "600002" not in selected["symbol"].tolist()

    strict = _select_strategy_candidates(
        strategy="phase1_phase2_phase4_all90",
        signal_date=date(2026, 1, 2),
        universe=universe,
        snapshots=snapshots,
        top_n=20,
        phase4_top_n=20,
    )

    assert strict.empty

    snapshots.phase4.loc[snapshots.phase4["symbol"].eq("600001"), "phase4_score_100"] = 91.0
    strict = _select_strategy_candidates(
        strategy="phase1_phase2_phase4_all90",
        signal_date=date(2026, 1, 2),
        universe=universe,
        snapshots=snapshots,
        top_n=20,
        phase4_top_n=20,
    )
    assert strict["symbol"].tolist() == ["600001"]
    assert strict["mixed_score"].tolist() == [129.0]

    snapshots.phase1.loc[snapshots.phase1["symbol"].eq("600003"), "phase1_score_100"] = 92.0
    snapshots.phase2.loc[snapshots.phase2["symbol"].eq("600003"), "phase2_score_100"] = 91.0
    snapshots.phase4.loc[snapshots.phase4["symbol"].eq("600003"), "phase4_score_100"] = 92.0
    strict = _select_strategy_candidates(
        strategy="phase1_phase2_phase4_all90",
        signal_date=date(2026, 1, 2),
        universe=universe,
        snapshots=snapshots,
        top_n=1,
        phase4_top_n=20,
    )
    assert strict["symbol"].tolist() == ["600001", "600003"]


def test_market_benchmark_simulates_forward_index_return_and_comparison() -> None:
    market = pd.DataFrame(
        [
            {"trade_date": "2026-01-02", "synthetic_equal_weight_index": 100.0},
            {"trade_date": "2026-01-05", "synthetic_equal_weight_index": 101.0},
            {"trade_date": "2026-01-06", "synthetic_equal_weight_index": 103.0},
            {"trade_date": "2026-01-07", "synthetic_equal_weight_index": 99.0},
        ]
    )

    benchmark = _simulate_market_benchmark_from_frame(
        market,
        signal_dates=[date(2026, 1, 2)],
        horizons=(3,),
        value_column="synthetic_equal_weight_index",
    )

    assert round(float(benchmark.loc[0, "raw_return"]), 6) == round(99.0 / 101.0 - 1.0, 6)
    assert round(float(benchmark.loc[0, "max_profit"]), 6) == round(103.0 / 101.0 - 1.0, 6)
    assert round(float(benchmark.loc[0, "max_drawdown"]), 6) == round(99.0 / 101.0 - 1.0, 6)

    summary = pd.DataFrame(
        [
            {
                "strategy": "phase1_phase2_phase4_mixed_top20",
                "horizon": 3,
                "trade_count": 2,
                "avg_raw_return": 0.05,
                "avg_max_profit": 0.08,
                "avg_max_drawdown": -0.02,
                "raw_win_rate": 0.5,
                "portfolio_compound_barrier_return": 0.04,
                "portfolio_max_drawdown": -0.01,
            }
        ]
    )
    comparison = _build_benchmark_comparison(summary=summary, benchmark=benchmark)

    assert comparison.loc[0, "strategy"] == "phase1_phase2_phase4_mixed_top20"
    assert "benchmark_avg_raw_return" in comparison.columns
    assert "excess_avg_raw_return" in comparison.columns
