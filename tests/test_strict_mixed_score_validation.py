from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from stocks_analyzer.strict_mixed_score_validation import (
    _effective_min_stock_count,
    _forward_outcome_frame,
    _oos_window_cache_path,
    build_selected_forward_paths,
    build_anchored_oos_windows,
    build_strict_strategy_trades,
    summarize_strict_strategy_trades,
)


def test_build_anchored_oos_windows_respects_embargo_and_min_train_days() -> None:
    trade_dates = pd.bdate_range("2020-01-01", periods=20).date

    windows = build_anchored_oos_windows(
        list(trade_dates),
        train_start=date(2020, 1, 1),
        test_start=date(2020, 1, 15),
        test_end=date(2020, 1, 28),
        min_train_days=5,
        test_window_days=3,
        step_days=3,
        embargo_days=2,
    )

    assert windows.loc[0, "train_start"] == "2020-01-01"
    assert windows.loc[0, "train_end"] == "2020-01-10"
    assert windows.loc[0, "test_start"] == "2020-01-15"
    assert windows.loc[0, "test_end"] == "2020-01-17"
    assert windows.loc[0, "train_days"] == 8
    assert windows.loc[1, "test_start"] == "2020-01-20"


def test_oos_window_cache_path_is_stable_and_zero_padded() -> None:
    assert _oos_window_cache_path(Path("cache"), 7) == Path("cache") / "oos_window_007.parquet"


def test_effective_min_stock_count_relaxes_when_limit_is_smaller() -> None:
    assert _effective_min_stock_count(limit=500, min_stock_count=500) == 400
    assert _effective_min_stock_count(limit=300, min_stock_count=500) == 240
    assert _effective_min_stock_count(limit=1000, min_stock_count=500) == 500
    assert _effective_min_stock_count(limit=None, min_stock_count=500) == 500


def test_forward_outcome_frame_uses_next_open_and_conservative_same_day_barrier() -> None:
    bars = pd.DataFrame(
        [
            {"trade_date": "2026-01-02", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0},
            {"trade_date": "2026-01-05", "open": 100.0, "high": 120.0, "low": 90.0, "close": 110.0},
            {"trade_date": "2026-01-06", "open": 111.0, "high": 112.0, "low": 108.0, "close": 109.0},
        ]
    )

    outcome = _forward_outcome_frame(
        bars,
        signal_dates=(date(2026, 1, 2),),
        horizons=(2,),
        stop_loss_pct=0.08,
        take_profit_pct=0.15,
    )

    assert outcome.loc[0, "entry_date"] == "2026-01-05"
    assert outcome.loc[0, "entry_open"] == 100.0
    assert round(float(outcome.loc[0, "return_2d"]), 6) == 0.09
    assert round(float(outcome.loc[0, "max_profit_2d"]), 6) == 0.20
    assert round(float(outcome.loc[0, "max_drawdown_2d"]), 6) == -0.10
    assert outcome.loc[0, "exit_reason_2d"] == "stop_loss_first"
    assert round(float(outcome.loc[0, "barrier_return_2d"]), 6) == -0.08
    assert bool(outcome.loc[0, "stop_loss_hit_2d"]) is True
    assert bool(outcome.loc[0, "take_profit_hit_2d"]) is False


def test_strict_strategy_trades_include_mixed_top20_and_unbounded_all90() -> None:
    panel = pd.DataFrame(
        [
            {
                "signal_date": "2026-01-02",
                "symbol": "600001",
                "name": "A",
                "phase1_score_100": 95.0,
                "phase2_score_100": 96.0,
                "phase4_score_100": 94.0,
                "mixed_010_score": 113.1,
                "mixed_score": 132.2,
                "phase1_center_score": 70.0,
                "phase2_center_score": 68.0,
                "centered_risk_score": 107.76,
                "all90_flag": True,
                "phase5_score_100": 80.0,
                "phase7_trade_permission": "allow",
                "entry_date": "2026-01-05",
                "entry_open": 10.0,
                "return_5d": 0.10,
                "barrier_return_5d": 0.10,
                "max_profit_5d": 0.16,
                "max_drawdown_5d": -0.03,
                "return_R_5d": 1.25,
                "barrier_R_5d": 1.25,
                "max_profit_R_5d": 2.0,
                "max_drawdown_R_5d": -0.375,
                "exit_reason_5d": "timeout",
                "stop_loss_hit_5d": False,
                "take_profit_hit_5d": False,
            },
            {
                "signal_date": "2026-01-02",
                "symbol": "600002",
                "name": "B",
                "phase1_score_100": 93.0,
                "phase2_score_100": 92.0,
                "phase4_score_100": 91.0,
                "mixed_010_score": 109.5,
                "mixed_score": 128.0,
                "phase1_center_score": 74.0,
                "phase2_center_score": 76.0,
                "centered_risk_score": 106.04,
                "all90_flag": True,
                "phase5_score_100": 30.0,
                "phase7_trade_permission": "allow",
                "entry_date": "2026-01-05",
                "entry_open": 10.0,
                "return_5d": -0.04,
                "barrier_return_5d": -0.04,
                "max_profit_5d": 0.02,
                "max_drawdown_5d": -0.06,
                "return_R_5d": -0.5,
                "barrier_R_5d": -0.5,
                "max_profit_R_5d": 0.25,
                "max_drawdown_R_5d": -0.75,
                "exit_reason_5d": "timeout",
                "stop_loss_hit_5d": False,
                "take_profit_hit_5d": False,
            },
            {
                "signal_date": "2026-01-02",
                "symbol": "600003",
                "name": "C",
                "phase1_score_100": 39.0,
                "phase2_score_100": 99.0,
                "phase4_score_100": 100.0,
                "mixed_010_score": 113.8,
                "mixed_score": 127.6,
                "phase1_center_score": 18.0,
                "phase2_center_score": 62.0,
                "centered_risk_score": 108.88,
                "all90_flag": False,
                "phase5_score_100": 90.0,
                "phase7_trade_permission": "allow",
                "entry_date": "2026-01-05",
                "entry_open": 10.0,
                "return_5d": 0.20,
                "barrier_return_5d": 0.15,
                "max_profit_5d": 0.21,
                "max_drawdown_5d": -0.01,
                "return_R_5d": 2.5,
                "barrier_R_5d": 1.875,
                "max_profit_R_5d": 2.625,
                "max_drawdown_R_5d": -0.125,
                "exit_reason_5d": "take_profit",
                "stop_loss_hit_5d": False,
                "take_profit_hit_5d": True,
            },
            {
                "signal_date": "2026-01-02",
                "symbol": "600004",
                "name": "D",
                "phase1_score_100": 80.0,
                "phase2_score_100": 80.0,
                "phase4_score_100": 92.0,
                "mixed_010_score": 108.0,
                "mixed_score": 124.0,
                "phase1_center_score": 100.0,
                "phase2_center_score": 100.0,
                "centered_risk_score": 112.0,
                "all90_flag": False,
                "phase5_score_100": 70.0,
                "phase7_trade_permission": "allow",
                "entry_date": "2026-01-05",
                "entry_open": 10.0,
                "return_5d": 0.08,
                "barrier_return_5d": 0.08,
                "max_profit_5d": 0.12,
                "max_drawdown_5d": -0.02,
                "return_R_5d": 1.0,
                "barrier_R_5d": 1.0,
                "max_profit_R_5d": 1.5,
                "max_drawdown_R_5d": -0.25,
                "exit_reason_5d": "timeout",
                "stop_loss_hit_5d": False,
                "take_profit_hit_5d": False,
            },
        ]
    )

    trades, counts = build_strict_strategy_trades(
        panel,
        strategies=("phase4_top20_p12_ge40", "mixed_010_top20", "mixed_top20", "centered_risk_top20", "all90", "mixed_top20_phase5_safe"),
        horizons=(5,),
        top_n=1,
        phase1_min_score=40.0,
        phase2_min_score=40.0,
        all90_min_score=90.0,
        phase5_safe_min_score=40.0,
    )

    assert counts.set_index("strategy").loc["phase4_top20_p12_ge40", "candidate_count"] == 1
    assert counts.set_index("strategy").loc["mixed_010_top20", "candidate_count"] == 1
    assert counts.set_index("strategy").loc["mixed_top20", "candidate_count"] == 1
    assert counts.set_index("strategy").loc["centered_risk_top20", "candidate_count"] == 1
    assert counts.set_index("strategy").loc["all90", "candidate_count"] == 2
    assert counts.set_index("strategy").loc["mixed_top20_phase5_safe", "candidate_count"] == 1
    assert trades[trades["strategy"].eq("all90")]["symbol"].tolist() == ["600001", "600002"]
    assert trades[trades["strategy"].eq("centered_risk_top20")]["symbol"].tolist() == ["600004"]

    summary = summarize_strict_strategy_trades(trades, candidate_counts=counts, signal_days=1)
    mixed = summary[summary["strategy"].eq("mixed_top20")].iloc[0]
    assert mixed["trade_count"] == 1
    assert round(float(mixed["avg_raw_return"]), 6) == 0.10
    assert round(float(mixed["raw_win_rate"]), 6) == 1.0


def test_build_selected_forward_paths_saves_one_path_per_selected_trade() -> None:
    class FakeStorage:
        def load_daily_bars(self, symbol: str) -> pd.DataFrame:
            assert symbol == "600001"
            return pd.DataFrame(
                [
                    {"trade_date": "2026-01-02", "open": 9.5, "high": 10.0, "low": 9.0, "close": 9.8},
                    {"trade_date": "2026-01-05", "open": 10.0, "high": 10.8, "low": 9.8, "close": 10.4},
                    {"trade_date": "2026-01-06", "open": 10.4, "high": 11.0, "low": 10.2, "close": 10.9},
                    {"trade_date": "2026-01-07", "open": 10.9, "high": 11.2, "low": 10.5, "close": 10.7},
                ]
            )

    trades = pd.DataFrame(
        [
            {
                "strategy": "mixed_top20",
                "signal_date": "2026-01-02",
                "symbol": "600001",
                "selected_rank": 1,
                "horizon": 5,
                "entry_date": "2026-01-05",
                "entry_open": 10.0,
                "phase1_score_100": 80.0,
                "phase2_score_100": 81.0,
                "phase4_score_100": 82.0,
                "mixed_score": 114.2,
            },
            {
                "strategy": "mixed_top20",
                "signal_date": "2026-01-02",
                "symbol": "600001",
                "selected_rank": 1,
                "horizon": 10,
                "entry_date": "2026-01-05",
                "entry_open": 10.0,
            },
        ]
    )

    paths = build_selected_forward_paths(
        storage=FakeStorage(),  # type: ignore[arg-type]
        strategy_trades=trades,
        max_horizon=2,
        strategies=("mixed_top20",),
    )

    assert paths["day_offset"].tolist() == [1, 2]
    assert paths["trade_date"].tolist() == ["2026-01-05", "2026-01-06"]
    assert paths["entry_open"].tolist() == [10.0, 10.0]
    assert paths["mixed_score"].tolist() == [114.2, 114.2]
