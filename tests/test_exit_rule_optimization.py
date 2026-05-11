from __future__ import annotations

from datetime import date

import pandas as pd

from stocks_analyzer.exit_rule_optimization import optimize_exit_rules, simulate_exit_rule_path


def test_simulate_exit_rule_path_uses_conservative_stop_first() -> None:
    path = pd.DataFrame(
        [
            {
                "day_offset": 1,
                "trade_date": "2026-01-05",
                "open": 10.0,
                "high": 11.5,
                "low": 9.0,
                "close": 10.5,
                "entry_open": 10.0,
            }
        ]
    )
    rule = {
        "rule_id": "fixed_sl08_tp10",
        "rule_family": "fixed",
        "initial_stop_pct": 0.08,
        "take_profit_pct": 0.10,
    }

    outcome = simulate_exit_rule_path(path, rule=rule, horizon=1)

    assert outcome is not None
    assert outcome["exit_reason"] == "stop_loss_first"
    assert round(float(outcome["rule_return"]), 6) == -0.08
    assert outcome["stop_loss_hit"] is True
    assert outcome["take_profit_hit"] is False


def test_fixed_horizon_rule_ignores_intraday_stop_and_take() -> None:
    path = pd.DataFrame(
        [
            {
                "day_offset": 1,
                "trade_date": "2026-01-05",
                "open": 10.0,
                "high": 11.5,
                "low": 9.0,
                "close": 10.5,
                "entry_open": 10.0,
            }
        ]
    )
    rule = {
        "rule_id": "fixed_horizon",
        "rule_family": "fixed_horizon",
        "initial_stop_pct": 0.08,
        "disable_stop_loss": True,
    }

    outcome = simulate_exit_rule_path(path, rule=rule, horizon=1)

    assert outcome is not None
    assert outcome["exit_reason"] == "timeout"
    assert round(float(outcome["rule_return"]), 6) == 0.05
    assert outcome["stop_loss_hit"] is False
    assert outcome["take_profit_hit"] is False


def test_optimize_exit_rules_selects_rule_from_tune_and_reports_test(tmp_path) -> None:
    rows = []
    for signal_date, high, low, close in [
        ("2022-01-03", 11.2, 9.7, 10.8),
        ("2024-01-03", 11.1, 9.8, 10.7),
    ]:
        rows.append(
            {
                "strategy": "mixed_top20",
                "signal_date": signal_date,
                "symbol": "600001",
                "selected_rank": 1,
                "day_offset": 1,
                "trade_date": signal_date,
                "open": 10.0,
                "high": high,
                "low": low,
                "close": close,
                "entry_open": 10.0,
            }
        )
    strict_dir = tmp_path / "strict"
    strict_dir.mkdir()
    pd.DataFrame(rows).to_parquet(strict_dir / "selected_forward_paths.parquet", index=False)

    result = optimize_exit_rules(
        strict_dir=strict_dir,
        strategies=("mixed_top20",),
        horizons=(1,),
        output_dir=tmp_path / "out",
        stop_grid=(0.08,),
        take_grid=(0.10,),
        trailing_grid=(),
        breakeven_trigger_grid=(),
        time_stop_days_grid=(),
        time_stop_min_return_grid=(),
        tune_end_date=date(2023, 12, 29),
        test_start_date=date(2024, 1, 1),
    )

    assert result.summary_path.exists()
    assert result.selection_report_path.exists()
    assert result.recommendations_path.exists()
    assert result.selection_report.loc[0, "rule_id"] == "fixed_sl08_tp10"
    assert round(float(result.selection_report.loc[0, "test_avg_return"]), 6) == 0.10
    assert "test_avg_daily_return" in result.selection_report.columns
