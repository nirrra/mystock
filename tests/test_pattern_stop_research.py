from __future__ import annotations

import pandas as pd

from stocks_analyzer.pattern_stop_research import research_pattern_stop_grid, select_best_pattern_stop_grid


def test_research_pattern_stop_grid_uses_stop_first_policy() -> None:
    forward_prices = _make_forward_prices(
        high_returns=[0.06, 0.08, 0.02, 0.01, 0.01],
        low_returns=[-0.06, -0.02, -0.01, -0.01, -0.01],
        close_returns=[0.01, 0.03, 0.01, 0.0, 0.0],
    )

    result = research_pattern_stop_grid(
        forward_prices,
        holding_days=[5],
        take_profits=[0.05],
        stop_losses=[0.05],
        same_day_policy="stop_first",
    )

    trades = result["trades"]
    assert trades.loc[0, "exit_reason"] == "stop_loss"
    assert trades.loc[0, "realized_return_pct"] == -0.05
    assert result["summary"].loc[0, "win_rate"] == 0.0


def test_research_pattern_stop_grid_uses_take_profit_first_policy() -> None:
    forward_prices = _make_forward_prices(
        high_returns=[0.06, 0.08, 0.02, 0.01, 0.01],
        low_returns=[-0.06, -0.02, -0.01, -0.01, -0.01],
        close_returns=[0.01, 0.03, 0.01, 0.0, 0.0],
    )

    result = research_pattern_stop_grid(
        forward_prices,
        holding_days=[5],
        take_profits=[0.05],
        stop_losses=[0.05],
        same_day_policy="take_profit_first",
    )

    trades = result["trades"]
    assert trades.loc[0, "exit_reason"] == "take_profit"
    assert trades.loc[0, "realized_return_pct"] == 0.05
    assert result["summary"].loc[0, "win_rate"] == 1.0


def test_research_pattern_stop_grid_summarizes_time_exit_and_best() -> None:
    first = _make_forward_prices(
        sample_id="sample_a",
        high_returns=[0.01, 0.02, 0.03, 0.04, 0.04],
        low_returns=[-0.01, -0.02, -0.02, -0.03, -0.03],
        close_returns=[0.01, 0.02, 0.03, 0.04, 0.04],
    )
    second = _make_forward_prices(
        sample_id="sample_b",
        high_returns=[0.01, 0.01, 0.01, 0.01, 0.01],
        low_returns=[-0.01, -0.01, -0.01, -0.01, -0.01],
        close_returns=[-0.01, -0.01, -0.01, -0.01, -0.01],
    )

    result = research_pattern_stop_grid(
        pd.concat([first, second], ignore_index=True),
        holding_days=[5],
        take_profits=[0.05, 0.10],
        stop_losses=[0.05],
    )

    summary = result["summary"]
    assert len(summary) == 2
    assert summary["sample_count"].tolist() == [2, 2]
    assert summary["time_exit_rate"].tolist() == [1.0, 1.0]
    assert summary["win_rate"].tolist() == [0.5, 0.5]

    best = select_best_pattern_stop_grid(summary, min_samples=2)
    assert len(best) == 1
    assert best.loc[0, "pattern_id"] == "1"
    assert best.loc[0, "holding_days"] == 5


def test_research_pattern_stop_grid_applies_ma20_stop_after_intraday_rules() -> None:
    forward_prices = _make_forward_prices(
        high_returns=[0.01, 0.02, 0.03, 0.03, 0.03],
        low_returns=[-0.01, -0.02, -0.02, -0.02, -0.02],
        close_returns=[0.01, -0.01, -0.02, -0.02, -0.02],
        closes=[10.1, 9.9, 9.8, 9.8, 9.8],
        ma20_values=[10.0, 10.0, 10.0, 10.0, 10.0],
    )

    result = research_pattern_stop_grid(
        forward_prices,
        holding_days=[5],
        take_profits=[0.05],
        stop_losses=[0.05],
        ma20_stop=True,
    )

    trades = result["trades"]
    assert trades.loc[0, "exit_reason"] == "ma20_stop"
    assert trades.loc[0, "exit_forward_day"] == 2
    assert trades.loc[0, "realized_return_pct"] == -0.01
    assert result["summary"].loc[0, "ma20_stop_rate"] == 1.0


def test_research_pattern_stop_grid_prioritizes_take_profit_before_ma20_stop() -> None:
    forward_prices = _make_forward_prices(
        high_returns=[0.06, 0.02, 0.03, 0.03, 0.03],
        low_returns=[-0.01, -0.02, -0.02, -0.02, -0.02],
        close_returns=[-0.01, -0.01, -0.02, -0.02, -0.02],
        closes=[9.9, 9.9, 9.8, 9.8, 9.8],
        ma20_values=[10.0, 10.0, 10.0, 10.0, 10.0],
    )

    result = research_pattern_stop_grid(
        forward_prices,
        holding_days=[5],
        take_profits=[0.05],
        stop_losses=[0.05],
        ma20_stop=True,
    )

    trades = result["trades"]
    assert trades.loc[0, "exit_reason"] == "take_profit"
    assert trades.loc[0, "realized_return_pct"] == 0.05


def _make_forward_prices(
    *,
    sample_id: str = "sample_1",
    high_returns: list[float],
    low_returns: list[float],
    close_returns: list[float],
    closes: list[float] | None = None,
    ma20_values: list[float] | None = None,
) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=len(high_returns), freq="D")
    if closes is None:
        closes = [10.0 * (1.0 + item) for item in close_returns]
    if ma20_values is None:
        ma20_values = [9.0 for _ in high_returns]
    return pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "symbol": "600000",
                "name": "测试",
                "pattern_id": "1",
                "strategy_name": "volume_top_pre_breakout",
                "signal_date": pd.Timestamp("2026-01-01"),
                "entry_date": pd.Timestamp("2026-01-02"),
                "entry_price": 10.0,
                "forward_day": index + 1,
                "trade_date": dates[index],
                "close": closes[index],
                "ma_20": ma20_values[index],
                "high_return_pct": high_returns[index],
                "low_return_pct": low_returns[index],
                "close_return_pct": close_returns[index],
            }
            for index in range(len(high_returns))
        ]
    )
