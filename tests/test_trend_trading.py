from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path

import pandas as pd

from stocks_analyzer.config import load_config
from stocks_analyzer.screener import Screener
from stocks_analyzer.trend_backtest import backtest_portfolios, backtest_signal_returns, summarize_signal_backtest
from stocks_analyzer.trend_indicator_scores import build_next_open_entries, score_symbol_trend_entries, select_tradable_entries
from stocks_analyzer.trend_signals import dedupe_trend_signals, generate_symbol_trend_signals
from stocks_analyzer.trend_universe import build_symbol_trend_frame, scan_trend_universe


ROOT = Path(__file__).resolve().parents[1]


def _load_test_config():
    return deepcopy(load_config(ROOT / "config" / "default.yaml"))


def _make_daily_bars(symbol: str, close_values: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    size = len(close_values)
    dates = pd.bdate_range("2025-01-02", periods=size)
    volume_values = volumes or [20_000_000.0] * size
    rows: list[dict[str, object]] = []
    previous_close = close_values[0]
    for index, close in enumerate(close_values):
        open_price = previous_close * 0.997 if index else close * 0.995
        high = max(open_price, close) * 1.01
        low = min(open_price, close) * 0.99
        rows.append(
            {
                "trade_date": dates[index],
                "symbol": symbol,
                "open": round(open_price, 4),
                "close": round(close, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "volume": volume_values[index],
                "amount": round(close * volume_values[index], 4),
                "pct_change": 0.0,
                "change": round(close - previous_close, 4) if index else 0.0,
                "amplitude": round((high - low) / close, 4),
                "turnover": 1.0,
                "trade_status": "1",
            }
        )
        previous_close = close
    return pd.DataFrame(rows)


def _make_breakout_bars(symbol: str = "600000") -> pd.DataFrame:
    uptrend = [10.0 + index * 0.06 for index in range(180)]
    platform = [20.6, 20.7, 20.75, 20.8, 20.78, 20.82, 20.79, 20.81, 20.77, 20.8] * 2
    breakout = [21.45]
    close_values = uptrend + platform + breakout
    volumes = [20_000_000.0] * (len(close_values) - 1) + [38_000_000.0]
    return _make_daily_bars(symbol, close_values, volumes)


def _make_pullback_bars(symbol: str = "600001") -> pd.DataFrame:
    uptrend = [10.0 + index * 0.065 for index in range(180)]
    pullback = [21.4, 21.3, 21.2, 21.1, 21.0, 20.95, 20.9, 20.92, 21.05, 21.3]
    close_values = uptrend + pullback
    volumes = [20_000_000.0] * 180 + [14_000_000.0, 13_000_000.0, 12_000_000.0, 11_000_000.0, 10_000_000.0, 10_000_000.0, 9_500_000.0, 9_000_000.0, 9_000_000.0, 8_500_000.0]
    frame = _make_daily_bars(symbol, close_values, volumes)
    # Enlarge the last-day lower shadow so the stabilization rule is satisfied.
    frame.loc[len(frame) - 1, "low"] = round(min(float(frame.loc[len(frame) - 1, "open"]), float(frame.loc[len(frame) - 1, "close"])) * 0.94, 4)
    return frame


def test_build_symbol_trend_frame_marks_trend_universe_candidate() -> None:
    config = _load_test_config()
    bars = _make_breakout_bars()

    trend_frame = build_symbol_trend_frame(bars, symbol="600000", name="测试突破", config=config.trend_universe)

    latest = trend_frame.iloc[-1]
    assert bool(latest["in_trend_universe"]) is True
    assert float(latest["trend_score"]) > 60.0


def test_scan_trend_universe_invokes_progress_callback(monkeypatch) -> None:
    config = _load_test_config()
    progress_calls: list[tuple[int, int]] = []
    bars = _make_breakout_bars()

    class FakeStorage:
        def load_universe(self) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {"symbol": "600000", "name": "甲"},
                    {"symbol": "600001", "name": "乙"},
                ]
            )

        def load_daily_bars(self, symbol: str) -> pd.DataFrame:
            if symbol == "600001":
                raise FileNotFoundError(symbol)
            return bars.copy()

    monkeypatch.setattr(
        "stocks_analyzer.trend_universe.build_symbol_trend_frame",
        lambda daily_bars, symbol, name, config: pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2026-04-10"),
                    "symbol": symbol,
                    "name": name,
                    "trend_score": 88.0,
                    "trend_direction_score": 80.0,
                    "trend_strength_score": 82.0,
                    "in_trend_universe": True,
                }
            ]
        ),
    )

    result = scan_trend_universe(
        FakeStorage(),
        config,
        as_of=date(2026, 4, 10),
        progress_callback=lambda current, total: progress_calls.append((current, total)),
    )

    assert result["symbol"].tolist() == ["600000"]
    assert progress_calls == [(1, 2), (2, 2)]


def test_screener_run_invokes_progress_callback(monkeypatch) -> None:
    config = _load_test_config()
    progress_calls: list[tuple[int, int]] = []
    bars = _make_breakout_bars()

    class FakeStorage:
        def load_universe(self) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {"symbol": "600000", "name": "甲"},
                    {"symbol": "600001", "name": "乙"},
                ]
            )

        def load_daily_bars(self, symbol: str) -> pd.DataFrame:
            if symbol == "600001":
                raise FileNotFoundError(symbol)
            return bars.copy()

    monkeypatch.setattr("stocks_analyzer.screener.required_history_days", lambda config, strategies: 1)
    monkeypatch.setattr(
        "stocks_analyzer.screener.add_indicators",
        lambda frame: frame.assign(amount_ma_20=frame["amount"].astype(float)),
    )
    monkeypatch.setattr(
        "stocks_analyzer.screener.evaluate_strategies",
        lambda cutoff, instrument, config, strategies: [
            {
                "trade_date": pd.Timestamp("2026-04-10"),
                "symbol": str(instrument["symbol"]).zfill(6),
                "name": instrument["name"],
                "strategy_name": "volume_top_pre_breakout",
                "close": 10.0,
                "reason": "demo",
            }
        ],
    )

    results = Screener(FakeStorage(), config).run(
        as_of=date(2026, 4, 10),
        progress_callback=lambda current, total: progress_calls.append((current, total)),
    )

    assert results["symbol"].tolist() == ["600000"]
    assert progress_calls == [(1, 2), (2, 2)]


def test_generate_symbol_trend_signals_detects_breakout_only_after_breakout_day() -> None:
    config = _load_test_config()
    bars = _make_breakout_bars()
    last_trade_date = pd.Timestamp(bars["trade_date"].iloc[-1]).date()
    prev_trade_date = pd.Timestamp(bars["trade_date"].iloc[-2]).date()

    signals_before = generate_symbol_trend_signals(
        bars,
        symbol="600000",
        name="测试突破",
        config=config,
        end_date=prev_trade_date,
    )
    signals_after = generate_symbol_trend_signals(
        bars,
        symbol="600000",
        name="测试突破",
        config=config,
        end_date=last_trade_date,
    )

    assert signals_before.empty
    assert signals_after["signal_type"].tolist() == ["breakout"]
    assert pd.Timestamp(signals_after.loc[0, "trade_date"]).date() == last_trade_date


def test_generate_symbol_trend_signals_detects_pullback_signal() -> None:
    config = _load_test_config()
    bars = _make_pullback_bars()
    last_trade_date = pd.Timestamp(bars["trade_date"].iloc[-1]).date()

    signals = generate_symbol_trend_signals(
        bars,
        symbol="600001",
        name="测试回踩",
        config=config,
    )

    assert set(signals["signal_type"].tolist()) == {"pullback"}
    assert last_trade_date in {pd.Timestamp(item).date() for item in signals["trade_date"].tolist()}
    assert float(signals.iloc[-1]["entry_score"]) > 0


def test_dedupe_trend_signals_prefers_pullback_on_equal_score() -> None:
    trade_date = pd.Timestamp("2026-04-10")
    frame = pd.DataFrame(
        [
            {"trade_date": trade_date, "symbol": "600000", "signal_type": "breakout", "entry_score": 80.0, "trend_score": 75.0},
            {"trade_date": trade_date, "symbol": "600000", "signal_type": "pullback", "entry_score": 80.0, "trend_score": 74.0},
        ]
    )

    deduped = dedupe_trend_signals(frame)

    assert len(deduped) == 1
    assert deduped.loc[0, "signal_type"] == "pullback"


def test_backtest_signal_returns_computes_fixed_holding_metrics() -> None:
    config = _load_test_config()
    config.trend_backtest.holding_days = (5,)
    bars = _make_daily_bars("600000", [10, 10.5, 11.0, 11.2, 11.5, 12.0, 12.5, 12.8])
    signal_trade_date = pd.Timestamp(bars["trade_date"].iloc[0])
    signals = pd.DataFrame(
        [
            {
                "trade_date": signal_trade_date,
                "symbol": "600000",
                "name": "测试回测",
                "signal_type": "breakout",
                "trend_score": 80.0,
                "entry_score": 82.0,
                "trigger_reason": "demo",
            }
        ]
    )

    results = backtest_signal_returns(signals, {"600000": bars}, config.trend_backtest)
    summary = summarize_signal_backtest(results)

    assert len(results) == 1
    assert results.loc[0, "holding_days"] == 5
    assert results.loc[0, "return_pct"] == 0.2
    assert not summary.empty
    assert summary.loc[0, "signal_count"] == 1


def test_backtest_portfolios_builds_summary_for_top_n_selection() -> None:
    config = _load_test_config()
    config.trend_backtest.holding_days = (5,)
    config.trend_backtest.portfolio_top_n = (1,)

    bars_a = _make_daily_bars("600000", [10, 10.5, 11.0, 11.2, 11.5, 12.0, 12.5, 12.8])
    bars_b = _make_daily_bars("600001", [10, 10.2, 10.1, 10.3, 10.35, 10.4, 10.5, 10.55])
    trade_date = pd.Timestamp(bars_a["trade_date"].iloc[0])
    signals = pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "symbol": "600000",
                "name": "甲",
                "signal_type": "breakout",
                "trend_score": 85.0,
                "entry_score": 90.0,
                "trigger_reason": "a",
            },
            {
                "trade_date": trade_date,
                "symbol": "600001",
                "name": "乙",
                "signal_type": "pullback",
                "trend_score": 70.0,
                "entry_score": 72.0,
                "trigger_reason": "b",
            },
        ]
    )

    outputs = backtest_portfolios(signals, {"600000": bars_a, "600001": bars_b}, config.trend_backtest)

    assert len(outputs["positions"]) == 1
    assert outputs["positions"].loc[0, "symbol"] == "600000"
    assert not outputs["equity"].empty
    assert outputs["summary"].loc[0, "portfolio_top_n"] == 1


def test_score_symbol_trend_entries_adds_buy_score_and_next_open_fields() -> None:
    config = _load_test_config()
    bars = _make_breakout_bars()

    scored = score_symbol_trend_entries(
        bars,
        symbol="600000",
        name="测试评分",
        config=config,
    )

    assert not scored.empty
    latest = scored.iloc[-1]
    assert float(latest["buy_score"]) > 0
    assert latest["entry_timing"] == "next_open"
    assert "buy_reason" in scored.columns
    assert latest["macd_cross_state"] in {"golden_cross", "dead_cross", "above_signal", "below_signal", "unknown"}


def test_build_next_open_entries_filters_rows_without_future_open() -> None:
    frame = pd.DataFrame(
        [
            {"trade_date": pd.Timestamp("2026-04-10"), "planned_entry_date": pd.Timestamp("2026-04-11"), "symbol": "600000"},
            {"trade_date": pd.Timestamp("2026-04-11"), "planned_entry_date": None, "symbol": "600001"},
        ]
    )

    entries = build_next_open_entries(frame)

    assert len(entries) == 1
    assert entries.loc[0, "symbol"] == "600000"
    assert entries.loc[0, "entry_timing"] == "next_open"


def test_select_tradable_entries_applies_threshold_rules() -> None:
    config = _load_test_config()
    frame = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2026-04-10"),
                "planned_entry_date": pd.Timestamp("2026-04-11"),
                "symbol": "600000",
                "buy_score": 82.0,
                "trend_base_score": 68.0,
                "price_action_score": 65.0,
                "macd_score": 42.0,
                "positive_indicator_count": 4,
            },
            {
                "trade_date": pd.Timestamp("2026-04-10"),
                "planned_entry_date": pd.Timestamp("2026-04-11"),
                "symbol": "600001",
                "buy_score": 79.9,
                "trend_base_score": 68.0,
                "price_action_score": 65.0,
                "macd_score": 42.0,
                "positive_indicator_count": 4,
            },
            {
                "trade_date": pd.Timestamp("2026-04-10"),
                "planned_entry_date": pd.Timestamp("2026-04-11"),
                "symbol": "600002",
                "buy_score": 88.0,
                "trend_base_score": 64.9,
                "price_action_score": 65.0,
                "macd_score": 42.0,
                "positive_indicator_count": 4,
            },
            {
                "trade_date": pd.Timestamp("2026-04-10"),
                "planned_entry_date": None,
                "symbol": "600003",
                "buy_score": 90.0,
                "trend_base_score": 75.0,
                "price_action_score": 75.0,
                "macd_score": 55.0,
                "positive_indicator_count": 5,
            },
        ]
    )

    entries = select_tradable_entries(frame, config)

    assert entries["symbol"].tolist() == ["600000"]
    assert entries.loc[0, "entry_timing"] == "next_open"


def test_load_config_applies_signal_specific_entry_thresholds() -> None:
    config = _load_test_config()

    assert config.trend_entry_rules.buy_score_min == 80.0
    assert config.trend_entry_rules.breakout.buy_score_min == 81.3308
    assert config.trend_entry_rules.breakout.price_action_score_min == 75.0373
    assert config.trend_entry_rules.pullback.buy_score_min is None


def test_select_tradable_entries_applies_breakout_override_and_pullback_fallback() -> None:
    config = _load_test_config()
    frame = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2026-04-10"),
                "planned_entry_date": pd.Timestamp("2026-04-11"),
                "symbol": "600010",
                "signal_type": "breakout",
                "buy_score": 81.4,
                "trend_base_score": 70.0,
                "price_action_score": 75.1,
                "macd_score": 40.0,
                "positive_indicator_count": 5,
            },
            {
                "trade_date": pd.Timestamp("2026-04-10"),
                "planned_entry_date": pd.Timestamp("2026-04-11"),
                "symbol": "600011",
                "signal_type": "breakout",
                "buy_score": 81.4,
                "trend_base_score": 70.0,
                "price_action_score": 74.9,
                "macd_score": 40.0,
                "positive_indicator_count": 5,
            },
            {
                "trade_date": pd.Timestamp("2026-04-10"),
                "planned_entry_date": pd.Timestamp("2026-04-11"),
                "symbol": "600012",
                "signal_type": "pullback",
                "buy_score": 80.1,
                "trend_base_score": 70.0,
                "price_action_score": 60.1,
                "macd_score": 35.1,
                "positive_indicator_count": 3,
            },
            {
                "trade_date": pd.Timestamp("2026-04-10"),
                "planned_entry_date": pd.Timestamp("2026-04-11"),
                "symbol": "600013",
                "signal_type": "pullback",
                "buy_score": 79.9,
                "trend_base_score": 70.0,
                "price_action_score": 70.0,
                "macd_score": 45.0,
                "positive_indicator_count": 6,
            },
        ]
    )

    entries = select_tradable_entries(frame, config)

    assert entries["symbol"].tolist() == ["600010", "600012"]


def test_backtest_signal_returns_next_open_uses_next_day_open() -> None:
    config = _load_test_config()
    config.trend_backtest.holding_days = (2,)
    bars = pd.DataFrame(
        [
            {"trade_date": pd.Timestamp("2026-04-10"), "symbol": "600000", "open": 10.0, "close": 10.0, "high": 10.2, "low": 9.8},
            {"trade_date": pd.Timestamp("2026-04-11"), "symbol": "600000", "open": 11.0, "close": 12.0, "high": 12.2, "low": 10.8},
            {"trade_date": pd.Timestamp("2026-04-14"), "symbol": "600000", "open": 12.1, "close": 12.5, "high": 12.8, "low": 11.9},
            {"trade_date": pd.Timestamp("2026-04-15"), "symbol": "600000", "open": 12.6, "close": 13.0, "high": 13.2, "low": 12.4},
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2026-04-10"),
                "symbol": "600000",
                "name": "测试次开",
                "signal_type": "breakout",
                "trend_score": 80.0,
                "entry_score": 82.0,
                "buy_score": 88.0,
                "trigger_reason": "demo",
            }
        ]
    )

    results = backtest_signal_returns(signals, {"600000": bars}, config.trend_backtest, entry_timing="next_open")

    assert len(results) == 1
    assert results.loc[0, "entry_price"] == 11.0
    assert results.loc[0, "exit_date"] == pd.Timestamp("2026-04-14")
    assert results.loc[0, "return_pct"] == round(12.5 / 11.0 - 1, 4)


def test_backtest_portfolios_next_open_ranks_by_buy_score() -> None:
    config = _load_test_config()
    config.trend_backtest.holding_days = (2,)
    config.trend_backtest.portfolio_top_n = (1,)
    bars_a = pd.DataFrame(
        [
            {"trade_date": pd.Timestamp("2026-04-10"), "symbol": "600000", "open": 10.0, "close": 10.0, "high": 10.2, "low": 9.8},
            {"trade_date": pd.Timestamp("2026-04-11"), "symbol": "600000", "open": 10.5, "close": 11.0, "high": 11.2, "low": 10.4},
            {"trade_date": pd.Timestamp("2026-04-14"), "symbol": "600000", "open": 11.0, "close": 11.3, "high": 11.5, "low": 10.9},
        ]
    )
    bars_b = pd.DataFrame(
        [
            {"trade_date": pd.Timestamp("2026-04-10"), "symbol": "600001", "open": 10.0, "close": 10.0, "high": 10.2, "low": 9.8},
            {"trade_date": pd.Timestamp("2026-04-11"), "symbol": "600001", "open": 10.2, "close": 10.1, "high": 10.3, "low": 10.0},
            {"trade_date": pd.Timestamp("2026-04-14"), "symbol": "600001", "open": 10.1, "close": 10.0, "high": 10.2, "low": 9.9},
        ]
    )
    entries = pd.DataFrame(
        [
            {"trade_date": pd.Timestamp("2026-04-10"), "symbol": "600000", "name": "甲", "signal_type": "breakout", "trend_score": 70.0, "entry_score": 72.0, "buy_score": 90.0, "trigger_reason": "a"},
            {"trade_date": pd.Timestamp("2026-04-10"), "symbol": "600001", "name": "乙", "signal_type": "pullback", "trend_score": 75.0, "entry_score": 76.0, "buy_score": 60.0, "trigger_reason": "b"},
        ]
    )

    outputs = backtest_portfolios(
        entries,
        {"600000": bars_a, "600001": bars_b},
        config.trend_backtest,
        entry_timing="next_open",
        rank_column="buy_score",
    )

    assert len(outputs["positions"]) == 1
    assert outputs["positions"].loc[0, "symbol"] == "600000"
