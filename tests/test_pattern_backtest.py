from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

from stocks_analyzer.config import load_config
from stocks_analyzer.pattern_backtest import (
    build_pattern_forward_price_frame,
    sample_pattern_backtest_trade_dates,
    scan_pattern_backtest_signals,
    summarize_pattern_backtest,
)
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.storage import Storage
from stocks_analyzer.strategies import TREND_PULLBACK, VOLUME_TOP_PRE_BREAKOUT
from stocks_analyzer.trend_backtest import backtest_signal_returns
from stocks_analyzer.trend_reporting import save_pattern_backtest_reports


ROOT = Path(__file__).resolve().parents[1]


def test_scan_pattern_backtest_signals_applies_five_trading_day_cooldown(monkeypatch) -> None:
    storage, config = _make_storage_and_config(_make_workspace_tmp_dir("pattern_backtest_cooldown"))
    storage.save_universe(pd.DataFrame([{"symbol": "600000", "name": "测试"}]))
    bars = _make_daily_bars(72)
    storage.save_daily_bars("600000", bars)

    def fake_evaluate_strategies(history_df, instrument, config, selected):
        latest = history_df.iloc[-1]
        return [
            {
                "trade_date": pd.Timestamp(latest["trade_date"]).date().isoformat(),
                "symbol": str(instrument["symbol"]),
                "name": str(instrument["name"]),
                "strategy_name": VOLUME_TOP_PRE_BREAKOUT,
                "close": float(latest["close"]),
                "reason": "always match",
            }
        ]

    monkeypatch.setattr("stocks_analyzer.pattern_backtest.evaluate_strategies", fake_evaluate_strategies)
    monkeypatch.setattr("stocks_analyzer.pattern_backtest.required_history_days", lambda config, strategies: 1)

    signals = scan_pattern_backtest_signals(
        storage,
        config,
        start_date=pd.Timestamp(bars["trade_date"].iloc[60]).date(),
        end_date=pd.Timestamp(bars["trade_date"].iloc[66]).date(),
        selected_strategies=[VOLUME_TOP_PRE_BREAKOUT],
        cooldown_trading_days=5,
    )

    assert signals["trade_date"].dt.date.tolist() == [
        pd.Timestamp(bars["trade_date"].iloc[60]).date(),
        pd.Timestamp(bars["trade_date"].iloc[65]).date(),
    ]
    assert signals["pattern_id"].tolist() == ["1", "1"]


def test_scan_pattern_backtest_signals_keeps_different_patterns_on_same_day(monkeypatch) -> None:
    storage, config = _make_storage_and_config(_make_workspace_tmp_dir("pattern_backtest_same_day"))
    storage.save_universe(pd.DataFrame([{"symbol": "600000", "name": "测试"}]))
    bars = _make_daily_bars(65)
    storage.save_daily_bars("600000", bars)

    def fake_evaluate_strategies(history_df, instrument, config, selected):
        latest = history_df.iloc[-1]
        trade_date = pd.Timestamp(latest["trade_date"]).date().isoformat()
        return [
            {
                "trade_date": trade_date,
                "symbol": str(instrument["symbol"]),
                "name": str(instrument["name"]),
                "strategy_name": VOLUME_TOP_PRE_BREAKOUT,
                "close": float(latest["close"]),
                "reason": "mode 1",
            },
            {
                "trade_date": trade_date,
                "symbol": str(instrument["symbol"]),
                "name": str(instrument["name"]),
                "strategy_name": TREND_PULLBACK,
                "close": float(latest["close"]),
                "reason": "mode 5",
            },
        ]

    monkeypatch.setattr("stocks_analyzer.pattern_backtest.evaluate_strategies", fake_evaluate_strategies)
    monkeypatch.setattr("stocks_analyzer.pattern_backtest.required_history_days", lambda config, strategies: 1)

    signal_date = pd.Timestamp(bars["trade_date"].iloc[60]).date()
    signals = scan_pattern_backtest_signals(
        storage,
        config,
        start_date=signal_date,
        end_date=signal_date,
        selected_strategies=[VOLUME_TOP_PRE_BREAKOUT, TREND_PULLBACK],
        cooldown_trading_days=5,
    )

    assert signals["pattern_id"].tolist() == ["1", "5"]


def test_scan_pattern_backtest_signals_only_evaluates_sampled_dates(monkeypatch) -> None:
    storage, config = _make_storage_and_config(_make_workspace_tmp_dir("pattern_backtest_sampled_dates"))
    storage.save_universe(pd.DataFrame([{"symbol": "600000", "name": "测试"}]))
    bars = _make_daily_bars(65)
    storage.save_daily_bars("600000", bars)
    evaluated_dates = []

    def fake_evaluate_strategies(history_df, instrument, config, selected):
        latest = history_df.iloc[-1]
        trade_date = pd.Timestamp(latest["trade_date"]).date()
        evaluated_dates.append(trade_date)
        return [
            {
                "trade_date": trade_date.isoformat(),
                "symbol": str(instrument["symbol"]),
                "name": str(instrument["name"]),
                "strategy_name": VOLUME_TOP_PRE_BREAKOUT,
                "close": float(latest["close"]),
                "reason": "sampled",
            }
        ]

    monkeypatch.setattr("stocks_analyzer.pattern_backtest.evaluate_strategies", fake_evaluate_strategies)
    monkeypatch.setattr("stocks_analyzer.pattern_backtest.required_history_days", lambda config, strategies: 1)

    sampled_dates = [
        pd.Timestamp(bars["trade_date"].iloc[60]).date(),
        pd.Timestamp(bars["trade_date"].iloc[62]).date(),
    ]
    signals = scan_pattern_backtest_signals(
        storage,
        config,
        start_date=pd.Timestamp(bars["trade_date"].iloc[58]).date(),
        end_date=pd.Timestamp(bars["trade_date"].iloc[64]).date(),
        selected_strategies=[VOLUME_TOP_PRE_BREAKOUT],
        sampled_trade_dates=sampled_dates,
        cooldown_trading_days=1,
    )

    assert evaluated_dates == sampled_dates
    assert signals["trade_date"].dt.date.tolist() == sampled_dates


def test_sample_pattern_backtest_trade_dates_is_seeded_and_sorted() -> None:
    storage, config = _make_storage_and_config(_make_workspace_tmp_dir("pattern_backtest_sample_dates"))
    storage.save_universe(pd.DataFrame([{"symbol": "600000", "name": "测试"}]))
    bars = _make_daily_bars(20)
    storage.save_daily_bars("600000", bars)

    sampled = sample_pattern_backtest_trade_dates(
        storage,
        start_date=pd.Timestamp(bars["trade_date"].iloc[5]).date(),
        end_date=pd.Timestamp(bars["trade_date"].iloc[15]).date(),
        sample_size=4,
        seed=7,
    )
    repeated = sample_pattern_backtest_trade_dates(
        storage,
        start_date=pd.Timestamp(bars["trade_date"].iloc[5]).date(),
        end_date=pd.Timestamp(bars["trade_date"].iloc[15]).date(),
        sample_size=4,
        seed=7,
    )

    assert sampled == sorted(sampled)
    assert sampled == repeated
    assert len(sampled) == 4
    assert sampled[0] >= pd.Timestamp(bars["trade_date"].iloc[5]).date()
    assert sampled[-1] <= pd.Timestamp(bars["trade_date"].iloc[15]).date()


def test_summarize_pattern_backtest_uses_next_open_returns() -> None:
    config = load_config("config/default.yaml")
    config.trend_backtest.holding_days = (2,)
    bars = _make_daily_bars(5, closes=[10.0, 11.0, 12.0, 9.0, 13.0], opens=[10.0, 10.0, 10.0, 10.0, 10.0])
    signals = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp(bars["trade_date"].iloc[0]),
                "symbol": "600000",
                "name": "测试",
                "signal_type": VOLUME_TOP_PRE_BREAKOUT,
                "trigger_reason": "mode 1",
                "trend_score": 0.0,
                "entry_score": 0.0,
            }
        ]
    )

    detail = backtest_signal_returns(signals, {"600000": bars}, config.trend_backtest, entry_timing="next_open")
    detail["pattern_id"] = detail["signal_type"].map({VOLUME_TOP_PRE_BREAKOUT: "1"})
    summary = summarize_pattern_backtest(detail)

    assert detail.loc[0, "entry_price"] == 10.0
    assert detail.loc[0, "return_pct"] == 0.2
    assert summary.loc[0, "pattern_id"] == "1"
    assert summary.loc[0, "win_rate"] == 1.0


def test_build_pattern_forward_price_frame_saves_40_day_daily_prices() -> None:
    bars = _make_daily_bars(45, closes=[10.0 + index for index in range(45)], opens=[10.5 + index for index in range(45)])
    signals = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp(bars["trade_date"].iloc[0]),
                "symbol": "600000",
                "name": "测试",
                "pattern_id": "1",
                "signal_type": VOLUME_TOP_PRE_BREAKOUT,
                "trigger_reason": "mode 1",
            }
        ]
    )

    forward_prices = build_pattern_forward_price_frame(signals, {"600000": bars}, forward_days=40, entry_timing="next_open")

    assert len(forward_prices) == 40
    assert forward_prices["forward_day"].tolist() == list(range(1, 41))
    assert forward_prices.loc[0, "entry_date"] == pd.Timestamp(bars["trade_date"].iloc[1])
    assert forward_prices.loc[0, "entry_price"] == 11.5
    assert forward_prices.loc[0, "open"] == 11.5
    assert forward_prices.loc[0, "open_return_pct"] == 0.0
    assert forward_prices.loc[1, "close_return_pct"] == round(float(bars["close"].iloc[2]) / 11.5 - 1, 4)
    assert "ma_20" in forward_prices.columns
    assert "close_below_ma20" in forward_prices.columns
    assert pd.notna(forward_prices.loc[18, "ma_20"])


def test_save_pattern_backtest_reports_writes_expected_files() -> None:
    config = load_config("config/default.yaml")
    tmp_path = _make_workspace_tmp_dir("pattern_backtest_reports")
    paths = ProjectPaths(tmp_path, config.storage)
    paths.ensure()
    detail = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2026-04-10"),
                "symbol": "600000",
                "pattern_id": "1",
                "return_pct": 0.05,
                "entry_note": "next open",
            }
        ]
    )
    summary = pd.DataFrame([{"pattern_id": "1", "holding_days": 5, "signal_count": 1, "win_rate": 1.0}])
    forward_prices = pd.DataFrame([{"sample_id": "20260410_600000_1", "forward_day": 1, "close": 10.0}])

    paths_map = save_pattern_backtest_reports(
        paths,
        report_date=date(2026, 4, 24),
        detail=detail,
        summary=summary,
        forward_prices=forward_prices,
        sampled_trade_dates=[date(2026, 4, 10)],
        sample_seed=42,
    )

    assert paths_map["detail_path"].exists()
    assert paths_map["summary_path"].exists()
    assert paths_map["json_path"].exists()
    assert paths_map["forward_prices_path"].exists()
    assert "pattern_backtest_details_2026-04-24.csv" in str(paths_map["detail_path"])


def _make_storage_and_config(tmp_path) -> tuple[Storage, object]:
    config = load_config("config/default.yaml")
    config.universe.min_avg_amount_20d = 0.0
    config.history_momentum_filter.lookback_days = 0
    config.trend_backtest.holding_days = (5,)
    paths = ProjectPaths(tmp_path, config.storage)
    return Storage(paths), config


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_daily_bars(
    periods: int,
    *,
    closes: list[float] | None = None,
    opens: list[float] | None = None,
) -> pd.DataFrame:
    if closes is None:
        closes = [10.0 + index * 0.1 for index in range(periods)]
    if opens is None:
        opens = closes.copy()
    dates = pd.date_range("2026-01-01", periods=periods, freq="D")
    rows = []
    for index in range(periods):
        close = float(closes[index])
        open_price = float(opens[index])
        rows.append(
            {
                "trade_date": dates[index],
                "symbol": "600000",
                "open": open_price,
                "close": close,
                "high": max(open_price, close) * 1.02,
                "low": min(open_price, close) * 0.98,
                "volume": 1_000_000 + index,
                "amount": (1_000_000 + index) * close,
                "pct_change": 0.0,
                "change": 0.0,
                "amplitude": 0.0,
                "turnover": 1.0,
            }
        )
    return pd.DataFrame(rows)
