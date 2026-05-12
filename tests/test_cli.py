from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

from stocks_analyzer.cli import (
    _append_recent_atr_summary,
    _append_recent_macd_summary,
    _command_needs_network,
    _prepare_pattern_results,
    _run_pattern,
    _run_update,
    build_parser,
)
from stocks_analyzer.config import load_config
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.storage import Storage


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_build_parser_accepts_daily_screening_main_commands() -> None:
    parser = build_parser()

    update = parser.parse_args(["update", "603588", "--start-date", "20240101"])
    assert update.command == "update"
    assert update.symbol == "603588"
    assert update.data_interface == "sina"
    assert update.update_index is False

    intraday = parser.parse_args(["intraday-update", "--data-interface", "sina_raw", "--limit", "5", "--watchlist-only"])
    assert intraday.command == "intraday-update"
    assert intraday.symbol is None
    assert intraday.data_interface == "sina_raw"
    assert intraday.limit == 5
    assert intraday.watchlist_only is True

    intraday_screening = parser.parse_args(["intraday-screening", "--date", "2026-05-08", "--skip-intraday-update", "--watchlist-only"])
    assert intraday_screening.command == "intraday-screening"
    assert intraday_screening.date == "2026-05-08"
    assert intraday_screening.data_interface == "sina_raw"
    assert intraday_screening.skip_intraday_update is True
    assert intraday_screening.watchlist_only is True
    assert intraday_screening.keep_report_dates == 10

    pattern = parser.parse_args(["pattern", "--1", "--5", "--as-of", "2026-05-07"])
    assert pattern.command == "pattern"
    assert pattern.pattern1 is True
    assert pattern.pattern5 is True

    daily = parser.parse_args(["daily-screening", "--date", "2026-05-07", "--start-date", "20150101"])
    assert daily.command == "daily-screening"
    assert daily.date == "2026-05-07"
    assert daily.start_date == "20150101"

    backtest = parser.parse_args(
        ["backtest-daily-screening-components", "--start-date", "2026-01-01", "--end-date", "2026-04-30"]
    )
    assert backtest.command == "backtest-daily-screening-components"
    assert backtest.horizons == "5,10,20,60"
    assert backtest.top_n == 20
    assert backtest.phase1_filter_rate == 0.2


def test_build_parser_exposes_current_phase_commands_only() -> None:
    parser = build_parser()

    assert parser.parse_args(["predict-tail-risk", "--date", "2026-05-07"]).command == "predict-tail-risk"
    assert parser.parse_args(["predict-barrier-risk", "--date", "2026-05-07"]).command == "predict-barrier-risk"
    assert parser.parse_args(["predict-alpha158-qlib-return", "--date", "2026-05-07"]).command == "predict-alpha158-qlib-return"
    assert parser.parse_args(["predict-trade-day-gate", "--date", "2026-05-07"]).command == "predict-trade-day-gate"
    assert parser.parse_args(["validate-mcd-crash-risk", "--end-date", "2026-05-07"]).command == "validate-mcd-crash-risk"
    assert parser.parse_args(
        ["backtest-daily-screening-components", "--start-date", "2026-01-01", "--end-date", "2026-04-30"]
    ).command == "backtest-daily-screening-components"


def test_command_needs_network_only_for_update() -> None:
    assert _command_needs_network("update") is True
    assert _command_needs_network("intraday-update") is True
    assert _command_needs_network("intraday-screening") is True
    assert _command_needs_network("macd") is False
    assert _command_needs_network("daily-screening") is False


def test_prepare_pattern_results_maps_pattern_ids_and_dedupes() -> None:
    rows = [
        {
            "trade_date": date(2026, 5, 7),
            "symbol": "600000",
            "name": "测试股份",
            "strategy_name": "volume_top_pre_breakout",
            "close": 10.0,
            "reason": "first",
        },
        {
            "trade_date": date(2026, 5, 7),
            "symbol": "600000",
            "name": "测试股份",
            "strategy_name": "volume_top_pre_breakout",
            "close": 10.1,
            "reason": "duplicate",
        },
        {
            "trade_date": date(2026, 5, 7),
            "symbol": "600001",
            "name": "测试股份2",
            "strategy_name": "trend_pullback",
            "close": 11.0,
            "reason": "second",
        },
    ]

    prepared = _prepare_pattern_results(pd.DataFrame(rows))

    assert len(prepared) == 2
    assert prepared["pattern_id"].tolist() == ["1", "5"]
    assert prepared["symbol"].tolist() == ['="600000"', '="600001"']


def test_append_recent_macd_summary_merges_state_columns_from_saved_report() -> None:
    tmp_path = _make_workspace_tmp_dir("macd_summary_merge")
    config = load_config(ROOT / "config" / "default.yaml")
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)

    macd_dir = paths.reports_dir / "macd"
    macd_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "symbol": '="600000"',
                "macd_cross_state": "golden_cross",
                "macd_divergence_state": "top_divergence",
                "volume_price_divergence_state": "bullish",
                "macd_top_divergence_15d": True,
                "macd_bottom_divergence_15d": False,
            }
        ]
    ).to_csv(macd_dir / "macd_2026-05-07.csv", index=False, encoding="utf-8-sig")

    exported = pd.DataFrame(
        [{"trade_date": "2026-05-07", "symbol": '="600000"', "name": "测试股份", "pattern_id": "1", "close": 10.0}]
    )

    enriched = _append_recent_macd_summary(storage, exported, as_of=date(2026, 5, 7))

    assert enriched.loc[0, "macd_cross_state"] == "golden_cross"
    assert enriched.loc[0, "macd_divergence_state"] == "top_divergence"
    assert enriched.loc[0, "volume_price_divergence_state"] == "bullish"
    assert bool(enriched.loc[0, "macd_top_divergence_15d"]) is True


def test_append_recent_atr_summary_merges_columns_from_saved_report() -> None:
    tmp_path = _make_workspace_tmp_dir("atr_summary_merge")
    config = load_config(ROOT / "config" / "default.yaml")
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)

    atr_dir = paths.reports_dir / "atr"
    atr_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "代码": '="600000"',
                "名称": "测试股份",
                "交易日期": "2026-05-07",
                "收盘价": 10.0,
                "ATR14": 0.8,
                "ATR%": 8.0,
                "1ATR止损参考": 9.2,
                "2ATR止损参考": 8.4,
                "2ATR止盈参考": 11.6,
                "3ATR止盈参考": 12.4,
                "波动分层": "高波动",
            }
        ]
    ).to_csv(atr_dir / "atr_2026-05-07.csv", index=False, encoding="utf-8-sig")

    exported = pd.DataFrame(
        [{"trade_date": "2026-05-07", "symbol": '="600000"', "name": "测试股份", "pattern_id": "1", "close": 10.0}]
    )

    enriched = _append_recent_atr_summary(storage, exported, as_of=date(2026, 5, 7))

    assert enriched.loc[0, "atr_14"] == 0.8
    assert enriched.loc[0, "atr_pct_14"] == 0.08
    assert enriched.loc[0, "atr_stop_loss_2x"] == 8.4
    assert enriched.loc[0, "atr_volatility_regime"] == "高波动"


def test_run_update_reports_progress_for_each_symbol(monkeypatch) -> None:
    progress_calls: list[tuple[int, int]] = []
    universe = pd.DataFrame([{"symbol": "600000"}, {"symbol": "600001"}])

    monkeypatch.setattr("stocks_analyzer.cli._refresh_or_load_universe", lambda storage, provider, exclude_st: universe.copy())
    monkeypatch.setattr("stocks_analyzer.cli._update_daily_cache_for_symbol", lambda **kwargs: Path("C:/tmp/daily.parquet"))
    monkeypatch.setattr("stocks_analyzer.cli._log_scan_progress", lambda stage_name, current, total: progress_calls.append((current, total)))

    _run_update(
        storage=object(),
        provider=object(),
        exclude_st=True,
        adjust="qfq",
        symbol=None,
        start_date="20240101",
        end_date="20260507",
        limit=None,
        update_indexes=False,
        index_interface="sina",
    )

    assert progress_calls == [(1, 2), (2, 2)]


def test_run_pattern_updates_pattern_watchlist(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("pattern_updates_watchlist")
    config = load_config(ROOT / "config" / "default.yaml")
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)
    exported = pd.DataFrame(
        [
            {
                "trade_date": "2026-05-07",
                "symbol": '="600000"',
                "name": "测试股份",
                "pattern_id": "1",
                "close": 10.0,
                "reason": "demo",
            }
        ]
    )

    monkeypatch.setattr("stocks_analyzer.cli._ensure_universe", lambda storage, provider_name, exclude_st: None)
    monkeypatch.setattr("stocks_analyzer.cli.Screener.run", lambda self, as_of, selected_strategies, symbols=None, progress_callback=None: pd.DataFrame())
    monkeypatch.setattr("stocks_analyzer.cli._prepare_pattern_results", lambda results: exported.copy())
    monkeypatch.setattr("stocks_analyzer.cli._append_recent_macd_summary", lambda storage, exported, as_of, symbols=None: exported)
    monkeypatch.setattr("stocks_analyzer.cli._append_recent_atr_summary", lambda storage, exported, as_of, symbols=None: exported)
    monkeypatch.setattr("stocks_analyzer.cli.append_daily_phase_display_columns", lambda exported, project_root, trade_date: exported)

    _run_pattern(
        storage=storage,
        provider_name=config.provider,
        config=config,
        as_of=date(2026, 5, 7),
        selected_patterns=["volume_top_pre_breakout"],
        limit=20,
        output=None,
    )

    watchlist = json.loads((tmp_path / "reports" / "watchlists" / "watchlist_2026-05-07.json").read_text(encoding="utf-8"))
    pattern_watchlist = json.loads((tmp_path / "reports" / "watchlists" / "watchlist_pattern_2026-05-07.json").read_text(encoding="utf-8"))
    assert watchlist["candidate_count"] == 1
    assert pattern_watchlist["candidate_count"] == 1
    assert pattern_watchlist["candidates"][0]["symbol"] == "600000"
