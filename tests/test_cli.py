import os
import json
from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd
from stocks_analyzer.cli import (
    PATTERN_LABEL_MAP,
    _append_recent_atr_summary,
    _append_recent_macd_summary,
    _append_recent_trend_universe_summary,
    _command_needs_network,
    _configure_network,
    _load_local_env,
    _append_recent_trend_summary,
    _run_pattern,
    _run_trend_universe,
    _run_trend,
    _run_update,
    _prepare_pattern_results,
    build_parser,
)
from stocks_analyzer.config import load_config
from stocks_analyzer.models import NetworkConfig
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.storage import Storage


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_build_parser_accepts_update_with_symbol() -> None:
    parser = build_parser()
    args = parser.parse_args(["update", "603588", "--start-date", "20240101"])

    assert args.command == "update"
    assert args.symbol == "603588"
    assert args.start_date == "20240101"


def test_build_parser_accepts_pattern_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["pattern", "--1", "--5", "--6", "--plot-all", "--as-of", "2026-04-10"])

    assert args.command == "pattern"
    assert args.pattern1 is True
    assert args.pattern5 is True
    assert args.pattern6 is True
    assert args.plot_all is True
    assert args.as_of == "2026-04-10"


def test_build_parser_accepts_plot_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["plot", "603588", "--start-date", "20240101"])

    assert args.command == "plot"
    assert args.symbol == "603588"
    assert args.start_date == "20240101"


def test_build_parser_accepts_train_prob_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["train-prob", "--train-end", "2025-12-31"])

    assert args.command == "train-prob"
    assert args.train_end == "2025-12-31"


def test_build_parser_accepts_predict_prob_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["predict-prob", "--date", "2026-04-10", "--top-n", "15"])

    assert args.command == "predict-prob"
    assert args.date == "2026-04-10"
    assert args.top_n == 15


def test_build_parser_accepts_tradingview_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["tradingview", "--date", "2026-04-10", "--top-n", "30"])

    assert args.command == "tradingview"
    assert args.date == "2026-04-10"
    assert args.top_n == 30


def test_build_parser_accepts_macd_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["macd", "--date", "2026-04-10", "--top-n", "30"])

    assert args.command == "macd"
    assert args.date == "2026-04-10"
    assert args.top_n == 30


def test_build_parser_accepts_atr_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["atr", "--date", "2026-04-10", "--top-n", "18"])

    assert args.command == "atr"
    assert args.date == "2026-04-10"
    assert args.top_n == 18


def test_build_parser_accepts_xueqiu_archive_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["xueqiu-archive", "--max-posts", "10", "--refresh", "--headed"])

    assert args.command == "xueqiu-archive"
    assert args.max_posts == 10
    assert args.refresh is True
    assert args.headed is True


def test_build_parser_accepts_daily_screening_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["daily-screening", "--date", "2026-04-10", "--start-date", "20240101"])

    assert args.command == "daily-screening"
    assert args.date == "2026-04-10"
    assert args.start_date == "20240101"


def test_build_parser_accepts_intraday_screening_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["intraday-screening", "--date", "2026-04-10", "--watchlist-date", "2026-04-09", "--top-n", "15"]
    )

    assert args.command == "intraday-screening"
    assert args.date == "2026-04-10"
    assert args.watchlist_date == "2026-04-09"
    assert args.top_n == 15


def test_build_parser_accepts_trend_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["trend", "--date", "2026-04-10", "--top-n", "14"])

    assert args.command == "trend"
    assert args.date == "2026-04-10"
    assert args.top_n == 14


def test_build_parser_accepts_trend_universe_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["trend-universe", "--date", "2026-04-10", "--top-n", "15"])

    assert args.command == "trend-universe"
    assert args.date == "2026-04-10"
    assert args.top_n == 15


def test_build_parser_accepts_trend_signals_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["trend-signals", "--date", "2026-04-10", "--top-n", "12"])

    assert args.command == "trend-signals"
    assert args.date == "2026-04-10"
    assert args.top_n == 12


def test_build_parser_accepts_trend_score_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["trend-score", "--date", "2026-04-10", "--top-n", "9"])

    assert args.command == "trend-score"
    assert args.date == "2026-04-10"
    assert args.top_n == 9


def test_build_parser_accepts_trend_entries_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["trend-entries", "--date", "2026-04-10", "--top-n", "7"])

    assert args.command == "trend-entries"
    assert args.date == "2026-04-10"
    assert args.top_n == 7


def test_build_parser_accepts_backtest_signals_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["backtest-signals", "--date", "2026-04-10", "--start-date", "2026-01-01"])

    assert args.command == "backtest-signals"
    assert args.date == "2026-04-10"
    assert args.start_date == "2026-01-01"


def test_build_parser_accepts_backtest_portfolio_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["backtest-portfolio", "--date", "2026-04-10", "--top-n", "8"])

    assert args.command == "backtest-portfolio"
    assert args.date == "2026-04-10"
    assert args.top_n == 8


def test_build_parser_accepts_backtest_entries_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["backtest-entries", "--date", "2026-04-10", "--start-date", "2026-01-01"])

    assert args.command == "backtest-entries"
    assert args.date == "2026-04-10"
    assert args.start_date == "2026-01-01"


def test_build_parser_accepts_backtest_patterns_command_and_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "backtest-patterns",
            "--date",
            "2026-04-10",
            "--start-date",
            "2026-01-01",
            "--1",
            "--3",
            "--sample-dates",
            "12",
            "--sample-seed",
            "7",
            "--save-forward-prices",
            "--forward-days",
            "40",
        ]
    )

    assert args.command == "backtest-patterns"
    assert args.date == "2026-04-10"
    assert args.start_date == "2026-01-01"
    assert args.pattern1 is True
    assert args.pattern3 is True
    assert args.sample_dates == 12
    assert args.sample_seed == 7
    assert args.save_forward_prices is True
    assert args.forward_days == 40


def test_build_parser_accepts_backtest_entries_portfolio_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["backtest-entries-portfolio", "--date", "2026-04-10", "--top-n", "6"])

    assert args.command == "backtest-entries-portfolio"
    assert args.date == "2026-04-10"
    assert args.top_n == 6


def test_build_parser_accepts_research_pattern_stops_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "research-pattern-stops",
            "--input",
            "reports/backtests/patterns/pattern_forward_prices_2026-04-24.csv",
            "--holding-days",
            "5,10,20,40",
            "--take-profits",
            "0.04,0.08",
            "--stop-losses",
            "0.03,0.05",
            "--same-day-policy",
            "take-profit-first",
            "--ma20-stop",
            "--ma20-stop-tolerance",
            "0.01",
        ]
    )

    assert args.command == "research-pattern-stops"
    assert args.input.endswith("pattern_forward_prices_2026-04-24.csv")
    assert args.holding_days == "5,10,20,40"
    assert args.take_profits == "0.04,0.08"
    assert args.stop_losses == "0.03,0.05"
    assert args.same_day_policy == "take-profit-first"
    assert args.ma20_stop is True
    assert args.ma20_stop_tolerance == 0.01


def test_command_needs_network_only_for_networked_commands() -> None:
    assert _command_needs_network("update") is True
    assert _command_needs_network("plot") is True
    assert _command_needs_network("intraday-screening") is True
    assert _command_needs_network("xueqiu-archive") is True
    assert _command_needs_network("macd") is False
    assert _command_needs_network("atr") is False
    assert _command_needs_network("trend-universe") is False
    assert _command_needs_network("trend") is False
    assert _command_needs_network("pattern") is False


def test_build_parser_accepts_research_thresholds_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "research-thresholds",
            "--date",
            "2026-04-10",
            "--start-date",
            "2025-01-01",
            "--sample-mode",
            "weekly",
            "--train-end-date",
            "2025-12-31",
        ]
    )

    assert args.command == "research-thresholds"
    assert args.date == "2026-04-10"
    assert args.start_date == "2025-01-01"
    assert args.sample_mode == "weekly"
    assert args.train_end_date == "2025-12-31"


def test_load_local_env_sets_missing_env_vars_only() -> None:
    root = _make_workspace_tmp_dir("local_env")
    env_path = root / ".env.local"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                "ITICK_TOKEN=file-token",
                "TUSHARE_TOKEN=file-ts-token",
            ]
        ),
        encoding="utf-8",
    )

    import os

    previous_itick = os.environ.get("ITICK_TOKEN")
    previous_tushare = os.environ.get("TUSHARE_TOKEN")
    os.environ["ITICK_TOKEN"] = "existing-token"
    os.environ.pop("TUSHARE_TOKEN", None)
    try:
        _load_local_env(env_path)
        assert os.environ["ITICK_TOKEN"] == "existing-token"
        assert os.environ["TUSHARE_TOKEN"] == "file-ts-token"
    finally:
        if previous_itick is None:
            os.environ.pop("ITICK_TOKEN", None)
        else:
            os.environ["ITICK_TOKEN"] = previous_itick
        if previous_tushare is None:
            os.environ.pop("TUSHARE_TOKEN", None)
        else:
            os.environ["TUSHARE_TOKEN"] = previous_tushare


def test_configure_network_prefers_existing_proxy_env(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://existing-http:8000")
    monkeypatch.setenv("HTTPS_PROXY", "http://existing-https:8443")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    _configure_network(
        NetworkConfig(
            http_proxy="http://127.0.0.1:7897",
            https_proxy="http://127.0.0.1:7897",
            socks5_proxy=None,
            no_proxy="127.0.0.1,localhost",
        )
    )

    assert os.environ["HTTP_PROXY"] == "http://existing-http:8000"
    assert os.environ["HTTPS_PROXY"] == "http://existing-https:8443"
    assert os.environ["http_proxy"] == "http://existing-http:8000"
    assert os.environ["https_proxy"] == "http://existing-https:8443"
    assert "NO_PROXY" not in os.environ
    assert "no_proxy" not in os.environ


def test_configure_network_skips_unreachable_local_proxy(monkeypatch, caplog) -> None:
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    def fake_create_connection(*args, **kwargs):
        raise OSError("refused")

    monkeypatch.setattr("stocks_analyzer.cli.socket.create_connection", fake_create_connection)

    with caplog.at_level("WARNING"):
        _configure_network(
            NetworkConfig(
                http_proxy="http://127.0.0.1:7897",
                https_proxy="http://localhost:7897",
                socks5_proxy=None,
                no_proxy="127.0.0.1,localhost",
            )
        )

    assert "HTTP_PROXY" not in os.environ
    assert "HTTPS_PROXY" not in os.environ
    assert "NO_PROXY" not in os.environ
    assert "fallback to direct connection" in caplog.text


def test_configure_network_applies_reachable_local_proxy(monkeypatch) -> None:
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    class FakeConnection:
        def close(self) -> None:
            return None

    monkeypatch.setattr("stocks_analyzer.cli.socket.create_connection", lambda *args, **kwargs: FakeConnection())

    _configure_network(
        NetworkConfig(
            http_proxy="http://127.0.0.1:7897",
            https_proxy="http://127.0.0.1:7897",
            socks5_proxy=None,
            no_proxy="127.0.0.1,localhost",
        )
    )

    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7897"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7897"
    assert os.environ["NO_PROXY"] == "127.0.0.1,localhost"


def test_prepare_pattern_results_maps_internal_type_to_pattern_id() -> None:
    results = pd.DataFrame(
        [
            {
                "trade_date": "2026-04-10",
                "symbol": "600000",
                "name": "测试股份",
                "strategy_name": "volume_top_pre_breakout",
                "close": 10.0,
                "old_high_date": "2026-03-01",
                "old_high_price": 10.8,
                "days_since_old_high": 28,
                "distance_to_old_high_pct": 0.0741,
                "reason": "demo",
            }
        ]
    )

    prepared = _prepare_pattern_results(results)

    assert prepared["pattern_id"].tolist() == [PATTERN_LABEL_MAP["volume_top_pre_breakout"]]
    assert "strategy_name" not in prepared.columns
    assert prepared.columns[:9].tolist() == [
        "trade_date",
        "symbol",
        "name",
        "pattern_id",
        "close",
        "old_high_date",
        "old_high_price",
        "days_since_old_high",
        "distance_to_old_high_pct",
    ]


def test_prepare_pattern_results_deduplicates_same_symbol_and_pattern() -> None:
    results = pd.DataFrame(
        [
            {
                "trade_date": "2026-04-10",
                "symbol": "600000",
                "name": "测试股份",
                "strategy_name": "volume_top_pre_breakout",
                "close": 10.0,
                "reason": "demo-a",
            },
            {
                "trade_date": "2026-04-10",
                "symbol": "600000",
                "name": "测试股份",
                "strategy_name": "volume_top_pre_breakout",
                "close": 10.0,
                "reason": "demo-b",
            },
            {
                "trade_date": "2026-04-10",
                "symbol": "600000",
                "name": "测试股份",
                "strategy_name": "trend_pullback",
                "close": 10.0,
                "reason": "demo-c",
            },
        ]
    )

    prepared = _prepare_pattern_results(results)

    assert len(prepared) == 2
    assert prepared["pattern_id"].tolist() == ["1", "5"]


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
    ).to_csv(macd_dir / "macd_2026-04-10.csv", index=False, encoding="utf-8-sig")

    exported = pd.DataFrame(
        [
            {
                "trade_date": "2026-04-10",
                "symbol": '="600000"',
                "name": "测试股份",
                "pattern_id": "1",
                "close": 10.0,
                "reason": "demo",
            }
        ]
    )

    enriched = _append_recent_macd_summary(storage, exported, as_of=date(2026, 4, 10))

    assert enriched.loc[0, "macd_cross_state"] == "golden_cross"
    assert enriched.loc[0, "macd_divergence_state"] == "top_divergence"
    assert enriched.loc[0, "volume_price_divergence_state"] == "bullish"
    assert bool(enriched.loc[0, "macd_top_divergence_15d"]) is True
    assert bool(enriched.loc[0, "macd_bottom_divergence_15d"]) is False


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
                "交易日期": "2026-04-10",
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
    ).to_csv(atr_dir / "atr_2026-04-10.csv", index=False, encoding="utf-8-sig")

    exported = pd.DataFrame(
        [
            {
                "trade_date": "2026-04-10",
                "symbol": '="600000"',
                "name": "测试股份",
                "pattern_id": "1",
                "close": 10.0,
            }
        ]
    )

    enriched = _append_recent_atr_summary(storage, exported, as_of=date(2026, 4, 10))

    assert enriched.loc[0, "atr_14"] == 0.8
    assert enriched.loc[0, "atr_pct_14"] == 0.08
    assert enriched.loc[0, "atr_stop_loss_2x"] == 8.4
    assert enriched.loc[0, "atr_volatility_regime"] == "高波动"


def test_append_recent_trend_summary_merges_score_columns_from_saved_report() -> None:
    tmp_path = _make_workspace_tmp_dir("trend_summary_merge")
    config = load_config(ROOT / "config" / "default.yaml")
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)

    trend_dir = paths.reports_dir / "trend"
    trend_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "symbol": "600000",
                "signal_type": "breakout",
                "buy_score": 72.5,
                "price_action_score": 61.2,
                "trend_base_score": 66.0,
                "macd_score": 44.0,
            }
        ]
    ).to_csv(trend_dir / "trend_2026-04-10.csv", index=False, encoding="utf-8-sig")

    exported = pd.DataFrame(
        [
            {
                "trade_date": "2026-04-10",
                "symbol": '="600000"',
                "name": "测试股份",
                "pattern_id": "1",
                "close": 10.0,
            }
        ]
    )

    enriched = _append_recent_trend_summary(storage, exported, config=config, as_of=date(2026, 4, 10))

    assert enriched.loc[0, "signal_type"] == "breakout"
    assert enriched.loc[0, "buy_score"] == 72.5
    assert enriched.loc[0, "price_action_score"] == 61.2


def test_append_recent_trend_summary_deduplicates_saved_report_rows_before_merge() -> None:
    tmp_path = _make_workspace_tmp_dir("trend_summary_merge_dedup")
    config = load_config(ROOT / "config" / "default.yaml")
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)

    trend_dir = paths.reports_dir / "trend"
    trend_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "symbol": "600000",
                "signal_type": "breakout",
                "buy_score": 72.5,
                "price_action_score": 61.2,
            },
            {
                "symbol": '="600000"',
                "signal_type": "breakout",
                "buy_score": 72.5,
                "price_action_score": 61.2,
            },
        ]
    ).to_csv(trend_dir / "trend_2026-04-10.csv", index=False, encoding="utf-8-sig")

    exported = pd.DataFrame(
        [
            {
                "trade_date": "2026-04-10",
                "symbol": '="600000"',
                "name": "测试股份",
                "pattern_id": "1",
                "close": 10.0,
            }
        ]
    )

    enriched = _append_recent_trend_summary(storage, exported, config=config, as_of=date(2026, 4, 10))

    assert len(enriched) == 1
    assert enriched.loc[0, "signal_type"] == "breakout"
    assert enriched.loc[0, "buy_score"] == 72.5


def test_append_recent_trend_universe_summary_merges_first_layer_fields() -> None:
    exported = pd.DataFrame(
        [
            {
                "trade_date": "2026-04-10",
                "symbol": '="600000"',
                "name": "测试股份",
                "pattern_id": "1",
                "close": 10.0,
            }
        ]
    )
    trend_universe = pd.DataFrame(
        [
            {
                "symbol": "600000",
                "in_trend_universe": True,
                "trend_score": 84.0,
                "trend_direction_score": 92.0,
            }
        ]
    )

    enriched = _append_recent_trend_universe_summary(exported, trend_universe)

    assert bool(enriched.loc[0, "in_trend_universe"]) is True
    assert enriched.loc[0, "trend_universe_score"] == 84.0
    assert enriched.loc[0, "trend_direction_score"] == 92.0


def test_run_pattern_updates_watchlist_for_same_trade_date(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("pattern_updates_watchlist")
    config = load_config(ROOT / "config" / "default.yaml")
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)

    exported = pd.DataFrame(
        [
            {
                "trade_date": "2026-04-10",
                "symbol": '="600000"',
                "name": "测试股份",
                "pattern_id": "1",
                "close": 10.0,
                "reason": "demo",
                "tradingview_all_rating_2026-04-08": 0.20,
                "tradingview_all_rating_2026-04-09": 0.30,
                "tradingview_all_rating_2026-04-10": 0.40,
                "tradingview_all_rating_2026-04-11": 0.50,
                "tradingview_all_rating_2026-04-12": 0.60,
                "tradingview_avg_all_rating_5d": 0.40,
                "tradingview_all_rating_label": "buy",
                "macd_top_divergence_15d": False,
                "macd_bottom_divergence_15d": True,
            }
        ]
    )

    monkeypatch.setattr("stocks_analyzer.cli._ensure_universe", lambda storage, provider_name, exclude_st: None)
    monkeypatch.setattr(
        "stocks_analyzer.cli.Screener.run",
        lambda self, as_of, selected_strategies, symbols=None, progress_callback=None: pd.DataFrame(),
    )
    monkeypatch.setattr("stocks_analyzer.cli._prepare_pattern_results", lambda results: exported.copy())
    monkeypatch.setattr("stocks_analyzer.cli._append_recent_tradingview_scores", lambda storage, exported, as_of, lookback_days, symbols=None: exported)
    monkeypatch.setattr("stocks_analyzer.cli._append_recent_macd_summary", lambda storage, exported, as_of, symbols=None: exported)
    monkeypatch.setattr("stocks_analyzer.cli._append_recent_atr_summary", lambda storage, exported, as_of, symbols=None: exported)
    monkeypatch.setattr(
        "stocks_analyzer.cli._load_or_build_trend_universe_summary",
        lambda storage, config, trade_date, symbols=None: pd.DataFrame([{"symbol": "600000", "in_trend_universe": True, "trend_score": 81.0}]),
    )
    monkeypatch.setattr(
        "stocks_analyzer.cli._append_recent_trend_summary",
        lambda storage, config, exported, as_of, symbols=None: exported,
    )

    _run_pattern(
        storage=storage,
        provider_name=config.provider,
        config=config,
        as_of=date(2026, 4, 10),
        selected_patterns=["volume_top_pre_breakout"],
        limit=20,
        output=None,
        plot_all=False,
    )

    watchlist = json.loads((tmp_path / "reports" / "watchlists" / "watchlist_2026-04-10.json").read_text(encoding="utf-8"))
    pattern_watchlist = json.loads((tmp_path / "reports" / "watchlists" / "watchlist_pattern_2026-04-10.json").read_text(encoding="utf-8"))
    assert watchlist["trade_date"] == "2026-04-10"
    assert watchlist["candidate_count"] == 1
    assert watchlist["candidates"][0]["symbol"] == "600000"
    assert pattern_watchlist["candidate_count"] == 1
    assert pattern_watchlist["candidates"][0]["symbol"] == "600000"


def test_run_trend_updates_trend_watchlist_for_same_trade_date(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("trend_updates_watchlist")
    config = load_config(ROOT / "config" / "default.yaml")
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)

    scored = pd.DataFrame(
        [
            {
                "trade_date": "2026-04-10",
                "symbol": "600000",
                "name": "测试趋势",
                "signal_type": "breakout",
                "buy_score": 76.0,
                "price_action_score": 60.0,
                "trend_score": 83.0,
                "trend_base_score": 70.0,
                "macd_cross_state": "golden_cross",
                "macd_divergence_state": "none",
                "volume_price_divergence_state": "none",
            }
        ]
    )

    monkeypatch.setattr(
        "stocks_analyzer.cli.scan_indicator_scored_entries",
        lambda storage, config, trade_date, progress_callback=None: scored.copy(),
    )

    _run_trend(
        storage=storage,
        config=config,
        paths=paths,
        trade_date=date(2026, 4, 10),
        top_n=20,
        output=None,
    )

    trend_watchlist = json.loads((tmp_path / "reports" / "watchlists" / "watchlist_trend_2026-04-10.json").read_text(encoding="utf-8"))
    assert trend_watchlist["trade_date"] == "2026-04-10"
    assert trend_watchlist["candidate_count"] == 1
    assert trend_watchlist["candidates"][0]["symbol"] == "600000"


def test_run_update_reports_progress_for_each_symbol(monkeypatch) -> None:
    progress_calls: list[tuple[int, int]] = []
    universe = pd.DataFrame([{"symbol": "600000"}, {"symbol": "600001"}])

    monkeypatch.setattr("stocks_analyzer.cli._refresh_or_load_universe", lambda storage, provider, exclude_st: universe.copy())
    monkeypatch.setattr(
        "stocks_analyzer.cli._update_daily_cache_for_symbol",
        lambda **kwargs: Path("C:/tmp/daily.parquet"),
    )
    monkeypatch.setattr("stocks_analyzer.cli._log_scan_progress", lambda stage_name, current, total: progress_calls.append((current, total)))

    _run_update(
        storage=object(),
        provider=object(),
        exclude_st=True,
        adjust="qfq",
        symbol=None,
        start_date="20240101",
        end_date="20260422",
        limit=None,
    )

    assert progress_calls == [(1, 2), (2, 2)]


def test_run_pattern_passes_progress_callback_to_screener(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("pattern_progress_callback")
    config = load_config(ROOT / "config" / "default.yaml")
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)
    callback_seen: list[bool] = []

    monkeypatch.setattr("stocks_analyzer.cli._ensure_universe", lambda storage, provider_name, exclude_st: None)

    def fake_run(self, as_of, selected_strategies, symbols=None, progress_callback=None):
        callback_seen.append(progress_callback is not None)
        return pd.DataFrame()

    monkeypatch.setattr("stocks_analyzer.cli.Screener.run", fake_run)
    monkeypatch.setattr("stocks_analyzer.cli._prepare_pattern_results", lambda results: pd.DataFrame())
    monkeypatch.setattr("stocks_analyzer.cli._append_recent_tradingview_scores", lambda storage, exported, as_of, lookback_days, symbols=None: exported)
    monkeypatch.setattr("stocks_analyzer.cli._append_recent_macd_summary", lambda storage, exported, as_of, symbols=None: exported)
    monkeypatch.setattr("stocks_analyzer.cli._append_recent_atr_summary", lambda storage, exported, as_of, symbols=None: exported)
    monkeypatch.setattr("stocks_analyzer.cli._load_or_build_trend_universe_summary", lambda storage, config, trade_date, symbols=None: pd.DataFrame())
    monkeypatch.setattr("stocks_analyzer.cli._append_recent_trend_summary", lambda storage, config, exported, as_of, symbols=None: exported)

    _run_pattern(
        storage=storage,
        provider_name=config.provider,
        config=config,
        as_of=date(2026, 4, 10),
        selected_patterns=["volume_top_pre_breakout"],
        limit=20,
        output=None,
        plot_all=False,
    )

    assert callback_seen == [True]


def test_run_trend_universe_passes_progress_callback(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("trend_universe_progress_callback")
    config = load_config(ROOT / "config" / "default.yaml")
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)
    callback_seen: list[bool] = []
    summary = pd.DataFrame(
        [
            {
                "trade_date": "2026-04-10",
                "symbol": "600000",
                "name": "测试趋势",
                "trend_score": 80.0,
                "trend_direction_score": 70.0,
                "trend_strength_score": 75.0,
            }
        ]
    )

    def fake_scan(storage, config, as_of, symbols=None, include_all=False, progress_callback=None):
        callback_seen.append(progress_callback is not None)
        return summary.copy()

    monkeypatch.setattr("stocks_analyzer.cli.scan_trend_universe", fake_scan)

    _run_trend_universe(
        storage=storage,
        config=config,
        paths=paths,
        trade_date=date(2026, 4, 10),
        top_n=20,
        output=None,
    )

    assert callback_seen == [True]


def test_run_trend_passes_progress_callback(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("trend_progress_callback")
    config = load_config(ROOT / "config" / "default.yaml")
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)
    callback_seen: list[bool] = []
    scored = pd.DataFrame(
        [
            {
                "trade_date": "2026-04-10",
                "symbol": "600000",
                "name": "测试趋势",
                "signal_type": "breakout",
                "buy_score": 76.0,
                "price_action_score": 60.0,
                "trend_score": 83.0,
                "trend_base_score": 70.0,
                "macd_cross_state": "golden_cross",
                "macd_divergence_state": "none",
                "volume_price_divergence_state": "none",
            }
        ]
    )

    def fake_scan(storage, config, trade_date, progress_callback=None):
        callback_seen.append(progress_callback is not None)
        return scored.copy()

    monkeypatch.setattr("stocks_analyzer.cli.scan_indicator_scored_entries", fake_scan)

    _run_trend(
        storage=storage,
        config=config,
        paths=paths,
        trade_date=date(2026, 4, 10),
        top_n=20,
        output=None,
    )

    assert callback_seen == [True]
