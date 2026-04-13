import json
from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd
from stocks_analyzer.cli import (
    PATTERN_LABEL_MAP,
    _append_recent_macd_divergence,
    _run_pattern,
    _prepare_pattern_results,
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


def test_build_parser_accepts_update_with_symbol() -> None:
    parser = build_parser()
    args = parser.parse_args(["update", "603588", "--start-date", "20240101"])

    assert args.command == "update"
    assert args.symbol == "603588"
    assert args.start_date == "20240101"


def test_build_parser_accepts_pattern_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["pattern", "--1", "--4", "--plot-all", "--as-of", "2026-04-10"])

    assert args.command == "pattern"
    assert args.pattern1 is True
    assert args.pattern4 is True
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


def test_build_parser_accepts_divergence_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["divergence", "--date", "2026-04-10", "--top-n", "30"])

    assert args.command == "divergence"
    assert args.date == "2026-04-10"
    assert args.top_n == 30


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


def test_prepare_pattern_results_maps_internal_type_to_pattern_id() -> None:
    results = pd.DataFrame(
        [
            {
                "trade_date": "2026-04-10",
                "symbol": "600000",
                "name": "测试股份",
                "strategy_name": "type1",
                "close": 10.0,
                "reason": "demo",
            }
        ]
    )

    prepared = _prepare_pattern_results(results)

    assert prepared["pattern_id"].tolist() == [PATTERN_LABEL_MAP["type1"]]
    assert "strategy_name" not in prepared.columns


def test_append_recent_macd_divergence_merges_two_flag_columns_from_saved_report() -> None:
    tmp_path = _make_workspace_tmp_dir("macd_divergence_merge")
    config = load_config(ROOT / "config" / "default.yaml")
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)

    divergence_dir = paths.reports_dir / "divergence"
    divergence_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "symbol": '="600000"',
                "macd_top_divergence_15d": True,
                "macd_bottom_divergence_15d": False,
            }
        ]
    ).to_csv(divergence_dir / "macd_divergence_2026-04-10.csv", index=False, encoding="utf-8-sig")

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

    enriched = _append_recent_macd_divergence(storage, exported, as_of=date(2026, 4, 10))

    assert bool(enriched.loc[0, "macd_top_divergence_15d"]) is True
    assert bool(enriched.loc[0, "macd_bottom_divergence_15d"]) is False


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
    monkeypatch.setattr("stocks_analyzer.cli.Screener.run", lambda self, as_of, selected_strategies, symbols=None: pd.DataFrame())
    monkeypatch.setattr("stocks_analyzer.cli._prepare_pattern_results", lambda results: exported.copy())
    monkeypatch.setattr("stocks_analyzer.cli._append_recent_tradingview_scores", lambda storage, exported, as_of, lookback_days, symbols=None: exported)
    monkeypatch.setattr("stocks_analyzer.cli._append_recent_macd_divergence", lambda storage, exported, as_of, symbols=None: exported)

    _run_pattern(
        storage=storage,
        provider_name=config.provider,
        config=config,
        as_of=date(2026, 4, 10),
        selected_patterns=["type1"],
        limit=20,
        output=None,
        plot_all=False,
    )

    watchlist = json.loads((tmp_path / "reports" / "watchlists" / "watchlist_2026-04-10.json").read_text(encoding="utf-8"))
    assert watchlist["trade_date"] == "2026-04-10"
    assert watchlist["candidate_count"] == 1
    assert watchlist["candidates"][0]["symbol"] == "600000"
