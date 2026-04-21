from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from stocks_analyzer.daily_screening import run_daily_screening
from stocks_analyzer.watchlist import watchlist_path, watchlist_pattern_path, watchlist_trend_path


def test_run_daily_screening_generates_watchlist_without_touching_picks(monkeypatch, tmp_path: Path, capsys) -> None:
    commands: list[list[str]] = []
    picks_path = tmp_path / "选股.md"
    picks_path.write_text("原始内容\n", encoding="utf-8")

    monkeypatch.setattr("stocks_analyzer.daily_screening.is_trading_day", lambda provider, trade_date: True)
    monkeypatch.setattr(
        "stocks_analyzer.daily_screening.load_config",
        lambda path: type("Config", (), {"provider": "mock"})(),
    )
    monkeypatch.setattr(
        "stocks_analyzer.daily_screening.load_watchlist",
        lambda project_root, trade_date: json.loads(watchlist_path(project_root, trade_date).read_text(encoding="utf-8")),
    )

    def fake_run_project_command(project_root: Path, args: list[str]) -> None:
        commands.append(args)
        if args[:2] == ["macd", "--date"]:
            target = tmp_path / "reports" / "macd" / "macd_2026-04-11.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("symbol,macd_cross_state\n002579,golden_cross\n", encoding="utf-8")
        if args[:2] == ["atr", "--date"]:
            target = tmp_path / "reports" / "atr" / "atr_2026-04-11.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("代码,ATR14\n002579,1.2\n", encoding="utf-8")
        if args[:2] == ["trend", "--date"]:
            trend_target = tmp_path / "reports" / "trend" / "trend_2026-04-11.csv"
            trend_target.parent.mkdir(parents=True, exist_ok=True)
            trend_target.write_text(
                "symbol,name,signal_type,buy_score,price_action_score,macd_cross_state,macd_divergence_state,volume_price_divergence_state\n"
                "600000,测试趋势,breakout,75.0,60.0,golden_cross,none,none\n",
                encoding="utf-8",
            )
            trend_watchlist = watchlist_trend_path(project_root, date.fromisoformat(args[2]))
            trend_watchlist.parent.mkdir(parents=True, exist_ok=True)
            trend_watchlist.write_text(
                json.dumps(
                    {
                        "trade_date": args[2],
                        "source_file": "reports/trend/trend_2026-04-11.csv",
                        "candidate_count": 1,
                        "candidates": [{"symbol": "600000", "name": "测试趋势", "source": "trend"}],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        if args[:2] == ["pattern", "--as-of"]:
            pattern_target = tmp_path / "reports" / "patterns" / "patterns_all_2026-04-11.csv"
            pattern_target.parent.mkdir(parents=True, exist_ok=True)
            pattern_target.write_text("symbol,name\n002579,中京电子\n", encoding="utf-8")
            target = watchlist_path(project_root, date.fromisoformat(args[2]))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(
                    {
                        "trade_date": args[2],
                        "source_file": "reports/patterns/patterns_all_2026-04-11.csv",
                        "candidate_count": 1,
                        "candidates": [{"symbol": "002579", "name": "中京电子", "tier": "第一梯队"}],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            pattern_watchlist = watchlist_pattern_path(project_root, date.fromisoformat(args[2]))
            pattern_watchlist.parent.mkdir(parents=True, exist_ok=True)
            pattern_watchlist.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr("stocks_analyzer.daily_screening._run_project_command", fake_run_project_command)

    result = run_daily_screening(project_root=tmp_path, trade_date=date(2026, 4, 11))
    output = capsys.readouterr().out

    assert result.skipped is False
    assert result.watchlist_path is not None
    assert result.watchlist_path.exists()
    assert picks_path.read_text(encoding="utf-8") == "原始内容\n"
    assert commands == [
        ["update", "--start-date", "20240101"],
        ["tradingview", "--date", "2026-04-11"],
        ["macd", "--date", "2026-04-11"],
        ["atr", "--date", "2026-04-11"],
        ["trend-universe", "--date", "2026-04-11"],
        ["trend", "--date", "2026-04-11"],
        ["pattern", "--as-of", "2026-04-11"],
    ]
    watchlist = json.loads(result.watchlist_path.read_text(encoding="utf-8"))
    assert watchlist["trade_date"] == "2026-04-11"
    assert watchlist["candidate_count"] == 1
    assert result.report_path is not None
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["watchlist_path"] == str(result.watchlist_path)
    assert report["watchlist_pattern_path"].endswith("watchlist_pattern_2026-04-11.json")
    assert report["watchlist_trend_path"].endswith("watchlist_trend_2026-04-11.json")
    assert report["macd_path"].endswith("macd_2026-04-11.csv")
    assert report["atr_path"].endswith("atr_2026-04-11.csv")
    assert report["pattern_path"].endswith("patterns_all_2026-04-11.csv")
    assert report["trend_path"].endswith("trend_2026-04-11.csv")
    assert "[0/7] 检查 2026-04-11 是否为交易日..." in output
    assert "[7/7] pattern 完成" in output


def test_run_daily_screening_uses_trend_universe_stage_when_enabled(monkeypatch, tmp_path: Path, capsys) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr("stocks_analyzer.daily_screening.is_trading_day", lambda provider, trade_date: True)
    monkeypatch.setattr(
        "stocks_analyzer.daily_screening.load_config",
        lambda path: type("Config", (), {"provider": "mock"})(),
    )
    monkeypatch.setattr(
        "stocks_analyzer.daily_screening.load_watchlist",
        lambda project_root, trade_date: json.loads(watchlist_path(project_root, trade_date).read_text(encoding="utf-8")),
    )

    def fake_run_project_command(project_root: Path, args: list[str]) -> None:
        commands.append(args)
        if args[:2] == ["pattern", "--as-of"]:
            pattern_target = tmp_path / "reports" / "patterns" / f"patterns_all_{args[2]}.csv"
            pattern_target.parent.mkdir(parents=True, exist_ok=True)
            pattern_target.write_text("symbol,name\n002579,中京电子\n", encoding="utf-8")
            target = watchlist_path(project_root, date.fromisoformat(args[2]))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(
                    {
                        "trade_date": args[2],
                        "source_file": "reports/patterns/patterns_all_2026-04-11.csv",
                        "candidate_count": 1,
                        "candidates": [
                            {"symbol": "002579", "name": "中京电子", "tier": "第一梯队", "in_trend_universe": True},
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            pattern_watchlist = watchlist_pattern_path(project_root, date.fromisoformat(args[2]))
            pattern_watchlist.parent.mkdir(parents=True, exist_ok=True)
            pattern_watchlist.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        if args[:2] == ["trend-universe", "--date"]:
            target = tmp_path / "reports" / "trend_universe" / "trend_universe_2026-04-11.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "\n".join(
                    [
                        "symbol,in_trend_universe,trend_score,trend_direction_score",
                        "002579,True,83.4,90.0",
                    ]
                ),
                encoding="utf-8",
            )
        if args[:2] == ["trend", "--date"]:
            trend_target = tmp_path / "reports" / "trend" / "trend_2026-04-11.csv"
            trend_target.parent.mkdir(parents=True, exist_ok=True)
            trend_target.write_text(
                "symbol,name,signal_type,buy_score,price_action_score,macd_cross_state,macd_divergence_state,volume_price_divergence_state\n"
                "600000,测试趋势,breakout,75.0,60.0,golden_cross,none,none\n",
                encoding="utf-8",
            )
            trend_watchlist = watchlist_trend_path(project_root, date.fromisoformat(args[2]))
            trend_watchlist.parent.mkdir(parents=True, exist_ok=True)
            trend_watchlist.write_text(
                json.dumps(
                    {
                        "trade_date": args[2],
                        "source_file": "reports/trend/trend_2026-04-11.csv",
                        "candidate_count": 1,
                        "candidates": [{"symbol": "600000", "name": "测试趋势", "source": "trend"}],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        if args[:2] == ["macd", "--date"]:
            target = tmp_path / "reports" / "macd" / "macd_2026-04-11.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("symbol,macd_cross_state\n002579,golden_cross\n", encoding="utf-8")
        if args[:2] == ["atr", "--date"]:
            target = tmp_path / "reports" / "atr" / "atr_2026-04-11.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("代码,ATR14\n002579,1.2\n", encoding="utf-8")

    monkeypatch.setattr("stocks_analyzer.daily_screening._run_project_command", fake_run_project_command)

    result = run_daily_screening(project_root=tmp_path, trade_date=date(2026, 4, 11))
    output = capsys.readouterr().out

    assert result.skipped is False
    assert commands == [
        ["update", "--start-date", "20240101"],
        ["tradingview", "--date", "2026-04-11"],
        ["macd", "--date", "2026-04-11"],
        ["atr", "--date", "2026-04-11"],
        ["trend-universe", "--date", "2026-04-11"],
        ["trend", "--date", "2026-04-11"],
        ["pattern", "--as-of", "2026-04-11"],
    ]
    watchlist = json.loads(result.watchlist_path.read_text(encoding="utf-8"))
    assert watchlist["candidate_count"] == 1
    assert watchlist["candidates"][0]["symbol"] == "002579"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["watchlist_pattern_path"].endswith("watchlist_pattern_2026-04-11.json")
    assert report["watchlist_trend_path"].endswith("watchlist_trend_2026-04-11.json")
    assert report["macd_path"].endswith("macd_2026-04-11.csv")
    assert report["atr_path"].endswith("atr_2026-04-11.csv")
    assert report["pattern_path"].endswith("patterns_all_2026-04-11.csv")
    assert report["trend_path"].endswith("trend_2026-04-11.csv")
    assert report["trend_universe_path"].endswith("trend_universe_2026-04-11.csv")
    assert "[0/7] 检查 2026-04-11 是否为交易日..." in output
    assert "[7/7] pattern 完成" in output
