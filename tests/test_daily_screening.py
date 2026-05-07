from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from uuid import uuid4

from stocks_analyzer.daily_screening import run_daily_screening
from stocks_analyzer.watchlist import watchlist_path, watchlist_pattern_path


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_run_daily_screening_generates_watchlist_without_touching_picks(monkeypatch, capsys) -> None:
    tmp_path = _make_workspace_tmp_dir("daily_screening")
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
        _write_stage_output(project_root, tmp_path, args)

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
        ["predict-model", "--date", "2026-04-11"],
        ["macd", "--date", "2026-04-11"],
        ["atr", "--date", "2026-04-11"],
        ["pattern", "--as-of", "2026-04-11"],
    ]

    watchlist = json.loads(result.watchlist_path.read_text(encoding="utf-8"))
    assert watchlist["trade_date"] == "2026-04-11"
    assert watchlist["candidate_count"] == 1

    assert result.report_path is not None
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["watchlist_path"] == str(result.watchlist_path)
    assert report["watchlist_pattern_path"].endswith("watchlist_pattern_2026-04-11.json")
    assert report["predict_model_path"].endswith("predictions_2026-04-11.csv")
    assert report["macd_path"].endswith("macd_2026-04-11.csv")
    assert report["atr_path"].endswith("atr_2026-04-11.csv")
    assert report["pattern_path"].endswith("patterns_all_2026-04-11.csv")
    assert "watchlist_trend_path" not in report
    assert "trend_path" not in report
    assert "trend_universe_path" not in report
    assert "[0/6] 检查 2026-04-11 是否为交易日..." in output
    assert "[6/6] pattern 完成" in output


def test_run_daily_screening_ignores_stale_trend_outputs(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("daily_screening_stale_trend")
    commands: list[list[str]] = []
    (tmp_path / "reports" / "trend").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "trend" / "trend_2026-04-11.csv").write_text("symbol,buy_score\n002579,80\n", encoding="utf-8")
    (tmp_path / "reports" / "trend_universe").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "trend_universe" / "trend_universe_2026-04-11.csv").write_text(
        "symbol,in_trend_universe\n002579,True\n",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "watchlists").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "watchlists" / "watchlist_trend_2026-04-11.json").write_text(
        json.dumps({"trade_date": "2026-04-11", "candidate_count": 1, "candidates": []}),
        encoding="utf-8",
    )

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
        _write_stage_output(project_root, tmp_path, args)

    monkeypatch.setattr("stocks_analyzer.daily_screening._run_project_command", fake_run_project_command)

    result = run_daily_screening(project_root=tmp_path, trade_date=date(2026, 4, 11))

    assert ["trend-universe", "--date", "2026-04-11"] not in commands
    assert ["trend", "--date", "2026-04-11"] not in commands
    assert result.report_path is not None
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert "watchlist_trend_path" not in report
    assert "trend_path" not in report
    assert "trend_universe_path" not in report


def _write_stage_output(project_root: Path, tmp_path: Path, args: list[str]) -> None:
    if args[:2] == ["macd", "--date"]:
        target = tmp_path / "reports" / "macd" / "macd_2026-04-11.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("symbol,macd_cross_state\n002579,golden_cross\n", encoding="utf-8")
    if args[:2] == ["atr", "--date"]:
        target = tmp_path / "reports" / "atr" / "atr_2026-04-11.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("代码,ATR14\n002579,1.2\n", encoding="utf-8")
    if args[:2] == ["predict-model", "--date"]:
        target = tmp_path / "reports" / "predict_model" / "predictions_2026-04-11.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "symbol,trade_date,action,trade_permission,risk_tier,risk_gate_reason,risk_score,long_upside_score,opportunity_rank_score,final_score_v42,buy_score_v42\n"
            "002579,2026-04-11,candidate,allow,low,passed,0.2,0.7,0.7,0.7,88.0\n",
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
