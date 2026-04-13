from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from stocks_analyzer.daily_screening import run_daily_screening
from stocks_analyzer.watchlist import watchlist_path


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
        if args[:2] == ["pattern", "--as-of"]:
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
        ["divergence", "--date", "2026-04-11"],
        ["pattern", "--as-of", "2026-04-11"],
    ]
    watchlist = json.loads(result.watchlist_path.read_text(encoding="utf-8"))
    assert watchlist["trade_date"] == "2026-04-11"
    assert watchlist["candidate_count"] == 1
    assert result.report_path is not None
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["watchlist_path"] == str(result.watchlist_path)
    assert "[0/4] 检查 2026-04-11 是否为交易日..." in output
    assert "[4/4] pattern 完成" in output
