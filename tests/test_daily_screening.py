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


def test_run_daily_screening_runs_current_pipeline_without_touching_picks(monkeypatch, capsys) -> None:
    tmp_path = _make_workspace_tmp_dir("daily_screening")
    commands: list[list[str]] = []
    picks_path = tmp_path / "选股.md"
    picks_path.write_text("原始内容\n", encoding="utf-8")

    monkeypatch.setattr("stocks_analyzer.daily_screening.is_trading_day", lambda provider, trade_date: True)
    monkeypatch.setattr("stocks_analyzer.daily_screening.load_config", lambda path: type("Config", (), {"provider": "mock"})())

    def fake_run_project_command(project_root: Path, args: list[str]) -> None:
        commands.append(args)
        _write_stage_output(project_root, args)

    def fake_phase_watchlist(stage_index: int, total_stages: int, project_root: Path, trade_date: date):
        target = watchlist_path(project_root, trade_date)
        payload = {
            "trade_date": trade_date.isoformat(),
            "candidate_count": 1,
            "candidates": [{"symbol": "002579", "name": "中京电子", "suggested_action": "candidate"}],
            "filter_summary": {"phase1_excluded": 0, "phase2_excluded": 0},
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target, payload

    monkeypatch.setattr("stocks_analyzer.daily_screening._run_project_command", fake_run_project_command)
    monkeypatch.setattr("stocks_analyzer.daily_screening._run_phase_watchlist_stage", fake_phase_watchlist)

    result = run_daily_screening(project_root=tmp_path, trade_date=date(2026, 4, 11))
    output = capsys.readouterr().out

    assert result.skipped is False
    assert result.watchlist_path is not None
    assert result.watchlist_path.exists()
    assert picks_path.read_text(encoding="utf-8") == "原始内容\n"
    assert commands == [
        ["update", "--start-date", "20240101", "--end-date", "20260411"],
        ["macd", "--date", "2026-04-11"],
        ["atr", "--date", "2026-04-11"],
        ["predict-tail-risk", "--date", "2026-04-11", "--latest-only", "--feature-lookback-bars", "61", "--compact-output"],
        ["predict-barrier-risk", "--date", "2026-04-11", "--latest-only", "--feature-lookback-bars", "61", "--compact-output"],
        ["predict-alpha158-qlib-return", "--date", "2026-04-11", "--latest-only", "--feature-lookback-bars", "61", "--compact-output"],
        ["predict-trade-day-gate", "--date", "2026-04-11"],
        ["validate-mcd-crash-risk", "--start-date", "2015-01-01", "--end-date", "2026-04-11"],
        ["pattern", "--as-of", "2026-04-11"],
        ["track-stock", "--date", "2026-04-11"],
    ]

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["watchlist_path"] == str(result.watchlist_path)
    assert report["watchlist_pattern_path"].endswith("watchlist_pattern_2026-04-11.json")
    assert report["macd_path"].endswith("macd_2026-04-11.csv")
    assert report["atr_path"].endswith("atr_2026-04-11.csv")
    assert report["pattern_path"].endswith("patterns_all_2026-04-11.csv")
    assert report["phase1_path"].endswith("tail_risk_predictions_2026-04-11.csv")
    assert report["phase2_path"].endswith("barrier_risk_predictions_2026-04-11.csv")
    assert report["phase4_path"].endswith("alpha158_qlib_return_predictions_2026-04-11.csv")
    assert report["phase8_path"] is None
    assert report["phase5_path"].endswith("mcd_crash_annual_measures.csv")
    assert report["phase7_path"].endswith("trade_day_gate_prediction_2026-04-11.csv")
    assert "[0/12] 检查 2026-04-11 是否为交易日..." in output
    assert "[7/12] phase8_limit_up_3d 跳过" in output
    assert "[10/12] pattern 完成" in output


def _write_stage_output(project_root: Path, args: list[str]) -> None:
    if args[:2] == ["macd", "--date"]:
        target = project_root / "reports" / "macd" / "macd_2026-04-11.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("symbol,macd_cross_state\n002579,golden_cross\n", encoding="utf-8")
    if args[:2] == ["atr", "--date"]:
        target = project_root / "reports" / "atr" / "atr_2026-04-11.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("代码,ATR14\n002579,1.2\n", encoding="utf-8")
    if args[:2] == ["predict-tail-risk", "--date"]:
        _write_full_market_csv(project_root / "reports" / "full_market_model" / "tail_risk_predictions_2026-04-11.csv")
    if args[:2] == ["predict-barrier-risk", "--date"]:
        _write_full_market_csv(project_root / "reports" / "full_market_model" / "barrier_risk_predictions_2026-04-11.csv")
    if args[:2] == ["predict-alpha158-qlib-return", "--date"]:
        _write_full_market_csv(project_root / "reports" / "full_market_model" / "alpha158_qlib_return_predictions_2026-04-11.csv")
    if args[:2] == ["predict-trade-day-gate", "--date"]:
        target = project_root / "reports" / "full_market_model" / "trade_day_gate_prediction_2026-04-11.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("trade_date,trade_permission,buy_day_risk_score\n2026-04-11,allow,0.2\n", encoding="utf-8")
    if args and args[0] == "validate-mcd-crash-risk":
        target = project_root / "reports" / "full_market_model" / "mcd_crash_annual_measures.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("symbol,year,NEGOUTLIER\n002579,2026,0\n", encoding="utf-8")
        (target.parent / "mcd_crash_config.json").write_text('{"end_date":"2026-04-11"}', encoding="utf-8")
    if args[:2] == ["pattern", "--as-of"]:
        pattern_target = project_root / "reports" / "patterns" / "patterns_all_2026-04-11.csv"
        pattern_target.parent.mkdir(parents=True, exist_ok=True)
        pattern_target.write_text("symbol,name,pattern_id\n002579,中京电子,1\n", encoding="utf-8")
        pattern_watchlist = watchlist_pattern_path(project_root, date.fromisoformat(args[2]))
        pattern_watchlist.parent.mkdir(parents=True, exist_ok=True)
        pattern_watchlist.write_text(
            json.dumps({"trade_date": args[2], "candidate_count": 1, "candidates": [{"symbol": "002579"}]}),
            encoding="utf-8",
        )
    if args[:2] == ["track-stock", "--date"]:
        (project_root / "track_stock.xlsx").write_bytes(b"placeholder")


def _write_full_market_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("symbol,score\n002579,0.2\n", encoding="utf-8")
