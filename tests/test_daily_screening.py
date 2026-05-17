from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from uuid import uuid4

from stocks_analyzer.daily_screening import run_daily_screening
from stocks_analyzer.sector_watchlist import watchlist_sectors_path
from stocks_analyzer.watchlist import intraday_pool_path, watchlist_stocks_path


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
        target = watchlist_stocks_path(project_root, trade_date)
        payload = {
            "trade_date": trade_date.isoformat(),
            "candidate_count": 1,
            "candidates": [{"symbol": "002579", "name": "中京电子", "suggested_action": "candidate"}],
            "filter_summary": {"phase1_excluded": 0, "phase2_excluded": 0},
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target, payload

    def fake_sector_watchlist(stage_index: int, total_stages: int, project_root: Path, trade_date: date):
        target = watchlist_sectors_path(project_root, trade_date)
        payload = {
            "trade_date": trade_date.isoformat(),
            "sector_count": 1,
            "sectors": [{"sector_name": "电子元件", "long_mainline_score_100": 80}],
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target, payload

    def fake_intraday_pool(stage_index: int, total_stages: int, project_root: Path, trade_date: date):
        target = intraday_pool_path(project_root, trade_date)
        payload = {
            "trade_date": trade_date.isoformat(),
            "candidate_count": 1,
            "selection_policy": {"source_scope": "daily_p124_top200"},
            "candidates": [{"symbol": "002579", "name": "中京电子", "source": "p124_top200"}],
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target, payload

    monkeypatch.setattr("stocks_analyzer.daily_screening._run_project_command", fake_run_project_command)
    monkeypatch.setattr("stocks_analyzer.daily_screening._run_phase_watchlist_stage", fake_phase_watchlist)
    monkeypatch.setattr("stocks_analyzer.daily_screening._run_intraday_pool_stage", fake_intraday_pool)
    monkeypatch.setattr("stocks_analyzer.daily_screening._run_sector_watchlist_stage", fake_sector_watchlist)

    result = run_daily_screening(project_root=tmp_path, trade_date=date(2026, 4, 11))
    output = capsys.readouterr().out

    assert result.skipped is False
    assert result.watchlist_path is not None
    assert result.watchlist_path.exists()
    assert picks_path.read_text(encoding="utf-8") == "原始内容\n"
    assert commands == [
        ["update", "--start-date", "20240101", "--end-date", "20260411"],
        ["update-sector-membership", "--date", "2026-04-11"],
        ["macd", "--date", "2026-04-11"],
        ["atr", "--date", "2026-04-11"],
        ["predict-tail-risk", "--date", "2026-04-11", "--latest-only", "--feature-lookback-bars", "61", "--compact-output"],
        ["predict-barrier-risk", "--date", "2026-04-11", "--latest-only", "--feature-lookback-bars", "61", "--compact-output"],
        ["predict-alpha158-qlib-return", "--date", "2026-04-11", "--latest-only", "--feature-lookback-bars", "61", "--compact-output"],
        ["pattern", "--as-of", "2026-04-11"],
        ["analyze-sector-leaders", "--date", "2026-04-11", "--top-n", "10"],
    ]

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["watchlist_stocks_path"] == str(result.watchlist_path)
    assert report["intraday_pool_path"] == str(result.intraday_pool_path)
    assert report["watchlist_sectors_path"].endswith("watchlist_sectors_2026-04-11.json")
    assert "watchlist_pattern_path" not in report
    assert report["macd_path"].endswith("macd_2026-04-11.csv")
    assert report["atr_path"].endswith("atr_2026-04-11.csv")
    assert report["pattern_path"].endswith("patterns_all_2026-04-11.csv")
    assert report["phase1_path"].endswith("tail_risk_predictions_2026-04-11.csv")
    assert report["phase2_path"].endswith("barrier_risk_predictions_2026-04-11.csv")
    assert report["phase4_path"].endswith("alpha158_qlib_return_predictions_2026-04-11.csv")
    assert report["full_market_daily_returns_path"].endswith("full_market_daily_returns_2026-04-11.csv")
    assert report["phase9_path"] is None
    assert report["track_stock_path"].endswith("track_stock.xlsx")
    assert report["sector_membership_path"].endswith("stock_sector_membership.csv")
    assert report["sector_performance_path"].endswith("sector_performance_2026-04-11.csv")
    assert "[0/15] 检查 2026-04-11 是否为交易日..." in output
    assert "[8/15] full_market_daily_returns 完成" in output
    assert "[11/15] phase9_sector_buy_score 跳过" in output
    assert "[15/15] track_stock_sheet2 完成" in output
    assert "[9/15] pattern 完成" in output


def _write_stage_output(project_root: Path, args: list[str]) -> None:
    if args[:2] == ["macd", "--date"]:
        target = project_root / "reports" / "macd" / "macd_2026-04-11.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("symbol,macd_cross_state\n002579,golden_cross\n", encoding="utf-8")
    if args[:2] == ["atr", "--date"]:
        target = project_root / "reports" / "atr" / "atr_2026-04-11.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("代码,ATR14\n002579,1.2\n", encoding="utf-8")
    if args and args[0] == "update-sector-membership":
        target = project_root / "data" / "sector_membership" / "stock_sector_membership.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "symbol,name,sector_type,sector_name,sector_label,source,updated_at\n"
            "002579,中京电子,industry,电子元件,881270,ths_industry,2026-04-11T17:30:00\n",
            encoding="utf-8",
        )
        perf = project_root / "reports" / "sectors" / "sector_performance_2026-04-11.csv"
        perf.parent.mkdir(parents=True, exist_ok=True)
        perf.write_text(
            "trade_date,sector_type,sector_name,sector_label,member_count,valid_count,avg_pct_change,amount_weighted_pct_change,up_count,up_ratio,total_amount\n"
            "2026-04-11,industry,电子元件,new_dzqj,1,1,2.0,2.0,1,1.0,1000000\n",
            encoding="utf-8",
        )
    if args[:2] == ["predict-tail-risk", "--date"]:
        _write_full_market_csv(project_root / "reports" / "full_market_model" / "tail_risk_predictions_2026-04-11.csv")
    if args[:2] == ["predict-barrier-risk", "--date"]:
        _write_full_market_csv(project_root / "reports" / "full_market_model" / "barrier_risk_predictions_2026-04-11.csv")
    if args[:2] == ["predict-alpha158-qlib-return", "--date"]:
        _write_full_market_csv(project_root / "reports" / "full_market_model" / "alpha158_qlib_return_predictions_2026-04-11.csv")
    if args[:2] == ["pattern", "--as-of"]:
        pattern_target = project_root / "reports" / "patterns" / "patterns_all_2026-04-11.csv"
        pattern_target.parent.mkdir(parents=True, exist_ok=True)
        pattern_target.write_text("symbol,name,pattern_id\n002579,中京电子,1\n", encoding="utf-8")
    if args[:2] == ["track-stock", "--date"]:
        (project_root / "track_stock.xlsx").write_bytes(b"placeholder")


def _write_full_market_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("symbol,score\n002579,0.2\n", encoding="utf-8")
