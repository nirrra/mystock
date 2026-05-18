from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from uuid import uuid4

from stocks_analyzer.concern_sectors import ConcernSectorResult
from stocks_analyzer.daily_screening import run_daily_screening
from stocks_analyzer.route_watchlists import RouteWatchlistResult
from stocks_analyzer.sector_watchlist import watchlist_sectors_path


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_run_daily_screening_runs_route_pipeline_without_track_stock(monkeypatch, capsys) -> None:
    tmp_path = _make_workspace_tmp_dir("daily_screening_routes")
    commands: list[list[str]] = []
    picks_path = tmp_path / "选股.md"
    picks_path.write_text("原始内容\n", encoding="utf-8")

    monkeypatch.setattr("stocks_analyzer.daily_screening.is_trading_day", lambda provider, trade_date: True)
    monkeypatch.setattr("stocks_analyzer.daily_screening.load_config", lambda path: type("Config", (), {"provider": "mock"})())

    def fake_run_project_command(project_root: Path, args: list[str]) -> None:
        commands.append(args)
        _write_stage_output(project_root, args)

    def fake_concern(stage_index: int, total_stages: int, project_root: Path, trade_date: date):
        stock = project_root / "reports" / "sectors" / f"stock_concern_sectors_{trade_date.isoformat()}.csv"
        member = project_root / "reports" / "sectors" / f"concern_sector_members_{trade_date.isoformat()}.csv"
        stock.parent.mkdir(parents=True, exist_ok=True)
        stock.write_text("交易日期,编号,名称,是否弱势股,关切板块,最高龙头指数,关切板块数量\n", encoding="utf-8")
        member.write_text("交易日期,编号,名称,板块类型,板块名称,板块代码,龙头指数\n", encoding="utf-8")
        return ConcernSectorResult(trade_date, stock, member, 1, 0, 1, 1)

    def fake_sector_watchlist(stage_index: int, total_stages: int, project_root: Path, trade_date: date):
        target = watchlist_sectors_path(project_root, trade_date)
        payload = {
            "trade_date": trade_date.isoformat(),
            "sector_count": 1,
            "sectors": [{"sector_name": "电子元件", "long_mainline_score_100": 80}],
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        target.with_suffix(".csv").write_text("板块名称,长期主线指数\n电子元件,80\n", encoding="utf-8")
        return target, payload

    def fake_routes(stage_index: int, total_stages: int, project_root: Path, trade_date: date):
        root = project_root / "reports" / "watchlists"
        root.mkdir(parents=True, exist_ok=True)
        a1 = root / f"watchlist_a1_recent_mainline_{trade_date.isoformat()}.json"
        a2 = root / f"watchlist_a2_rotation_expected_{trade_date.isoformat()}.json"
        b = root / f"watchlist_b_pattern_{trade_date.isoformat()}.json"
        pool = root / f"watchlist_sector_leader_pool_{trade_date.isoformat()}.json"
        for path in (a1, a2, b, pool):
            path.write_text('{"candidates":[]}', encoding="utf-8")
            path.with_suffix(".csv").write_text("", encoding="utf-8")
        return RouteWatchlistResult(trade_date, a1, a2, b, pool, 1, 2, 3, 4, 5)

    monkeypatch.setattr("stocks_analyzer.daily_screening._run_project_command", fake_run_project_command)
    monkeypatch.setattr("stocks_analyzer.daily_screening._run_concern_sector_stage", fake_concern)
    monkeypatch.setattr("stocks_analyzer.daily_screening._run_sector_watchlist_stage", fake_sector_watchlist)
    monkeypatch.setattr("stocks_analyzer.daily_screening._run_route_watchlist_stage", fake_routes)

    result = run_daily_screening(project_root=tmp_path, trade_date=date(2026, 4, 11))
    output = capsys.readouterr().out

    assert result.skipped is False
    assert result.a1_watchlist_path is not None
    assert result.sector_leader_pool_path is not None
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
    assert report["watchlist_a1_recent_mainline_path"] == str(result.a1_watchlist_path)
    assert report["watchlist_a2_rotation_expected_path"] == str(result.a2_watchlist_path)
    assert report["watchlist_b_pattern_path"] == str(result.b_watchlist_path)
    assert report["watchlist_sector_leader_pool_path"] == str(result.sector_leader_pool_path)
    assert "track_stock_path" not in report
    assert "watchlist_stocks_path" not in report
    assert "intraday_pool_path" not in report
    assert report["a1_candidate_count"] == 1
    assert report["a2_candidate_count"] == 2
    assert report["b_candidate_count"] == 3
    assert "[0/15] 检查 2026-04-11 是否为交易日..." in output
    assert "[12/15] phase9_sector_buy_score 跳过" in output
    assert "[15/15] sector_reports_cleanup" in output


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
        target.write_text("symbol,name,sector_type,sector_name,sector_label,source,updated_at\n", encoding="utf-8")
        perf = project_root / "reports" / "sectors" / "sector_performance_2026-04-11.csv"
        perf.parent.mkdir(parents=True, exist_ok=True)
        perf.write_text("trade_date,sector_type,sector_name,sector_label,avg_pct_change\n", encoding="utf-8")
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
    if args[:2] == ["analyze-sector-leaders", "--date"]:
        target = project_root / "reports" / "sectors" / "sector_leaders_2026-04-11.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("sector_name,symbol,combined_leader_score\n", encoding="utf-8")
        all_scores = project_root / "reports" / "sectors" / "sector_leader_scores_all_2026-04-11.csv"
        all_scores.write_text("symbol,name,sector_type,sector_name,sector_label,combined_leader_score\n", encoding="utf-8")


def _write_full_market_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("symbol,score\n002579,0.2\n", encoding="utf-8")
