from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

from stocks_analyzer.sector_report_cleanup import cleanup_sector_reports


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_cleanup_sector_reports_keeps_tracking_and_latest_core_outputs() -> None:
    project_root = _make_workspace_tmp_dir("sector_report_cleanup")
    sector_dir = project_root / "reports" / "sectors"
    sector_dir.mkdir(parents=True)
    keep = {
        "sector_mainline_daily_tracking.xlsx",
        "sector_mainline_intraday_tracking.xlsx",
        "sector_performance_2026-05-15.csv",
        "sector_leader_scores_all_2026-05-15.csv",
        "sector_phase9_buy_score_predictions_2026-05-15.csv",
    }
    remove = {
        "sector_performance_2026-05-14.csv",
        "sector_leader_scores_all_2026-05-14.csv",
        "sector_phase9_buy_score_predictions_2026-05-14.csv",
        "sector_leaders_2026-05-15.csv",
        "sector_leaders_summary_2026-05-15.csv",
        "skipped_sector_leaders_2026-05-15.csv",
        "sector_pullback_metrics_2026-05-15.csv",
        "sector_mainline_scores_2026-05-15.csv",
    }
    untouched = {
        "manual_notes.csv",
    }
    for name in keep | remove | untouched:
        (sector_dir / name).write_text(name, encoding="utf-8")
    validation_dir = sector_dir / "phase9_buy_score_validation"
    validation_dir.mkdir()
    (validation_dir / "summary.csv").write_text("keep", encoding="utf-8")

    result = cleanup_sector_reports(project_root=project_root, trade_date=date(2026, 5, 15))

    for name in keep | untouched:
        assert (sector_dir / name).exists()
    for name in remove:
        assert not (sector_dir / name).exists()
        assert name in result.deleted_files
    assert (validation_dir / "summary.csv").exists()
