from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from .config import load_config
from .concern_sectors import (
    ConcernSectorResult,
    concern_sector_members_path,
    stock_concern_sectors_path,
    write_concern_sector_frames_from_files,
)
from .daily_returns import write_full_market_daily_returns
from .full_market_return import alpha158_qlib_return_predictions_path
from .full_market_risk import barrier_risk_predictions_path, tail_risk_predictions_path
from .route_watchlists import RouteWatchlistResult, write_route_watchlists_from_files
from .sector_membership import sector_membership_path, sector_performance_path
from .sector_leaders import sector_leader_scores_all_path, sector_leaders_path
from .sector_phase9 import sector_phase9_model_path, sector_phase9_predictions_path
from .sector_report_cleanup import cleanup_obsolete_daily_reports, cleanup_sector_reports
from .sector_tracking_workbook import write_sector_daily_tracking_workbook
from .sector_watchlist import (
    build_sector_tracking_payload_from_files,
    build_sector_watchlist_from_files,
    watchlist_sectors_path,
    write_sector_watchlist,
)
from .trading_calendar import is_trading_day


PICKS_FILENAME = "选股.md"


@dataclass
class ScreeningResult:
    trade_date: date
    skipped: bool
    message: str
    report_path: Path | None = None
    a1_watchlist_path: Path | None = None
    a2_watchlist_path: Path | None = None
    b_watchlist_path: Path | None = None
    sector_leader_pool_path: Path | None = None


def run_daily_screening(
    *,
    project_root: Path,
    trade_date: date,
    start_date: str = "20240101",
    picks_filename: str = PICKS_FILENAME,
) -> ScreeningResult:
    _ = picks_filename  # kept only for backward compatibility with existing callers
    config = load_config(project_root / "config" / "default.yaml")
    total_stages = 15
    print(f"[0/{total_stages}] 检查 {trade_date.isoformat()} 是否为交易日...", flush=True)
    trading_day = is_trading_day(config.provider, trade_date)
    if not trading_day:
        print(f"[0/{total_stages}] {trade_date.isoformat()} 不是交易日，跳过每日筛选。", flush=True)
        return ScreeningResult(
            trade_date=trade_date,
            skipped=True,
            message=f"{trade_date.isoformat()} 不是交易日，已跳过每日筛选。",
        )
    print(f"[0/{total_stages}] {trade_date.isoformat()} 是交易日，开始执行每日筛选。", flush=True)

    _run_project_stage(
        1,
        total_stages,
        "update",
        project_root,
        ["update", "--start-date", start_date, "--end-date", trade_date.strftime("%Y%m%d")],
    )
    _run_project_stage(2, total_stages, "sector_membership", project_root, ["update-sector-membership", "--date", trade_date.isoformat()])
    _run_project_stage(3, total_stages, "macd", project_root, ["macd", "--date", trade_date.isoformat()])
    _run_project_stage(4, total_stages, "atr", project_root, ["atr", "--date", trade_date.isoformat()])
    fast_prediction_args = ["--latest-only", "--feature-lookback-bars", "61", "--compact-output"]
    _run_project_stage(
        5,
        total_stages,
        "phase1_tail_risk",
        project_root,
        ["predict-tail-risk", "--date", trade_date.isoformat(), *fast_prediction_args],
    )
    _run_project_stage(
        6,
        total_stages,
        "phase2_barrier_risk",
        project_root,
        ["predict-barrier-risk", "--date", trade_date.isoformat(), *fast_prediction_args],
    )
    _run_project_stage(
        7,
        total_stages,
        "phase4_alpha158_return",
        project_root,
        ["predict-alpha158-qlib-return", "--date", trade_date.isoformat(), *fast_prediction_args],
    )
    full_market_returns_path = _run_full_market_daily_returns_stage(8, total_stages, project_root, trade_date)
    _run_project_stage(9, total_stages, "pattern", project_root, ["pattern", "--as-of", trade_date.isoformat()])
    _run_project_stage(10, total_stages, "sector_leaders", project_root, ["analyze-sector-leaders", "--date", trade_date.isoformat(), "--top-n", "10"])
    concern_result = _run_concern_sector_stage(11, total_stages, project_root, trade_date)
    _run_phase9_stage(12, total_stages, project_root, trade_date)
    generated_sector_watchlist_path, sector_watchlist_payload = _run_sector_watchlist_stage(13, total_stages, project_root, trade_date)
    route_result = _run_route_watchlist_stage(
        14,
        total_stages,
        project_root,
        trade_date,
    )
    _run_sector_report_cleanup_stage(15, total_stages, project_root, trade_date)

    macd_path = _macd_report_path(project_root, trade_date)
    atr_path = _atr_report_path(project_root, trade_date)
    pattern_path = _pattern_report_path(project_root, trade_date)
    phase1_path = tail_risk_predictions_path(project_root, trade_date)
    phase2_path = barrier_risk_predictions_path(project_root, trade_date)
    phase4_path = alpha158_qlib_return_predictions_path(project_root, trade_date)
    sector_path = sector_membership_path(project_root)
    sector_perf_path = sector_performance_path(project_root, trade_date)
    sector_leaders_report_path = sector_leaders_path(project_root, trade_date)
    sector_leader_scores_all_report_path = sector_leader_scores_all_path(project_root, trade_date)
    phase9_path = sector_phase9_predictions_path(project_root, trade_date)
    report_path = _write_run_report(
        project_root,
        trade_date,
        sector_watchlist_payload=sector_watchlist_payload,
        sector_watchlist_path=generated_sector_watchlist_path,
        concern_result=concern_result,
        route_result=route_result,
        macd_path=macd_path if macd_path.exists() else None,
        atr_path=atr_path if atr_path.exists() else None,
        pattern_path=pattern_path if pattern_path.exists() else None,
        phase1_path=phase1_path if phase1_path.exists() else None,
        phase2_path=phase2_path if phase2_path.exists() else None,
        phase4_path=phase4_path if phase4_path.exists() else None,
        full_market_daily_returns_path=full_market_returns_path if full_market_returns_path.exists() else None,
        sector_path=sector_path if sector_path.exists() else None,
        sector_performance_path=sector_perf_path if sector_perf_path.exists() else None,
        sector_leaders_path=sector_leaders_report_path if sector_leaders_report_path.exists() else None,
        sector_leader_scores_all_path=sector_leader_scores_all_report_path if sector_leader_scores_all_report_path.exists() else None,
        phase9_path=phase9_path if phase9_path.exists() else None,
    )
    return ScreeningResult(
        trade_date=trade_date,
        skipped=False,
        message=(
            f"已完成 {trade_date.isoformat()} 每日筛选，并生成 A1/A2/B 路线 watchlist、"
            f"{generated_sector_watchlist_path} 和 {route_result.sector_leader_pool_path}"
        ),
        report_path=report_path,
        a1_watchlist_path=route_result.a1_path,
        a2_watchlist_path=route_result.a2_path,
        b_watchlist_path=route_result.b_path,
        sector_leader_pool_path=route_result.sector_leader_pool_path,
    )


def _run_project_command(project_root: Path, args: list[str]) -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    src_path = str(project_root / "src")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else src_path + os.pathsep + existing_pythonpath
    completed = subprocess.run(
        [sys.executable, "-m", "stocks_analyzer", "--project-root", str(project_root), *args],
        cwd=project_root,
        env=env,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(args)}")


def _run_project_stage(stage_index: int, total_stages: int, stage_name: str, project_root: Path, args: list[str]) -> None:
    started_at = perf_counter()
    print(f"[{stage_index}/{total_stages}] 开始 {stage_name}...", flush=True)
    _run_project_command(project_root, args)
    elapsed = perf_counter() - started_at
    print(f"[{stage_index}/{total_stages}] {stage_name} 完成，用时 {elapsed:.1f}s。", flush=True)


def _run_phase9_stage(stage_index: int, total_stages: int, project_root: Path, trade_date: date) -> None:
    if not sector_phase9_model_path(project_root).exists():
        print(f"[{stage_index}/{total_stages}] phase9_sector_buy_score 跳过：模型文件尚未训练完成。", flush=True)
        return
    _run_project_stage(
        stage_index,
        total_stages,
        "phase9_sector_buy_score",
        project_root,
        ["predict-sector-phase9-buy-score", "--date", trade_date.isoformat()],
    )


def _run_full_market_daily_returns_stage(stage_index: int, total_stages: int, project_root: Path, trade_date: date) -> Path:
    started_at = perf_counter()
    print(f"[{stage_index}/{total_stages}] 开始 full_market_daily_returns...", flush=True)
    target = write_full_market_daily_returns(project_root=project_root, trade_date=trade_date)
    elapsed = perf_counter() - started_at
    print(f"[{stage_index}/{total_stages}] full_market_daily_returns 完成：{target}，用时 {elapsed:.1f}s。", flush=True)
    return target


def _run_concern_sector_stage(
    stage_index: int,
    total_stages: int,
    project_root: Path,
    trade_date: date,
) -> ConcernSectorResult:
    started_at = perf_counter()
    print(f"[{stage_index}/{total_stages}] 开始 concern_sectors...", flush=True)
    result = write_concern_sector_frames_from_files(project_root=project_root, trade_date=trade_date)
    elapsed = perf_counter() - started_at
    print(
        f"[{stage_index}/{total_stages}] concern_sectors 完成，关系 {result.relation_count} 条，弱势股 {result.weak_stock_count} 只，用时 {elapsed:.1f}s。",
        flush=True,
    )
    return result


def _run_route_watchlist_stage(
    stage_index: int,
    total_stages: int,
    project_root: Path,
    trade_date: date,
) -> RouteWatchlistResult:
    started_at = perf_counter()
    print(f"[{stage_index}/{total_stages}] 开始 route_watchlists...", flush=True)
    result = write_route_watchlists_from_files(project_root=project_root, trade_date=trade_date)
    elapsed = perf_counter() - started_at
    print(
        f"[{stage_index}/{total_stages}] route_watchlists 完成，A1={result.a1_count} A2={result.a2_count} B={result.b_count}，盘中源池 {result.sector_leader_count} 只，用时 {elapsed:.1f}s。",
        flush=True,
    )
    return result


def _run_sector_watchlist_stage(
    stage_index: int,
    total_stages: int,
    project_root: Path,
    trade_date: date,
) -> tuple[Path, dict[str, object]]:
    started_at = perf_counter()
    print(f"[{stage_index}/{total_stages}] 开始 watchlist_sectors...", flush=True)
    payload = build_sector_watchlist_from_files(project_root=project_root, trade_date=trade_date)
    target = write_sector_watchlist(project_root=project_root, trade_date=trade_date, payload=payload)
    tracking_payload = build_sector_tracking_payload_from_files(project_root=project_root, trade_date=trade_date)
    tracking_path = write_sector_daily_tracking_workbook(
        project_root=project_root,
        trade_date=trade_date,
        sector_payload=tracking_payload,
    )
    written_payload = json.loads(target.read_text(encoding="utf-8"))
    elapsed = perf_counter() - started_at
    print(
        f"[{stage_index}/{total_stages}] watchlist_sectors 完成，主线跟踪表 {tracking_path}，用时 {elapsed:.1f}s。",
        flush=True,
    )
    return target, written_payload


def _run_sector_report_cleanup_stage(stage_index: int, total_stages: int, project_root: Path, trade_date: date) -> None:
    started_at = perf_counter()
    print(f"[{stage_index}/{total_stages}] 开始 sector_reports_cleanup...", flush=True)
    result = cleanup_sector_reports(project_root=project_root, trade_date=trade_date)
    obsolete = cleanup_obsolete_daily_reports(project_root=project_root)
    elapsed = perf_counter() - started_at
    print(
        f"[{stage_index}/{total_stages}] sector_reports_cleanup 完成，删除 {len(result.deleted_files) + len(obsolete)} 个文件，用时 {elapsed:.1f}s。",
        flush=True,
    )


def _write_run_report(
    project_root: Path,
    trade_date: date,
    *,
    sector_watchlist_payload: dict[str, object],
    sector_watchlist_path: Path,
    concern_result: ConcernSectorResult,
    route_result: RouteWatchlistResult,
    macd_path: Path | None,
    atr_path: Path | None,
    pattern_path: Path | None,
    phase1_path: Path | None,
    phase2_path: Path | None,
    phase4_path: Path | None,
    full_market_daily_returns_path: Path | None,
    sector_path: Path | None,
    sector_performance_path: Path | None,
    sector_leaders_path: Path | None,
    sector_leader_scores_all_path: Path | None,
    phase9_path: Path | None,
) -> Path:
    target = project_root / "reports" / "daily_screening"
    target.mkdir(parents=True, exist_ok=True)
    report_path = target / f"daily_screening_{trade_date.isoformat()}.json"
    report = {
        "trade_date": trade_date.isoformat(),
        "watchlist_sectors_path": str(sector_watchlist_path),
        "watchlist_sectors_csv_path": str(sector_watchlist_path.with_suffix(".csv")),
        "stock_concern_sectors_path": str(concern_result.stock_path),
        "stock_concern_sectors_csv_path": str(concern_result.stock_path),
        "concern_sector_members_path": str(concern_result.member_path),
        "concern_sector_members_csv_path": str(concern_result.member_path),
        "watchlist_a1_recent_mainline_path": str(route_result.a1_path),
        "watchlist_a1_recent_mainline_csv_path": str(route_result.a1_path.with_suffix(".csv")),
        "watchlist_a2_rotation_expected_path": str(route_result.a2_path),
        "watchlist_a2_rotation_expected_csv_path": str(route_result.a2_path.with_suffix(".csv")),
        "watchlist_b_pattern_path": str(route_result.b_path),
        "watchlist_b_pattern_csv_path": str(route_result.b_path.with_suffix(".csv")),
        "watchlist_sector_leader_pool_path": str(route_result.sector_leader_pool_path),
        "watchlist_sector_leader_pool_csv_path": str(route_result.sector_leader_pool_path.with_suffix(".csv")),
        "macd_path": str(macd_path) if macd_path is not None else None,
        "atr_path": str(atr_path) if atr_path is not None else None,
        "pattern_path": str(pattern_path) if pattern_path is not None else None,
        "phase1_path": str(phase1_path) if phase1_path is not None else None,
        "phase2_path": str(phase2_path) if phase2_path is not None else None,
        "phase4_path": str(phase4_path) if phase4_path is not None else None,
        "full_market_daily_returns_path": str(full_market_daily_returns_path) if full_market_daily_returns_path is not None else None,
        "phase9_path": str(phase9_path) if phase9_path is not None else None,
        "sector_membership_path": str(sector_path) if sector_path is not None else None,
        "sector_performance_path": str(sector_performance_path) if sector_performance_path is not None else None,
        "sector_leaders_path": str(sector_leaders_path) if sector_leaders_path is not None else None,
        "sector_leader_scores_all_path": str(sector_leader_scores_all_path) if sector_leader_scores_all_path is not None else None,
        "sector_candidate_count": len(sector_watchlist_payload.get("sectors", []))
        if isinstance(sector_watchlist_payload.get("sectors"), list)
        else 0,
        "concern_relation_count": concern_result.relation_count,
        "concern_weak_stock_count": concern_result.weak_stock_count,
        "a1_candidate_count": route_result.a1_count,
        "a2_candidate_count": route_result.a2_count,
        "b_candidate_count": route_result.b_count,
        "sector_leader_pool_candidate_count": route_result.sector_leader_count,
        "deprecated_phases": ["P3", "P5", "P7", "P8", "P10"],
        "deprecated_outputs": ["track_stock.xlsx", "watchlist_stocks", "watchlist_mainline_stocks", "intraday_pool"],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def _macd_report_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "macd" / f"macd_{trade_date.isoformat()}.csv"


def _atr_report_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "atr" / f"atr_{trade_date.isoformat()}.csv"


def _pattern_report_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "patterns" / f"patterns_all_{trade_date.isoformat()}.csv"


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required daily-screening input not found: {path}")
    return pd.read_csv(path)


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _parse_optional_iso_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def _count_local_trading_days_between(project_root: Path, previous: date, current: date) -> int:
    if current <= previous:
        return 0
    dates = _load_local_trading_dates(project_root, previous=previous, current=current)
    if dates:
        return sum(1 for item in dates if previous < item <= current)
    return len(pd.bdate_range(previous, current, inclusive="right"))


def _load_local_trading_dates(project_root: Path, *, previous: date, current: date) -> list[date]:
    collected: set[date] = set()
    market_path = project_root / "reports" / "full_market_model" / "synthetic_market.csv"
    if market_path.exists():
        try:
            market = pd.read_csv(market_path, usecols=["trade_date"])
            collected.update(pd.to_datetime(market["trade_date"], errors="coerce").dropna().dt.date.tolist())
        except Exception:
            collected.clear()
    if collected and min(collected) <= previous and max(collected) >= current:
        return sorted(collected)

    daily_dir = project_root / "data" / "daily"
    for index, path in enumerate(sorted(daily_dir.glob("*.parquet")), start=1):
        try:
            frame = pd.read_parquet(path, columns=["trade_date"])
        except Exception:
            continue
        collected.update(pd.to_datetime(frame["trade_date"], errors="coerce").dropna().dt.date.tolist())
        if index >= 50 and collected and max(collected) >= current:
            break
    return sorted(collected)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the daily stock screening workflow and generate a watchlist.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--date", default=None, help="目标日期，格式 YYYY-MM-DD，默认今天")
    parser.add_argument("--start-date", default="20240101", help="更新数据的起始日期，格式 YYYYMMDD")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    trade_date = datetime.fromisoformat(args.date).date() if args.date else date.today()
    result = run_daily_screening(
        project_root=Path(args.project_root).resolve(),
        trade_date=trade_date,
        start_date=args.start_date,
    )
    print(result.message)
    if result.report_path:
        print(f"报告文件：{result.report_path}")


if __name__ == "__main__":
    main()
