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
from .daily_returns import write_full_market_daily_returns
from .full_market_return import alpha158_qlib_return_predictions_path
from .full_market_risk import barrier_risk_predictions_path, tail_risk_predictions_path
from .sector_membership import sector_membership_path, sector_performance_path
from .sector_leaders import sector_leader_scores_all_path, sector_leaders_path
from .sector_phase9 import sector_phase9_model_path, sector_phase9_predictions_path
from .sector_tracking_workbook import write_sector_daily_tracking_workbook
from .sector_watchlist import (
    build_sector_tracking_payload_from_files,
    build_sector_watchlist_from_files,
    watchlist_sectors_path,
    write_sector_watchlist,
)
from .track_stock import update_track_stock_workbook
from .trading_calendar import is_trading_day
from .watchlist import (
    build_intraday_pool_candidates,
    build_phase_daily_watchlist_candidates,
    load_watchlist,
    write_intraday_pool,
    write_watchlist_stocks,
)


PICKS_FILENAME = "选股.md"


@dataclass
class ScreeningResult:
    trade_date: date
    skipped: bool
    message: str
    report_path: Path | None = None
    watchlist_path: Path | None = None
    intraday_pool_path: Path | None = None
    track_stock_path: Path | None = None


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
    _run_phase9_stage(11, total_stages, project_root, trade_date)
    generated_watchlist_path, watchlist_payload = _run_phase_watchlist_stage(12, total_stages, project_root, trade_date)
    generated_intraday_pool_path, intraday_pool_payload = _run_intraday_pool_stage(13, total_stages, project_root, trade_date)
    generated_sector_watchlist_path, sector_watchlist_payload = _run_sector_watchlist_stage(14, total_stages, project_root, trade_date)
    track_stock_path = _run_track_stock_stage(15, total_stages, project_root, trade_date)

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
        watchlist_payload,
        generated_watchlist_path,
        intraday_pool_payload=intraday_pool_payload,
        intraday_pool_path=generated_intraday_pool_path,
        sector_watchlist_payload=sector_watchlist_payload,
        sector_watchlist_path=generated_sector_watchlist_path,
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
        track_stock_path=track_stock_path if track_stock_path.exists() else None,
        filter_summary=watchlist_payload.get("filter_summary") if isinstance(watchlist_payload, dict) else None,
    )
    return ScreeningResult(
        trade_date=trade_date,
        skipped=False,
        message=f"已完成 {trade_date.isoformat()} 每日筛选，并生成 {generated_watchlist_path}、{generated_intraday_pool_path} 和 {generated_sector_watchlist_path}",
        report_path=report_path,
        watchlist_path=generated_watchlist_path,
        intraday_pool_path=generated_intraday_pool_path,
        track_stock_path=track_stock_path,
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


def _run_phase_watchlist_stage(
    stage_index: int,
    total_stages: int,
    project_root: Path,
    trade_date: date,
) -> tuple[Path, dict[str, object]]:
    started_at = perf_counter()
    print(f"[{stage_index}/{total_stages}] 开始 watchlist_stocks...", flush=True)
    pattern_path = _pattern_report_path(project_root, trade_date)
    phase1_path = tail_risk_predictions_path(project_root, trade_date)
    phase2_path = barrier_risk_predictions_path(project_root, trade_date)
    phase4_path = alpha158_qlib_return_predictions_path(project_root, trade_date)
    macd_path = _macd_report_path(project_root, trade_date)
    atr_path = _atr_report_path(project_root, trade_date)

    payload = build_phase_daily_watchlist_candidates(
        trade_date=trade_date,
        pattern_frame=_read_required_csv(pattern_path),
        phase1_predictions=_read_required_csv(phase1_path),
        phase2_predictions=_read_required_csv(phase2_path),
        phase4_predictions=_read_required_csv(phase4_path),
        macd_frame=_read_optional_csv(macd_path),
        atr_frame=_read_optional_csv(atr_path),
        source_files={
            "pattern": str(pattern_path),
            "phase1": str(phase1_path),
            "phase2": str(phase2_path),
            "phase4": str(phase4_path),
            "macd": str(macd_path),
            "atr": str(atr_path),
        },
        phase_filter_rate=0.2,
        phase4_top_n=20,
    )
    target = write_watchlist_stocks(project_root=project_root, trade_date=trade_date, picker_payload=payload)
    written_payload = load_watchlist(project_root=project_root, trade_date=trade_date, kind="stocks")
    elapsed = perf_counter() - started_at
    print(f"[{stage_index}/{total_stages}] watchlist_stocks 完成，用时 {elapsed:.1f}s。", flush=True)
    return target, written_payload


def _run_intraday_pool_stage(
    stage_index: int,
    total_stages: int,
    project_root: Path,
    trade_date: date,
) -> tuple[Path, dict[str, object]]:
    started_at = perf_counter()
    print(f"[{stage_index}/{total_stages}] 开始 intraday_pool_top200...", flush=True)
    pattern_path = _pattern_report_path(project_root, trade_date)
    phase1_path = tail_risk_predictions_path(project_root, trade_date)
    phase2_path = barrier_risk_predictions_path(project_root, trade_date)
    phase4_path = alpha158_qlib_return_predictions_path(project_root, trade_date)
    macd_path = _macd_report_path(project_root, trade_date)
    atr_path = _atr_report_path(project_root, trade_date)

    payload = build_intraday_pool_candidates(
        trade_date=trade_date,
        pattern_frame=_read_required_csv(pattern_path),
        phase1_predictions=_read_required_csv(phase1_path),
        phase2_predictions=_read_required_csv(phase2_path),
        phase4_predictions=_read_required_csv(phase4_path),
        phase7_prediction=pd.DataFrame(),
        macd_frame=_read_optional_csv(macd_path),
        atr_frame=_read_optional_csv(atr_path),
        source_files={
            "pattern": str(pattern_path),
            "phase1": str(phase1_path),
            "phase2": str(phase2_path),
            "phase4": str(phase4_path),
            "macd": str(macd_path),
            "atr": str(atr_path),
        },
        pattern_limit=0,
        p124_top_n=200,
        pool_size=200,
    )
    selection_policy = dict(payload.get("selection_policy", {}))
    selection_policy["source_scope"] = "daily_p124_top200"
    selection_policy["purpose"] = "next trading day's ordinary intraday-screening source pool"
    payload["selection_policy"] = selection_policy
    target = write_intraday_pool(project_root=project_root, trade_date=trade_date, picker_payload=payload)
    written_payload = json.loads(target.read_text(encoding="utf-8"))
    elapsed = perf_counter() - started_at
    print(f"[{stage_index}/{total_stages}] intraday_pool_top200 完成，用时 {elapsed:.1f}s。", flush=True)
    return target, written_payload


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


def _run_track_stock_stage(stage_index: int, total_stages: int, project_root: Path, trade_date: date) -> Path:
    started_at = perf_counter()
    print(f"[{stage_index}/{total_stages}] 开始 track_stock_sheet2...", flush=True)
    result = update_track_stock_workbook(project_root=project_root, trade_date=trade_date, mode="daily")
    elapsed = perf_counter() - started_at
    print(
        f"[{stage_index}/{total_stages}] track_stock_sheet2 完成，写入 {result.output_rows} 行，用时 {elapsed:.1f}s。",
        flush=True,
    )
    return result.workbook_path


def _write_run_report(
    project_root: Path,
    trade_date: date,
    watchlist_payload: dict[str, object],
    watchlist_path: Path,
    *,
    intraday_pool_payload: dict[str, object],
    intraday_pool_path: Path,
    sector_watchlist_payload: dict[str, object],
    sector_watchlist_path: Path,
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
    track_stock_path: Path | None,
    filter_summary: object | None,
) -> Path:
    target = project_root / "reports" / "daily_screening"
    target.mkdir(parents=True, exist_ok=True)
    report_path = target / f"daily_screening_{trade_date.isoformat()}.json"
    report = {
        "trade_date": trade_date.isoformat(),
        "source_file": watchlist_payload.get("source_file"),
        "watchlist_stocks_path": str(watchlist_path),
        "watchlist_stocks_csv_path": str(watchlist_path.with_suffix(".csv")),
        "intraday_pool_path": str(intraday_pool_path),
        "intraday_pool_csv_path": str(intraday_pool_path.with_suffix(".csv")),
        "watchlist_sectors_path": str(sector_watchlist_path),
        "watchlist_sectors_csv_path": str(sector_watchlist_path.with_suffix(".csv")),
        "macd_path": str(macd_path) if macd_path is not None else None,
        "atr_path": str(atr_path) if atr_path is not None else None,
        "pattern_path": str(pattern_path) if pattern_path is not None else None,
        "phase1_path": str(phase1_path) if phase1_path is not None else None,
        "phase2_path": str(phase2_path) if phase2_path is not None else None,
        "phase4_path": str(phase4_path) if phase4_path is not None else None,
        "full_market_daily_returns_path": str(full_market_daily_returns_path) if full_market_daily_returns_path is not None else None,
        "phase9_path": str(phase9_path) if phase9_path is not None else None,
        "track_stock_path": str(track_stock_path) if track_stock_path is not None else None,
        "sector_membership_path": str(sector_path) if sector_path is not None else None,
        "sector_performance_path": str(sector_performance_path) if sector_performance_path is not None else None,
        "sector_leaders_path": str(sector_leaders_path) if sector_leaders_path is not None else None,
        "sector_leader_scores_all_path": str(sector_leader_scores_all_path) if sector_leader_scores_all_path is not None else None,
        "filter_summary": filter_summary,
        "candidate_count": len(watchlist_payload.get("candidates", []))
        if isinstance(watchlist_payload.get("candidates"), list)
        else 0,
        "intraday_pool_candidate_count": len(intraday_pool_payload.get("candidates", []))
        if isinstance(intraday_pool_payload.get("candidates"), list)
        else 0,
        "sector_candidate_count": len(sector_watchlist_payload.get("sectors", []))
        if isinstance(sector_watchlist_payload.get("sectors"), list)
        else 0,
        "deprecated_phases": ["P3", "P5", "P7", "P8", "P10"],
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
