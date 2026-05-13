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
from .full_market_limit_up_3d import limit_up_3d_model_path, limit_up_3d_predictions_path
from .full_market_return import alpha158_qlib_return_predictions_path
from .full_market_risk import barrier_risk_predictions_path, tail_risk_predictions_path
from .full_market_trade_day import trade_day_gate_prediction_path
from .trading_calendar import is_trading_day
from .watchlist import build_phase_daily_watchlist_candidates, load_watchlist, watchlist_path, watchlist_pattern_path, write_watchlist


PICKS_FILENAME = "选股.md"


@dataclass
class ScreeningResult:
    trade_date: date
    skipped: bool
    message: str
    report_path: Path | None = None
    watchlist_path: Path | None = None


def run_daily_screening(
    *,
    project_root: Path,
    trade_date: date,
    start_date: str = "20240101",
    picks_filename: str = PICKS_FILENAME,
) -> ScreeningResult:
    _ = picks_filename  # kept only for backward compatibility with existing callers
    config = load_config(project_root / "config" / "default.yaml")
    total_stages = 12
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

    phase5_needed, phase5_reason = _phase5_needs_refresh(project_root, trade_date)
    _run_project_stage(
        1,
        total_stages,
        "update",
        project_root,
        ["update", "--start-date", start_date, "--end-date", trade_date.strftime("%Y%m%d")],
    )
    _run_project_stage(2, total_stages, "macd", project_root, ["macd", "--date", trade_date.isoformat()])
    _run_project_stage(3, total_stages, "atr", project_root, ["atr", "--date", trade_date.isoformat()])
    fast_prediction_args = ["--latest-only", "--feature-lookback-bars", "61", "--compact-output"]
    _run_project_stage(
        4,
        total_stages,
        "phase1_tail_risk",
        project_root,
        ["predict-tail-risk", "--date", trade_date.isoformat(), *fast_prediction_args],
    )
    _run_project_stage(
        5,
        total_stages,
        "phase2_barrier_risk",
        project_root,
        ["predict-barrier-risk", "--date", trade_date.isoformat(), *fast_prediction_args],
    )
    _run_project_stage(
        6,
        total_stages,
        "phase4_alpha158_return",
        project_root,
        ["predict-alpha158-qlib-return", "--date", trade_date.isoformat(), *fast_prediction_args],
    )
    _run_phase8_stage(7, total_stages, project_root, trade_date, fast_prediction_args)
    _run_project_stage(8, total_stages, "phase7_trade_day_gate", project_root, ["predict-trade-day-gate", "--date", trade_date.isoformat()])
    _run_phase5_stage(
        9,
        total_stages,
        project_root,
        trade_date,
        refresh_needed=phase5_needed,
        reason=phase5_reason,
    )
    _run_project_stage(10, total_stages, "pattern", project_root, ["pattern", "--as-of", trade_date.isoformat()])
    generated_watchlist_path, watchlist_payload = _run_phase_watchlist_stage(11, total_stages, project_root, trade_date)
    _run_project_stage(12, total_stages, "track_stock", project_root, ["track-stock", "--date", trade_date.isoformat()])

    watchlist_pattern = watchlist_pattern_path(project_root, trade_date)
    macd_path = _macd_report_path(project_root, trade_date)
    atr_path = _atr_report_path(project_root, trade_date)
    pattern_path = _pattern_report_path(project_root, trade_date)
    phase1_path = tail_risk_predictions_path(project_root, trade_date)
    phase2_path = barrier_risk_predictions_path(project_root, trade_date)
    phase4_path = alpha158_qlib_return_predictions_path(project_root, trade_date)
    phase8_path = limit_up_3d_predictions_path(project_root, trade_date)
    phase5_path = _phase5_annual_measures_path(project_root)
    phase7_path = trade_day_gate_prediction_path(project_root, trade_date)
    track_stock_path = project_root / "track_stock.xlsx"
    report_path = _write_run_report(
        project_root,
        trade_date,
        watchlist_payload,
        generated_watchlist_path,
        watchlist_pattern_path=watchlist_pattern if watchlist_pattern.exists() else None,
        macd_path=macd_path if macd_path.exists() else None,
        atr_path=atr_path if atr_path.exists() else None,
        pattern_path=pattern_path if pattern_path.exists() else None,
        phase1_path=phase1_path if phase1_path.exists() else None,
        phase2_path=phase2_path if phase2_path.exists() else None,
        phase4_path=phase4_path if phase4_path.exists() else None,
        phase8_path=phase8_path if phase8_path.exists() else None,
        phase5_path=phase5_path if phase5_path.exists() else None,
        phase7_path=phase7_path if phase7_path.exists() else None,
        track_stock_path=track_stock_path if track_stock_path.exists() else None,
        filter_summary=watchlist_payload.get("filter_summary") if isinstance(watchlist_payload, dict) else None,
    )
    return ScreeningResult(
        trade_date=trade_date,
        skipped=False,
        message=f"已完成 {trade_date.isoformat()} 每日筛选，并生成 {generated_watchlist_path}",
        report_path=report_path,
        watchlist_path=generated_watchlist_path,
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


def _run_phase5_stage(
    stage_index: int,
    total_stages: int,
    project_root: Path,
    trade_date: date,
    *,
    refresh_needed: bool,
    reason: str,
) -> None:
    if not refresh_needed:
        print(f"[{stage_index}/{total_stages}] phase5_mcd_crash 跳过：{reason}", flush=True)
        return
    _run_project_stage(
        stage_index,
        total_stages,
        "phase5_mcd_crash",
        project_root,
        [
            "validate-mcd-crash-risk",
            "--start-date",
            "2015-01-01",
            "--end-date",
            trade_date.isoformat(),
        ],
    )


def _run_phase8_stage(stage_index: int, total_stages: int, project_root: Path, trade_date: date, fast_prediction_args: list[str]) -> None:
    if not limit_up_3d_model_path(project_root).exists():
        print(f"[{stage_index}/{total_stages}] phase8_limit_up_3d 跳过：模型文件尚未训练完成。", flush=True)
        return
    _run_project_stage(
        stage_index,
        total_stages,
        "phase8_limit_up_3d",
        project_root,
        ["predict-limit-up-3d-opportunity", "--date", trade_date.isoformat(), *fast_prediction_args],
    )


def _run_phase_watchlist_stage(
    stage_index: int,
    total_stages: int,
    project_root: Path,
    trade_date: date,
) -> tuple[Path, dict[str, object]]:
    started_at = perf_counter()
    print(f"[{stage_index}/{total_stages}] 开始 phase_watchlist...", flush=True)
    pattern_path = _pattern_report_path(project_root, trade_date)
    phase1_path = tail_risk_predictions_path(project_root, trade_date)
    phase2_path = barrier_risk_predictions_path(project_root, trade_date)
    phase4_path = alpha158_qlib_return_predictions_path(project_root, trade_date)
    phase8_path = limit_up_3d_predictions_path(project_root, trade_date)
    phase5_path = _phase5_annual_measures_path(project_root)
    phase7_path = trade_day_gate_prediction_path(project_root, trade_date)
    macd_path = _macd_report_path(project_root, trade_date)
    atr_path = _atr_report_path(project_root, trade_date)

    payload = build_phase_daily_watchlist_candidates(
        trade_date=trade_date,
        pattern_frame=_read_required_csv(pattern_path),
        phase1_predictions=_read_required_csv(phase1_path),
        phase2_predictions=_read_required_csv(phase2_path),
        phase4_predictions=_read_required_csv(phase4_path),
        phase8_predictions=_read_optional_csv(phase8_path),
        phase7_prediction=_read_required_csv(phase7_path),
        phase5_measures=_read_optional_csv(phase5_path),
        macd_frame=_read_optional_csv(macd_path),
        atr_frame=_read_optional_csv(atr_path),
        source_files={
            "pattern": str(pattern_path),
            "phase1": str(phase1_path),
            "phase2": str(phase2_path),
            "phase4": str(phase4_path),
            "phase8": str(phase8_path),
            "phase5": str(phase5_path),
            "phase7": str(phase7_path),
            "macd": str(macd_path),
            "atr": str(atr_path),
        },
        phase_filter_rate=0.2,
        phase4_top_n=20,
    )
    target = write_watchlist(project_root=project_root, trade_date=trade_date, picker_payload=payload)
    written_payload = load_watchlist(project_root=project_root, trade_date=trade_date)
    elapsed = perf_counter() - started_at
    print(f"[{stage_index}/{total_stages}] phase_watchlist 完成，用时 {elapsed:.1f}s。", flush=True)
    return target, written_payload


def _write_run_report(
    project_root: Path,
    trade_date: date,
    watchlist_payload: dict[str, object],
    watchlist_path: Path,
    *,
    watchlist_pattern_path: Path | None,
    macd_path: Path | None,
    atr_path: Path | None,
    pattern_path: Path | None,
    phase1_path: Path | None,
    phase2_path: Path | None,
    phase4_path: Path | None,
    phase8_path: Path | None,
    phase5_path: Path | None,
    phase7_path: Path | None,
    track_stock_path: Path | None,
    filter_summary: object | None,
) -> Path:
    target = project_root / "reports" / "daily_screening"
    target.mkdir(parents=True, exist_ok=True)
    report_path = target / f"daily_screening_{trade_date.isoformat()}.json"
    report = {
        "trade_date": trade_date.isoformat(),
        "source_file": watchlist_payload.get("source_file"),
        "watchlist_path": str(watchlist_path),
        "watchlist_csv_path": str(watchlist_path.with_suffix(".csv")),
        "watchlist_pattern_path": str(watchlist_pattern_path) if watchlist_pattern_path is not None else None,
        "macd_path": str(macd_path) if macd_path is not None else None,
        "atr_path": str(atr_path) if atr_path is not None else None,
        "pattern_path": str(pattern_path) if pattern_path is not None else None,
        "phase1_path": str(phase1_path) if phase1_path is not None else None,
        "phase2_path": str(phase2_path) if phase2_path is not None else None,
        "phase4_path": str(phase4_path) if phase4_path is not None else None,
        "phase8_path": str(phase8_path) if phase8_path is not None else None,
        "phase5_path": str(phase5_path) if phase5_path is not None else None,
        "phase7_path": str(phase7_path) if phase7_path is not None else None,
        "track_stock_path": str(track_stock_path) if track_stock_path is not None else None,
        "trade_permission": watchlist_payload.get("trade_permission"),
        "filter_summary": filter_summary,
        "candidate_count": len(watchlist_payload.get("candidates", []))
        if isinstance(watchlist_payload.get("candidates"), list)
        else 0,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def _macd_report_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "macd" / f"macd_{trade_date.isoformat()}.csv"


def _atr_report_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "atr" / f"atr_{trade_date.isoformat()}.csv"


def _pattern_report_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "patterns" / f"patterns_all_{trade_date.isoformat()}.csv"


def _phase5_annual_measures_path(project_root: Path) -> Path:
    return project_root / "reports" / "full_market_model" / "mcd_crash_annual_measures.csv"


def _phase5_config_path(project_root: Path) -> Path:
    return project_root / "reports" / "full_market_model" / "mcd_crash_config.json"


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required daily-screening input not found: {path}")
    return pd.read_csv(path)


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _phase5_needs_refresh(project_root: Path, trade_date: date) -> tuple[bool, str]:
    annual_path = _phase5_annual_measures_path(project_root)
    config_path = _phase5_config_path(project_root)
    if not annual_path.exists() or not config_path.exists():
        return True, "phase5 result missing"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True, "phase5 config unreadable"
    latest = _parse_optional_iso_date(config.get("end_date"))
    if latest is None:
        return True, "phase5 end_date missing"
    if latest >= trade_date:
        return False, f"phase5 covers {latest.isoformat()}"
    lag = _count_local_trading_days_between(project_root, latest, trade_date)
    if lag > 6:
        return True, f"phase5 stale by {lag} local trading days"
    return False, f"phase5 lag {lag} local trading days"


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
