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

from .config import load_config
from .trading_calendar import is_trading_day
from .watchlist import load_watchlist, watchlist_path


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
    print(f"[0/4] 检查 {trade_date.isoformat()} 是否为交易日...", flush=True)
    trading_day = is_trading_day(config.provider, trade_date)
    if not trading_day:
        print(f"[0/4] {trade_date.isoformat()} 不是交易日，跳过每日筛选。", flush=True)
        return ScreeningResult(
            trade_date=trade_date,
            skipped=True,
            message=f"{trade_date.isoformat()} 不是交易日，已跳过每日筛选。",
        )
    print(f"[0/4] {trade_date.isoformat()} 是交易日，开始执行每日筛选。", flush=True)

    _run_project_stage(1, 4, "update", project_root, ["update", "--start-date", start_date])
    _run_project_stage(2, 4, "tradingview", project_root, ["tradingview", "--date", trade_date.isoformat()])
    _run_project_stage(3, 4, "divergence", project_root, ["divergence", "--date", trade_date.isoformat()])
    _run_project_stage(4, 4, "pattern", project_root, ["pattern", "--as-of", trade_date.isoformat()])

    generated_watchlist_path = watchlist_path(project_root, trade_date)
    watchlist_payload = load_watchlist(project_root=project_root, trade_date=trade_date)
    report_path = _write_run_report(project_root, trade_date, watchlist_payload, generated_watchlist_path)
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


def _write_run_report(
    project_root: Path,
    trade_date: date,
    watchlist_payload: dict[str, object],
    watchlist_path: Path,
) -> Path:
    target = project_root / "reports" / "daily_screening"
    target.mkdir(parents=True, exist_ok=True)
    report_path = target / f"daily_screening_{trade_date.isoformat()}.json"
    report = {
        "trade_date": trade_date.isoformat(),
        "source_file": watchlist_payload.get("source_file"),
        "watchlist_path": str(watchlist_path),
        "candidate_count": len(watchlist_payload.get("candidates", []))
        if isinstance(watchlist_payload.get("candidates"), list)
        else 0,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


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
