from __future__ import annotations

import argparse
import logging
import os
import socket
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from .atr import build_atr_export_frame, build_atr_snapshot_row, normalize_atr_summary_frame
from .config import load_config
from .daily_screening import run_daily_screening
from .daily_screening_backtest import (
    DEFAULT_BACKTEST_STRATEGIES,
    backtest_daily_screening_components,
    format_backtest_summary,
)
from .data_sources import create_data_provider
from .full_market_crash import validate_mcd_crash_risk
from .full_market_panel import audit_full_market_data, format_full_market_audit_summary
from .full_market_return import (
    format_alpha158_qlib_return_prediction_table,
    predict_alpha158_qlib_return,
    train_alpha158_qlib_return_model,
    validate_alpha158_qlib_return,
)
from .full_market_risk import (
    DEFAULT_BARRIER_MODEL_NAMES,
    DEFAULT_PANEL_MODEL_NAMES,
    format_barrier_risk_prediction_table,
    format_tail_risk_prediction_table,
    predict_barrier_risk,
    predict_tail_risk,
    reproduce_tail_risk,
    summarize_tail_risk_walkforward,
    train_barrier_risk_model,
    train_tail_risk_model,
    validate_barrier_risk_grid,
    validate_barrier_risk_walkforward,
    validate_tail_risk_walkforward,
)
from .full_market_trade_day import (
    DEFAULT_TRADE_DAY_MODEL_NAMES,
    format_trade_day_gate_prediction_table,
    predict_trade_day_gate,
    train_trade_day_gate_model,
    validate_trade_day_gate,
)
from .indicators import add_indicators
from .intraday_screening import DEFAULT_INTRADAY_REPORT_KEEP_DATES, run_intraday_screening
from .intraday_update import INTRADAY_DATA_INTERFACES, run_intraday_update
from .macd_divergence import summarize_recent_macd_divergence
from .models import NetworkConfig
from .paths import ProjectPaths
from .phase_display import PHASE_DISPLAY_COLUMNS, append_daily_phase_display_columns
from .reporting import format_multi_pattern_summary, format_report
from .screener import Screener, parse_as_of
from .storage import DailyBarsReadError, Storage
from .strategies import STRATEGY_NAMES
from .synthetic_market import build_synthetic_market_index
from .track_stock import DEFAULT_TRACK_STOCK_FILENAME, update_track_stock_workbook
from .trend_reporting import save_atr_report, save_macd_report
from .universe import build_main_board_universe
from .watchlist import build_watchlist_candidates_from_patterns, write_watchlist


PATTERN_FLAG_MAP = {
    "pattern1": "volume_top_pre_breakout",
    "pattern2": "volume_top_breakout",
    "pattern3": "volume_top_follow_through",
    "pattern4": "duck_nostril_cross",
    "pattern5": "trend_pullback",
    "pattern6": "double_volume_support_rebound",
}
PATTERN_LABEL_MAP = {
    "volume_top_pre_breakout": "1",
    "volume_top_breakout": "2",
    "volume_top_follow_through": "3",
    "duck_nostril_cross": "4",
    "trend_pullback": "5",
    "double_volume_support_rebound": "6",
}
PROGRESS_LOG_INTERVAL = 100
LOCAL_PROXY_HOSTS = {"127.0.0.1", "localhost", "::1"}
DEFAULT_INDEX_SYMBOLS = ("sh000001", "sh000300", "sh000905", "sh000906", "sz399001", "sz399006")


def build_parser() -> argparse.ArgumentParser:
    _localize_argparse()
    parser = argparse.ArgumentParser(
        description="A 股 daily-screening 主线工具",
        epilog=(
            "最常用：\n"
            "  python -m stocks_analyzer --project-root . daily-screening\n\n"
            "拆分运行：\n"
            "  python -m stocks_analyzer --project-root . update --start-date 20150101\n"
            "  python -m stocks_analyzer --project-root . macd --date 2026-05-07\n"
            "  python -m stocks_analyzer --project-root . atr --date 2026-05-07\n"
            "  python -m stocks_analyzer --project-root . predict-tail-risk --date 2026-05-07\n"
            "  python -m stocks_analyzer --project-root . predict-barrier-risk --date 2026-05-07\n"
            "  python -m stocks_analyzer --project-root . predict-alpha158-qlib-return --date 2026-05-07\n"
            "  python -m stocks_analyzer --project-root . predict-trade-day-gate --date 2026-05-07\n"
            "  python -m stocks_analyzer --project-root . pattern --as-of 2026-05-07\n"
            "  python -m stocks_analyzer --project-root . track-stock --date 2026-05-07\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", default="config/default.yaml", help="YAML 配置文件路径")
    parser.add_argument("--project-root", default=".", help="项目根目录")
    parser.add_argument("--log-level", default="INFO", help="日志级别")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.title = "子命令"

    _add_update_parser(subparsers)
    _add_pattern_parsers(subparsers)
    _add_technical_parsers(subparsers)
    _add_model_maintenance_parsers(subparsers)
    _add_daily_parsers(subparsers)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    project_root = Path(args.project_root).resolve()
    _load_local_env(project_root / ".env.local")
    config = load_config(project_root / args.config)
    if _command_needs_network(args.command):
        _configure_network(config.network)
    paths = ProjectPaths(project_root, config.storage)
    storage = Storage(paths)

    if args.command == "update":
        provider = _create_update_data_provider(args.data_interface)
        try:
            _run_update(
                storage,
                provider,
                config.universe.exclude_st,
                config.adjustment,
                args.symbol,
                args.start_date,
                args.end_date,
                args.limit,
                index_symbols=tuple(_parse_str_list(args.index_symbols)),
                update_indexes=args.update_index,
                index_interface=args.index_interface,
            )
        finally:
            provider.close()
        return

    if args.command == "intraday-update":
        symbols = _parse_str_list(args.symbol) if args.symbol else None
        result = run_intraday_update(
            storage=storage,
            project_root=project_root,
            source=args.data_interface,
            symbols=symbols,
            limit=args.limit,
            watchlist_only=args.watchlist_only,
            timeout_seconds=args.timeout,
            chunk_size=args.chunk_size,
        )
        print("Intraday update complete.")
        print(f"Source: {result.source}")
        if result.source_watchlist_path is not None:
            print(f"Source watchlist: {result.source_watchlist_path}")
        print(f"Requested symbols: {len(result.requested_symbols)}")
        print(f"Updated symbols: {len(result.updated_symbols)}")
        print(f"Data directory: {paths.intraday_dir}")
        if result.failed_symbols:
            print(f"Missing symbols: {', '.join(result.failed_symbols[:20])}")
        return

    if args.command == "intraday-screening":
        trade_date = _parse_required_date(args.date) if args.date else date.today()
        result = run_intraday_screening(
            storage=storage,
            project_root=project_root,
            trade_date=trade_date,
            data_interface=args.data_interface,
            limit=args.limit,
            watchlist_only=args.watchlist_only,
            skip_intraday_update=args.skip_intraday_update,
            timeout_seconds=args.timeout,
            chunk_size=args.chunk_size,
            output=Path(args.output).resolve() if args.output else None,
            report_keep_dates=args.keep_report_dates,
        )
        print("Intraday screening complete.")
        print(f"Trade date: {result.trade_date.isoformat()}")
        print(f"Source watchlist: {result.source_watchlist_path}")
        print(f"Candidates: {result.candidate_count}")
        print(f"Intraday updated: {result.intraday_updated_count}")
        if result.focus_output_path is not None:
            print(f"Previous Top20 output: {result.focus_output_path}")
            print(f"Previous Top20 candidates: {result.focus_candidate_count}")
        print(f"Output: {result.output_path}")
        print(f"Top20 focus: {result.top20_path}")
        if result.cleaned_report_files:
            print(f"Cleaned report files: {result.cleaned_report_files}")
        if result.missing_intraday_symbols:
            print(f"Missing intraday symbols: {', '.join(result.missing_intraday_symbols[:20])}")
        return

    if args.command == "pattern":
        _run_pattern(
            storage,
            config.provider,
            config,
            parse_as_of(args.as_of),
            _selected_patterns(args),
            args.limit,
            args.output,
        )
        return

    if args.command == "report":
        _run_report(storage, config, _parse_required_date(args.date), args.limit)
        return

    if args.command == "macd":
        _run_macd(storage, config, paths, _parse_required_date(args.date), args.top_n, args.output)
        return

    if args.command == "atr":
        _run_atr(storage, config, paths, _parse_required_date(args.date), args.top_n, args.output)
        return

    if args.command == "audit-full-market-data":
        _run_audit_full_market(storage, project_root, args)
        return

    if args.command == "build-synthetic-market":
        _run_build_synthetic_market(storage, project_root, args)
        return

    if args.command == "reproduce-tail-risk":
        _run_reproduce_tail_risk(storage, project_root, args)
        return

    if args.command == "validate-tail-risk-walkforward":
        _run_validate_tail_risk(storage, project_root, args)
        return

    if args.command == "train-tail-risk-model":
        _run_train_tail_risk(storage, project_root, args)
        return

    if args.command == "predict-tail-risk":
        _run_predict_tail_risk(storage, project_root, args)
        return

    if args.command == "reproduce-barrier-risk":
        _run_validate_barrier_risk(storage, project_root, args)
        return

    if args.command == "validate-barrier-risk-grid":
        _run_validate_barrier_grid(storage, project_root, args)
        return

    if args.command == "train-barrier-risk-model":
        _run_train_barrier_risk(storage, project_root, args)
        return

    if args.command == "predict-barrier-risk":
        _run_predict_barrier_risk(storage, project_root, args)
        return

    if args.command == "validate-alpha158-qlib-return":
        _run_validate_alpha158_return(storage, project_root, args)
        return

    if args.command == "train-alpha158-qlib-return-model":
        _run_train_alpha158_return(storage, project_root, args)
        return

    if args.command == "predict-alpha158-qlib-return":
        _run_predict_alpha158_return(storage, project_root, args)
        return

    if args.command == "validate-mcd-crash-risk":
        _run_validate_mcd_crash(storage, project_root, args)
        return

    if args.command == "validate-trade-day-gate":
        _run_validate_trade_day_gate(storage, project_root, args)
        return

    if args.command == "train-trade-day-gate-model":
        _run_train_trade_day_gate(storage, project_root, args)
        return

    if args.command == "predict-trade-day-gate":
        _run_predict_trade_day_gate(storage, project_root, args)
        return

    if args.command == "backtest-daily-screening-components":
        _run_backtest_daily_screening_components(storage, project_root, config, args)
        return

    if args.command == "daily-screening":
        trade_date = _parse_required_date(args.date) if args.date else date.today()
        result = run_daily_screening(project_root=project_root, trade_date=trade_date, start_date=args.start_date)
        print(result.message)
        if result.report_path:
            print(f"报告文件：{result.report_path}")
        return

    if args.command == "track-stock":
        trade_date = _parse_required_date(args.date) if args.date else date.today()
        result = update_track_stock_workbook(
            project_root=project_root,
            trade_date=trade_date,
            workbook_path=Path(args.workbook),
        )
        print("Track stock workbook updated.")
        print(f"Workbook: {result.workbook_path}")
        print(f"Trade date: {result.trade_date.isoformat()}")
        print(f"Tracked symbols: {result.tracked_count}")
        print(f"Sheet2 rows: {result.output_rows}")
        return

    parser.error(f"Unknown command: {args.command}")


def _add_update_parser(subparsers: argparse._SubParsersAction) -> None:
    update = subparsers.add_parser("update", help="更新股票池和日线数据")
    update.add_argument("symbol", nargs="?", help="可选的 6 位股票代码，只更新该股票")
    update.add_argument("--start-date", default="20230101", help="开始日期，格式 YYYYMMDD")
    update.add_argument("--end-date", default=datetime.today().strftime("%Y%m%d"), help="结束日期，格式 YYYYMMDD")
    update.add_argument("--limit", type=int, default=None, help="仅更新前 N 只股票")
    update.add_argument("--data-interface", choices=["baostock", "sina", "eastmoney"], default="sina", help="日线数据接口")
    update.add_argument("--update-index", action="store_true", help="额外更新外部指数日线，默认不更新")
    update.add_argument("--index-interface", choices=["baostock", "sina", "eastmoney"], default="sina", help="指数日线接口")
    update.add_argument("--index-symbols", default=",".join(DEFAULT_INDEX_SYMBOLS), help="外部指数代码，逗号分隔")

    intraday = subparsers.add_parser("intraday-update", help="盘中更新全市场或 watchlist 的临时日 K 数据")
    intraday.add_argument("symbol", nargs="?", help="可选的 6 位股票代码或逗号分隔代码；默认读取全市场")
    intraday.add_argument(
        "--data-interface",
        choices=INTRADAY_DATA_INTERFACES,
        default="eastmoney_direct",
        help="盘中数据接口",
    )
    intraday.add_argument("--limit", type=int, default=None, help="仅更新前 N 只股票")
    intraday.add_argument("--watchlist-only", action="store_true", help="只更新最新主 watchlist 股票；默认更新全市场")
    intraday.add_argument("--timeout", type=float, default=15.0, help="单次网络请求超时秒数")
    intraday.add_argument("--chunk-size", type=int, default=50, help="批量请求每批股票数量")

    intraday_screening = subparsers.add_parser("intraday-screening", help="用盘中临时日 K 做全市场或 watchlist 分析")
    intraday_screening.add_argument("--date", default=None, help="目标日期，格式 YYYY-MM-DD，默认今天")
    intraday_screening.add_argument(
        "--data-interface",
        choices=INTRADAY_DATA_INTERFACES,
        default="eastmoney_direct",
        help="盘中数据接口",
    )
    intraday_screening.add_argument("--limit", type=int, default=None, help="仅分析前 N 只股票")
    intraday_screening.add_argument("--watchlist-only", action="store_true", help="只分析前一日主 watchlist；默认分析全市场")
    intraday_screening.add_argument("--skip-intraday-update", action="store_true", help="不刷新盘中数据，直接使用 data/intraday")
    intraday_screening.add_argument("--timeout", type=float, default=15.0, help="单次网络请求超时秒数")
    intraday_screening.add_argument("--chunk-size", type=int, default=50, help="批量请求每批股票数量")
    intraday_screening.add_argument("--output", default=None, help="可选 CSV 输出路径")
    intraday_screening.add_argument(
        "--keep-report-dates",
        type=int,
        default=DEFAULT_INTRADAY_REPORT_KEEP_DATES,
        help="reports/intraday_screening 中保留最近多少个正式报告日期，默认 10；过程预测文件不写入 reports",
    )


def _add_pattern_parsers(subparsers: argparse._SubParsersAction) -> None:
    pattern = subparsers.add_parser("pattern", help="识别本地日线数据中的 1 到 6 号模式")
    for item in ("1", "2", "3", "4", "5", "6"):
        pattern.add_argument(f"--{item}", dest=f"pattern{item}", action="store_true", help=f"只识别模式 {item}")
    pattern.add_argument("--as-of", default=None, help="分析截止日期，格式 YYYY-MM-DD")
    pattern.add_argument("--limit", type=int, default=None, help="终端最多显示多少行")
    pattern.add_argument("--output", default=None, help="可选 CSV 输出路径")

    report = subparsers.add_parser("report", help="读取已保存的模式识别结果")
    report.add_argument("--date", required=True, help="结果日期，格式 YYYY-MM-DD")
    report.add_argument("--limit", type=int, default=None, help="终端最多显示多少行")


def _add_technical_parsers(subparsers: argparse._SubParsersAction) -> None:
    macd = subparsers.add_parser("macd", help="生成指定日期的 MACD/量价统一技术状态表")
    macd.add_argument("--date", required=True, help="识别日期，格式 YYYY-MM-DD")
    macd.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")
    macd.add_argument("--output", default=None, help="可选 CSV 输出路径")

    atr = subparsers.add_parser("atr", help="生成指定日期的 ATR 风险辅助表")
    atr.add_argument("--date", required=True, help="识别日期，格式 YYYY-MM-DD")
    atr.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")
    atr.add_argument("--output", default=None, help="可选 CSV 输出路径")


def _add_model_maintenance_parsers(subparsers: argparse._SubParsersAction) -> None:
    audit = subparsers.add_parser("audit-full-market-data", help="审计全市场日线数据是否足够支持主线模型")
    audit.add_argument("--limit", type=int, default=None)
    audit.add_argument("--min-exact-history-days", type=int, default=900)
    audit.add_argument("--tail-lookback-days", type=int, default=100)
    audit.add_argument("--max-horizon-days", type=int, default=20)
    audit.add_argument("--output-dir", default=None)

    synthetic = subparsers.add_parser("build-synthetic-market", help="基于本地个股日线构建市场代理指数")
    synthetic.add_argument("--start-date", default=None)
    synthetic.add_argument("--end-date", default=None)
    synthetic.add_argument("--limit", type=int, default=None)
    synthetic.add_argument("--min-stock-count", type=int, default=500)
    synthetic.add_argument("--output", default=None)

    _add_phase1_parsers(subparsers)
    _add_phase2_parsers(subparsers)
    _add_phase4_parsers(subparsers)
    _add_phase5_parser(subparsers)
    _add_phase7_parsers(subparsers)
    _add_backtest_parser(subparsers)


def _add_phase1_parsers(subparsers: argparse._SubParsersAction) -> None:
    validate_tail = subparsers.add_parser("validate-tail-risk-walkforward", help="多窗口验证 Phase1 尾部风险模型")
    validate_tail.add_argument("--start-date", default=None)
    validate_tail.add_argument("--end-date", default=None)
    validate_tail.add_argument("--limit", type=int, default=None)
    validate_tail.add_argument("--train-days", type=int, default=1000)
    validate_tail.add_argument("--valid-days", type=int, default=250)
    validate_tail.add_argument("--step-days", type=int, default=250)
    validate_tail.add_argument("--embargo-days", type=int, default=None)
    validate_tail.add_argument("--max-windows", type=int, default=None)
    validate_tail.add_argument("--lookback-days", type=int, default=100)
    validate_tail.add_argument("--quantile", type=float, default=0.05)
    validate_tail.add_argument("--horizon-days", type=int, default=1)
    validate_tail.add_argument("--min-training-rows", type=int, default=200)
    validate_tail.add_argument("--panel-models", default=",".join(DEFAULT_PANEL_MODEL_NAMES))
    validate_tail.add_argument("--filter-rates", default="0.2")
    validate_tail.add_argument("--return-tolerance", type=float, default=0.001)
    validate_tail.add_argument("--allow-short-sample", action="store_true")

    reproduce_tail = subparsers.add_parser("reproduce-tail-risk", help="复现 Phase1 尾部风险分类模型")
    reproduce_tail.add_argument("--start-date", default=None)
    reproduce_tail.add_argument("--end-date", default=None)
    reproduce_tail.add_argument("--train-end", required=True)
    reproduce_tail.add_argument("--valid-end", required=True)
    reproduce_tail.add_argument("--limit", type=int, default=None)
    reproduce_tail.add_argument("--lookback-days", type=int, default=100)
    reproduce_tail.add_argument("--quantile", type=float, default=0.05)
    reproduce_tail.add_argument("--horizon-days", type=int, default=1)
    reproduce_tail.add_argument("--min-training-rows", type=int, default=200)
    reproduce_tail.add_argument("--index-source-column", default="synthetic_equal_weight_index")
    reproduce_tail.add_argument("--panel-models", default=",".join(DEFAULT_PANEL_MODEL_NAMES))
    reproduce_tail.add_argument("--skip-index", action="store_true")
    reproduce_tail.add_argument("--skip-panel", action="store_true")
    reproduce_tail.add_argument("--allow-short-sample", action="store_true")

    train_tail = subparsers.add_parser("train-tail-risk-model", help="训练 Phase1 尾部风险部署模型")
    train_tail.add_argument("--start-date", default=None)
    train_tail.add_argument("--end-date", default=None)
    train_tail.add_argument("--model-name", default="logistic_regression")
    train_tail.add_argument("--limit", type=int, default=None)
    train_tail.add_argument("--lookback-days", type=int, default=100)
    train_tail.add_argument("--quantile", type=float, default=0.05)
    train_tail.add_argument("--horizon-days", type=int, default=1)
    train_tail.add_argument("--min-training-rows", type=int, default=200)

    predict_tail = subparsers.add_parser("predict-tail-risk", help="使用 Phase1 artifact 对指定日期全市场打分")
    predict_tail.add_argument("--date", required=True)
    predict_tail.add_argument("--limit", type=int, default=None)
    predict_tail.add_argument("--top-n", type=int, default=20)
    predict_tail.add_argument("--output", default=None)


def _add_phase2_parsers(subparsers: argparse._SubParsersAction) -> None:
    reproduce_barrier = subparsers.add_parser("reproduce-barrier-risk", help="复现 Phase2 triple-barrier 风险模型")
    reproduce_barrier.add_argument("--start-date", default=None)
    reproduce_barrier.add_argument("--end-date", default=None)
    reproduce_barrier.add_argument("--limit", type=int, default=None)
    reproduce_barrier.add_argument("--train-days", type=int, default=1000)
    reproduce_barrier.add_argument("--valid-days", type=int, default=250)
    reproduce_barrier.add_argument("--step-days", type=int, default=250)
    reproduce_barrier.add_argument("--embargo-days", type=int, default=None)
    reproduce_barrier.add_argument("--max-windows", type=int, default=None)
    reproduce_barrier.add_argument("--horizon-days", type=int, default=20)
    reproduce_barrier.add_argument("--downside-atr-mult", type=float, default=1.0)
    reproduce_barrier.add_argument("--upside-atr-mult", type=float, default=2.0)
    reproduce_barrier.add_argument("--downside-pct", type=float, default=None)
    reproduce_barrier.add_argument("--upside-pct", type=float, default=None)
    reproduce_barrier.add_argument("--no-upside-barrier", action="store_true")
    reproduce_barrier.add_argument("--label-variant", choices=["barrier_down_first", "max_drawdown_exceed"], default="barrier_down_first")
    reproduce_barrier.add_argument("--label-method", choices=["a_share_daily", "mlfin_cusum"], default="a_share_daily")
    reproduce_barrier.add_argument("--volatility-lookback", type=int, default=100)
    reproduce_barrier.add_argument("--pt-mult", type=float, default=1.0)
    reproduce_barrier.add_argument("--sl-mult", type=float, default=1.0)
    reproduce_barrier.add_argument("--min-ret", type=float, default=0.005)
    reproduce_barrier.add_argument("--cusum-threshold", type=float, default=None)
    reproduce_barrier.add_argument("--cusum-threshold-mult", type=float, default=1.0)
    reproduce_barrier.add_argument("--min-training-rows", type=int, default=200)
    reproduce_barrier.add_argument("--models", default=",".join(DEFAULT_BARRIER_MODEL_NAMES))
    reproduce_barrier.add_argument("--filter-rates", default="0.2")
    reproduce_barrier.add_argument("--return-tolerance", type=float, default=0.001)
    reproduce_barrier.add_argument("--allow-short-sample", action="store_true")

    barrier_grid = subparsers.add_parser("validate-barrier-risk-grid", help="验证 Phase2 mlfin_cusum 参数稳健性")
    barrier_grid.add_argument("--start-date", default=None)
    barrier_grid.add_argument("--end-date", default=None)
    barrier_grid.add_argument("--limit", type=int, default=None)
    barrier_grid.add_argument("--train-days", type=int, default=1000)
    barrier_grid.add_argument("--valid-days", type=int, default=250)
    barrier_grid.add_argument("--step-days", type=int, default=250)
    barrier_grid.add_argument("--max-windows", type=int, default=None)
    barrier_grid.add_argument("--horizon-days-grid", default="5,10")
    barrier_grid.add_argument("--pt-sl-grid", default="1:1,2:2")
    barrier_grid.add_argument("--min-ret-grid", default="0.003,0.005")
    barrier_grid.add_argument("--volatility-lookback", type=int, default=100)
    barrier_grid.add_argument("--models", default="lightgbm_classifier")
    barrier_grid.add_argument("--filter-rates", default="0.2")
    barrier_grid.add_argument("--return-tolerance", type=float, default=0.001)
    barrier_grid.add_argument("--min-training-rows", type=int, default=200)
    barrier_grid.add_argument("--allow-short-sample", action="store_true")

    train_barrier = subparsers.add_parser("train-barrier-risk-model", help="训练 Phase2 triple-barrier 风险部署模型")
    train_barrier.add_argument("--start-date", default="2015-01-01")
    train_barrier.add_argument("--end-date", default=date.today().isoformat())
    train_barrier.add_argument("--model-name", default="lightgbm_classifier")
    train_barrier.add_argument("--limit", type=int, default=None)
    train_barrier.add_argument("--horizon-days", type=int, default=5)
    train_barrier.add_argument("--volatility-lookback", type=int, default=100)
    train_barrier.add_argument("--pt-mult", type=float, default=1.0)
    train_barrier.add_argument("--sl-mult", type=float, default=1.0)
    train_barrier.add_argument("--min-ret", type=float, default=0.005)
    train_barrier.add_argument("--min-training-rows", type=int, default=200)

    predict_barrier = subparsers.add_parser("predict-barrier-risk", help="使用 Phase2 artifact 对指定日期全市场打分")
    predict_barrier.add_argument("--date", required=True)
    predict_barrier.add_argument("--limit", type=int, default=None)
    predict_barrier.add_argument("--top-n", type=int, default=20)
    predict_barrier.add_argument("--output", default=None)


def _add_phase4_parsers(subparsers: argparse._SubParsersAction) -> None:
    validate_return = subparsers.add_parser("validate-alpha158-qlib-return", help="验证 Phase4 Qlib Alpha158 收益回归模型")
    validate_return.add_argument("--start-date", default="2015-01-01")
    validate_return.add_argument("--end-date", default="2026-05-07")
    validate_return.add_argument("--train-end", required=True)
    validate_return.add_argument("--valid-end", required=True)
    validate_return.add_argument("--limit", type=int, default=None)
    validate_return.add_argument("--topk", type=int, default=50)
    validate_return.add_argument("--drop", type=int, default=5)
    validate_return.add_argument("--min-training-rows", type=int, default=200)

    train_return = subparsers.add_parser("train-alpha158-qlib-return-model", help="训练 Phase4 Qlib Alpha158 收益部署模型")
    train_return.add_argument("--start-date", default="2015-01-01")
    train_return.add_argument("--end-date", default=date.today().isoformat())
    train_return.add_argument("--limit", type=int, default=None)
    train_return.add_argument("--min-training-rows", type=int, default=200)

    predict_return = subparsers.add_parser("predict-alpha158-qlib-return", help="使用 Phase4 artifact 对指定日期全市场打分")
    predict_return.add_argument("--date", required=True)
    predict_return.add_argument("--limit", type=int, default=None)
    predict_return.add_argument("--top-n", type=int, default=20)
    predict_return.add_argument("--output", default=None)


def _add_phase5_parser(subparsers: argparse._SubParsersAction) -> None:
    mcd = subparsers.add_parser("validate-mcd-crash-risk", help="复现 Phase5 MCD stock price crash-risk 指标")
    mcd.add_argument("--start-date", default="2015-01-01")
    mcd.add_argument("--end-date", default="2026-05-07")
    mcd.add_argument("--limit", type=int, default=None)
    mcd.add_argument("--min-weeks-per-year", type=int, default=26)
    mcd.add_argument("--mcd-support-fraction", type=float, default=0.75)
    mcd.add_argument("--mcd-contamination", type=float, default=0.04)


def _add_phase7_parsers(subparsers: argparse._SubParsersAction) -> None:
    validate_trade_day = subparsers.add_parser("validate-trade-day-gate", help="验证 Phase7 交易日买点风险 gate")
    validate_trade_day.add_argument("--start-date", default="2015-01-01")
    validate_trade_day.add_argument("--end-date", default=date.today().isoformat())
    validate_trade_day.add_argument("--limit", type=int, default=None)
    validate_trade_day.add_argument("--min-stock-count", type=int, default=500)
    validate_trade_day.add_argument("--train-days", type=int, default=1000)
    validate_trade_day.add_argument("--valid-days", type=int, default=250)
    validate_trade_day.add_argument("--step-days", type=int, default=250)
    validate_trade_day.add_argument("--max-windows", type=int, default=None)
    validate_trade_day.add_argument("--horizon-days-grid", default="5,10")
    validate_trade_day.add_argument("--drawdown-threshold-grid", default="-0.02,-0.03,-0.05")
    validate_trade_day.add_argument("--return-threshold-grid", default="-0.01,-0.02,-0.03")
    validate_trade_day.add_argument("--models", default=",".join(DEFAULT_TRADE_DAY_MODEL_NAMES))
    validate_trade_day.add_argument("--filter-rates", default="0.2,0.3")
    validate_trade_day.add_argument("--min-training-rows", type=int, default=200)
    validate_trade_day.add_argument("--allow-short-sample", action="store_true")

    train_trade_day = subparsers.add_parser("train-trade-day-gate-model", help="训练 Phase7 交易日买点风险部署模型")
    train_trade_day.add_argument("--start-date", default="2015-01-01")
    train_trade_day.add_argument("--end-date", default=date.today().isoformat())
    train_trade_day.add_argument("--limit", type=int, default=None)
    train_trade_day.add_argument("--min-stock-count", type=int, default=500)
    train_trade_day.add_argument("--model-name", default="naive_bayes")
    train_trade_day.add_argument("--horizon-days", type=int, default=10)
    train_trade_day.add_argument("--drawdown-threshold", type=float, default=-0.02)
    train_trade_day.add_argument("--return-threshold", type=float, default=-0.01)
    train_trade_day.add_argument("--block-rate", type=float, default=0.2)
    train_trade_day.add_argument("--min-training-rows", type=int, default=200)

    predict_trade_day = subparsers.add_parser("predict-trade-day-gate", help="使用 Phase7 artifact 判断指定日期是否适合交易")
    predict_trade_day.add_argument("--date", required=True)
    predict_trade_day.add_argument("--limit", type=int, default=None)
    predict_trade_day.add_argument("--output", default=None)


def _add_daily_parsers(subparsers: argparse._SubParsersAction) -> None:
    daily = subparsers.add_parser("daily-screening", help="运行完整每日筛选主线")
    daily.add_argument("--date", default=None, help="目标日期，格式 YYYY-MM-DD，默认今天")
    daily.add_argument("--start-date", default="20240101", help="update 起始日期，格式 YYYYMMDD")

    track = subparsers.add_parser("track-stock", help="更新 track_stock.xlsx 的 Sheet2")
    track.add_argument("--date", default=None, help="目标日期，格式 YYYY-MM-DD，默认今天")
    track.add_argument("--workbook", default=DEFAULT_TRACK_STOCK_FILENAME, help="跟踪表路径")


def _add_backtest_parser(subparsers: argparse._SubParsersAction) -> None:
    backtest = subparsers.add_parser("backtest-daily-screening-components", help="小样本回测 daily-screening 组件消融")
    backtest.add_argument("--start-date", required=True, help="信号开始日期，格式 YYYY-MM-DD")
    backtest.add_argument("--end-date", required=True, help="信号结束日期，格式 YYYY-MM-DD")
    backtest.add_argument("--strategies", default=",".join(DEFAULT_BACKTEST_STRATEGIES), help="策略组合，逗号分隔")
    backtest.add_argument("--horizons", default="5,10,20,60", help="持有交易日窗口，逗号分隔")
    backtest.add_argument("--top-n", type=int, default=20, help="每个策略每天最多买入数量")
    backtest.add_argument("--phase4-top-n", type=int, default=20, help="当前 watchlist 逻辑中 Phase4 补入数量")
    backtest.add_argument("--phase1-filter-rate", type=float, default=0.2, help="Phase1 排除最高风险比例")
    backtest.add_argument("--phase2-filter-rate", type=float, default=0.2, help="Phase2 排除最高风险比例")
    backtest.add_argument("--stop-loss-pct", type=float, default=0.08, help="止损比例，例如 0.08")
    backtest.add_argument("--take-profit-pct", type=float, default=0.15, help="止盈比例，例如 0.15")
    backtest.add_argument("--max-signal-days", type=int, default=30, help="最多回测多少个信号日，取区间内最近 N 个")
    backtest.add_argument("--symbol-limit", type=int, default=500, help="小样本股票数量，默认取 universe 前 500 只")
    backtest.add_argument("--output-dir", default="reports/daily_screening_smoke_backtest", help="输出目录")
    backtest.add_argument("--no-cache", action="store_true", help="不复用 output-dir/cache 下的历史预测缓存")
    backtest.add_argument("--progress", action="store_true", help="显示日期级进度")


def _run_audit_full_market(storage: Storage, project_root: Path, args: argparse.Namespace) -> None:
    result = audit_full_market_data(
        storage=storage,
        project_root=project_root,
        limit=args.limit,
        min_exact_history_days=args.min_exact_history_days,
        tail_lookback_days=args.tail_lookback_days,
        max_horizon_days=args.max_horizon_days,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
    )
    print(format_full_market_audit_summary(result.summary))
    print(f"Saved detail: {result.detail_path}")
    print(f"Saved summary: {result.summary_path}")


def _run_build_synthetic_market(storage: Storage, project_root: Path, args: argparse.Namespace) -> None:
    result = build_synthetic_market_index(
        storage=storage,
        project_root=project_root,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
        min_stock_count=args.min_stock_count,
        output=Path(args.output).resolve() if args.output else None,
    )
    print("Synthetic market index complete.")
    if not result.frame.empty:
        print(result.frame.tail(min(10, len(result.frame))).to_string(index=False))
    if not result.skipped.empty:
        print(f"Skipped symbols: {len(result.skipped)}")
    print(f"Saved output: {result.output_path}")


def _run_reproduce_tail_risk(storage: Storage, project_root: Path, args: argparse.Namespace) -> None:
    result = reproduce_tail_risk(
        storage=storage,
        project_root=project_root,
        start_date=_parse_optional_date(args.start_date),
        end_date=_parse_optional_date(args.end_date),
        train_end=_parse_required_date(args.train_end),
        valid_end=_parse_required_date(args.valid_end),
        limit=args.limit,
        lookback_days=args.lookback_days,
        quantile=args.quantile,
        horizon_days=args.horizon_days,
        min_training_rows=args.min_training_rows,
        allow_short_sample=args.allow_short_sample,
        index_source_column=args.index_source_column,
        run_index=not args.skip_index,
        run_panel=not args.skip_panel,
        panel_model_names=_parse_tail_risk_panel_models(args.panel_models),
    )
    print("Tail-risk reproduction complete.")
    if not result.index_reproduction.empty:
        print(result.index_reproduction.to_string(index=False))
    if not result.metrics.empty:
        print(result.metrics.to_string(index=False))


def _run_validate_tail_risk(storage: Storage, project_root: Path, args: argparse.Namespace) -> None:
    result = validate_tail_risk_walkforward(
        storage=storage,
        project_root=project_root,
        start_date=_parse_optional_date(args.start_date),
        end_date=_parse_optional_date(args.end_date),
        limit=args.limit,
        train_days=args.train_days,
        valid_days=args.valid_days,
        step_days=args.step_days,
        embargo_days=args.embargo_days,
        max_windows=args.max_windows,
        lookback_days=args.lookback_days,
        quantile=args.quantile,
        horizon_days=args.horizon_days,
        min_training_rows=args.min_training_rows,
        allow_short_sample=args.allow_short_sample,
        panel_model_names=_parse_tail_risk_panel_models(args.panel_models),
        filter_rates=tuple(_parse_float_list(args.filter_rates)),
        return_tolerance=args.return_tolerance,
    )
    print("Tail-risk walk-forward validation complete.")
    print(result.summary.to_string(index=False))


def _run_train_tail_risk(storage: Storage, project_root: Path, args: argparse.Namespace) -> None:
    result = train_tail_risk_model(
        storage=storage,
        project_root=project_root,
        start_date=_parse_optional_date(args.start_date),
        end_date=_parse_optional_date(args.end_date),
        model_name=args.model_name,
        limit=args.limit,
        lookback_days=args.lookback_days,
        quantile=args.quantile,
        horizon_days=args.horizon_days,
        min_training_rows=args.min_training_rows,
    )
    print("Tail-risk deployment training complete.")
    print(f"Model: {result.model_name}")
    print(f"Rows: {result.train_rows}")
    print(f"Saved model: {result.model_path}")
    print(f"Saved metadata: {result.metadata_path}")


def _run_predict_tail_risk(storage: Storage, project_root: Path, args: argparse.Namespace) -> None:
    result = predict_tail_risk(
        storage=storage,
        project_root=project_root,
        trade_date=_parse_required_date(args.date),
        output=Path(args.output).resolve() if args.output else None,
        limit=args.limit,
    )
    print("Tail-risk prediction complete.")
    print(format_tail_risk_prediction_table(result.predictions, top_n=args.top_n))
    print(f"Saved predictions: {result.output_path}")


def _run_validate_barrier_risk(storage: Storage, project_root: Path, args: argparse.Namespace) -> None:
    result = validate_barrier_risk_walkforward(
        storage=storage,
        project_root=project_root,
        start_date=_parse_optional_date(args.start_date),
        end_date=_parse_optional_date(args.end_date),
        limit=args.limit,
        train_days=args.train_days,
        valid_days=args.valid_days,
        step_days=args.step_days,
        embargo_days=args.embargo_days,
        max_windows=args.max_windows,
        horizon_days=args.horizon_days,
        downside_atr_mult=args.downside_atr_mult,
        upside_atr_mult=None if args.no_upside_barrier else args.upside_atr_mult,
        downside_pct=args.downside_pct,
        upside_pct=args.upside_pct,
        label_variant=args.label_variant,
        label_method=args.label_method,
        volatility_lookback=args.volatility_lookback,
        pt_mult=args.pt_mult,
        sl_mult=args.sl_mult,
        min_ret=args.min_ret,
        cusum_threshold=args.cusum_threshold,
        cusum_threshold_mult=args.cusum_threshold_mult,
        min_training_rows=args.min_training_rows,
        allow_short_sample=args.allow_short_sample,
        model_names=tuple(_parse_str_list(args.models)),
        filter_rates=tuple(_parse_float_list(args.filter_rates)),
        return_tolerance=args.return_tolerance,
    )
    print("Barrier-risk walk-forward validation complete.")
    print(summarize_tail_risk_walkforward(result.metrics, result.deciles).to_string(index=False))


def _run_validate_barrier_grid(storage: Storage, project_root: Path, args: argparse.Namespace) -> None:
    result = validate_barrier_risk_grid(
        storage=storage,
        project_root=project_root,
        start_date=_parse_optional_date(args.start_date),
        end_date=_parse_optional_date(args.end_date),
        limit=args.limit,
        train_days=args.train_days,
        valid_days=args.valid_days,
        step_days=args.step_days,
        max_windows=args.max_windows,
        horizon_days_grid=tuple(_parse_int_list(args.horizon_days_grid)),
        pt_sl_grid=tuple(_parse_pt_sl_grid(args.pt_sl_grid)),
        min_ret_grid=tuple(_parse_float_list(args.min_ret_grid)),
        volatility_lookback=args.volatility_lookback,
        model_names=tuple(_parse_str_list(args.models)),
        filter_rates=tuple(_parse_float_list(args.filter_rates)),
        return_tolerance=args.return_tolerance,
        allow_short_sample=args.allow_short_sample,
        min_training_rows=args.min_training_rows,
    )
    print("Barrier-risk parameter grid validation complete.")
    print(result.summary.to_string(index=False))


def _run_train_barrier_risk(storage: Storage, project_root: Path, args: argparse.Namespace) -> None:
    result = train_barrier_risk_model(
        storage=storage,
        project_root=project_root,
        start_date=_parse_optional_date(args.start_date),
        end_date=_parse_optional_date(args.end_date),
        model_name=args.model_name,
        limit=args.limit,
        horizon_days=args.horizon_days,
        volatility_lookback=args.volatility_lookback,
        pt_mult=args.pt_mult,
        sl_mult=args.sl_mult,
        min_ret=args.min_ret,
        min_training_rows=args.min_training_rows,
    )
    print("Barrier-risk deployment training complete.")
    print(f"Model: {result.model_name}")
    print(f"Rows: {result.train_rows}")
    print(f"Saved model: {result.model_path}")
    print(f"Saved metadata: {result.metadata_path}")


def _run_predict_barrier_risk(storage: Storage, project_root: Path, args: argparse.Namespace) -> None:
    result = predict_barrier_risk(
        storage=storage,
        project_root=project_root,
        trade_date=_parse_required_date(args.date),
        output=Path(args.output).resolve() if args.output else None,
        limit=args.limit,
    )
    print("Barrier-risk prediction complete.")
    print(format_barrier_risk_prediction_table(result.predictions, top_n=args.top_n))
    print(f"Saved predictions: {result.output_path}")


def _run_validate_alpha158_return(storage: Storage, project_root: Path, args: argparse.Namespace) -> None:
    result = validate_alpha158_qlib_return(
        storage=storage,
        project_root=project_root,
        start_date=_parse_optional_date(args.start_date),
        end_date=_parse_optional_date(args.end_date),
        train_end=_parse_required_date(args.train_end),
        valid_end=_parse_required_date(args.valid_end),
        limit=args.limit,
        topk=args.topk,
        n_drop=args.drop,
        min_training_rows=args.min_training_rows,
    )
    print("Alpha158 Qlib return validation complete.")
    print(result.signal_metrics.to_string(index=False))
    print(result.topk_summary.to_string(index=False))


def _run_train_alpha158_return(storage: Storage, project_root: Path, args: argparse.Namespace) -> None:
    result = train_alpha158_qlib_return_model(
        storage=storage,
        project_root=project_root,
        start_date=_parse_optional_date(args.start_date),
        end_date=_parse_optional_date(args.end_date),
        limit=args.limit,
        min_training_rows=args.min_training_rows,
    )
    print("Alpha158 Qlib return deployment training complete.")
    print(f"Rows: {result.train_rows}")
    print(f"Saved model: {result.model_path}")
    print(f"Saved metadata: {result.metadata_path}")


def _run_predict_alpha158_return(storage: Storage, project_root: Path, args: argparse.Namespace) -> None:
    result = predict_alpha158_qlib_return(
        storage=storage,
        project_root=project_root,
        trade_date=_parse_required_date(args.date),
        output=Path(args.output).resolve() if args.output else None,
        limit=args.limit,
    )
    print("Alpha158 Qlib return prediction complete.")
    print(format_alpha158_qlib_return_prediction_table(result.predictions, top_n=args.top_n))
    print(f"Saved predictions: {result.output_path}")


def _run_validate_mcd_crash(storage: Storage, project_root: Path, args: argparse.Namespace) -> None:
    result = validate_mcd_crash_risk(
        storage=storage,
        project_root=project_root,
        start_date=_parse_optional_date(args.start_date),
        end_date=_parse_optional_date(args.end_date),
        limit=args.limit,
        min_weeks_per_year=args.min_weeks_per_year,
        mcd_support_fraction=args.mcd_support_fraction,
        mcd_contamination=args.mcd_contamination,
    )
    print("MCD crash-risk label reproduction complete.")
    print(result.distribution.tail(12).to_string(index=False))
    print(f"Saved annual measures: {result.annual_measures_path}")


def _run_validate_trade_day_gate(storage: Storage, project_root: Path, args: argparse.Namespace) -> None:
    result = validate_trade_day_gate(
        storage=storage,
        project_root=project_root,
        start_date=_parse_optional_date(args.start_date),
        end_date=_parse_optional_date(args.end_date),
        limit=args.limit,
        min_stock_count=args.min_stock_count,
        train_days=args.train_days,
        valid_days=args.valid_days,
        step_days=args.step_days,
        max_windows=args.max_windows,
        horizon_days_grid=tuple(_parse_int_list(args.horizon_days_grid)),
        drawdown_threshold_grid=tuple(_parse_float_list(args.drawdown_threshold_grid)),
        return_threshold_grid=tuple(_parse_float_list(args.return_threshold_grid)),
        model_names=tuple(_parse_str_list(args.models)),
        filter_rates=tuple(_parse_float_list(args.filter_rates)),
        min_training_rows=args.min_training_rows,
        allow_short_sample=args.allow_short_sample,
    )
    print("Trade-day gate validation complete.")
    print(result.summary.to_string(index=False) if not result.summary.empty else "No summary rows.")


def _run_train_trade_day_gate(storage: Storage, project_root: Path, args: argparse.Namespace) -> None:
    result = train_trade_day_gate_model(
        storage=storage,
        project_root=project_root,
        start_date=_parse_optional_date(args.start_date),
        end_date=_parse_optional_date(args.end_date),
        limit=args.limit,
        min_stock_count=args.min_stock_count,
        model_name=args.model_name,
        horizon_days=args.horizon_days,
        drawdown_threshold=args.drawdown_threshold,
        return_threshold=args.return_threshold,
        block_rate=args.block_rate,
        min_training_rows=args.min_training_rows,
    )
    print("Trade-day gate deployment training complete.")
    print(f"Model: {result.model_name}")
    print(f"Rows: {result.train_rows}")
    print(f"Selected threshold: {result.selected_threshold:.6f}")
    print(f"Saved model: {result.model_path}")
    print(f"Saved metadata: {result.metadata_path}")


def _run_predict_trade_day_gate(storage: Storage, project_root: Path, args: argparse.Namespace) -> None:
    result = predict_trade_day_gate(
        storage=storage,
        project_root=project_root,
        trade_date=_parse_required_date(args.date),
        output=Path(args.output).resolve() if args.output else None,
        limit=args.limit,
    )
    print("Trade-day gate prediction complete.")
    print(format_trade_day_gate_prediction_table(result.prediction))
    print(f"Saved prediction: {result.output_path}")


def _run_backtest_daily_screening_components(
    storage: Storage,
    project_root: Path,
    config,
    args: argparse.Namespace,
) -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    result = backtest_daily_screening_components(
        storage=storage,
        project_root=project_root,
        config=config,
        start_date=_parse_required_date(args.start_date),
        end_date=_parse_required_date(args.end_date),
        strategies=tuple(_parse_str_list(args.strategies)),
        horizons=tuple(_parse_int_list(args.horizons)),
        top_n=args.top_n,
        phase1_filter_rate=args.phase1_filter_rate,
        phase2_filter_rate=args.phase2_filter_rate,
        phase4_top_n=args.phase4_top_n,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        max_signal_days=args.max_signal_days,
        symbol_limit=args.symbol_limit,
        output_dir=output_dir,
        progress=args.progress,
        use_cache=not args.no_cache,
    )
    print("Daily-screening component backtest complete.")
    print(format_backtest_summary(result.summary))
    print(f"Saved trades: {result.trades_path}")
    print(f"Saved daily portfolio: {result.daily_portfolio_path}")
    print(f"Saved summary: {result.summary_path}")
    print(f"Saved comparison: {result.comparison_path}")


def _run_update(
    storage: Storage,
    provider,
    exclude_st: bool,
    adjust: str,
    symbol: str | None,
    start_date: str,
    end_date: str,
    limit: int | None,
    *,
    index_symbols: tuple[str, ...] = DEFAULT_INDEX_SYMBOLS,
    update_indexes: bool = False,
    index_interface: str = "sina",
) -> None:
    if symbol:
        _update_daily_cache_for_symbol(
            storage=storage,
            provider=provider,
            symbol=str(symbol).zfill(6),
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        return

    universe = _refresh_or_load_universe(storage, provider, exclude_st)
    symbols = universe["symbol"].tolist()
    if limit is not None:
        symbols = symbols[:limit]

    success_count = 0
    failed_symbols: list[str] = []
    total_symbols = len(symbols)
    for index, item_symbol in enumerate(symbols, start=1):
        try:
            _update_daily_cache_for_symbol(
                storage=storage,
                provider=provider,
                symbol=item_symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
            success_count += 1
        except Exception as exc:
            failed_symbols.append(item_symbol)
            logging.warning("[%s/%s] failed to fetch %s: %s", index, len(symbols), item_symbol, exc)
        _log_scan_progress("Update", index, total_symbols)

    logging.info("Daily update finished: success=%s failed=%s", success_count, len(failed_symbols))
    if failed_symbols:
        logging.warning("Failed symbols sample: %s", ", ".join(failed_symbols[:20]))

    if update_indexes:
        index_provider = _create_update_data_provider(index_interface)
        try:
            _run_update_indexes(storage=storage, provider=index_provider, index_symbols=index_symbols, start_date=start_date, end_date=end_date)
        finally:
            index_provider.close()


def _update_daily_cache_for_symbol(
    *,
    storage: Storage,
    provider,
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
) -> Path:
    try:
        cached = storage.load_daily_bars(symbol)
    except (FileNotFoundError, DailyBarsReadError):
        fresh = provider.get_daily_bars(symbol, start_date=start_date, end_date=end_date, adjust=adjust)
        target = storage.save_daily_bars(symbol, fresh)
        logging.info("Initialized %s rows for %s to %s", len(fresh), symbol, target)
        return target

    cached_frame = cached.copy()
    cached_frame["trade_date"] = pd.to_datetime(cached_frame["trade_date"], errors="coerce")
    valid_dates = cached_frame["trade_date"].dropna()
    if cached_frame.empty or valid_dates.empty:
        fresh = provider.get_daily_bars(symbol, start_date=start_date, end_date=end_date, adjust=adjust)
        target = storage.save_daily_bars(symbol, fresh)
        logging.info("Rebuilt %s rows for %s to %s", len(fresh), symbol, target)
        return target

    requested_start_date = datetime.strptime(start_date, "%Y%m%d").date()
    requested_end_date = datetime.strptime(end_date, "%Y%m%d").date()
    missing_ranges = _missing_cache_ranges(
        requested_start_date=requested_start_date,
        requested_end_date=requested_end_date,
        cached_first_date=valid_dates.min().date(),
        cached_last_date=valid_dates.max().date(),
    )
    target = storage.paths.daily_dir / f"{symbol}.parquet"
    if not missing_ranges:
        logging.info("Skip %s because cached daily bars already cover requested range", symbol)
        return target

    fresh_parts = []
    for range_start, range_end in missing_ranges:
        fresh = provider.get_daily_bars(
            symbol,
            start_date=range_start.strftime("%Y%m%d"),
            end_date=range_end.strftime("%Y%m%d"),
            adjust=adjust,
        )
        if not fresh.empty:
            fresh_parts.append(fresh)
        else:
            logging.info("No daily bars returned for %s from %s to %s", symbol, range_start.isoformat(), range_end.isoformat())
    if not fresh_parts:
        return target

    merged = _merge_daily_cache_frames(cached_frame, fresh_parts)
    target = storage.save_daily_bars(symbol, merged)
    logging.info("Merged %s fetched rows for %s into %s", sum(len(frame) for frame in fresh_parts), symbol, target)
    return target


def _run_update_indexes(
    *,
    storage: Storage,
    provider,
    index_symbols: tuple[str, ...],
    start_date: str,
    end_date: str,
) -> None:
    symbols = tuple(dict.fromkeys(str(symbol).strip() for symbol in index_symbols if str(symbol).strip()))
    for index, index_symbol in enumerate(symbols, start=1):
        try:
            _update_index_daily_cache(storage=storage, provider=provider, index_symbol=index_symbol, start_date=start_date, end_date=end_date)
        except Exception as exc:
            logging.warning("[%s/%s] failed to fetch index %s: %s", index, len(symbols), index_symbol, exc)
        _log_scan_progress("Index update", index, len(symbols))


def _update_index_daily_cache(*, storage: Storage, provider, index_symbol: str, start_date: str, end_date: str) -> Path:
    try:
        cached = storage.load_index_daily_bars(index_symbol)
    except (FileNotFoundError, DailyBarsReadError):
        fresh = provider.get_index_daily_bars(index_symbol, start_date=start_date, end_date=end_date)
        target = storage.save_index_daily_bars(index_symbol, fresh)
        logging.info("Initialized %s rows for index %s to %s", len(fresh), index_symbol, target)
        return target

    cached_frame = cached.copy()
    cached_frame["trade_date"] = pd.to_datetime(cached_frame["trade_date"], errors="coerce")
    valid_dates = cached_frame["trade_date"].dropna()
    if cached_frame.empty or valid_dates.empty:
        fresh = provider.get_index_daily_bars(index_symbol, start_date=start_date, end_date=end_date)
        target = storage.save_index_daily_bars(index_symbol, fresh)
        logging.info("Rebuilt %s rows for index %s to %s", len(fresh), index_symbol, target)
        return target

    requested_start_date = datetime.strptime(start_date, "%Y%m%d").date()
    requested_end_date = datetime.strptime(end_date, "%Y%m%d").date()
    missing_ranges = _missing_cache_ranges(
        requested_start_date=requested_start_date,
        requested_end_date=requested_end_date,
        cached_first_date=valid_dates.min().date(),
        cached_last_date=valid_dates.max().date(),
    )
    normalized = _normalize_index_symbol_for_update(index_symbol)
    target = storage.paths.index_daily_dir / f"{normalized}.parquet"
    if not missing_ranges:
        logging.info("Skip index %s because cached daily bars already cover requested range", normalized)
        return target

    fresh_parts = []
    for range_start, range_end in missing_ranges:
        fresh = provider.get_index_daily_bars(
            index_symbol,
            start_date=range_start.strftime("%Y%m%d"),
            end_date=range_end.strftime("%Y%m%d"),
        )
        if not fresh.empty:
            fresh_parts.append(fresh)
    if not fresh_parts:
        return target
    target = storage.save_index_daily_bars(index_symbol, _merge_daily_cache_frames(cached_frame, fresh_parts))
    logging.info("Merged %s fetched rows for index %s into %s", sum(len(frame) for frame in fresh_parts), normalized, target)
    return target


def _run_pattern(
    storage: Storage,
    provider_name: str,
    config,
    as_of: date,
    selected_patterns: list[str],
    limit: int | None,
    output: str | None,
    symbols: list[str] | None = None,
) -> None:
    _ensure_universe(storage, provider_name, config.universe.exclude_st)
    screener = Screener(storage, config)
    results = screener.run(
        as_of=as_of,
        selected_strategies=selected_patterns,
        symbols=symbols,
        progress_callback=lambda current, total: _log_scan_progress("Pattern", current, total),
    )
    exported = _prepare_pattern_results(results)
    exported = _append_recent_macd_summary(storage, exported, as_of=as_of, symbols=symbols)
    exported = _append_recent_atr_summary(storage, exported, as_of=as_of, symbols=symbols)
    exported = append_daily_phase_display_columns(exported, project_root=storage.paths.root, trade_date=as_of)
    for column in reversed(PHASE_DISPLAY_COLUMNS):
        exported = _move_column_after(exported, column, "风险情况")

    output_path = Path(output) if output else _default_pattern_output_path(storage, as_of=as_of, selected_patterns=selected_patterns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(output_path, index=False, encoding="utf-8-sig")

    pattern_watchlist_payload = build_watchlist_candidates_from_patterns(exported, source_file=str(output_path), limit=None)
    write_watchlist(project_root=storage.paths.root, trade_date=as_of, picker_payload=pattern_watchlist_payload)
    pattern_watchlist_target = write_watchlist(
        project_root=storage.paths.root,
        trade_date=as_of,
        picker_payload=pattern_watchlist_payload,
        kind="pattern",
    )
    logging.info("Saved pattern watchlist to %s", pattern_watchlist_target)

    if exported.empty:
        print(f"No patterns matched. Saved empty CSV to {output_path}")
        print(f"Saved pattern watchlist to {pattern_watchlist_target}")
        return
    multi_pattern_summary = format_multi_pattern_summary(exported)
    if multi_pattern_summary:
        print(multi_pattern_summary)
        print()
    print(format_report(exported, limit=limit or config.screening.output_limit))
    print(f"\nSaved pattern report to {output_path}")
    print(f"Saved pattern watchlist to {pattern_watchlist_target}")


def _run_report(storage: Storage, config, trade_date: date, limit: int | None) -> None:
    report_path = _default_pattern_output_path(storage, as_of=trade_date, selected_patterns=STRATEGY_NAMES)
    if report_path.exists():
        print(format_report(pd.read_csv(report_path), limit=limit or config.screening.output_limit))
        return
    signals_path = storage.paths.signals_dir / f"signals_{trade_date.isoformat()}.parquet"
    if signals_path.exists():
        print(format_report(_prepare_pattern_results(storage.load_signals(trade_date)), limit=limit or config.screening.output_limit))
        return
    raise FileNotFoundError(f"Pattern report not found for {trade_date.isoformat()}: {report_path}")


def _run_macd(storage: Storage, config, paths: ProjectPaths, trade_date: date, top_n: int, output: str | None) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    summary = _load_or_build_macd_summary(storage, trade_date=trade_date)
    if summary.empty:
        raise RuntimeError(f"No MACD summary could be generated for {trade_date.isoformat()}")
    report_paths = save_macd_report(paths, trade_date=trade_date, dataframe=summary, output=output)
    display_columns = [
        "symbol",
        "name",
        "macd_cross_state",
        "macd_divergence_state",
        "volume_price_divergence_state",
        "macd_top_divergence_15d",
        "macd_bottom_divergence_15d",
    ]
    available = [column for column in display_columns if column in summary.columns]
    print(summary.loc[:, available].head(top_n).to_string(index=False))
    print(f"\nMACD 状态文件：{report_paths['detail_path']}")


def _run_atr(storage: Storage, config, paths: ProjectPaths, trade_date: date, top_n: int, output: str | None) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    summary = _load_or_build_atr_summary(storage, trade_date=trade_date)
    if summary.empty:
        raise RuntimeError(f"No ATR summary could be generated for {trade_date.isoformat()}")
    report_paths = save_atr_report(paths, trade_date=trade_date, dataframe=summary, output=output)
    print(build_atr_export_frame(summary).head(top_n).to_string(index=False))
    print(f"\nATR 风险辅助文件：{report_paths['detail_path']}")


def _prepare_pattern_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["trade_date", "symbol", "name", "pattern_id", "风险情况", "close", "reason"])
    exported = results.copy()
    exported["symbol"] = exported["symbol"].map(_format_symbol_for_excel)
    exported["pattern_id"] = exported["strategy_name"].map(PATTERN_LABEL_MAP)
    if "风险情况" not in exported.columns:
        exported["风险情况"] = "模型风险待合并"
    exported = exported.drop(columns=["strategy_name"], errors="ignore")
    dedupe_keys = [column for column in ("trade_date", "symbol", "pattern_id") if column in exported.columns]
    if dedupe_keys:
        exported = exported.drop_duplicates(subset=dedupe_keys, keep="first")
    preferred = ["trade_date", "symbol", "name", "pattern_id", "风险情况", "close", "reason"]
    available = [column for column in preferred if column in exported.columns]
    remaining = [column for column in exported.columns if column not in available]
    return exported.loc[:, available + remaining].sort_values(["pattern_id", "symbol"]).reset_index(drop=True)


def _append_recent_macd_summary(storage: Storage, exported: pd.DataFrame, *, as_of: date, symbols: list[str] | None = None) -> pd.DataFrame:
    if exported.empty or "symbol" not in exported.columns:
        return exported
    summary = _load_or_build_macd_summary(storage, trade_date=as_of, symbols=symbols)
    if summary.empty:
        return exported
    merge_columns = [
        "symbol",
        "macd_cross_state",
        "macd_divergence_state",
        "volume_price_divergence_state",
        "macd_top_divergence_15d",
        "macd_bottom_divergence_15d",
        "macd_top_divergence_signal_date",
        "macd_bottom_divergence_signal_date",
        "bullish_volume_price_divergence_flag",
        "bearish_volume_price_divergence_flag",
        "macd",
        "macd_signal_line",
        "macd_hist",
    ]
    return _merge_symbol_report(
        exported,
        summary,
        merge_columns=merge_columns,
        bool_columns=[
            "macd_top_divergence_15d",
            "macd_bottom_divergence_15d",
            "bullish_volume_price_divergence_flag",
            "bearish_volume_price_divergence_flag",
        ],
    )


def _append_recent_atr_summary(storage: Storage, exported: pd.DataFrame, *, as_of: date, symbols: list[str] | None = None) -> pd.DataFrame:
    if exported.empty or "symbol" not in exported.columns:
        return exported
    summary = _load_or_build_atr_summary(storage, trade_date=as_of, symbols=symbols)
    if summary.empty:
        return exported
    merge_columns = [
        "symbol",
        "atr_14",
        "atr_pct_14",
        "atr_stop_loss_1x",
        "atr_stop_loss_2x",
        "atr_take_profit_2x",
        "atr_take_profit_3x",
        "atr_volatility_regime",
    ]
    return _merge_symbol_report(exported, summary, merge_columns=merge_columns, bool_columns=[])


def _merge_symbol_report(exported: pd.DataFrame, report: pd.DataFrame, *, merge_columns: list[str], bool_columns: list[str]) -> pd.DataFrame:
    columns = [column for column in merge_columns if column in report.columns]
    if "symbol" not in columns:
        return exported
    summary = _dedupe_symbol_report_rows(report.loc[:, columns].copy())
    enriched = exported.copy()
    enriched["_normalized_symbol"] = enriched["symbol"].map(_normalize_exported_symbol)
    enriched = enriched.merge(summary, how="left", left_on="_normalized_symbol", right_on="symbol")
    enriched = enriched.drop(columns=["_normalized_symbol", "symbol_y"], errors="ignore")
    enriched = enriched.rename(columns={"symbol_x": "symbol"})
    for column in bool_columns:
        if column in enriched.columns:
            enriched[column] = enriched[column].fillna(False).astype(bool)
    return enriched


def _dedupe_symbol_report_rows(report: pd.DataFrame) -> pd.DataFrame:
    if report.empty or "symbol" not in report.columns:
        return report
    deduped = report.copy()
    deduped["symbol"] = deduped["symbol"].map(_normalize_exported_symbol)
    return deduped.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)


def _load_or_build_macd_summary(storage: Storage, trade_date: date, symbols: list[str] | None = None) -> pd.DataFrame:
    if not symbols:
        default_path = storage.paths.reports_dir / "macd" / f"macd_{trade_date.isoformat()}.csv"
        if default_path.exists():
            return pd.read_csv(default_path)
    return _build_macd_summary(storage, trade_date=trade_date, symbols=symbols)


def _load_or_build_atr_summary(storage: Storage, trade_date: date, symbols: list[str] | None = None) -> pd.DataFrame:
    if not symbols:
        default_path = storage.paths.reports_dir / "atr" / f"atr_{trade_date.isoformat()}.csv"
        if default_path.exists():
            return normalize_atr_summary_frame(pd.read_csv(default_path))
    return _build_atr_summary(storage, trade_date=trade_date, symbols=symbols)


def _build_macd_summary(storage: Storage, trade_date: date, symbols: list[str] | None = None) -> pd.DataFrame:
    instruments = _load_instruments(storage, symbols=symbols)
    rows: list[dict[str, object]] = []
    logging.info("MACD summary scan started for %s: %s symbols", trade_date.isoformat(), len(instruments))
    for index, instrument in enumerate(instruments, start=1):
        symbol = str(instrument["symbol"]).zfill(6)
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError) as exc:
            if isinstance(exc, DailyBarsReadError):
                logging.warning("Skip MACD for %s because cached daily bars are unreadable: %s", symbol, exc)
            _log_scan_progress("MACD", index, len(instruments))
            continue
        cutoff = bars[pd.to_datetime(bars["trade_date"]).dt.date <= trade_date].reset_index(drop=True)
        if cutoff.empty:
            _log_scan_progress("MACD", index, len(instruments))
            continue
        macd_frame = _prepare_daily_macd_frame(cutoff)
        latest = macd_frame.iloc[-1]
        divergence_row = summarize_recent_macd_divergence(macd_frame)
        bullish_volume_divergence, bearish_volume_divergence = _detect_daily_volume_price_divergence(macd_frame)
        rows.append(
            {
                "trade_date": trade_date.isoformat(),
                "symbol": _format_symbol_for_excel(symbol),
                "name": instrument.get("name", ""),
                "macd": _safe_float_or_none(latest.get("macd")),
                "macd_signal_line": _safe_float_or_none(latest.get("macd_signal_line")),
                "macd_hist": _safe_float_or_none(latest.get("macd_hist")),
                "macd_cross_state": _describe_macd_cross_state(macd_frame),
                "macd_divergence_state": _describe_macd_divergence_state(divergence_row),
                "volume_price_divergence_state": _describe_volume_price_divergence_state(bullish_volume_divergence, bearish_volume_divergence),
                "macd_top_divergence_15d": bool(divergence_row.get("macd_top_divergence_15d", False)),
                "macd_bottom_divergence_15d": bool(divergence_row.get("macd_bottom_divergence_15d", False)),
                "macd_top_divergence_signal_date": divergence_row.get("macd_top_divergence_signal_date"),
                "macd_bottom_divergence_signal_date": divergence_row.get("macd_bottom_divergence_signal_date"),
                "bullish_volume_price_divergence_flag": bool(bullish_volume_divergence),
                "bearish_volume_price_divergence_flag": bool(bearish_volume_divergence),
            }
        )
        _log_scan_progress("MACD", index, len(instruments))
    return pd.DataFrame(rows)


def _build_atr_summary(storage: Storage, trade_date: date, symbols: list[str] | None = None) -> pd.DataFrame:
    instruments = _load_instruments(storage, symbols=symbols)
    rows: list[dict[str, object]] = []
    logging.info("ATR summary scan started for %s: %s symbols", trade_date.isoformat(), len(instruments))
    for index, instrument in enumerate(instruments, start=1):
        symbol = str(instrument["symbol"]).zfill(6)
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError) as exc:
            if isinstance(exc, DailyBarsReadError):
                logging.warning("Skip ATR for %s because cached daily bars are unreadable: %s", symbol, exc)
            _log_scan_progress("ATR", index, len(instruments))
            continue
        cutoff = bars[pd.to_datetime(bars["trade_date"]).dt.date <= trade_date].reset_index(drop=True)
        if cutoff.empty:
            _log_scan_progress("ATR", index, len(instruments))
            continue
        snapshot = build_atr_snapshot_row(cutoff, symbol=_format_symbol_for_excel(symbol), name=str(instrument.get("name", "")), trade_date=trade_date)
        if snapshot is not None:
            rows.append(snapshot)
        _log_scan_progress("ATR", index, len(instruments))
    return pd.DataFrame(rows)


def _prepare_daily_macd_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy().sort_values("trade_date").reset_index(drop=True)
    if not {"macd_dif", "macd_dea", "macd_hist"}.issubset(frame.columns):
        frame = add_indicators(frame).sort_values("trade_date").reset_index(drop=True)
    if "macd" not in frame.columns and "macd_dif" in frame.columns:
        frame["macd"] = frame["macd_dif"]
    if "macd_signal_line" not in frame.columns and "macd_dea" in frame.columns:
        frame["macd_signal_line"] = frame["macd_dea"]
    return frame


def _describe_macd_cross_state(dataframe: pd.DataFrame) -> str:
    normalized = _prepare_daily_macd_frame(dataframe)
    if normalized.empty or "macd" not in normalized.columns or "macd_signal_line" not in normalized.columns:
        return "unknown"
    recent = normalized.tail(3).reset_index(drop=True)
    recent_cross_up = False
    recent_cross_down = False
    for offset in range(1, len(recent)):
        prev_row = recent.iloc[offset - 1]
        current_row = recent.iloc[offset]
        if pd.isna(prev_row.get("macd")) or pd.isna(prev_row.get("macd_signal_line")):
            continue
        if pd.isna(current_row.get("macd")) or pd.isna(current_row.get("macd_signal_line")):
            continue
        if float(prev_row["macd"]) <= float(prev_row["macd_signal_line"]) and float(current_row["macd"]) > float(current_row["macd_signal_line"]):
            recent_cross_up = True
        if float(prev_row["macd"]) >= float(prev_row["macd_signal_line"]) and float(current_row["macd"]) < float(current_row["macd_signal_line"]):
            recent_cross_down = True
    latest = recent.iloc[-1]
    macd = _safe_float_or_none(latest.get("macd"))
    signal_line = _safe_float_or_none(latest.get("macd_signal_line"))
    if recent_cross_up:
        return "golden_cross"
    if recent_cross_down:
        return "dead_cross"
    if macd is None or signal_line is None:
        return "unknown"
    return "above_signal" if macd >= signal_line else "below_signal"


def _detect_daily_volume_price_divergence(dataframe: pd.DataFrame) -> tuple[bool, bool]:
    if dataframe.empty or len(dataframe) < 6:
        return False, False
    frame = dataframe.copy().sort_values("trade_date").reset_index(drop=True)
    recent = frame.tail(5).reset_index(drop=True)
    previous = frame.iloc[-6]
    latest = recent.iloc[-1]
    recent_avg_volume = pd.to_numeric(recent["volume"], errors="coerce").mean()
    previous_volume = _safe_float_or_none(previous.get("volume"))
    latest_close = _safe_float_or_none(latest.get("close"))
    previous_close = _safe_float_or_none(previous.get("close"))
    if previous_volume is None or latest_close is None or previous_close is None or pd.isna(recent_avg_volume):
        return False, False
    bullish = latest_close > previous_close and float(recent_avg_volume) < previous_volume * 0.9
    bearish = latest_close < previous_close and float(recent_avg_volume) > previous_volume * 1.1
    return bullish, bearish


def _describe_macd_divergence_state(macd_summary: dict[str, object]) -> str:
    if bool(macd_summary.get("macd_bottom_divergence_15d", False)):
        return "bottom_divergence"
    if bool(macd_summary.get("macd_top_divergence_15d", False)):
        return "top_divergence"
    return "none"


def _describe_volume_price_divergence_state(bullish: bool, bearish: bool) -> str:
    if bullish and not bearish:
        return "bullish"
    if bearish and not bullish:
        return "bearish"
    return "none"


def _safe_float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _load_instruments(storage: Storage, symbols: list[str] | None = None) -> list[dict[str, object]]:
    universe = storage.load_universe().copy()
    universe["symbol"] = universe["symbol"].astype(str).str.zfill(6)
    if symbols:
        symbol_set = {str(symbol).zfill(6) for symbol in symbols}
        universe = universe[universe["symbol"].isin(symbol_set)].reset_index(drop=True)
    return universe.to_dict("records")


def _selected_patterns(args: argparse.Namespace) -> list[str]:
    selected = [pattern for field, pattern in PATTERN_FLAG_MAP.items() if getattr(args, field)]
    return selected or list(STRATEGY_NAMES)


def _move_column_after(frame: pd.DataFrame, column: str, after: str) -> pd.DataFrame:
    if column not in frame.columns or after not in frame.columns:
        return frame
    columns = [item for item in frame.columns if item != column]
    index = columns.index(after) + 1
    columns.insert(index, column)
    return frame.loc[:, columns].copy()


def _default_pattern_filename(as_of: date, selected_patterns: list[str]) -> str:
    if len(selected_patterns) == len(STRATEGY_NAMES):
        label = "all"
    else:
        label = "-".join(PATTERN_LABEL_MAP[item] for item in selected_patterns)
    return f"patterns_{label}_{as_of.isoformat()}.csv"


def _default_pattern_output_path(storage: Storage, *, as_of: date, selected_patterns: list[str]) -> Path:
    return storage.paths.reports_dir / "patterns" / _default_pattern_filename(as_of, selected_patterns)


def _format_symbol_for_excel(value: object) -> str:
    return f'="{str(value).zfill(6)}"'


def _normalize_exported_symbol(value: object) -> str:
    text = str(value).strip()
    if text.startswith('="') and text.endswith('"'):
        text = text[2:-1]
    return text.zfill(6)


def _missing_cache_ranges(
    *,
    requested_start_date: date,
    requested_end_date: date,
    cached_first_date: date,
    cached_last_date: date,
) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    if requested_start_date < cached_first_date:
        ranges.append((requested_start_date, cached_first_date - timedelta(days=1)))
    if requested_end_date > cached_last_date:
        ranges.append((cached_last_date + timedelta(days=1), requested_end_date))
    return [(start, end) for start, end in ranges if start <= end]


def _merge_daily_cache_frames(cached_frame: pd.DataFrame, fresh_parts: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat([cached_frame, *fresh_parts], ignore_index=True)
    merged["trade_date"] = pd.to_datetime(merged["trade_date"], errors="coerce")
    merged = merged.dropna(subset=["trade_date"]).drop_duplicates(subset=["trade_date"], keep="last")
    return merged.sort_values("trade_date").reset_index(drop=True)


def _normalize_index_symbol_for_update(index_symbol: str) -> str:
    text = str(index_symbol).strip().lower().replace(".", "")
    if text.startswith(("sh", "sz")):
        return f"{text[:2]}{text[2:].zfill(6)}"
    code = text.zfill(6)
    return f"{'sz' if code.startswith('399') else 'sh'}{code}"


def _parse_optional_date(value: str | None) -> date | None:
    if value is None:
        return None
    return datetime.fromisoformat(value).date()


def _parse_required_date(value: str) -> date:
    return datetime.fromisoformat(value).date()


def _parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer value")
    return [int(item) for item in items]


def _parse_float_list(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one float value")
    return [float(item) for item in items]


def _parse_pt_sl_grid(value: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for item in str(value).split(","):
        text = item.strip()
        if not text:
            continue
        if ":" in text:
            pt_text, sl_text = text.split(":", 1)
        elif "/" in text:
            pt_text, sl_text = text.split("/", 1)
        else:
            pt_text = sl_text = text
        pairs.append((float(pt_text.strip()), float(sl_text.strip())))
    if not pairs:
        raise ValueError("Expected at least one pt:sl pair")
    return pairs


def _parse_str_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one text value")
    return items


def _parse_tail_risk_panel_models(value: str) -> tuple[str, ...]:
    if str(value).strip().lower() == "all":
        from .full_market_risk import ALL_TAIL_RISK_MODEL_NAMES

        return ALL_TAIL_RISK_MODEL_NAMES
    return tuple(_parse_str_list(value))


def _proxy_env_value(name: str) -> str | None:
    return os.environ.get(name) or os.environ.get(name.lower())


def _set_proxy_env(name: str, value: str) -> None:
    os.environ[name] = value
    os.environ[name.lower()] = value


def _is_local_proxy_url(proxy_url: str) -> bool:
    parsed = urlparse(proxy_url)
    return (parsed.hostname or "").lower() in LOCAL_PROXY_HOSTS


def _is_proxy_reachable(proxy_url: str, timeout: float = 0.3) -> bool:
    parsed = urlparse(proxy_url)
    host = parsed.hostname
    port = parsed.port
    if not host or port is None:
        return False
    try:
        connection = socket.create_connection((host, port), timeout=timeout)
    except OSError:
        return False
    connection.close()
    return True


def _maybe_apply_proxy(env_name: str, configured_proxy: str | None) -> tuple[str | None, bool]:
    existing = _proxy_env_value(env_name)
    if existing:
        return existing, False
    if not configured_proxy:
        return None, False
    if _is_local_proxy_url(configured_proxy) and not _is_proxy_reachable(configured_proxy):
        logging.warning("Skipping unreachable local proxy for %s: %s; fallback to direct connection", env_name, configured_proxy)
        return None, False
    _set_proxy_env(env_name, configured_proxy)
    return configured_proxy, True


def _configure_network(network: NetworkConfig) -> None:
    http_proxy, applied_http_proxy = _maybe_apply_proxy("HTTP_PROXY", network.http_proxy)
    https_proxy, applied_https_proxy = _maybe_apply_proxy("HTTPS_PROXY", network.https_proxy)
    baostock_socks_proxy, _ = _maybe_apply_proxy("BAOSTOCK_SOCKS_PROXY", network.socks5_proxy)
    if network.no_proxy and not _proxy_env_value("NO_PROXY") and (applied_http_proxy or applied_https_proxy):
        _set_proxy_env("NO_PROXY", network.no_proxy)
    if http_proxy or https_proxy:
        logging.info("Configured proxy: http=%s https=%s", http_proxy or "-", https_proxy or "-")
    if baostock_socks_proxy:
        logging.info("Configured BaoStock SOCKS proxy: %s", baostock_socks_proxy)


def _load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()


def _command_needs_network(command_name: str) -> bool:
    return command_name in {"update", "intraday-update", "intraday-screening"}


def _create_update_data_provider(data_interface: str):
    normalized = str(data_interface or "sina").strip().lower()
    if normalized == "baostock":
        return create_data_provider("baostock")
    if normalized in {"sina", "eastmoney"}:
        from .data_sources import AKShareDataProvider

        return AKShareDataProvider(daily_backend=normalized)
    raise ValueError(f"Unsupported update data interface: {data_interface}")


def _refresh_universe(storage: Storage, provider, exclude_st: bool) -> pd.DataFrame:
    instruments = provider.get_instruments()
    universe = build_main_board_universe(instruments, exclude_st=exclude_st)
    if universe.empty:
        logging.warning("Universe refresh returned no symbols; skip overwriting cached universe.")
        return universe
    target = storage.save_universe(universe)
    logging.info("Saved %s symbols to %s", len(universe), target)
    return universe


def _refresh_or_load_universe(storage: Storage, provider, exclude_st: bool) -> pd.DataFrame:
    try:
        universe = _refresh_universe(storage, provider, exclude_st)
    except Exception as exc:
        try:
            cached = storage.load_universe()
        except FileNotFoundError:
            raise
        logging.warning("Failed to refresh universe, fallback to cached universe: %s", exc)
        return cached
    if not universe.empty:
        return universe
    try:
        return storage.load_universe()
    except FileNotFoundError:
        raise RuntimeError("Universe refresh returned no symbols and no cached universe is available.")


def _ensure_universe(storage: Storage, provider_name: str, exclude_st: bool) -> None:
    try:
        storage.load_universe()
        return
    except FileNotFoundError:
        pass
    provider = create_data_provider(provider_name)
    try:
        _refresh_universe(storage, provider, exclude_st)
    finally:
        provider.close()


def _log_scan_progress(stage_name: str, current: int, total: int) -> None:
    if total <= 0:
        return
    if current == 1 or current == total or current % PROGRESS_LOG_INTERVAL == 0:
        logging.info("%s progress: %s/%s", stage_name, current, total)


def _localize_argparse() -> None:
    if getattr(argparse, "_stocks_analyzer_localized", False):
        return
    translations = {
        "usage: ": "用法：",
        "positional arguments": "位置参数",
        "options": "可选参数",
        "show this help message and exit": "显示此帮助信息并退出",
    }
    original = getattr(argparse, "_", None)
    if callable(original):
        argparse._ = lambda text: translations.get(original(text), original(text))
    else:
        argparse._ = lambda text: translations.get(text, text)
    argparse._stocks_analyzer_localized = True
