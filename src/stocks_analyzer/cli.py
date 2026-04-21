from __future__ import annotations

import argparse
import json
import logging
import os
import socket
from datetime import date, datetime, timedelta
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

import pandas as pd

from .atr import build_atr_export_frame, build_atr_snapshot_row, normalize_atr_summary_frame
from .config import load_config
from .data_sources import create_data_provider
from .daily_screening import run_daily_screening
from .macd_divergence import summarize_recent_macd_divergence
from .features import build_feature_frame
from .indicators import add_indicators
from .intraday_ranking import save_intraday_rankings
from .ml_dataset import build_probability_dataset, infer_split_dates, split_probability_dataset
from .ml_evaluation import evaluate_trained_artifact
from .ml_models import load_model_artifact, normalize_model_names, predict_with_model, train_and_save_models
from .models import NetworkConfig
from .paths import ProjectPaths
from .plotting import default_start_date, filter_by_date, load_or_fetch_daily, plot_candles_and_volume
from .probability_reporting import (
    format_evaluation_summary,
    format_prediction_summary,
    format_tradingview_summary,
    format_training_summary,
    save_evaluation_reports,
    save_predictions_report,
)
from .reporting import format_multi_pattern_summary, format_report
from .screener import Screener, parse_as_of
from .storage import Storage
from .strategies import STRATEGY_NAMES
from .trend_backtest import backtest_portfolios, backtest_signal_returns, summarize_signal_backtest
from .trend_indicator_scores import build_next_open_entries, scan_indicator_scored_entries, select_tradable_entries
from .trend_reporting import (
    save_atr_report,
    save_entry_backtest_reports,
    save_entry_portfolio_backtest_reports,
    save_macd_report,
    save_portfolio_backtest_reports,
    save_signal_backtest_reports,
    save_threshold_research_reports,
    save_trend_report,
    save_trend_entries_report,
    save_trend_scores_report,
    save_trend_signals_report,
    save_trend_universe_report,
)
from .trend_signals import scan_trend_signals
from .trend_threshold_research import (
    build_default_threshold_candidates,
    build_combo_threshold_candidates,
    build_threshold_research_dataset,
    derive_threshold_candidates,
    evaluate_combo_thresholds,
    evaluate_threshold_candidates,
    summarize_indicator_distributions,
)
from .trend_universe import scan_trend_universe
from .universe import build_main_board_universe
from .watchlist import (
    build_watchlist_candidates_from_patterns,
    build_watchlist_candidates_from_trend,
    extract_watchlist_symbols,
    find_latest_watchlist_before,
    load_watchlist,
    watchlist_path as build_watchlist_path,
    write_watchlist,
)
from .xueqiu_archive import archive_xueqiu_user_1155695148


def _localize_argparse() -> None:
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


PATTERN_FLAG_MAP = {
    "pattern1": "volume_top_pre_breakout",
    "pattern2": "volume_top_breakout",
    "pattern3": "volume_top_follow_through",
    "pattern4": "platform_breakout",
    "pattern5": "trend_pullback",
    "pattern6": "second_wave",
}
PATTERN_LABEL_MAP = {
    "volume_top_pre_breakout": "1",
    "volume_top_breakout": "2",
    "volume_top_follow_through": "3",
    "platform_breakout": "4",
    "trend_pullback": "5",
    "second_wave": "6",
}
PROGRESS_LOG_INTERVAL = 100
LOCAL_PROXY_HOSTS = {"127.0.0.1", "localhost", "::1"}


def build_parser() -> argparse.ArgumentParser:
    _localize_argparse()
    parser = argparse.ArgumentParser(
        description="A 股主板技术分析命令行工具",
        epilog=(
            "第一次使用建议顺序：\n"
            "  1. mystock update --start-date 20240101\n"
            "     更新主板股票池并拉取本地日线数据。\n"
            "  2. mystock pattern\n"
            "     扫描本地全部股票，识别 1 到 6 号模式并生成 CSV。\n"
            "  3. mystock plot 603588\n"
            "     查看单只股票近两年的 K 线和成交量图。\n\n"
            "常见示例：\n"
            "  mystock update 603588 --start-date 20240101\n"
            "  mystock pattern --1 --4\n"
            "  mystock report --date 2026-04-10\n"
            "  mystock tradingview --date 2026-04-10\n"
            "  mystock macd --date 2026-04-10\n"
            "  mystock atr --date 2026-04-10\n"
            "  mystock train-prob\n"
            "  mystock predict-prob --date 2026-04-10\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", default="config/default.yaml", help="YAML 配置文件路径")
    parser.add_argument("--project-root", default=".", help="项目根目录")
    parser.add_argument("--log-level", default="INFO", help="日志级别")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.title = "子命令"

    update = subparsers.add_parser(
        "update",
        help="更新股票池和日线数据，或只更新单只股票",
        description=(
            "更新本地数据。\n"
            "不传股票代码时，会先刷新主板股票池，再批量更新所有股票的日线数据；\n"
            "传入股票代码时，只更新该股票。"
        ),
        epilog=(
            "常见示例：\n"
            "  mystock update --start-date 20240101\n"
            "  mystock update 603588 --start-date 20240101\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    update.add_argument("symbol", nargs="?", help="可选的 6 位股票代码，只更新该股票")
    update.add_argument("--start-date", default="20230101", help="开始日期，格式 YYYYMMDD")
    update.add_argument("--end-date", default=datetime.today().strftime("%Y%m%d"), help="结束日期，格式 YYYYMMDD")
    update.add_argument("--limit", type=int, default=None, help="仅更新前 N 只股票，便于小范围测试")
    pattern = subparsers.add_parser(
        "pattern",
        help="识别本地日线数据中的 1 到 6 号模式",
        description=(
            "扫描本地缓存的全部股票日线数据，识别模式 1 到 6。\n"
            "默认识别全部模式；如果传入 --1 --2 --3 --4 --5 --6 中的任意组合，则只识别指定模式。"
        ),
        epilog=(
            "常见示例：\n"
            "  mystock pattern\n"
            "  mystock pattern --1\n"
            "  mystock pattern --2 --5\n"
            "  mystock pattern --as-of 2026-04-10 --output reports/my_patterns.csv\n"
            "  mystock pattern --plot-all\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    pattern.add_argument("--1", dest="pattern1", action="store_true", help="只识别模式 1")
    pattern.add_argument("--2", dest="pattern2", action="store_true", help="只识别模式 2")
    pattern.add_argument("--3", dest="pattern3", action="store_true", help="只识别模式 3")
    pattern.add_argument("--4", dest="pattern4", action="store_true", help="只识别模式 4")
    pattern.add_argument("--5", dest="pattern5", action="store_true", help="只识别模式 5")
    pattern.add_argument("--6", dest="pattern6", action="store_true", help="只识别模式 6")
    pattern.add_argument("--as-of", default=None, help="分析截止日期，格式 YYYY-MM-DD")
    pattern.add_argument("--limit", type=int, default=None, help="终端最多显示多少行")
    pattern.add_argument("--output", default=None, help="可选的 CSV 输出路径")
    pattern.add_argument("--plot-all", action="store_true", help="为所有命中股票批量生成图形")

    plot = subparsers.add_parser(
        "plot",
        help="绘制单只股票的 K 线和成交量图",
        description="绘制单只股票的日 K 线和成交量图。默认时间范围为近两年。",
        epilog=(
            "常见示例：\n"
            "  mystock plot 603588\n"
            "  mystock plot 603588 --start-date 20240101 --end-date 20260410\n"
            "  mystock plot 603588 --output reports/plots/603588_custom.png\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    plot.add_argument("symbol", help="6 位股票代码，例如 603588")
    plot.add_argument("--start-date", default=default_start_date(), help="开始日期，格式 YYYYMMDD，默认近两年")
    plot.add_argument("--end-date", default=datetime.today().strftime("%Y%m%d"), help="结束日期，格式 YYYYMMDD，默认今天")
    plot.add_argument("--output", default=None, help="可选的 PNG 输出路径")

    report = subparsers.add_parser(
        "report",
        help="读取已保存的模式识别结果",
        description="读取此前由 pattern 命令生成的 CSV 结果，并在终端中展示。",
        epilog=(
            "常见示例：\n"
            "  mystock report --date 2026-04-10\n"
            "  mystock report --date 2026-04-10 --limit 30\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    report.add_argument("--date", required=True, help="结果日期，格式 YYYY-MM-DD")
    report.add_argument("--limit", type=int, default=None, help="终端最多显示多少行")

    tradingview = subparsers.add_parser(
        "tradingview",
        help="计算指定日期的全市场 TradingView 技术评分",
        description="读取本地主板日线数据，按指定日期汇总每只股票的 TradingView 风格技术评分。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    tradingview.add_argument("--date", required=True, help="评分日期，格式 YYYY-MM-DD")
    tradingview.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")
    tradingview.add_argument("--output", default=None, help="可选的 CSV 输出路径")

    macd = subparsers.add_parser(
        "macd",
        help="生成指定日期的 MACD/量价统一技术状态表",
        description="读取本地主板日线数据，输出金叉死叉、MACD 背离和量价背离的统一状态表。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    macd.add_argument("--date", required=True, help="识别日期，格式 YYYY-MM-DD")
    macd.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")
    macd.add_argument("--output", default=None, help="可选的 CSV 输出路径")

    atr = subparsers.add_parser(
        "atr",
        help="生成指定日期的 ATR 风险辅助表",
        description="读取本地主板日线数据，输出 ATR14、ATR% 和止损止盈参考价。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    atr.add_argument("--date", required=True, help="识别日期，格式 YYYY-MM-DD")
    atr.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")
    atr.add_argument("--output", default=None, help="可选的 CSV 输出路径")

    train_prob = subparsers.add_parser(
        "train-prob",
        help="训练中短期上涨概率模型",
        description="基于本地主板日线数据构建样本，并训练 XGBoost 模型。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    train_prob.add_argument("--start-date", default=None, help="样本开始日期，格式 YYYY-MM-DD")
    train_prob.add_argument("--end-date", default=None, help="样本结束日期，格式 YYYY-MM-DD")
    train_prob.add_argument("--train-end", default=None, help="训练集结束日期，格式 YYYY-MM-DD")
    train_prob.add_argument("--valid-end", default=None, help="验证集结束日期，格式 YYYY-MM-DD")
    train_prob.add_argument("--test-end", default=None, help="测试集结束日期，格式 YYYY-MM-DD")
    train_prob.add_argument("--limit", type=int, default=None, help="仅使用前 N 只股票，便于快速测试")

    predict_prob = subparsers.add_parser(
        "predict-prob",
        help="生成指定日期的全市场上涨概率排序",
        description="读取已训练模型，对指定日期的主板股票生成概率排序结果。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    predict_prob.add_argument("--date", required=True, help="预测日期，格式 YYYY-MM-DD")
    predict_prob.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")
    predict_prob.add_argument("--output", default=None, help="可选的预测结果输出路径")

    xueqiu_archive = subparsers.add_parser(
        "xueqiu-archive",
        help="归档雪球博主 1155695148 的公开历史帖子",
        description="使用浏览器驱动抓取雪球博主 1155695148 的公开帖子，并导出为 Markdown。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    xueqiu_archive.add_argument("--output", default=None, help="可选的 Markdown 输出路径")
    xueqiu_archive.add_argument("--max-posts", type=int, default=None, help="仅抓取前 N 条帖子，便于测试")
    xueqiu_archive.add_argument("--refresh", action="store_true", help="忽略本地链接缓存并重新发现帖子")
    xueqiu_archive.add_argument("--headed", action="store_true", help="打开可见浏览器，便于手动完成滑动验证")

    daily_screening = subparsers.add_parser(
        "daily-screening",
        help="按交易日执行每日筛选，并生成当日 watchlist",
        description="自动判断是否为交易日，串行执行 update/tradingview/macd/trend-universe/pattern，再生成当日 watchlist。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    daily_screening.add_argument("--date", default=None, help="目标日期，格式 YYYY-MM-DD，默认今天")
    daily_screening.add_argument("--start-date", default="20240101", help="更新数据的起始日期，格式 YYYYMMDD")

    intraday_screening = subparsers.add_parser(
        "intraday-screening",
        help="读取 watchlist，只对候选股执行盘中复筛",
        description="自动读取上一交易日或指定日期的 watchlist，只更新候选股并执行盘中复筛。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    intraday_screening.add_argument("--date", default=None, help="目标日期，格式 YYYY-MM-DD，默认今天")
    intraday_screening.add_argument("--watchlist-date", default=None, help="watchlist 日期，格式 YYYY-MM-DD")
    intraday_screening.add_argument("--start-date", default="20240101", help="更新数据的起始日期，格式 YYYYMMDD")
    intraday_screening.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")

    trend = subparsers.add_parser(
        "trend",
        help="输出指定日期的全市场趋势评分结果",
        description="扫描全市场并输出指定日期的趋势评分结果，供 daily-screening 与 watchlist 复核复用。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    trend.add_argument("--date", required=True, help="目标日期，格式 YYYY-MM-DD")
    trend.add_argument("--top-n", type=int, default=30, help="终端展示前 N 行")
    trend.add_argument("--output", default=None, help="可选的趋势评分 CSV 输出路径")

    trend_universe = subparsers.add_parser(
        "trend-universe",
        help="生成指定日期的趋势股池和趋势评分",
        description="扫描本地主板日线数据，识别趋势股池并输出趋势评分。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    trend_universe.add_argument("--date", required=True, help="目标日期，格式 YYYY-MM-DD")
    trend_universe.add_argument("--top-n", type=int, default=30, help="终端展示前 N 行")
    trend_universe.add_argument("--output", default=None, help="可选的趋势股池 CSV 输出路径")

    trend_signals = subparsers.add_parser(
        "trend-signals",
        help="在趋势股池上生成 breakout 和 pullback 信号",
        description="读取本地日线数据，先识别趋势股池，再生成 breakout 和 pullback 两类趋势信号。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    trend_signals.add_argument("--date", required=True, help="目标日期，格式 YYYY-MM-DD")
    trend_signals.add_argument("--top-n", type=int, default=30, help="终端展示前 N 行")
    trend_signals.add_argument("--output", default=None, help="可选的趋势信号 CSV 输出路径")

    trend_score = subparsers.add_parser(
        "trend-score",
        help="对趋势 setup 做多指标打分",
        description="在 breakout 和 pullback setup 上叠加 MACD/RSI/BOLL/KDJ/ATR/量价等指标，输出买入评分。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    trend_score.add_argument("--date", required=True, help="目标日期，格式 YYYY-MM-DD")
    trend_score.add_argument("--top-n", type=int, default=30, help="终端展示前 N 行")
    trend_score.add_argument("--output", default=None, help="可选的评分 CSV 输出路径")

    trend_entries = subparsers.add_parser(
        "trend-entries",
        help="输出按次日开盘执行的趋势买入候选",
        description="读取收盘后的多指标评分结果，生成 planned_entry_date 为次日开盘的趋势买入候选。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    trend_entries.add_argument("--date", required=True, help="目标日期，格式 YYYY-MM-DD")
    trend_entries.add_argument("--top-n", type=int, default=30, help="终端展示前 N 行")
    trend_entries.add_argument("--output", default=None, help="可选的次日开盘候选 CSV 输出路径")

    backtest_signals = subparsers.add_parser(
        "backtest-signals",
        help="运行趋势信号的固定持有回测",
        description="对 breakout 和 pullback 信号做 5/10/20/40 日固定持有回测。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    backtest_signals.add_argument("--date", required=True, help="回测截止日期，格式 YYYY-MM-DD")
    backtest_signals.add_argument("--start-date", default=None, help="回测开始日期，格式 YYYY-MM-DD")
    backtest_signals.add_argument("--output", default=None, help="可选的回测明细 CSV 输出路径")
    backtest_signals.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")

    backtest_portfolio = subparsers.add_parser(
        "backtest-portfolio",
        help="运行趋势信号的组合回测",
        description="每天按评分选择前 N 只趋势信号，执行固定持有组合回测。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    backtest_portfolio.add_argument("--date", required=True, help="回测截止日期，格式 YYYY-MM-DD")
    backtest_portfolio.add_argument("--start-date", default=None, help="回测开始日期，格式 YYYY-MM-DD")
    backtest_portfolio.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")

    backtest_entries = subparsers.add_parser(
        "backtest-entries",
        help="运行次日开盘入场的趋势回测",
        description="基于收盘评分，在次日开盘买入并执行固定持有回测。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    backtest_entries.add_argument("--date", required=True, help="回测截止日期，格式 YYYY-MM-DD")
    backtest_entries.add_argument("--start-date", default=None, help="回测开始日期，格式 YYYY-MM-DD")
    backtest_entries.add_argument("--output", default=None, help="可选的回测明细 CSV 输出路径")
    backtest_entries.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")

    backtest_entries_portfolio = subparsers.add_parser(
        "backtest-entries-portfolio",
        help="运行次日开盘入场的组合回测",
        description="基于收盘评分，在次日开盘执行前 N 等权组合回测。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    backtest_entries_portfolio.add_argument("--date", required=True, help="回测截止日期，格式 YYYY-MM-DD")
    backtest_entries_portfolio.add_argument("--start-date", default=None, help="回测开始日期，格式 YYYY-MM-DD")
    backtest_entries_portfolio.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")

    research_thresholds = subparsers.add_parser(
        "research-thresholds",
        help="研究趋势买点评分阈值",
        description="基于历史样本比较强弱组指标分布，生成阈值候选和阈值回测对比表。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    research_thresholds.add_argument("--date", required=True, help="研究截止日期，格式 YYYY-MM-DD")
    research_thresholds.add_argument("--start-date", required=True, help="研究开始日期，格式 YYYY-MM-DD")
    research_thresholds.add_argument(
        "--sample-mode",
        choices=["daily", "weekly", "monthly"],
        default="monthly",
        help="历史截面抽样方式，默认 monthly",
    )
    research_thresholds.add_argument("--train-end-date", default=None, help="可选的样本内结束日期，格式 YYYY-MM-DD")
    research_thresholds.add_argument("--output", default=None, help="可选的样本明细 CSV 输出路径")
    research_thresholds.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    project_root = Path(args.project_root).resolve()
    _load_local_env(project_root / ".env.local")
    config = load_config(project_root / args.config)
    _configure_network(config.network)
    paths = ProjectPaths(project_root, config.storage)
    storage = Storage(paths)

    if args.command == "update":
        provider = create_data_provider(config.provider)
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
            )
        finally:
            provider.close()
        return

    if args.command == "pattern":
        selected = _selected_patterns(args)
        as_of = parse_as_of(args.as_of)
        _run_pattern(storage, config.provider, config, as_of, selected, args.limit, args.output, args.plot_all)
        return

    if args.command == "plot":
        _run_plot(storage, config, args.symbol, args.start_date, args.end_date, args.output)
        return

    if args.command == "report":
        trade_date = datetime.fromisoformat(args.date).date()
        _run_report(storage, config, trade_date, args.limit)
        return

    if args.command == "tradingview":
        _run_tradingview(
            storage=storage,
            config=config,
            trade_date=datetime.fromisoformat(args.date).date(),
            top_n=args.top_n,
            output=args.output,
        )
        return

    if args.command == "macd":
        _run_macd(
            storage=storage,
            config=config,
            paths=paths,
            trade_date=datetime.fromisoformat(args.date).date(),
            top_n=args.top_n,
            output=args.output,
        )
        return

    if args.command == "atr":
        _run_atr(
            storage=storage,
            config=config,
            paths=paths,
            trade_date=datetime.fromisoformat(args.date).date(),
            top_n=args.top_n,
            output=args.output,
        )
        return

    if args.command == "train-prob":
        _run_train_prob(
            storage=storage,
            config=config,
            start_date=_parse_optional_date(args.start_date),
            end_date=_parse_optional_date(args.end_date),
            train_end=_parse_optional_date(args.train_end),
            valid_end=_parse_optional_date(args.valid_end),
            test_end=_parse_optional_date(args.test_end),
            limit=args.limit,
        )
        return

    if args.command == "predict-prob":
        _run_predict_prob(
            storage=storage,
            config=config,
            trade_date=datetime.fromisoformat(args.date).date(),
            top_n=args.top_n,
            output=args.output,
        )
        return

    if args.command == "xueqiu-archive":
        _run_xueqiu_archive(paths, output=args.output, max_posts=args.max_posts, refresh=args.refresh, headed=args.headed)
        return

    if args.command == "daily-screening":
        trade_date = datetime.fromisoformat(args.date).date() if args.date else date.today()
        result = run_daily_screening(
            project_root=project_root,
            trade_date=trade_date,
            start_date=args.start_date,
        )
        print(result.message)
        if result.report_path:
            print(f"报告文件：{result.report_path}")
        return

    if args.command == "intraday-screening":
        trade_date = datetime.fromisoformat(args.date).date() if args.date else date.today()
        watchlist_date = datetime.fromisoformat(args.watchlist_date).date() if args.watchlist_date else None
        result = _run_intraday_screening(
            storage=storage,
            config=config,
            project_root=project_root,
            trade_date=trade_date,
            watchlist_date=watchlist_date,
            start_date=args.start_date,
            top_n=args.top_n,
        )
        print(result["message"])
        print(f"报告文件：{result['report_path']}")
        return

    if args.command == "trend":
        _run_trend(
            storage=storage,
            config=config,
            paths=paths,
            trade_date=datetime.fromisoformat(args.date).date(),
            top_n=args.top_n,
            output=args.output,
        )
        return

    if args.command == "trend-universe":
        _run_trend_universe(
            storage=storage,
            config=config,
            paths=paths,
            trade_date=datetime.fromisoformat(args.date).date(),
            top_n=args.top_n,
            output=args.output,
        )
        return

    if args.command == "trend-signals":
        _run_trend_signals(
            storage=storage,
            config=config,
            paths=paths,
            trade_date=datetime.fromisoformat(args.date).date(),
            top_n=args.top_n,
            output=args.output,
        )
        return

    if args.command == "trend-score":
        _run_trend_score(
            storage=storage,
            config=config,
            paths=paths,
            trade_date=datetime.fromisoformat(args.date).date(),
            top_n=args.top_n,
            output=args.output,
        )
        return

    if args.command == "trend-entries":
        _run_trend_entries(
            storage=storage,
            config=config,
            paths=paths,
            trade_date=datetime.fromisoformat(args.date).date(),
            top_n=args.top_n,
            output=args.output,
        )
        return

    if args.command == "backtest-signals":
        _run_backtest_signals(
            storage=storage,
            config=config,
            paths=paths,
            end_date=datetime.fromisoformat(args.date).date(),
            start_date=_parse_optional_date(args.start_date),
            output=args.output,
            top_n=args.top_n,
        )
        return

    if args.command == "backtest-portfolio":
        _run_backtest_portfolio(
            storage=storage,
            config=config,
            paths=paths,
            end_date=datetime.fromisoformat(args.date).date(),
            start_date=_parse_optional_date(args.start_date),
            top_n=args.top_n,
        )
        return

    if args.command == "backtest-entries":
        _run_backtest_entries(
            storage=storage,
            config=config,
            paths=paths,
            end_date=datetime.fromisoformat(args.date).date(),
            start_date=_parse_optional_date(args.start_date),
            output=args.output,
            top_n=args.top_n,
        )
        return

    if args.command == "backtest-entries-portfolio":
        _run_backtest_entries_portfolio(
            storage=storage,
            config=config,
            paths=paths,
            end_date=datetime.fromisoformat(args.date).date(),
            start_date=_parse_optional_date(args.start_date),
            top_n=args.top_n,
        )
        return

    if args.command == "research-thresholds":
        _run_research_thresholds(
            storage=storage,
            config=config,
            paths=paths,
            end_date=datetime.fromisoformat(args.date).date(),
            start_date=datetime.fromisoformat(args.start_date).date(),
            sample_mode=args.sample_mode,
            train_end_date=_parse_optional_date(args.train_end_date),
            output=args.output,
            top_n=args.top_n,
        )
        return

    parser.error(f"Unknown command: {args.command}")


def _run_update(
    storage: Storage,
    provider,
    exclude_st: bool,
    adjust: str,
    symbol: str | None,
    start_date: str,
    end_date: str,
    limit: int | None,
) -> None:
    if symbol:
        normalized_symbol = str(symbol).zfill(6)
        _update_daily_cache_for_symbol(
            storage=storage,
            provider=provider,
            symbol=normalized_symbol,
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

    logging.info(
        "Daily update finished: success=%s failed=%s",
        success_count,
        len(failed_symbols),
    )
    if failed_symbols:
        logging.warning("Failed symbols sample: %s", ", ".join(failed_symbols[:20]))


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
    except FileNotFoundError:
        fresh = provider.get_daily_bars(symbol, start_date=start_date, end_date=end_date, adjust=adjust)
        target = storage.save_daily_bars(symbol, fresh)
        logging.info("Initialized %s rows for %s to %s", len(fresh), symbol, target)
        return target

    cached_frame = cached.copy()
    if cached_frame.empty:
        fresh = provider.get_daily_bars(symbol, start_date=start_date, end_date=end_date, adjust=adjust)
        target = storage.save_daily_bars(symbol, fresh)
        logging.info("Initialized %s rows for %s to %s", len(fresh), symbol, target)
        return target

    cached_frame["trade_date"] = pd.to_datetime(cached_frame["trade_date"], errors="coerce")
    valid_dates = cached_frame["trade_date"].dropna()
    if valid_dates.empty:
        logging.warning("Cached daily bars for %s have no valid trade_date values; rebuilding from %s", symbol, start_date)
        fresh = provider.get_daily_bars(symbol, start_date=start_date, end_date=end_date, adjust=adjust)
        target = storage.save_daily_bars(symbol, fresh)
        logging.info("Rebuilt %s rows for %s to %s", len(fresh), symbol, target)
        return target

    last_trade_date = valid_dates.max().date()
    requested_end_date = datetime.strptime(end_date, "%Y%m%d").date()
    incremental_start_date = last_trade_date + timedelta(days=1)
    target = storage.paths.daily_dir / f"{symbol}.parquet"
    if incremental_start_date > requested_end_date:
        logging.info(
            "Skip %s because cached daily bars already cover %s (last=%s)",
            symbol,
            requested_end_date.isoformat(),
            last_trade_date.isoformat(),
        )
        return target

    incremental_start = incremental_start_date.strftime("%Y%m%d")
    fresh = provider.get_daily_bars(symbol, start_date=incremental_start, end_date=end_date, adjust=adjust)
    if fresh.empty:
        logging.info(
            "No new daily bars returned for %s from %s to %s",
            symbol,
            incremental_start_date.isoformat(),
            requested_end_date.isoformat(),
        )
        return target

    merged = pd.concat([cached_frame, fresh], ignore_index=True)
    merged["trade_date"] = pd.to_datetime(merged["trade_date"], errors="coerce")
    merged = merged.dropna(subset=["trade_date"]).drop_duplicates(subset=["trade_date"], keep="last")
    merged = merged.sort_values("trade_date").reset_index(drop=True)
    target = storage.save_daily_bars(symbol, merged)
    logging.info(
        "Appended %s rows for %s from %s to %s",
        len(fresh),
        symbol,
        incremental_start,
        target,
    )
    return target


def _run_pattern(
    storage: Storage,
    provider_name: str,
    config,
    as_of: date,
    selected_patterns: list[str],
    limit: int | None,
    output: str | None,
    plot_all: bool,
    symbols: list[str] | None = None,
) -> None:
    _ensure_universe(storage, provider_name, config.universe.exclude_st)
    screener = Screener(storage, config)
    results = screener.run(as_of=as_of, selected_strategies=selected_patterns, symbols=symbols)
    exported = _prepare_pattern_results(results)
    exported = _append_recent_tradingview_scores(storage, exported, as_of=as_of, lookback_days=5, symbols=symbols)
    exported = _append_recent_macd_summary(storage, exported, as_of=as_of, symbols=symbols)
    exported = _append_recent_atr_summary(storage, exported, as_of=as_of, symbols=symbols)
    trend_universe = _load_or_build_trend_universe_summary(storage, config=config, trade_date=as_of, symbols=symbols)
    exported = _append_recent_trend_universe_summary(exported, trend_universe)
    exported = _append_recent_trend_summary(storage, config=config, exported=exported, as_of=as_of, symbols=symbols)

    output_path = Path(output) if output else _default_pattern_output_path(
        storage,
        as_of=as_of,
        selected_patterns=selected_patterns,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(output_path, index=False, encoding="utf-8-sig")
    watchlist_payload = build_watchlist_candidates_from_patterns(
        exported,
        source_file=str(output_path),
        limit=config.screening.output_limit,
    )
    write_watchlist(
        project_root=storage.paths.root,
        trade_date=as_of,
        picker_payload=watchlist_payload,
    )
    pattern_watchlist_target = write_watchlist(
        project_root=storage.paths.root,
        trade_date=as_of,
        picker_payload=watchlist_payload,
        kind="pattern",
    )
    logging.info("Saved pattern watchlist to %s", pattern_watchlist_target)

    if exported.empty:
        logging.info("No patterns matched for %s", as_of.isoformat())
        logging.info("Saved empty pattern report to %s", output_path)
        print(f"No patterns matched. Saved empty CSV to {output_path}")
        print(f"Saved pattern watchlist to {pattern_watchlist_target}")
        return

    multi_pattern_summary = format_multi_pattern_summary(exported)
    if multi_pattern_summary:
        print(multi_pattern_summary)
        print()
    print(format_report(exported, limit=limit or config.screening.output_limit))
    if plot_all:
        plots_dir = _plot_pattern_matches(storage, config, as_of, results)
        print(f"\n已生成图形目录: {plots_dir}")
    print(f"\nSaved pattern watchlist to {pattern_watchlist_target}")
    logging.info("Saved %s pattern rows to %s", len(exported), output_path)


def _run_plot(
    storage: Storage,
    config,
    symbol: str,
    start_date: str,
    end_date: str,
    output: str | None,
) -> None:
    normalized_symbol = str(symbol).zfill(6)
    dataframe = load_or_fetch_daily(
        storage=storage,
        provider_name=config.provider,
        symbol=normalized_symbol,
        start_date=start_date,
        end_date=end_date,
        adjust=config.adjustment,
    )
    filtered = filter_by_date(dataframe, start_date, end_date)
    if filtered.empty:
        raise RuntimeError(f"No data available for {normalized_symbol} in {start_date} to {end_date}")

    output_path = Path(output) if output else storage.paths.reports_dir / "plots" / f"{normalized_symbol}_2y.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_candles_and_volume(filtered, normalized_symbol, output_path)
    print(f"Saved plot to {output_path}")


def _run_report(storage: Storage, config, trade_date: date, limit: int | None) -> None:
    report_path = _default_pattern_output_path(storage, as_of=trade_date, selected_patterns=STRATEGY_NAMES)
    if report_path.exists():
        results = pd.read_csv(report_path)
        print(format_report(results, limit=limit or config.screening.output_limit))
        return

    signals_path = storage.paths.signals_dir / f"signals_{trade_date.isoformat()}.parquet"
    if signals_path.exists():
        results = storage.load_signals(trade_date)
        results = _prepare_pattern_results(results)
        print(format_report(results, limit=limit or config.screening.output_limit))
        return

    raise FileNotFoundError(f"Pattern report not found for {trade_date.isoformat()}: {report_path}")


def _run_tradingview(
    storage: Storage,
    config,
    trade_date: date,
    top_n: int,
    output: str | None,
    symbols: list[str] | None = None,
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    summary, daily_frames = _load_or_build_tradingview_summary(
        storage,
        trade_date=trade_date,
        lookback_days=5,
        symbols=symbols,
    )
    if summary.empty:
        raise RuntimeError(f"No TradingView ratings could be generated for {trade_date.isoformat()}")

    output_path = Path(output) if output else storage.paths.reports_dir / "tradingview" / f"tradingview_avg5_{trade_date.isoformat()}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    for snapshot in daily_frames:
        daily_path = output_path.parent / f"tradingview_{snapshot['trade_date']}.csv"
        snapshot["data"].to_csv(daily_path, index=False, encoding="utf-8-sig")

    print(format_tradingview_summary(summary, limit=top_n))
    print(f"\nSaved TradingView 5-day summary to {output_path}")
    print(f"Saved {len(daily_frames)} daily TradingView files to {output_path.parent}")


def _run_macd(
    storage: Storage,
    config,
    paths: ProjectPaths,
    trade_date: date,
    top_n: int,
    output: str | None,
    symbols: list[str] | None = None,
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    summary = _load_or_build_macd_summary(storage, trade_date=trade_date, symbols=symbols)
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


def _run_atr(
    storage: Storage,
    config,
    paths: ProjectPaths,
    trade_date: date,
    top_n: int,
    output: str | None,
    symbols: list[str] | None = None,
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    summary = _load_or_build_atr_summary(storage, trade_date=trade_date, symbols=symbols)
    if summary.empty:
        raise RuntimeError(f"No ATR summary could be generated for {trade_date.isoformat()}")
    report_paths = save_atr_report(paths, trade_date=trade_date, dataframe=summary, output=output)

    display = build_atr_export_frame(summary).head(top_n)
    print(display.to_string(index=False))
    print(f"\nATR 风险辅助文件：{report_paths['detail_path']}")


def _run_trend_universe(
    storage: Storage,
    config,
    paths: ProjectPaths,
    trade_date: date,
    top_n: int,
    output: str | None,
) -> None:
    summary = scan_trend_universe(storage, config, as_of=trade_date)
    report_paths = save_trend_universe_report(paths, trade_date=trade_date, dataframe=summary, output=output)
    print(
        _format_dataframe(
            summary,
            ["trade_date", "symbol", "name", "trend_score", "trend_direction_score", "trend_strength_score"],
            top_n,
        )
    )
    print(f"\n趋势股池文件：{report_paths['detail_path']}")


def _run_trend_signals(
    storage: Storage,
    config,
    paths: ProjectPaths,
    trade_date: date,
    top_n: int,
    output: str | None,
) -> None:
    signals = scan_trend_signals(storage, config, trade_date=trade_date)
    report_paths = save_trend_signals_report(paths, trade_date=trade_date, dataframe=signals, output=output)
    print(
        _format_dataframe(
            signals,
            ["trade_date", "symbol", "name", "signal_type", "trend_score", "entry_score", "trigger_reason"],
            top_n,
        )
    )
    print(f"\n趋势信号文件：{report_paths['detail_path']}")


def _run_trend_score(
    storage: Storage,
    config,
    paths: ProjectPaths,
    trade_date: date,
    top_n: int,
    output: str | None,
) -> None:
    scored = scan_indicator_scored_entries(storage, config, trade_date=trade_date)
    report_paths = save_trend_scores_report(paths, trade_date=trade_date, dataframe=scored, output=output)
    print(
        _format_dataframe(
            scored,
            ["trade_date", "symbol", "name", "setup_type", "buy_score", "trend_base_score", "price_action_score", "macd_score"],
            top_n,
        )
    )
    print(f"\n趋势评分文件：{report_paths['detail_path']}")


def _run_trend(
    storage: Storage,
    config,
    paths: ProjectPaths,
    trade_date: date,
    top_n: int,
    output: str | None,
) -> None:
    scored = scan_indicator_scored_entries(storage, config, trade_date=trade_date)
    report_paths = save_trend_report(paths, trade_date=trade_date, dataframe=scored, output=output)
    watchlist_payload = build_watchlist_candidates_from_trend(
        scored,
        source_file=str(report_paths["detail_path"]),
        thresholds=config.pick_trend_watchlist,
        limit=config.screening.output_limit,
    )
    trend_watchlist_target = write_watchlist(
        project_root=paths.root,
        trade_date=trade_date,
        picker_payload=watchlist_payload,
        kind="trend",
    )
    print(
        _format_dataframe(
            scored,
            [
                "trade_date",
                "symbol",
                "name",
                "signal_type",
                "buy_score",
                "price_action_score",
                "macd_cross_state",
                "macd_divergence_state",
                "volume_price_divergence_state",
            ],
            top_n,
        )
    )
    print(f"\n趋势复核文件：{report_paths['detail_path']}")
    print(f"趋势候选文件：{trend_watchlist_target}")


def _run_trend_entries(
    storage: Storage,
    config,
    paths: ProjectPaths,
    trade_date: date,
    top_n: int,
    output: str | None,
) -> None:
    scored = scan_indicator_scored_entries(storage, config, trade_date=trade_date)
    entries = select_tradable_entries(scored, config)
    report_paths = save_trend_entries_report(paths, trade_date=trade_date, dataframe=entries, output=output)
    print(
        _format_dataframe(
            entries,
            ["trade_date", "planned_entry_date", "symbol", "name", "setup_type", "buy_score", "positive_indicator_count", "buy_reason"],
            top_n,
        )
    )
    print(f"\n次日开盘候选文件：{report_paths['detail_path']}")


def _run_backtest_signals(
    storage: Storage,
    config,
    paths: ProjectPaths,
    end_date: date,
    start_date: date | None,
    output: str | None,
    top_n: int,
) -> None:
    signals = scan_trend_signals(storage, config, start_date=start_date, end_date=end_date)
    daily_history = _load_daily_history_map(storage, signals["symbol"].astype(str).tolist() if not signals.empty else [])
    detail = backtest_signal_returns(signals, daily_history, config.trend_backtest)
    summary = summarize_signal_backtest(detail)
    report_paths = save_signal_backtest_reports(paths, report_date=end_date, detail=detail, summary=summary, output=output)
    print(
        _format_dataframe(
            summary,
            ["sample_group", "signal_type", "holding_days", "signal_count", "win_rate", "avg_return_pct", "avg_max_drawdown_pct"],
            top_n,
        )
    )
    print(f"\n信号回测明细：{report_paths['detail_path']}")
    print(f"信号回测汇总：{report_paths['summary_path']}")


def _run_backtest_entries(
    storage: Storage,
    config,
    paths: ProjectPaths,
    end_date: date,
    start_date: date | None,
    output: str | None,
    top_n: int,
) -> None:
    scored = scan_indicator_scored_entries(storage, config, start_date=start_date, end_date=end_date)
    entries = select_tradable_entries(scored, config)
    daily_history = _load_daily_history_map(storage, entries["symbol"].astype(str).tolist() if not entries.empty else [])
    detail = backtest_signal_returns(entries, daily_history, config.trend_backtest, entry_timing="next_open")
    summary = summarize_signal_backtest(detail)
    report_paths = save_entry_backtest_reports(paths, report_date=end_date, detail=detail, summary=summary, output=output)
    print(
        _format_dataframe(
            summary,
            ["sample_group", "signal_type", "holding_days", "signal_count", "win_rate", "avg_return_pct", "avg_buy_score"],
            top_n,
        )
    )
    print(f"\n次日开盘回测明细：{report_paths['detail_path']}")
    print(f"次日开盘回测汇总：{report_paths['summary_path']}")


def _run_backtest_portfolio(
    storage: Storage,
    config,
    paths: ProjectPaths,
    end_date: date,
    start_date: date | None,
    top_n: int,
) -> None:
    signals = scan_trend_signals(storage, config, start_date=start_date, end_date=end_date)
    daily_history = _load_daily_history_map(storage, signals["symbol"].astype(str).tolist() if not signals.empty else [])
    backtest_outputs = backtest_portfolios(signals, daily_history, config.trend_backtest)
    report_paths = save_portfolio_backtest_reports(
        paths,
        report_date=end_date,
        positions=backtest_outputs["positions"],
        equity=backtest_outputs["equity"],
        summary=backtest_outputs["summary"],
    )
    print(
        _format_dataframe(
            backtest_outputs["summary"],
            ["portfolio_top_n", "holding_days", "position_count", "win_rate", "final_net_value", "max_drawdown"],
            top_n,
        )
    )
    print(f"\n组合回测持仓：{report_paths['positions_path']}")
    print(f"组合回测净值：{report_paths['equity_path']}")
    print(f"组合回测汇总：{report_paths['summary_path']}")


def _run_backtest_entries_portfolio(
    storage: Storage,
    config,
    paths: ProjectPaths,
    end_date: date,
    start_date: date | None,
    top_n: int,
) -> None:
    scored = scan_indicator_scored_entries(storage, config, start_date=start_date, end_date=end_date)
    entries = select_tradable_entries(scored, config)
    daily_history = _load_daily_history_map(storage, entries["symbol"].astype(str).tolist() if not entries.empty else [])
    backtest_outputs = backtest_portfolios(
        entries,
        daily_history,
        config.trend_backtest,
        entry_timing="next_open",
        rank_column="buy_score",
    )
    report_paths = save_entry_portfolio_backtest_reports(
        paths,
        report_date=end_date,
        positions=backtest_outputs["positions"],
        equity=backtest_outputs["equity"],
        summary=backtest_outputs["summary"],
    )
    print(
        _format_dataframe(
            backtest_outputs["summary"],
            ["portfolio_top_n", "holding_days", "position_count", "win_rate", "final_net_value", "avg_buy_score"],
            top_n,
        )
    )
    print(f"\n次日开盘组合持仓：{report_paths['positions_path']}")
    print(f"次日开盘组合净值：{report_paths['equity_path']}")
    print(f"次日开盘组合汇总：{report_paths['summary_path']}")


def _run_research_thresholds(
    storage: Storage,
    config,
    paths: ProjectPaths,
    end_date: date,
    start_date: date,
    sample_mode: str,
    train_end_date: date | None,
    output: str | None,
    top_n: int,
) -> None:
    dataset = build_threshold_research_dataset(
        storage,
        config,
        start_date=start_date,
        end_date=end_date,
        sample_mode=sample_mode,
        train_end_date=train_end_date,
    )
    if dataset.empty:
        raise RuntimeError("No threshold research samples could be generated for the requested range")

    logging.info("Threshold research stage: summarizing indicator distributions")
    distributions = summarize_indicator_distributions(dataset)
    logging.info("Threshold research stage: deriving threshold candidates")
    candidates = derive_threshold_candidates(distributions)
    logging.info("Threshold research stage: evaluating single-metric thresholds")
    candidate_evaluation = evaluate_threshold_candidates(dataset, candidates)
    logging.info("Threshold research stage: building combo threshold candidates")
    combo_candidates = build_combo_threshold_candidates(candidates, config)
    logging.info("Threshold research stage: evaluating combo thresholds")
    combo_evaluation = evaluate_combo_thresholds(dataset, combo_candidates)
    logging.info("Threshold research stage: summarizing signal-specific default candidates")
    default_candidates = build_default_threshold_candidates(candidates, combo_candidates, combo_evaluation)
    logging.info("Threshold research stage: writing reports")
    report_paths = save_threshold_research_reports(
        paths,
        report_date=end_date,
        samples=dataset,
        distributions=distributions,
        candidates=candidates,
        candidate_evaluation=candidate_evaluation,
        combo_candidates=combo_candidates,
        combo_evaluation=combo_evaluation,
        default_candidates=default_candidates,
        output=output,
    )

    display = candidates[
        (candidates["dataset_split"] == "all_period") & (candidates["signal_scope"] == "all") & (candidates["candidate_type"] == "balanced")
    ].copy()
    if display.empty:
        display = candidates.copy()
    print(
        _format_dataframe(
            display,
            ["dataset_split", "holding_days", "metric", "candidate_type", "threshold", "separation_gap", "strong_p20", "weak_p80"],
            top_n,
        )
    )

    if not combo_evaluation.empty:
        combo_display = combo_evaluation[
            (combo_evaluation["dataset_split"] == "all_period") & (combo_evaluation["signal_scope"] == "all")
        ].copy()
        if combo_display.empty:
            combo_display = combo_evaluation.copy()
        print(
            "\n组合阈值对比：\n"
            + _format_dataframe(
                combo_display,
                ["dataset_split", "holding_days", "combo_name", "selected_count", "coverage", "win_rate", "avg_return_pct"],
                min(top_n, 10),
            )
        )

    if not default_candidates.empty:
        default_display = default_candidates[default_candidates["dataset_split"] == "all_period"].copy()
        if default_display.empty:
            default_display = default_candidates.copy()
        print(
            "\n分信号候选默认阈值：\n"
            + _format_dataframe(
                default_display,
                [
                    "signal_scope",
                    "holding_days",
                    "recommended_combo_name",
                    "buy_score_min",
                    "trend_base_score_min",
                    "price_action_score_min",
                    "macd_score_min",
                    "positive_indicator_count_min",
                    "avg_return_pct",
                    "current_default_avg_return_pct",
                ],
                top_n,
            )
        )

    print(f"\n阈值研究样本：{report_paths['samples_path']}")
    print(f"指标分布对比：{report_paths['distributions_path']}")
    print(f"候选阈值：{report_paths['candidates_path']}")
    print(f"单指标阈值回测：{report_paths['candidate_evaluation_path']}")
    print(f"组合阈值候选：{report_paths['combo_candidates_path']}")
    print(f"组合阈值回测：{report_paths['combo_evaluation_path']}")
    print(f"分信号默认阈值：{report_paths['default_candidates_path']}")


def _run_intraday_screening(
    *,
    storage: Storage,
    config,
    project_root: Path,
    trade_date: date,
    watchlist_date: date | None,
    start_date: str,
    top_n: int,
) -> dict[str, object]:
    resolved_watchlist_date = watchlist_date
    if resolved_watchlist_date is None:
        resolved_watchlist_date, resolved_watchlist_path = find_latest_watchlist_before(
            project_root=project_root,
            trade_date=trade_date,
        )
    else:
        resolved_watchlist_path = build_watchlist_path(project_root, resolved_watchlist_date)

    watchlist_payload = load_watchlist(project_root=project_root, trade_date=resolved_watchlist_date)
    symbols = extract_watchlist_symbols(watchlist_payload)
    if not symbols:
        raise RuntimeError(f"Watchlist {resolved_watchlist_date.isoformat()} contains no candidate symbols.")

    intraday_dir = storage.paths.reports_dir / "intraday_screening" / trade_date.isoformat()
    intraday_dir.mkdir(parents=True, exist_ok=True)
    intraday_rank_path = intraday_dir / f"intraday_rank_{trade_date.isoformat()}.csv"
    ranking_result = save_intraday_rankings(
        trade_date=trade_date,
        intraday_provider=config.intraday_provider,
        adjust=config.adjustment,
        watchlist_payload=watchlist_payload,
        output_path=intraday_rank_path,
    )

    report_path = intraday_dir / f"intraday_screening_{trade_date.isoformat()}.json"
    report = {
        "trade_date": trade_date.isoformat(),
        "watchlist_date": resolved_watchlist_date.isoformat(),
        "watchlist_path": str(resolved_watchlist_path),
        "symbol_count": len(symbols),
        "symbols": symbols,
        "intraday_rank_path": str(intraday_rank_path),
        "successful_symbol_count": int(ranking_result["processed_count"]),
        "failed_symbol_count": int(ranking_result["failed_count"]),
        "failed_symbols": ranking_result["failed_symbols"],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    failed_count = int(ranking_result["failed_count"])
    failure_note = f"，失败 {failed_count} 只" if failed_count else ""
    return {
        "message": (
            f"已完成 {trade_date.isoformat()} 盘中复筛，基于 {resolved_watchlist_date.isoformat()} watchlist "
            f"共处理 {len(symbols)} 只股票，成功 {int(ranking_result['processed_count'])} 只{failure_note}。"
        ),
        "report_path": report_path,
    }


def _run_train_prob(
    storage: Storage,
    config,
    start_date: date | None,
    end_date: date | None,
    train_end: date | None,
    valid_end: date | None,
    test_end: date | None,
    limit: int | None,
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    started_at = perf_counter()
    logging.info("Building probability dataset from local daily bars")
    dataset = build_probability_dataset(
        storage=storage,
        config=config,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    if dataset.empty:
        raise RuntimeError("No probability dataset could be built from the current local data.")
    logging.info("Built probability dataset with %s rows", len(dataset))

    resolved_train_end, resolved_valid_end, resolved_test_end = _resolve_probability_split_dates(
        dataset,
        train_end,
        valid_end,
        test_end,
    )
    split = split_probability_dataset(
        dataset=dataset,
        train_end=resolved_train_end,
        valid_end=resolved_valid_end,
        test_end=resolved_test_end,
    )
    logging.info(
        "Probability split resolved: train<=%s valid<=%s test<=%s",
        resolved_train_end.isoformat(),
        resolved_valid_end.isoformat(),
        resolved_test_end.isoformat(),
    )
    logging.info(
        "Probability split sizes: train=%s valid=%s test=%s features=%s",
        len(split.train),
        len(split.valid),
        len(split.test),
        len(split.feature_columns),
    )
    artifacts = train_and_save_models(
        split=split,
        model_names=["xgboost"],
        output_dir=storage.paths.ml_models_dir,
    )
    print(format_training_summary(artifacts))
    evaluation_reports = [
        evaluate_trained_artifact(artifact, split=split, top_n_list=config.probability.top_n_list) for artifact in artifacts
    ]
    print()
    print(format_evaluation_summary(evaluation_reports))
    saved_reports = save_evaluation_reports(evaluation_reports, storage.paths.probability_reports_dir)
    logging.info("Saved %s probability evaluation reports to %s", len(saved_reports), storage.paths.probability_reports_dir)
    logging.info("Probability training finished in %.2fs", perf_counter() - started_at)


def _run_predict_prob(
    storage: Storage,
    config,
    trade_date: date,
    top_n: int,
    output: str | None,
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    model_name = "xgboost"
    model_path = storage.paths.ml_models_dir / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    artifact = load_model_artifact(model_path)

    universe = storage.load_universe()
    rows: list[pd.DataFrame] = []
    for instrument in universe.to_dict("records"):
        symbol = str(instrument["symbol"])
        try:
            bars = storage.load_daily_bars(symbol)
        except FileNotFoundError:
            continue

        feature_frame = build_feature_frame(bars)
        feature_frame["trade_date"] = pd.to_datetime(feature_frame["trade_date"])
        current = feature_frame[feature_frame["trade_date"].dt.date == trade_date].copy()
        if current.empty:
            continue
        if pd.isna(current.iloc[-1]["amount_ma_20"]) or current.iloc[-1]["amount_ma_20"] < config.universe.min_avg_amount_20d:
            continue

        current["symbol"] = symbol
        current["name"] = instrument["name"]
        rows.append(current)

    if not rows:
        raise RuntimeError(f"No feature rows found for prediction date {trade_date.isoformat()}")

    frame = pd.concat(rows, ignore_index=True)
    predictions = predict_with_model(artifact, frame)
    output_path = Path(output) if output else storage.paths.probability_reports_dir / f"probability_{trade_date.isoformat()}_{model_name}.csv"
    save_predictions_report(predictions, output_path)
    print(format_prediction_summary(predictions, limit=top_n))
    print(f"\nSaved probability ranking to {output_path}")


def _load_daily_history_map(storage: Storage, symbols: list[str]) -> dict[str, pd.DataFrame]:
    history: dict[str, pd.DataFrame] = {}
    for symbol in {str(item).zfill(6) for item in symbols}:
        try:
            history[symbol] = storage.load_daily_bars(symbol)
        except FileNotFoundError:
            continue
    return history


def _format_dataframe(dataframe: pd.DataFrame, columns: list[str], top_n: int) -> str:
    if dataframe.empty:
        return "No rows matched."
    available = [column for column in columns if column in dataframe.columns]
    if not available:
        return dataframe.head(top_n).to_string(index=False)
    return dataframe.loc[:, available].head(top_n).to_string(index=False)


def _run_xueqiu_archive(
    paths: ProjectPaths,
    *,
    output: str | None,
    max_posts: int | None,
    refresh: bool,
    headed: bool,
) -> None:
    started_at = perf_counter()
    result = archive_xueqiu_user_1155695148(paths, output=output, max_posts=max_posts, refresh=refresh, headed=headed)
    elapsed = perf_counter() - started_at
    cache_text = "yes" if result.used_cache else "no"
    print("雪球归档完成")
    print(f"候选链接数：{result.candidate_count}")
    print(f"成功归档数：{result.archived_count}")
    print(f"失败数：{result.failed_count}")
    print(f"使用缓存：{cache_text}")
    print(f"输出文件：{result.output_path}")
    print(f"耗时：{elapsed:.1f}s")


def _selected_patterns(args: argparse.Namespace) -> list[str]:
    selected = [pattern for field, pattern in PATTERN_FLAG_MAP.items() if getattr(args, field)]
    return selected or list(STRATEGY_NAMES)


def _prepare_pattern_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["trade_date", "symbol", "name", "pattern_id", "close", "reason"])

    exported = results.copy()
    exported["symbol"] = exported["symbol"].map(_format_symbol_for_excel)
    exported["pattern_id"] = exported["strategy_name"].map(PATTERN_LABEL_MAP)
    exported = exported.drop(columns=["strategy_name"], errors="ignore")
    dedupe_keys = [column for column in ("trade_date", "symbol", "pattern_id") if column in exported.columns]
    if dedupe_keys:
        exported = exported.drop_duplicates(subset=dedupe_keys, keep="first")
    preferred_order = [
        "trade_date",
        "symbol",
        "name",
        "pattern_id",
        "close",
        "old_high_date",
        "old_high_price",
        "days_since_old_high",
        "max_drawdown_since_old_high",
        "distance_to_old_high_pct",
        "extension_above_old_high_pct",
        "breakout_date",
        "breakout_volume_ratio",
        "days_after_breakout",
        "platform_window_days",
        "platform_range_pct",
        "distance_to_platform_high_pct",
        "distance_to_ma20",
        "drawdown_15d",
        "consolidation_days",
        "consolidation_range_pct",
        "consolidation_volume_ratio",
        "volume_ratio_20",
        "reason",
    ]
    available = [column for column in preferred_order if column in exported.columns]
    remaining = [column for column in exported.columns if column not in available]
    return exported.loc[:, available + remaining].sort_values(["pattern_id", "symbol"]).reset_index(drop=True)


def _append_recent_tradingview_scores(
    storage: Storage,
    exported: pd.DataFrame,
    as_of: date,
    lookback_days: int,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    if exported.empty or "symbol" not in exported.columns:
        return exported

    summary, _ = _load_or_build_tradingview_summary(
        storage,
        trade_date=as_of,
        lookback_days=lookback_days,
        symbols=symbols,
    )
    if summary.empty:
        return exported

    rating_date_columns = sorted(column for column in summary.columns if column.startswith("all_rating_20"))
    if len(rating_date_columns) != lookback_days:
        return exported

    merge_columns = ["symbol", *rating_date_columns, "avg_all_rating_5d", "all_rating_label"]
    tradingview = summary.loc[:, [column for column in merge_columns if column in summary.columns]].copy()
    tradingview = _dedupe_symbol_report_rows(tradingview)
    tradingview = tradingview.rename(
        columns={
            **{column: f"tradingview_{column}" for column in rating_date_columns},
            "avg_all_rating_5d": "tradingview_avg_all_rating_5d",
            "all_rating_label": "tradingview_all_rating_label",
        }
    )

    enriched = exported.copy()
    enriched["_normalized_symbol"] = enriched["symbol"].map(_normalize_exported_symbol)
    enriched = enriched.merge(
        tradingview,
        how="left",
        left_on="_normalized_symbol",
        right_on="symbol",
    )
    enriched = enriched.drop(columns=["_normalized_symbol", "symbol_y"], errors="ignore")
    enriched = enriched.rename(columns={"symbol_x": "symbol"})
    return enriched


def _append_recent_macd_summary(
    storage: Storage,
    exported: pd.DataFrame,
    *,
    as_of: date,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
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
    return _merge_symbol_report(exported, summary, merge_columns=merge_columns, bool_columns=[
        "macd_top_divergence_15d",
        "macd_bottom_divergence_15d",
        "bullish_volume_price_divergence_flag",
        "bearish_volume_price_divergence_flag",
    ])


def _append_recent_atr_summary(
    storage: Storage,
    exported: pd.DataFrame,
    *,
    as_of: date,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
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


def _append_recent_trend_universe_summary(
    exported: pd.DataFrame,
    trend_universe: pd.DataFrame,
) -> pd.DataFrame:
    if exported.empty or "symbol" not in exported.columns or trend_universe.empty:
        return exported

    universe = trend_universe.copy()
    rename_map = {
        "trend_score": "trend_universe_score",
    }
    universe = universe.rename(columns=rename_map)
    merge_columns = [
        "symbol",
        "in_trend_universe",
        "trend_universe_score",
        "trend_direction_score",
        "trend_strength_score",
        "trend_quality_score",
        "trend_liquidity_score",
    ]
    enriched = _merge_symbol_report(exported, universe, merge_columns=merge_columns, bool_columns=["in_trend_universe"])
    if "in_trend_universe" in enriched.columns:
        enriched["in_trend_universe"] = enriched["in_trend_universe"].fillna(False).astype(bool)
    return enriched


def _append_recent_trend_summary(
    storage: Storage,
    exported: pd.DataFrame,
    *,
    config,
    as_of: date,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    if exported.empty or "symbol" not in exported.columns:
        return exported

    trend = _load_or_build_trend_summary(storage, config=config, trade_date=as_of, symbols=symbols)
    if trend.empty:
        return exported

    merge_columns = [
        "symbol",
        "signal_type",
        "trend_score",
        "entry_score",
        "trend_base_score",
        "price_action_score",
        "macd_score",
        "buy_score",
        "positive_indicator_count",
        "trigger_reason",
        "buy_reason",
    ]
    return _merge_symbol_report(exported, trend, merge_columns=merge_columns, bool_columns=[])


def _merge_symbol_report(
    exported: pd.DataFrame,
    report: pd.DataFrame,
    *,
    merge_columns: list[str],
    bool_columns: list[str],
) -> pd.DataFrame:
    columns = [column for column in merge_columns if column in report.columns]
    if "symbol" not in columns:
        return exported
    summary = _dedupe_symbol_report_rows(report.loc[:, columns].copy())

    enriched = exported.copy()
    enriched["_normalized_symbol"] = enriched["symbol"].map(_normalize_exported_symbol)
    enriched = enriched.merge(
        summary,
        how="left",
        left_on="_normalized_symbol",
        right_on="symbol",
    )
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


def _load_or_build_macd_summary(
    storage: Storage,
    trade_date: date,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    if symbols:
        return _build_macd_summary(storage, trade_date=trade_date, symbols=symbols)
    default_path = storage.paths.reports_dir / "macd" / f"macd_{trade_date.isoformat()}.csv"
    if default_path.exists():
        return pd.read_csv(default_path)
    return _build_macd_summary(storage, trade_date=trade_date)


def _load_or_build_atr_summary(
    storage: Storage,
    trade_date: date,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    if symbols:
        return _build_atr_summary(storage, trade_date=trade_date, symbols=symbols)
    default_path = storage.paths.reports_dir / "atr" / f"atr_{trade_date.isoformat()}.csv"
    if default_path.exists():
        return normalize_atr_summary_frame(pd.read_csv(default_path))
    return _build_atr_summary(storage, trade_date=trade_date)


def _load_or_build_trend_universe_summary(
    storage: Storage,
    config,
    trade_date: date,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    default_path = storage.paths.reports_dir / "trend_universe" / f"trend_universe_{trade_date.isoformat()}.csv"
    if default_path.exists() and not symbols:
        return pd.read_csv(default_path)
    trend = scan_trend_universe(storage, config, as_of=trade_date, symbols=symbols)
    if symbols and not trend.empty:
        symbol_set = {str(symbol).zfill(6) for symbol in symbols}
        trend = trend[trend["symbol"].astype(str).str.zfill(6).isin(symbol_set)].reset_index(drop=True)
    return trend


def _load_or_build_trend_summary(
    storage: Storage,
    *,
    config,
    trade_date: date,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    default_path = storage.paths.reports_dir / "trend" / f"trend_{trade_date.isoformat()}.csv"
    if default_path.exists() and not symbols:
        return pd.read_csv(default_path)
    trend = scan_indicator_scored_entries(storage, config, trade_date=trade_date)
    if symbols and not trend.empty:
        symbol_set = {str(symbol).zfill(6) for symbol in symbols}
        trend = trend[trend["symbol"].astype(str).str.zfill(6).isin(symbol_set)].reset_index(drop=True)
    return trend


def _load_existing_trend_summary(
    storage: Storage,
    *,
    trade_date: date,
) -> pd.DataFrame:
    default_path = storage.paths.reports_dir / "trend" / f"trend_{trade_date.isoformat()}.csv"
    if default_path.exists():
        return pd.read_csv(default_path)
    return pd.DataFrame()


def _load_or_build_tradingview_summary(
    storage: Storage,
    *,
    trade_date: date,
    lookback_days: int,
    symbols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if symbols:
        return _build_tradingview_snapshots(storage, trade_date=trade_date, lookback_days=lookback_days, symbols=symbols)
    default_path = storage.paths.reports_dir / "tradingview" / f"tradingview_avg5_{trade_date.isoformat()}.csv"
    if lookback_days == 5 and default_path.exists():
        return pd.read_csv(default_path), []
    return _build_tradingview_snapshots(storage, trade_date=trade_date, lookback_days=lookback_days)


def _default_pattern_filename(as_of: date, selected_patterns: list[str]) -> str:
    if len(selected_patterns) == len(STRATEGY_NAMES):
        label = "all"
    else:
        pattern_labels = [PATTERN_LABEL_MAP[item] for item in selected_patterns]
        label = "-".join(pattern_labels)
    return f"patterns_{label}_{as_of.isoformat()}.csv"


def _default_pattern_output_path(storage: Storage, *, as_of: date, selected_patterns: list[str]) -> Path:
    return storage.paths.reports_dir / "patterns" / _default_pattern_filename(as_of, selected_patterns)


def _format_symbol_for_excel(value: object) -> str:
    symbol = str(value).zfill(6)
    return f'="{symbol}"'


def _normalize_exported_symbol(value: object) -> str:
    text = str(value).strip()
    if text.startswith('="') and text.endswith('"'):
        text = text[2:-1]
    return text.zfill(6)


def _build_tradingview_snapshots(
    storage: Storage,
    trade_date: date,
    lookback_days: int,
    symbols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    instruments = _load_instruments(storage, symbols=symbols)
    target_dates = _resolve_recent_trading_dates(storage, as_of=trade_date, lookback_days=lookback_days, symbols=symbols)
    if len(target_dates) != lookback_days:
        return pd.DataFrame(), []

    summary_rows: list[dict[str, object]] = []
    daily_rows: dict[str, list[dict[str, object]]] = {}
    target_set = set(target_dates)
    total_instruments = len(instruments)
    logging.info(
        "TradingView scan started for %s: %s symbols, %s recent trading days",
        trade_date.isoformat(),
        total_instruments,
        lookback_days,
    )

    for index, instrument in enumerate(instruments, start=1):
        symbol = str(instrument["symbol"])
        try:
            bars = storage.load_daily_bars(symbol)
        except FileNotFoundError:
            _log_scan_progress("TradingView", index, total_instruments)
            continue

        feature_frame = build_feature_frame(bars)
        feature_frame["trade_date"] = pd.to_datetime(feature_frame["trade_date"])
        recent = feature_frame[feature_frame["trade_date"].dt.date.isin(target_set)].dropna(subset=["all_rating"]).copy()
        available_dates = sorted(recent["trade_date"].dt.date.tolist())
        if available_dates != target_dates:
            _log_scan_progress("TradingView", index, total_instruments)
            continue

        recent = recent.sort_values("trade_date").reset_index(drop=True)
        all_ratings = [float(value) for value in recent["all_rating"].tolist()]
        latest = recent.iloc[-1]
        summary_row: dict[str, object] = {
            "trade_date": pd.Timestamp(latest["trade_date"]).date().isoformat(),
            "symbol": _format_symbol_for_excel(symbol),
            "name": instrument["name"],
            "avg_all_rating_5d": sum(all_ratings) / lookback_days,
            "ma_rating": float(latest["ma_rating"]) if pd.notna(latest.get("ma_rating")) else None,
            "osc_rating": float(latest["osc_rating"]) if pd.notna(latest.get("osc_rating")) else None,
            "all_rating": float(latest["all_rating"]) if pd.notna(latest.get("all_rating")) else None,
            "ma_rating_label": latest.get("ma_rating_label"),
            "osc_rating_label": latest.get("osc_rating_label"),
            "all_rating_label": latest.get("all_rating_label"),
        }
        for _, row in recent.iterrows():
            row_trade_date = pd.Timestamp(row["trade_date"]).date()
            summary_row[f"all_rating_{row_trade_date.isoformat()}"] = float(row["all_rating"])
        summary_rows.append(summary_row)

        for _, row in recent.iterrows():
            row_trade_date = pd.Timestamp(row["trade_date"]).date().isoformat()
            daily_rows.setdefault(row_trade_date, []).append(
                {
                    "trade_date": row_trade_date,
                    "symbol": _format_symbol_for_excel(symbol),
                    "name": instrument["name"],
                    "ma_rating": float(row["ma_rating"]) if pd.notna(row.get("ma_rating")) else None,
                    "osc_rating": float(row["osc_rating"]) if pd.notna(row.get("osc_rating")) else None,
                    "all_rating": float(row["all_rating"]) if pd.notna(row.get("all_rating")) else None,
                    "ma_rating_label": row.get("ma_rating_label"),
                    "osc_rating_label": row.get("osc_rating_label"),
                    "all_rating_label": row.get("all_rating_label"),
                }
            )
        _log_scan_progress("TradingView", index, total_instruments)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return summary, []
    summary = summary.sort_values(["avg_all_rating_5d", "all_rating", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
    rating_date_columns = sorted(column for column in summary.columns if column.startswith("all_rating_20"))
    ordered_columns = [
        column
        for column in (
            "symbol",
            "name",
            *rating_date_columns,
            "avg_all_rating_5d",
            "ma_rating",
            "osc_rating",
            "all_rating",
            "ma_rating_label",
            "osc_rating_label",
            "all_rating_label",
            "trade_date",
        )
        if column in summary.columns
    ]
    remaining_columns = [column for column in summary.columns if column not in ordered_columns]
    summary = summary.loc[:, ordered_columns + remaining_columns]

    ordered_daily_dates = [item_date.isoformat() for item_date in target_dates]
    daily_frames = [
        {
            "trade_date": item_date,
            "data": pd.DataFrame(daily_rows.get(item_date, [])).sort_values(["all_rating", "ma_rating", "symbol"], ascending=[False, False, True]).reset_index(drop=True),
        }
        for item_date in ordered_daily_dates
        if daily_rows.get(item_date)
    ]
    return summary, daily_frames


def _build_macd_summary(
    storage: Storage,
    trade_date: date,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    tradingview_summary, _ = _load_or_build_tradingview_summary(
        storage,
        trade_date=trade_date,
        lookback_days=5,
        symbols=symbols,
    )
    if tradingview_summary.empty:
        return pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    total_rows = len(tradingview_summary)
    logging.info("MACD summary scan started for %s: %s symbols", trade_date.isoformat(), total_rows)
    for index, (_, row) in enumerate(tradingview_summary.iterrows(), start=1):
        symbol = _normalize_exported_symbol(row["symbol"])
        try:
            bars = storage.load_daily_bars(symbol)
        except FileNotFoundError:
            _log_scan_progress("MACD", index, total_rows)
            continue

        cutoff = bars[pd.to_datetime(bars["trade_date"]).dt.date <= trade_date].reset_index(drop=True)
        if cutoff.empty:
            _log_scan_progress("MACD", index, total_rows)
            continue

        macd_frame = _prepare_daily_macd_frame(cutoff)
        latest = macd_frame.iloc[-1]
        divergence_row = summarize_recent_macd_divergence(macd_frame)
        cross_state = _describe_macd_cross_state(macd_frame)
        bullish_volume_divergence, bearish_volume_divergence = _detect_daily_volume_price_divergence(macd_frame)
        summary_rows.append(
            {
                "trade_date": trade_date.isoformat(),
                "symbol": _format_symbol_for_excel(symbol),
                "name": row.get("name", ""),
                "macd": _safe_float_or_none(latest.get("macd")),
                "macd_signal_line": _safe_float_or_none(latest.get("macd_signal_line")),
                "macd_hist": _safe_float_or_none(latest.get("macd_hist")),
                "macd_cross_state": cross_state,
                "macd_divergence_state": _describe_macd_divergence_state(divergence_row),
                "volume_price_divergence_state": _describe_volume_price_divergence_state(
                    bullish_volume_divergence,
                    bearish_volume_divergence,
                ),
                "macd_top_divergence_15d": bool(divergence_row.get("macd_top_divergence_15d", False)),
                "macd_bottom_divergence_15d": bool(divergence_row.get("macd_bottom_divergence_15d", False)),
                "macd_top_divergence_signal_date": divergence_row.get("macd_top_divergence_signal_date"),
                "macd_bottom_divergence_signal_date": divergence_row.get("macd_bottom_divergence_signal_date"),
                "bullish_volume_price_divergence_flag": bool(bullish_volume_divergence),
                "bearish_volume_price_divergence_flag": bool(bearish_volume_divergence),
            }
        )
        _log_scan_progress("MACD", index, total_rows)

    if not summary_rows:
        return pd.DataFrame()

    summary = pd.DataFrame(summary_rows)
    for column in (
        "macd_top_divergence_15d",
        "macd_bottom_divergence_15d",
        "bullish_volume_price_divergence_flag",
        "bearish_volume_price_divergence_flag",
    ):
        if column in summary.columns:
            summary[column] = summary[column].fillna(False).astype(bool)
    return summary


def _build_atr_summary(
    storage: Storage,
    trade_date: date,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    tradingview_summary, _ = _load_or_build_tradingview_summary(
        storage,
        trade_date=trade_date,
        lookback_days=5,
        symbols=symbols,
    )
    if tradingview_summary.empty:
        return pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    total_rows = len(tradingview_summary)
    logging.info("ATR summary scan started for %s: %s symbols", trade_date.isoformat(), total_rows)
    for index, (_, row) in enumerate(tradingview_summary.iterrows(), start=1):
        symbol = _normalize_exported_symbol(row["symbol"])
        try:
            bars = storage.load_daily_bars(symbol)
        except FileNotFoundError:
            _log_scan_progress("ATR", index, total_rows)
            continue

        cutoff = bars[pd.to_datetime(bars["trade_date"]).dt.date <= trade_date].reset_index(drop=True)
        if cutoff.empty:
            _log_scan_progress("ATR", index, total_rows)
            continue

        snapshot = build_atr_snapshot_row(
            cutoff,
            symbol=_format_symbol_for_excel(symbol),
            name=str(row.get("name", "")),
            trade_date=trade_date,
        )
        if snapshot is not None:
            summary_rows.append(snapshot)
        _log_scan_progress("ATR", index, total_rows)

    if not summary_rows:
        return pd.DataFrame()
    return pd.DataFrame(summary_rows)


def _prepare_daily_macd_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy().sort_values("trade_date").reset_index(drop=True)
    required = {"macd_dif", "macd_dea", "macd_hist"}
    if not required.issubset(frame.columns):
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
        if float(prev_row["macd"]) <= float(prev_row["macd_signal_line"]) and float(current_row["macd"]) > float(
            current_row["macd_signal_line"]
        ):
            recent_cross_up = True
        if float(prev_row["macd"]) >= float(prev_row["macd_signal_line"]) and float(current_row["macd"]) < float(
            current_row["macd_signal_line"]
        ):
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


def _safe_float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _log_scan_progress(stage_name: str, current: int, total: int) -> None:
    if total <= 0:
        return
    if current == 1 or current == total or current % PROGRESS_LOG_INTERVAL == 0:
        logging.info("%s progress: %s/%s", stage_name, current, total)


def _resolve_recent_trading_dates(
    storage: Storage,
    as_of: date,
    lookback_days: int,
    symbols: list[str] | None = None,
) -> list[date]:
    candidate_dates: set[date] = set()
    for instrument in _load_instruments(storage, symbols=symbols):
        symbol = str(instrument["symbol"])
        try:
            bars = storage.load_daily_bars(symbol)
        except FileNotFoundError:
            continue
        trade_dates = pd.to_datetime(bars["trade_date"]).dt.date
        recent_dates = sorted(item for item in trade_dates.unique() if item <= as_of)
        candidate_dates.update(recent_dates[-lookback_days:])

    return sorted(candidate_dates)[-lookback_days:]


def _load_instruments(storage: Storage, symbols: list[str] | None = None) -> list[dict[str, object]]:
    universe = storage.load_universe().copy()
    universe["symbol"] = universe["symbol"].astype(str).str.zfill(6)
    if symbols:
        symbol_set = {str(symbol).zfill(6) for symbol in symbols}
        universe = universe[universe["symbol"].isin(symbol_set)].reset_index(drop=True)
    return universe.to_dict("records")


def _plot_pattern_matches(storage: Storage, config, as_of: date, results: pd.DataFrame) -> Path:
    plots_dir = storage.paths.reports_dir / "plots" / as_of.isoformat()
    plots_dir.mkdir(parents=True, exist_ok=True)
    if results.empty:
        return plots_dir

    grouped = results.groupby("symbol", sort=True)
    start_date = (as_of - timedelta(days=365 * 2)).strftime("%Y%m%d")
    end_date = as_of.strftime("%Y%m%d")

    for symbol, frame in grouped:
        normalized_symbol = str(symbol).zfill(6)
        try:
            daily_bars = storage.load_daily_bars(normalized_symbol)
        except FileNotFoundError:
            logging.warning("Skip plot for %s because local daily bars are missing", normalized_symbol)
            continue
        filtered = filter_by_date(daily_bars, start_date, end_date)
        if filtered.empty:
            continue

        pattern_ids = "-".join(sorted({PATTERN_LABEL_MAP[str(item)] for item in frame["strategy_name"].tolist()}))
        output_path = plots_dir / f"{normalized_symbol}_pattern_{pattern_ids}.png"
        plot_candles_and_volume(filtered, normalized_symbol, output_path)

    return plots_dir


def _parse_optional_date(value: str | None) -> date | None:
    if value is None:
        return None
    return datetime.fromisoformat(value).date()


def _resolve_probability_split_dates(
    dataset: pd.DataFrame,
    train_end: date | None,
    valid_end: date | None,
    test_end: date | None,
) -> tuple[date, date, date]:
    if train_end and valid_end and test_end:
        return train_end, valid_end, test_end
    return infer_split_dates(dataset)


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
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip()


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
        cached = storage.load_universe()
    except FileNotFoundError:
        raise RuntimeError("Universe refresh returned no symbols and no cached universe is available.")

    logging.warning("Universe refresh returned no symbols, fallback to cached universe.")
    return cached


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
