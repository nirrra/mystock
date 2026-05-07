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
from .event_risk_ranker import (
    EVENT_RISK_RANKER_VERSION,
    build_event_risk_ranker_dataset,
    event_risk_ranker_predictions_path,
    event_risk_ranker_report_dir,
    format_event_ranker_prediction_table,
    predict_event_risk_ranker,
    train_event_risk_ranker_model,
    validate_event_risk_ranker_walkforward,
)
from .event_labels import EventLabelConfig
from .full_market_panel import audit_full_market_data, format_full_market_audit_summary
from .full_market_risk import reproduce_tail_risk
from .macd_divergence import summarize_recent_macd_divergence
from .features import build_feature_frame
from .indicators import add_indicators
from .intraday_ranking import save_intraday_rankings
from .models import NetworkConfig
from .pattern_backtest import (
    build_pattern_forward_price_frame,
    sample_pattern_backtest_trade_dates,
    scan_pattern_backtest_signals,
    summarize_pattern_backtest,
)
from .pattern_stop_research import (
    DEFAULT_HOLDING_DAYS,
    DEFAULT_STOP_LOSSES,
    DEFAULT_TAKE_PROFITS,
    research_pattern_stop_grid,
    select_best_pattern_stop_grid,
)
from .paths import ProjectPaths
from .predict_model import (
    load_predict_model_predictions,
    save_predict_model_predictions,
)
from .reporting import format_multi_pattern_summary, format_report
from .screener import Screener, parse_as_of
from .stacked_trade_value import (
    candidate_ranker_report_dir,
    format_candidate_ranker_prediction_table,
    format_metric_table,
    format_opportunity_ranker_prediction_table,
    format_volume_price_fusion_prediction_table,
    model_walkforward_report_dir,
    opportunity_ranker_report_dir,
    predict_candidate_ranker,
    predict_opportunity_ranker,
    predict_volume_price_fusion,
    train_candidate_ranker_model,
    train_opportunity_ranker_model,
    train_volume_price_fusion_model,
    validate_model_walkforward,
    volume_price_fusion_report_dir,
)
from .storage import DailyBarsReadError, Storage
from .strategies import STRATEGY_NAMES
from .trend_backtest import backtest_portfolios, backtest_signal_returns, summarize_signal_backtest
from .trend_indicator_scores import build_next_open_entries, scan_indicator_scored_entries, select_tradable_entries
from .trend_reporting import (
    save_atr_report,
    save_entry_backtest_reports,
    save_entry_portfolio_backtest_reports,
    save_macd_report,
    save_pattern_backtest_reports,
    save_pattern_stop_research_reports,
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
from .tradingview_factor_research import (
    DEFAULT_FACTOR_FIELDS,
    DEFAULT_HORIZONS,
    DEFAULT_RANK_FIELDS,
    DEFAULT_TOP_N,
    run_tradingview_factor_research,
    save_tradingview_factor_research_reports,
)
from .trend_universe import scan_trend_universe
from .universe import build_main_board_universe
from .watchlist import (
    build_watchlist_candidates_from_patterns,
    build_daily_watchlist_candidates,
    build_watchlist_candidates_from_trend,
    extract_watchlist_symbols,
    find_latest_watchlist_before,
    load_watchlist,
    watchlist_path as build_watchlist_path,
    write_watchlist,
)


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
        description="A 股主板技术分析命令行工具",
        epilog=(
            "第一次使用建议顺序：\n"
            "  1. mystock update --start-date 20240101\n"
            "     更新主板股票池并拉取本地日线数据。\n"
            "  2. mystock pattern\n"
            "     扫描本地全部股票，识别 1 到 6 号模式并生成 CSV。\n"
            "  3. mystock predict-model --date 2026-04-10\n"
            "     生成当前主版本模型预测文件。\n\n"
            "常见示例：\n"
            "  mystock update 603588 --start-date 20240101\n"
            "  mystock pattern --1 --4\n"
            "  mystock report --date 2026-04-10\n"
            "  mystock tradingview --date 2026-04-10\n"
            "  mystock macd --date 2026-04-10\n"
            "  mystock atr --date 2026-04-10\n"
            "  mystock predict-model --date 2026-04-10\n"
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
    update.add_argument(
        "--data-interface",
        choices=["baostock", "sina", "eastmoney"],
        default="sina",
        help="日线数据接口：baostock/sina/eastmoney，默认 sina",
    )
    update.add_argument(
        "--index-symbols",
        default=",".join(DEFAULT_INDEX_SYMBOLS),
        help="批量更新股票后同步更新的指数代码，逗号分隔；默认沪深300/中证500/中证800等",
    )
    update.add_argument("--skip-index", action="store_true", help="批量 update 时跳过指数日线更新")
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

    audit_full_market = subparsers.add_parser(
        "audit-full-market-data",
        help="审计全市场日线数据是否足够支持风险/收益模型复现",
        description="读取本地 data/daily parquet，统计每只股票历史长度、OHLCV 缺失、涨跌停可识别情况和复现资格。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    audit_full_market.add_argument("--limit", type=int, default=None, help="仅审计前 N 只股票，便于快速测试")
    audit_full_market.add_argument("--min-exact-history-days", type=int, default=900, help="严格复现所需最少交易日，默认 900")
    audit_full_market.add_argument("--tail-lookback-days", type=int, default=100, help="尾部风险标签滚动窗口，默认 100")
    audit_full_market.add_argument("--max-horizon-days", type=int, default=20, help="最长 forward horizon，默认 20")
    audit_full_market.add_argument("--output-dir", default=None, help="可选输出目录，默认 reports/full_market_model")

    reproduce_tail = subparsers.add_parser(
        "reproduce-tail-risk",
        help="复现日线尾部下跌风险分类模型",
        description="按 Noh 2026 的滚动 100 日 5% 分位风险标签构建全市场日线风险分类基线。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    reproduce_tail.add_argument("--start-date", default=None, help="样本开始日期，格式 YYYY-MM-DD")
    reproduce_tail.add_argument("--end-date", default=None, help="样本结束日期，格式 YYYY-MM-DD")
    reproduce_tail.add_argument("--train-end", required=True, help="训练集结束日期，格式 YYYY-MM-DD")
    reproduce_tail.add_argument("--valid-end", required=True, help="验证集结束日期，格式 YYYY-MM-DD")
    reproduce_tail.add_argument("--limit", type=int, default=None, help="仅使用前 N 只股票，便于快速测试")
    reproduce_tail.add_argument("--lookback-days", type=int, default=100, help="滚动分位窗口，默认 100")
    reproduce_tail.add_argument("--quantile", type=float, default=0.05, help="尾部风险分位，默认 0.05")
    reproduce_tail.add_argument("--horizon-days", type=int, default=1, help="预测未来第 N 个交易日风险，默认 1")
    reproduce_tail.add_argument("--min-training-rows", type=int, default=200, help="最少训练样本行数，默认 200")
    reproduce_tail.add_argument("--allow-short-sample", action="store_true", help="允许短样本 smoke test；正式复现不要使用")

    train_opportunity = subparsers.add_parser(
        "train-opportunity-ranker",
        help="训练 V4.2 日期机会过滤 + 条件收益排序模型",
        description=(
            "复用 V4 风险过滤器，先训练日期级 opportunity gate 允许 no-trade，"
            "再只在历史好机会日训练低风险池内的条件排序器。"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    train_opportunity.add_argument("--start-date", default=None, help="样本开始日期，格式 YYYY-MM-DD")
    train_opportunity.add_argument("--end-date", default=None, help="样本结束日期，格式 YYYY-MM-DD")
    train_opportunity.add_argument("--train-end", default=None, help="训练集结束日期，格式 YYYY-MM-DD")
    train_opportunity.add_argument("--valid-end", default=None, help="验证集结束日期，格式 YYYY-MM-DD")
    train_opportunity.add_argument("--test-end", default=None, help="测试集结束日期，格式 YYYY-MM-DD")
    train_opportunity.add_argument("--limit", type=int, default=None, help="仅使用前 N 只股票，便于快速测试")
    train_opportunity.add_argument("--max-iter", type=int, default=80, help="每个模型的最大迭代轮数")
    train_opportunity.add_argument("--top-n", default="20,50", help="评估 TopN 列表，逗号分隔，默认 20,50")
    train_opportunity.add_argument("--predict-date", default=None, help="训练后顺便生成该日期的预测排序，格式 YYYY-MM-DD")

    predict_opportunity = subparsers.add_parser(
        "predict-opportunity-ranker",
        help="生成 V4.2 opportunity-gated 排序结果",
        description="读取已训练的 V4.2 opportunity ranker，对指定日期先判断是否交易，再生成低风险池排序。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    predict_opportunity.add_argument("--date", required=True, help="预测日期，格式 YYYY-MM-DD")
    predict_opportunity.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")
    predict_opportunity.add_argument("--output", default=None, help="可选的预测结果输出路径")
    predict_opportunity.add_argument(
        "--rank-source",
        choices=["v42", "v4"],
        default="v42",
        help="排序来源：v42 使用条件排序器；v4 使用原 long_upside_score，默认 v42",
    )

    train_v5 = subparsers.add_parser(
        "train-volume-price-fusion",
        help="训练 V5 量价融合模型",
        description="在当前 V4.2 hybrid 基座上训练量价风险、量价质量和 V5 融合排序模型。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    train_v5.add_argument("--start-date", default=None, help="样本开始日期，格式 YYYY-MM-DD")
    train_v5.add_argument("--end-date", default=None, help="样本结束日期，格式 YYYY-MM-DD")
    train_v5.add_argument("--train-end", default=None, help="训练集结束日期，格式 YYYY-MM-DD")
    train_v5.add_argument("--valid-end", default=None, help="验证集结束日期，格式 YYYY-MM-DD")
    train_v5.add_argument("--test-end", default=None, help="测试集结束日期，格式 YYYY-MM-DD")
    train_v5.add_argument("--limit", type=int, default=None, help="仅使用前 N 只股票，便于快速测试")
    train_v5.add_argument("--max-iter", type=int, default=80, help="每个模型的最大迭代轮数")
    train_v5.add_argument("--top-n", default="20,50", help="评估 TopN 列表，逗号分隔，默认 20,50")
    train_v5.add_argument("--predict-date", default=None, help="训练后顺便生成该日期的预测排序，格式 YYYY-MM-DD")
    train_v5.add_argument(
        "--reuse-base-artifact",
        action="store_true",
        help="复用已训练 V4.2 hybrid 基座，只训练 V5 量价子模型和融合层；用于快速评估",
    )

    predict_v5 = subparsers.add_parser(
        "predict-volume-price-fusion",
        help="生成 V5 量价融合模型预测结果",
        description="读取已训练的 V5 量价融合模型，对指定日期生成排序和量价解释字段。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    predict_v5.add_argument("--date", required=True, help="预测日期，格式 YYYY-MM-DD")
    predict_v5.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")
    predict_v5.add_argument("--output", default=None, help="可选的预测结果输出路径")

    train_v51 = subparsers.add_parser(
        "train-candidate-ranker",
        aliases=["train-v51-candidate-ranker"],
        help="训练 V5.1 候选池内排序模型",
        description="读取已训练 V5 量价融合 artifact，只在通过机会日和风险过滤的候选池内训练横截面排序模型。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    train_v51.add_argument("--start-date", default=None, help="样本开始日期，格式 YYYY-MM-DD")
    train_v51.add_argument("--end-date", default=None, help="样本结束日期，格式 YYYY-MM-DD")
    train_v51.add_argument("--train-end", default=None, help="训练集结束日期，格式 YYYY-MM-DD")
    train_v51.add_argument("--valid-end", default=None, help="验证集结束日期，格式 YYYY-MM-DD")
    train_v51.add_argument("--test-end", default=None, help="测试集结束日期，格式 YYYY-MM-DD")
    train_v51.add_argument("--limit", type=int, default=None, help="仅使用前 N 只股票，便于快速测试")
    train_v51.add_argument("--max-iter", type=int, default=80, help="排序模型最大迭代轮数")
    train_v51.add_argument("--top-n", default="20,50", help="评估 TopN 列表，逗号分隔，默认 20,50")
    train_v51.add_argument("--predict-date", default=None, help="训练后顺便生成该日期的预测排序，格式 YYYY-MM-DD")

    walkforward = subparsers.add_parser(
        "validate-model-walkforward",
        aliases=["walkforward-validate-model"],
        help="对当前主线模型做轻量 walk-forward 泛化验证",
        description="按时间顺序生成多个 train/valid/test 窗口，重复训练并评估 V4.2 hybrid 主线模型的泛化稳定性。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    walkforward.add_argument("--model", choices=["v42"], default="v42", help="验证模型，第一版仅支持 v42")
    walkforward.add_argument("--start-date", default=None, help="样本开始日期，格式 YYYY-MM-DD")
    walkforward.add_argument("--end-date", default=None, help="样本结束日期，格式 YYYY-MM-DD")
    walkforward.add_argument("--limit", type=int, default=None, help="仅使用前 N 只股票，便于快速测试")
    walkforward.add_argument("--windows", type=int, default=8, help="滚动窗口数量，默认 8")
    walkforward.add_argument("--train-days", type=int, default=280, help="每个窗口训练交易日数量，默认 280")
    walkforward.add_argument("--valid-days", type=int, default=60, help="每个窗口验证交易日数量，默认 60")
    walkforward.add_argument("--test-days", type=int, default=60, help="每个窗口测试交易日数量，默认 60")
    walkforward.add_argument("--min-train-days", type=int, default=220, help="允许缩短后的最小训练交易日数量，默认 220")
    walkforward.add_argument("--max-iter", type=int, default=40, help="每个模型的最大迭代轮数，默认 40")
    walkforward.add_argument("--top-n", default="20,50", help="评估 TopN 列表，逗号分隔，默认 20,50")

    predict_v51 = subparsers.add_parser(
        "predict-candidate-ranker",
        aliases=["predict-v51-candidate-ranker"],
        help="生成 V5.1 候选池内排序预测结果",
        description="读取已训练 V5.1 候选池内排序模型，对指定日期生成排序和解释字段。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    predict_v51.add_argument("--date", required=True, help="预测日期，格式 YYYY-MM-DD")
    predict_v51.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")
    predict_v51.add_argument("--output", default=None, help="可选的预测结果输出路径")

    predict_model = subparsers.add_parser(
        "predict-model",
        aliases=["predict_model"],
        help="生成每日通用模型预测结果",
        description="当前使用 V4.2 opportunity gate + V4 long-upside 排序生成每日预测文件，供 daily-screening 与 watchlist 复用。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    predict_model.add_argument("--date", required=True, help="预测日期，格式 YYYY-MM-DD")
    predict_model.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")
    predict_model.add_argument("--output", default=None, help="可选的预测结果输出路径")

    build_event_labels = subparsers.add_parser(
        "build-event-labels",
        help="构建 pattern 事件 triple-barrier 标签",
        description="扫描历史 pattern 事件，按 ATR 动态止盈止损构建 event_risk_ranker 训练标签。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    build_event_labels.add_argument("--start-date", required=True, help="样本开始日期，格式 YYYY-MM-DD")
    build_event_labels.add_argument("--end-date", required=True, help="样本结束日期，格式 YYYY-MM-DD")
    build_event_labels.add_argument("--limit", type=int, default=None, help="仅使用前 N 只股票，便于快速测试")
    build_event_labels.add_argument("--stop-atr", type=float, default=1.2, help="止损 ATR 倍数，默认 1.2")
    build_event_labels.add_argument("--take-atr", type=float, default=2.5, help="止盈 ATR 倍数，默认 2.5")
    build_event_labels.add_argument("--max-holding-days", type=int, default=20, help="最长持有交易日，默认 20")

    train_event = subparsers.add_parser(
        "train-event-risk-ranker",
        help="训练 pattern 事件风险筛查与 R 倍数排序模型",
        description="完全独立于 TradingView 和旧 predict-model，基于 pattern 事件、triple-barrier 标签和 R 倍数排序训练。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    train_event.add_argument("--start-date", default="2022-01-01", help="样本开始日期，格式 YYYY-MM-DD")
    train_event.add_argument("--end-date", default=None, help="样本结束日期，格式 YYYY-MM-DD，默认今天")
    train_event.add_argument("--limit", type=int, default=None, help="仅使用前 N 只股票，便于快速测试")
    train_event.add_argument("--max-iter", type=int, default=80, help="HGB 模型最大迭代轮数")
    train_event.add_argument("--top-n", default="10,20", help="评估 TopN 列表，逗号分隔，默认 10,20")
    train_event.add_argument("--stop-atr-grid", default="1.0,1.2,1.5", help="止损 ATR 网格")
    train_event.add_argument("--take-atr-grid", default="2.0,2.5,3.0", help="止盈 ATR 网格")
    train_event.add_argument("--holding-days-grid", default="10,20,40", help="最长持有交易日网格")
    train_event.add_argument("--predict-date", default=None, help="训练后顺便生成该日期预测，格式 YYYY-MM-DD")

    predict_event = subparsers.add_parser(
        "predict-event-risk-ranker",
        help="生成 event_risk_ranker 每日预测和 watchlist_event",
        description="读取已训练 event_risk_ranker artifact，对指定日期 pattern 事件生成风险/R 排序结果。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    predict_event.add_argument("--date", required=True, help="预测日期，格式 YYYY-MM-DD")
    predict_event.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")
    predict_event.add_argument("--output", default=None, help="可选的预测 CSV 输出路径")

    validate_event = subparsers.add_parser(
        "validate-event-risk-ranker",
        help="walk-forward 验证 event_risk_ranker",
        description="按时间窗口训练并测试 event_risk_ranker，输出 R 倍数 TopN 指标。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    validate_event.add_argument("--start-date", default="2022-01-01", help="样本开始日期，格式 YYYY-MM-DD")
    validate_event.add_argument("--end-date", default=None, help="样本结束日期，格式 YYYY-MM-DD，默认今天")
    validate_event.add_argument("--limit", type=int, default=None, help="仅使用前 N 只股票，便于快速测试")
    validate_event.add_argument("--windows", type=int, default=8, help="滚动窗口数量，默认 8")
    validate_event.add_argument("--train-days", type=int, default=280, help="每个窗口训练交易日数量")
    validate_event.add_argument("--valid-days", type=int, default=60, help="每个窗口验证交易日数量")
    validate_event.add_argument("--test-days", type=int, default=60, help="每个窗口测试交易日数量")
    validate_event.add_argument("--max-iter", type=int, default=40, help="HGB 模型最大迭代轮数")
    validate_event.add_argument("--top-n", default="10,20", help="评估 TopN 列表，逗号分隔，默认 10,20")
    validate_event.add_argument("--stop-atr-grid", default="1.0,1.2,1.5", help="止损 ATR 网格")
    validate_event.add_argument("--take-atr-grid", default="2.0,2.5,3.0", help="止盈 ATR 网格")
    validate_event.add_argument("--holding-days-grid", default="10,20,40", help="最长持有交易日网格")

    daily_screening = subparsers.add_parser(
        "daily-screening",
        help="按交易日执行每日筛选，并生成当日 watchlist",
        description="自动判断是否为交易日，串行执行 update/tradingview/predict-model/macd/atr/pattern，再生成当日 watchlist。",
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

    backtest_patterns = subparsers.add_parser(
        "backtest-patterns",
        help="运行六种模式的纯模式胜率回测",
        description="扫描历史模式 1 到 6 命中，按同股同模式 5 个交易日冷却去重后执行次日开盘固定持有回测。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    backtest_patterns.add_argument("--date", required=True, help="回测截止日期，格式 YYYY-MM-DD")
    backtest_patterns.add_argument("--start-date", required=True, help="回测开始日期，格式 YYYY-MM-DD")
    backtest_patterns.add_argument("--1", dest="pattern1", action="store_true", help="只回测模式 1")
    backtest_patterns.add_argument("--2", dest="pattern2", action="store_true", help="只回测模式 2")
    backtest_patterns.add_argument("--3", dest="pattern3", action="store_true", help="只回测模式 3")
    backtest_patterns.add_argument("--4", dest="pattern4", action="store_true", help="只回测模式 4")
    backtest_patterns.add_argument("--5", dest="pattern5", action="store_true", help="只回测模式 5")
    backtest_patterns.add_argument("--6", dest="pattern6", action="store_true", help="只回测模式 6")
    backtest_patterns.add_argument("--output", default=None, help="可选的回测明细 CSV 输出路径")
    backtest_patterns.add_argument("--sample-dates", type=int, default=None, help="随机抽样的历史信号日数量；不传则全量扫描")
    backtest_patterns.add_argument("--sample-seed", type=int, default=42, help="抽样随机种子，默认 42")
    backtest_patterns.add_argument("--save-forward-prices", action="store_true", help="保存每个买入点后续每日价格")
    backtest_patterns.add_argument("--forward-days", type=int, default=40, help="保存买入后多少个交易日的每日价格，默认 40")
    backtest_patterns.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")

    research_pattern_stops = subparsers.add_parser(
        "research-pattern-stops",
        help="研究模式回测样本的止盈止损网格",
        description="读取 pattern_forward_prices CSV，按模式和持有周期回测止盈止损组合。默认覆盖 5/10/20/40 周期。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    research_pattern_stops.add_argument("--input", required=True, help="pattern_forward_prices_YYYY-MM-DD.csv 路径")
    research_pattern_stops.add_argument("--date", default=None, help="报告日期，格式 YYYY-MM-DD；不传则尝试从输入文件名推断")
    research_pattern_stops.add_argument(
        "--holding-days",
        default=",".join(str(item) for item in DEFAULT_HOLDING_DAYS),
        help="逗号分隔的持有周期，默认 5,10,20,40",
    )
    research_pattern_stops.add_argument(
        "--take-profits",
        default=",".join(f"{item:.2f}" for item in DEFAULT_TAKE_PROFITS),
        help="逗号分隔的止盈比例，默认 0.04,0.06,0.08,0.10,0.12,0.15,0.20,0.25",
    )
    research_pattern_stops.add_argument(
        "--stop-losses",
        default=",".join(f"{item:.2f}" for item in DEFAULT_STOP_LOSSES),
        help="逗号分隔的止损比例，默认 0.03,0.05,0.07,0.10,0.12",
    )
    research_pattern_stops.add_argument(
        "--same-day-policy",
        choices=["stop-first", "take-profit-first"],
        default="stop-first",
        help="同一天同时触发止盈止损时的处理，默认 stop-first",
    )
    research_pattern_stops.add_argument("--ma20-stop", action="store_true", help="启用收盘跌破 MA20 止损")
    research_pattern_stops.add_argument(
        "--ma20-stop-tolerance",
        type=float,
        default=0.0,
        help="MA20 止损容忍度，0.00 表示收盘跌破 MA20 即止损，0.01 表示低于 MA20*0.99 才止损",
    )
    research_pattern_stops.add_argument("--min-samples", type=int, default=30, help="best 表挑选组合时的最小样本数，默认 30")
    research_pattern_stops.add_argument("--output", default=None, help="可选的完整网格 CSV 输出路径")
    research_pattern_stops.add_argument("--top-n", type=int, default=30, help="终端展示前 N 行")

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

    research_tradingview_factor = subparsers.add_parser(
        "research-tradingview-factor",
        help="验证 TradingView 分数的单因子有效性",
        description=(
            "基于本地日线回算 TradingView 风格评分，按 t+1 开盘买入评估未来收益，"
            "并输出分组、Rank IC、标签表现和每日 TopN 后续表现。"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    research_tradingview_factor.add_argument("--start-date", required=True, help="研究开始日期，格式 YYYY-MM-DD")
    research_tradingview_factor.add_argument("--end-date", required=True, help="研究结束日期，格式 YYYY-MM-DD")
    research_tradingview_factor.add_argument(
        "--horizons",
        default=",".join(str(item) for item in DEFAULT_HORIZONS),
        help="逗号分隔的观察周期，默认 1,5,10,20",
    )
    research_tradingview_factor.add_argument(
        "--factor-fields",
        default=",".join(DEFAULT_FACTOR_FIELDS),
        help="逗号分隔的单因子字段，默认 all_rating,avg_all_rating_5d,ma_rating,osc_rating",
    )
    research_tradingview_factor.add_argument(
        "--rank-fields",
        default=",".join(DEFAULT_RANK_FIELDS),
        help="逗号分隔的 TopN 排序字段，默认 all_rating,avg_all_rating_5d",
    )
    research_tradingview_factor.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="每日最高分股票数量，默认 10")
    research_tradingview_factor.add_argument("--quantiles", type=int, default=5, help="每日横截面分组数量，默认 5")
    research_tradingview_factor.add_argument(
        "--symbols",
        default=None,
        help="可选的逗号分隔股票代码列表，用于小样本验证，例如 600000,000001",
    )

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
                update_indexes=not args.skip_index,
            )
        finally:
            provider.close()
        return

    if args.command == "pattern":
        selected = _selected_patterns(args)
        as_of = parse_as_of(args.as_of)
        _run_pattern(storage, config.provider, config, as_of, selected, args.limit, args.output)
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

    if args.command == "audit-full-market-data":
        output_dir = Path(args.output_dir).resolve() if args.output_dir else None
        result = audit_full_market_data(
            storage=storage,
            project_root=project_root,
            limit=args.limit,
            min_exact_history_days=args.min_exact_history_days,
            tail_lookback_days=args.tail_lookback_days,
            max_horizon_days=args.max_horizon_days,
            output_dir=output_dir,
        )
        print(format_full_market_audit_summary(result.summary))
        print(f"Saved detail: {result.detail_path}")
        print(f"Saved summary: {result.summary_path}")
        return

    if args.command == "reproduce-tail-risk":
        result = reproduce_tail_risk(
            storage=storage,
            project_root=project_root,
            start_date=_parse_optional_date(args.start_date),
            end_date=_parse_optional_date(args.end_date),
            train_end=datetime.fromisoformat(args.train_end).date(),
            valid_end=datetime.fromisoformat(args.valid_end).date(),
            limit=args.limit,
            lookback_days=args.lookback_days,
            quantile=args.quantile,
            horizon_days=args.horizon_days,
            min_training_rows=args.min_training_rows,
            allow_short_sample=args.allow_short_sample,
        )
        print("Tail-risk reproduction complete.")
        print(result.metrics.to_string(index=False))
        print(f"Saved dataset: {result.dataset_path}")
        print(f"Saved metrics: {result.metrics_path}")
        print(f"Saved deciles: {result.deciles_path}")
        return

    if args.command == "train-opportunity-ranker":
        _run_train_opportunity_ranker(
            storage=storage,
            config=config,
            project_root=project_root,
            start_date=_parse_optional_date(args.start_date),
            end_date=_parse_optional_date(args.end_date),
            train_end=_parse_optional_date(args.train_end),
            valid_end=_parse_optional_date(args.valid_end),
            test_end=_parse_optional_date(args.test_end),
            limit=args.limit,
            max_iter=args.max_iter,
            top_n_list=tuple(_parse_int_list(args.top_n)),
            prediction_date=_parse_optional_date(args.predict_date),
        )
        return

    if args.command == "predict-opportunity-ranker":
        _run_predict_opportunity_ranker(
            storage=storage,
            config=config,
            project_root=project_root,
            trade_date=datetime.fromisoformat(args.date).date(),
            top_n=args.top_n,
            output=args.output,
            rank_source=args.rank_source,
        )
        return

    if args.command == "train-volume-price-fusion":
        _run_train_volume_price_fusion(
            storage=storage,
            config=config,
            project_root=project_root,
            start_date=_parse_optional_date(args.start_date),
            end_date=_parse_optional_date(args.end_date),
            train_end=_parse_optional_date(args.train_end),
            valid_end=_parse_optional_date(args.valid_end),
            test_end=_parse_optional_date(args.test_end),
            limit=args.limit,
            max_iter=args.max_iter,
            top_n_list=tuple(_parse_int_list(args.top_n)),
            prediction_date=_parse_optional_date(args.predict_date),
            reuse_base_artifact=args.reuse_base_artifact,
        )
        return

    if args.command == "predict-volume-price-fusion":
        _run_predict_volume_price_fusion(
            storage=storage,
            config=config,
            project_root=project_root,
            trade_date=datetime.fromisoformat(args.date).date(),
            top_n=args.top_n,
            output=args.output,
        )
        return

    if args.command in {"train-candidate-ranker", "train-v51-candidate-ranker"}:
        _run_train_candidate_ranker(
            storage=storage,
            config=config,
            project_root=project_root,
            start_date=_parse_optional_date(args.start_date),
            end_date=_parse_optional_date(args.end_date),
            train_end=_parse_optional_date(args.train_end),
            valid_end=_parse_optional_date(args.valid_end),
            test_end=_parse_optional_date(args.test_end),
            limit=args.limit,
            max_iter=args.max_iter,
            top_n_list=tuple(_parse_int_list(args.top_n)),
            prediction_date=_parse_optional_date(args.predict_date),
        )
        return

    if args.command in {"validate-model-walkforward", "walkforward-validate-model"}:
        _run_validate_model_walkforward(
            storage=storage,
            config=config,
            project_root=project_root,
            model=args.model,
            start_date=_parse_optional_date(args.start_date),
            end_date=_parse_optional_date(args.end_date),
            limit=args.limit,
            windows=args.windows,
            train_days=args.train_days,
            valid_days=args.valid_days,
            test_days=args.test_days,
            min_train_days=args.min_train_days,
            max_iter=args.max_iter,
            top_n_list=tuple(_parse_int_list(args.top_n)),
        )
        return

    if args.command in {"predict-candidate-ranker", "predict-v51-candidate-ranker"}:
        _run_predict_candidate_ranker(
            storage=storage,
            config=config,
            project_root=project_root,
            trade_date=datetime.fromisoformat(args.date).date(),
            top_n=args.top_n,
            output=args.output,
        )
        return

    if args.command in {"predict-model", "predict_model"}:
        _run_predict_model(
            storage=storage,
            config=config,
            project_root=project_root,
            trade_date=datetime.fromisoformat(args.date).date(),
            top_n=args.top_n,
            output=args.output,
        )
        return

    if args.command == "build-event-labels":
        _run_build_event_labels(
            storage=storage,
            config=config,
            project_root=project_root,
            start_date=datetime.fromisoformat(args.start_date).date(),
            end_date=datetime.fromisoformat(args.end_date).date(),
            limit=args.limit,
            stop_atr=args.stop_atr,
            take_atr=args.take_atr,
            max_holding_days=args.max_holding_days,
        )
        return

    if args.command == "train-event-risk-ranker":
        _run_train_event_risk_ranker(
            storage=storage,
            config=config,
            project_root=project_root,
            start_date=datetime.fromisoformat(args.start_date).date(),
            end_date=_parse_optional_date(args.end_date) or date.today(),
            limit=args.limit,
            max_iter=args.max_iter,
            top_n_list=tuple(_parse_int_list(args.top_n)),
            stop_atr_grid=tuple(_parse_float_list(args.stop_atr_grid)),
            take_atr_grid=tuple(_parse_float_list(args.take_atr_grid)),
            holding_days_grid=tuple(_parse_int_list(args.holding_days_grid)),
            prediction_date=_parse_optional_date(args.predict_date),
        )
        return

    if args.command == "predict-event-risk-ranker":
        _run_predict_event_risk_ranker(
            storage=storage,
            config=config,
            project_root=project_root,
            trade_date=datetime.fromisoformat(args.date).date(),
            top_n=args.top_n,
            output=args.output,
        )
        return

    if args.command == "validate-event-risk-ranker":
        _run_validate_event_risk_ranker(
            storage=storage,
            config=config,
            project_root=project_root,
            start_date=datetime.fromisoformat(args.start_date).date(),
            end_date=_parse_optional_date(args.end_date) or date.today(),
            limit=args.limit,
            windows=args.windows,
            train_days=args.train_days,
            valid_days=args.valid_days,
            test_days=args.test_days,
            max_iter=args.max_iter,
            top_n_list=tuple(_parse_int_list(args.top_n)),
            stop_atr_grid=tuple(_parse_float_list(args.stop_atr_grid)),
            take_atr_grid=tuple(_parse_float_list(args.take_atr_grid)),
            holding_days_grid=tuple(_parse_int_list(args.holding_days_grid)),
        )
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

    if args.command == "backtest-patterns":
        _run_backtest_patterns(
            storage=storage,
            config=config,
            paths=paths,
            end_date=datetime.fromisoformat(args.date).date(),
            start_date=datetime.fromisoformat(args.start_date).date(),
            selected_patterns=_selected_patterns(args),
            output=args.output,
            sample_dates=args.sample_dates,
            sample_seed=args.sample_seed,
            save_forward_prices=args.save_forward_prices,
            forward_days=args.forward_days,
            top_n=args.top_n,
        )
        return

    if args.command == "research-pattern-stops":
        input_path = Path(args.input)
        _run_research_pattern_stops(
            paths=paths,
            input_path=input_path,
            report_date=datetime.fromisoformat(args.date).date() if args.date else _infer_report_date_from_path(input_path),
            holding_days=_parse_int_list(args.holding_days),
            take_profits=_parse_float_list(args.take_profits),
            stop_losses=_parse_float_list(args.stop_losses),
            same_day_policy=args.same_day_policy.replace("-", "_"),
            ma20_stop=args.ma20_stop,
            ma20_stop_tolerance=args.ma20_stop_tolerance,
            min_samples=args.min_samples,
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

    if args.command == "research-tradingview-factor":
        _run_research_tradingview_factor(
            storage=storage,
            paths=paths,
            start_date=datetime.fromisoformat(args.start_date).date(),
            end_date=datetime.fromisoformat(args.end_date).date(),
            horizons=tuple(_parse_int_list(args.horizons)),
            factor_fields=tuple(_parse_str_list(args.factor_fields)),
            rank_fields=tuple(_parse_str_list(args.rank_fields)),
            top_n=args.top_n,
            quantiles=args.quantiles,
            symbols=_parse_optional_symbol_list(args.symbols),
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
    *,
    index_symbols: tuple[str, ...] = DEFAULT_INDEX_SYMBOLS,
    update_indexes: bool = True,
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

    logging.info(
        "Daily update finished: success=%s failed=%s",
        success_count,
        len(failed_symbols),
    )
    if failed_symbols:
        logging.warning("Failed symbols sample: %s", ", ".join(failed_symbols[:20]))

    if update_indexes:
        _run_update_indexes(
            storage=storage,
            provider=provider,
            index_symbols=index_symbols,
            start_date=start_date,
            end_date=end_date,
        )


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
    except DailyBarsReadError as exc:
        logging.warning("Cached daily bars for %s are unreadable; rebuilding from %s: %s", symbol, start_date, exc)
        fresh = provider.get_daily_bars(symbol, start_date=start_date, end_date=end_date, adjust=adjust)
        target = storage.save_daily_bars(symbol, fresh)
        logging.info("Rebuilt %s rows for %s to %s", len(fresh), symbol, target)
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

    first_trade_date = valid_dates.min().date()
    last_trade_date = valid_dates.max().date()
    requested_start_date = datetime.strptime(start_date, "%Y%m%d").date()
    requested_end_date = datetime.strptime(end_date, "%Y%m%d").date()
    target = storage.paths.daily_dir / f"{symbol}.parquet"
    missing_ranges = _missing_cache_ranges(
        requested_start_date=requested_start_date,
        requested_end_date=requested_end_date,
        cached_first_date=first_trade_date,
        cached_last_date=last_trade_date,
    )
    if not missing_ranges:
        logging.info(
            "Skip %s because cached daily bars already cover %s to %s (cached=%s to %s)",
            symbol,
            requested_start_date.isoformat(),
            requested_end_date.isoformat(),
            first_trade_date.isoformat(),
            last_trade_date.isoformat(),
        )
        return target

    fresh_parts = []
    for range_start, range_end in missing_ranges:
        fresh = provider.get_daily_bars(
            symbol,
            start_date=range_start.strftime("%Y%m%d"),
            end_date=range_end.strftime("%Y%m%d"),
            adjust=adjust,
        )
        if fresh.empty:
            logging.info("No daily bars returned for %s from %s to %s", symbol, range_start.isoformat(), range_end.isoformat())
            continue
        fresh_parts.append(fresh)

    if not fresh_parts:
        return target

    merged = _merge_daily_cache_frames(cached_frame, fresh_parts)
    target = storage.save_daily_bars(symbol, merged)
    logging.info(
        "Merged %s fetched rows for %s into %s (requested=%s to %s)",
        sum(len(frame) for frame in fresh_parts),
        symbol,
        target,
        requested_start_date.isoformat(),
        requested_end_date.isoformat(),
    )
    return target


def _run_update_indexes(
    *,
    storage: Storage,
    provider,
    index_symbols: tuple[str, ...],
    start_date: str,
    end_date: str,
) -> None:
    success_count = 0
    failed_indexes: list[str] = []
    symbols = tuple(dict.fromkeys(str(symbol).strip() for symbol in index_symbols if str(symbol).strip()))
    total_symbols = len(symbols)
    for index, index_symbol in enumerate(symbols, start=1):
        try:
            _update_index_daily_cache(
                storage=storage,
                provider=provider,
                index_symbol=index_symbol,
                start_date=start_date,
                end_date=end_date,
            )
            success_count += 1
        except Exception as exc:
            failed_indexes.append(index_symbol)
            logging.warning("[%s/%s] failed to fetch index %s: %s", index, total_symbols, index_symbol, exc)
        _log_scan_progress("Index update", index, total_symbols)
    logging.info("Index daily update finished: success=%s failed=%s", success_count, len(failed_indexes))
    if failed_indexes:
        logging.warning("Failed indexes: %s", ", ".join(failed_indexes))


def _update_index_daily_cache(
    *,
    storage: Storage,
    provider,
    index_symbol: str,
    start_date: str,
    end_date: str,
) -> Path:
    try:
        cached = storage.load_index_daily_bars(index_symbol)
    except FileNotFoundError:
        fresh = provider.get_index_daily_bars(index_symbol, start_date=start_date, end_date=end_date)
        target = storage.save_index_daily_bars(index_symbol, fresh)
        logging.info("Initialized %s rows for index %s to %s", len(fresh), index_symbol, target)
        return target
    except DailyBarsReadError as exc:
        logging.warning("Cached index daily bars for %s are unreadable; rebuilding from %s: %s", index_symbol, start_date, exc)
        fresh = provider.get_index_daily_bars(index_symbol, start_date=start_date, end_date=end_date)
        target = storage.save_index_daily_bars(index_symbol, fresh)
        logging.info("Rebuilt %s rows for index %s to %s", len(fresh), index_symbol, target)
        return target

    cached_frame = cached.copy()
    if cached_frame.empty:
        fresh = provider.get_index_daily_bars(index_symbol, start_date=start_date, end_date=end_date)
        target = storage.save_index_daily_bars(index_symbol, fresh)
        logging.info("Initialized %s rows for index %s to %s", len(fresh), index_symbol, target)
        return target

    cached_frame["trade_date"] = pd.to_datetime(cached_frame["trade_date"], errors="coerce")
    valid_dates = cached_frame["trade_date"].dropna()
    if valid_dates.empty:
        logging.warning("Cached index daily bars for %s have no valid trade_date values; rebuilding from %s", index_symbol, start_date)
        fresh = provider.get_index_daily_bars(index_symbol, start_date=start_date, end_date=end_date)
        target = storage.save_index_daily_bars(index_symbol, fresh)
        logging.info("Rebuilt %s rows for index %s to %s", len(fresh), index_symbol, target)
        return target

    first_trade_date = valid_dates.min().date()
    last_trade_date = valid_dates.max().date()
    requested_start_date = datetime.strptime(start_date, "%Y%m%d").date()
    requested_end_date = datetime.strptime(end_date, "%Y%m%d").date()
    normalized_index_symbol = _normalize_index_symbol_for_update(index_symbol)
    target = storage.paths.index_daily_dir / f"{normalized_index_symbol}.parquet"
    missing_ranges = _missing_cache_ranges(
        requested_start_date=requested_start_date,
        requested_end_date=requested_end_date,
        cached_first_date=first_trade_date,
        cached_last_date=last_trade_date,
    )
    if not missing_ranges:
        logging.info(
            "Skip index %s because cached daily bars already cover %s to %s (cached=%s to %s)",
            normalized_index_symbol,
            requested_start_date.isoformat(),
            requested_end_date.isoformat(),
            first_trade_date.isoformat(),
            last_trade_date.isoformat(),
        )
        return target

    fresh_parts = []
    for range_start, range_end in missing_ranges:
        fresh = provider.get_index_daily_bars(
            index_symbol,
            start_date=range_start.strftime("%Y%m%d"),
            end_date=range_end.strftime("%Y%m%d"),
        )
        if fresh.empty:
            logging.info(
                "No index daily bars returned for %s from %s to %s",
                normalized_index_symbol,
                range_start.isoformat(),
                range_end.isoformat(),
            )
            continue
        fresh_parts.append(fresh)

    if not fresh_parts:
        return target

    merged = _merge_daily_cache_frames(cached_frame, fresh_parts)
    target = storage.save_index_daily_bars(index_symbol, merged)
    logging.info(
        "Merged %s fetched rows for index %s into %s (requested=%s to %s)",
        sum(len(frame) for frame in fresh_parts),
        normalized_index_symbol,
        target,
        requested_start_date.isoformat(),
        requested_end_date.isoformat(),
    )
    return target


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
    prefix = "sz" if code.startswith("399") else "sh"
    return f"{prefix}{code}"


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
    exported = _append_recent_tradingview_scores(storage, exported, as_of=as_of, lookback_days=5, symbols=symbols)
    exported = _append_recent_macd_summary(storage, exported, as_of=as_of, symbols=symbols)
    exported = _append_recent_atr_summary(storage, exported, as_of=as_of, symbols=symbols)

    try:
        model_predictions = load_predict_model_predictions(project_root=storage.paths.root, trade_date=as_of)
    except FileNotFoundError:
        if not exported.empty:
            raise
        model_predictions = pd.DataFrame()
    exported = _append_predict_model_risk_summary(exported, model_predictions)

    output_path = Path(output) if output else _default_pattern_output_path(
        storage,
        as_of=as_of,
        selected_patterns=selected_patterns,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(output_path, index=False, encoding="utf-8-sig")
    predict_model_path = storage.paths.root / "reports" / "predict_model" / f"predictions_{as_of.isoformat()}.csv"
    if model_predictions.empty:
        pattern_watchlist_payload = build_watchlist_candidates_from_patterns(
            exported,
            source_file=str(output_path),
            limit=None,
            model_predictions=None,
        )
        watchlist_payload = pattern_watchlist_payload
    else:
        pattern_watchlist_payload = build_watchlist_candidates_from_patterns(
            exported,
            source_file=str(output_path),
            limit=None,
            model_predictions=model_predictions,
        )
        watchlist_payload = build_daily_watchlist_candidates(
            exported,
            model_predictions=model_predictions,
            pattern_source_file=str(output_path),
            model_source_file=str(predict_model_path),
            model_top_n=20,
        )
    write_watchlist(
        project_root=storage.paths.root,
        trade_date=as_of,
        picker_payload=watchlist_payload,
    )
    pattern_watchlist_target = write_watchlist(
        project_root=storage.paths.root,
        trade_date=as_of,
        picker_payload=pattern_watchlist_payload,
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
    print(f"\nSaved pattern watchlist to {pattern_watchlist_target}")
    logging.info("Saved %s pattern rows to %s", len(exported), output_path)


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
    summary = scan_trend_universe(
        storage,
        config,
        as_of=trade_date,
        progress_callback=lambda current, total: _log_scan_progress("Trend-universe", current, total),
    )
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
    scored = scan_indicator_scored_entries(
        storage,
        config,
        trade_date=trade_date,
        progress_callback=lambda current, total: _log_scan_progress("Trend-score", current, total),
    )
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
    scored = scan_indicator_scored_entries(
        storage,
        config,
        trade_date=trade_date,
        progress_callback=lambda current, total: _log_scan_progress("Trend", current, total),
    )
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
    scored = scan_indicator_scored_entries(
        storage,
        config,
        trade_date=trade_date,
        progress_callback=lambda current, total: _log_scan_progress("Trend-entries", current, total),
    )
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


def _run_backtest_patterns(
    storage: Storage,
    config,
    paths: ProjectPaths,
    end_date: date,
    start_date: date,
    selected_patterns: list[str],
    output: str | None,
    sample_dates: int | None,
    sample_seed: int | None,
    save_forward_prices: bool,
    forward_days: int,
    top_n: int,
) -> None:
    sampled_trade_dates = None
    if sample_dates is not None:
        sampled_trade_dates = sample_pattern_backtest_trade_dates(
            storage,
            start_date=start_date,
            end_date=end_date,
            sample_size=sample_dates,
            seed=sample_seed,
        )
        logging.info(
            "Pattern-backtest sampled %s trade dates from %s to %s with seed=%s",
            len(sampled_trade_dates),
            start_date.isoformat(),
            end_date.isoformat(),
            sample_seed,
        )

    signals = scan_pattern_backtest_signals(
        storage,
        config,
        start_date=start_date,
        end_date=end_date,
        selected_strategies=selected_patterns,
        sampled_trade_dates=sampled_trade_dates,
        cooldown_trading_days=5,
        progress_callback=lambda current, total: _log_scan_progress("Pattern-backtest", current, total),
    )
    daily_history = _load_daily_history_map(storage, signals["symbol"].astype(str).tolist() if not signals.empty else [])
    forward_prices = (
        build_pattern_forward_price_frame(signals, daily_history, forward_days=forward_days, entry_timing="next_open")
        if save_forward_prices
        else None
    )
    detail = backtest_signal_returns(signals, daily_history, config.trend_backtest, entry_timing="next_open")
    if not detail.empty:
        detail["pattern_id"] = detail["signal_type"].map(PATTERN_LABEL_MAP)
        detail = detail.rename(columns={"signal_type": "strategy_name"})
        preferred = [
            "trade_date",
            "signal_date",
            "entry_date",
            "exit_date",
            "symbol",
            "name",
            "pattern_id",
            "strategy_name",
            "holding_days",
            "entry_price",
            "exit_price",
            "return_pct",
            "max_upside_pct",
            "max_drawdown_pct",
            "min_return_pct",
            "trigger_reason",
            "entry_timing",
            "entry_note",
        ]
        available = [column for column in preferred if column in detail.columns]
        remaining = [column for column in detail.columns if column not in available]
        detail = detail.loc[:, available + remaining]
    summary = summarize_pattern_backtest(detail.rename(columns={"strategy_name": "signal_type"}) if not detail.empty else detail)
    report_paths = save_pattern_backtest_reports(
        paths,
        report_date=end_date,
        detail=detail,
        summary=summary,
        forward_prices=forward_prices,
        sampled_trade_dates=sampled_trade_dates,
        sample_seed=sample_seed if sample_dates is not None else None,
        output=output,
    )
    print(
        _format_dataframe(
            summary,
            [
                "pattern_id",
                "strategy_name",
                "holding_days",
                "signal_count",
                "win_rate",
                "avg_return_pct",
                "avg_max_upside_pct",
                "avg_max_drawdown_pct",
            ],
            top_n,
        )
    )
    print(f"\n模式回测明细：{report_paths['detail_path']}")
    print(f"模式回测汇总：{report_paths['summary_path']}")
    if "forward_prices_path" in report_paths:
        print(f"买入后每日价格：{report_paths['forward_prices_path']}")


def _run_research_pattern_stops(
    paths: ProjectPaths,
    *,
    input_path: Path,
    report_date: date,
    holding_days: list[int],
    take_profits: list[float],
    stop_losses: list[float],
    same_day_policy: str,
    ma20_stop: bool,
    ma20_stop_tolerance: float,
    min_samples: int,
    output: str | None,
    top_n: int,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Pattern forward price file not found: {input_path}")

    forward_prices = pd.read_csv(input_path)
    if ma20_stop and "ma_20" not in forward_prices.columns:
        raise RuntimeError(
            "MA20 stop requires a forward price file with ma_20. "
            "Please rerun backtest-patterns with --save-forward-prices using the updated code."
        )
    result = research_pattern_stop_grid(
        forward_prices,
        holding_days=holding_days,
        take_profits=take_profits,
        stop_losses=stop_losses,
        same_day_policy=same_day_policy,
        ma20_stop=ma20_stop,
        ma20_stop_tolerance=ma20_stop_tolerance,
    )
    summary = result["summary"]
    trades = result["trades"]
    best = result["best"]
    if min_samples != 30 and not summary.empty:
        best = select_best_pattern_stop_grid(summary, min_samples=min_samples)

    report_paths = save_pattern_stop_research_reports(
        paths,
        report_date=report_date,
        trades=trades,
        summary=summary,
        best=best,
        input_path=input_path,
        output=output,
    )

    display = best.copy()
    if display.empty:
        display = summary.copy()
    print(
        _format_dataframe(
            display,
            [
                "pattern_id",
                "strategy_name",
                "holding_days",
                "take_profit_pct",
                "stop_loss_pct",
                "sample_count",
                "win_rate",
                "avg_return_pct",
                "take_profit_rate",
                "stop_loss_rate",
                "ma20_stop_rate",
                "time_exit_rate",
                "avg_exit_day",
            ],
            top_n,
        )
    )
    print(f"\n止盈止损完整网格：{report_paths['summary_path']}")
    print(f"止盈止损最佳组合：{report_paths['best_path']}")
    print(f"止盈止损逐笔明细：{report_paths['trades_path']}")


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


def _run_train_opportunity_ranker(
    *,
    storage: Storage,
    config,
    project_root: Path,
    start_date: date | None,
    end_date: date | None,
    train_end: date | None,
    valid_end: date | None,
    test_end: date | None,
    limit: int | None,
    max_iter: int,
    top_n_list: tuple[int, ...],
    prediction_date: date | None,
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    result = train_opportunity_ranker_model(
        storage=storage,
        config=config,
        project_root=project_root,
        start_date=start_date,
        end_date=end_date,
        train_end=train_end,
        valid_end=valid_end,
        test_end=test_end,
        limit=limit,
        max_iter=max_iter,
        top_n_list=top_n_list,
        prediction_date=prediction_date,
    )
    print("[V4.2 Opportunity Ranker] Selected risk params")
    print(json.dumps(result.selected_risk_params, ensure_ascii=False, indent=2))
    print()
    print("[V4.2 Opportunity Ranker] Selected opportunity params")
    print(json.dumps(result.selected_opportunity_params, ensure_ascii=False, indent=2))
    print()
    print("[V4.2 Opportunity Ranker] Selected hybrid V4-rank opportunity params")
    print(json.dumps(result.selected_hybrid_opportunity_params, ensure_ascii=False, indent=2))
    print()
    print("[V4.2 Opportunity Ranker] Split metrics")
    print(format_metric_table(result.split_metrics))
    print()
    print("[V4.2 Opportunity Ranker] TopN metrics")
    print(format_metric_table(result.topn_metrics))
    print()
    print("[V4.2 Opportunity Ranker] Risk filter metrics")
    print(format_metric_table(result.risk_filter_metrics))
    print()
    print("[V4.2 Opportunity Ranker] Opportunity metrics")
    print(format_metric_table(result.opportunity_metrics))
    print()
    print("[V4.2 Opportunity Ranker] Ranker metrics")
    print(format_metric_table(result.ranker_metrics))
    print()
    print("[V4.2 Opportunity Ranker] V4 baseline comparison")
    print(format_metric_table(result.comparison_metrics))
    print()
    print("[V4.2 Opportunity Ranker] Threshold grid")
    if result.threshold_grid.empty:
        print("No threshold grid.")
    else:
        print(format_metric_table(result.threshold_grid.sort_values("objective", ascending=False).head(10)))
    if result.prediction_path is not None:
        print()
        print("[V4.2 Opportunity Ranker] Latest predictions")
        print(format_opportunity_ranker_prediction_table(result.latest_predictions, limit=20))
        print(f"\nSaved V4.2 opportunity predictions to {result.prediction_path}")
    print(f"\nSaved V4.2 opportunity model to {result.model_path}")
    print(f"Saved V4.2 opportunity metadata to {result.metadata_path}")
    print(f"Saved V4.2 opportunity reports to {opportunity_ranker_report_dir(project_root)}")


def _run_predict_opportunity_ranker(
    *,
    storage: Storage,
    config,
    project_root: Path,
    trade_date: date,
    top_n: int,
    output: str | None,
    rank_source: str,
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    predictions = predict_opportunity_ranker(
        storage=storage,
        config=config,
        project_root=project_root,
        trade_date=trade_date,
        rank_source=rank_source,
    )
    if predictions.empty:
        raise RuntimeError(f"No V4.2 opportunity predictions generated for {trade_date.isoformat()}")
    output_path = (
        Path(output)
        if output
        else opportunity_ranker_report_dir(project_root)
        / (
            f"predictions_{trade_date.isoformat()}.csv"
            if rank_source == "v42"
            else f"predictions_v4_rank_{trade_date.isoformat()}.csv"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(format_opportunity_ranker_prediction_table(predictions, limit=top_n))
    print(f"\nSaved V4.2 opportunity ranking to {output_path}")


def _run_train_volume_price_fusion(
    *,
    storage: Storage,
    config,
    project_root: Path,
    start_date: date | None,
    end_date: date | None,
    train_end: date | None,
    valid_end: date | None,
    test_end: date | None,
    limit: int | None,
    max_iter: int,
    top_n_list: tuple[int, ...],
    prediction_date: date | None,
    reuse_base_artifact: bool,
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    result = train_volume_price_fusion_model(
        storage=storage,
        config=config,
        project_root=project_root,
        start_date=start_date,
        end_date=end_date,
        train_end=train_end,
        valid_end=valid_end,
        test_end=test_end,
        limit=limit,
        max_iter=max_iter,
        top_n_list=top_n_list,
        prediction_date=prediction_date,
        reuse_base_artifact=reuse_base_artifact,
    )
    print("[V5 Volume-Price Fusion] TopN metrics")
    print(format_metric_table(result.topn_metrics))
    print()
    print("[V5 Volume-Price Fusion] Volume-price risk metrics")
    print(format_metric_table(result.volume_price_risk_metrics))
    print()
    print("[V5 Volume-Price Fusion] Volume-price quality metrics")
    print(format_metric_table(result.volume_price_quality_metrics))
    print()
    print("[V5 Volume-Price Fusion] V4.2 hybrid comparison")
    print(format_metric_table(result.comparison_metrics))
    if result.prediction_path is not None:
        print()
        print("[V5 Volume-Price Fusion] Latest predictions")
        print(format_volume_price_fusion_prediction_table(result.latest_predictions, limit=20))
        print(f"\nSaved V5 predictions to {result.prediction_path}")
    print(f"\nSaved V5 volume-price fusion model to {result.model_path}")
    print(f"Saved V5 volume-price fusion metadata to {result.metadata_path}")
    print(f"Saved V5 volume-price fusion reports to {volume_price_fusion_report_dir(project_root)}")


def _run_predict_volume_price_fusion(
    *,
    storage: Storage,
    config,
    project_root: Path,
    trade_date: date,
    top_n: int,
    output: str | None,
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    predictions = predict_volume_price_fusion(
        storage=storage,
        config=config,
        project_root=project_root,
        trade_date=trade_date,
    )
    if predictions.empty:
        raise RuntimeError(f"No V5 volume-price fusion predictions generated for {trade_date.isoformat()}")
    output_path = (
        Path(output)
        if output
        else volume_price_fusion_report_dir(project_root) / f"predictions_{trade_date.isoformat()}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(format_volume_price_fusion_prediction_table(predictions, limit=top_n))
    print(f"\nSaved V5 volume-price fusion ranking to {output_path}")


def _run_train_candidate_ranker(
    *,
    storage: Storage,
    config,
    project_root: Path,
    start_date: date | None,
    end_date: date | None,
    train_end: date | None,
    valid_end: date | None,
    test_end: date | None,
    limit: int | None,
    max_iter: int,
    top_n_list: tuple[int, ...],
    prediction_date: date | None,
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    result = train_candidate_ranker_model(
        storage=storage,
        config=config,
        project_root=project_root,
        start_date=start_date,
        end_date=end_date,
        train_end=train_end,
        valid_end=valid_end,
        test_end=test_end,
        limit=limit,
        max_iter=max_iter,
        top_n_list=top_n_list,
        prediction_date=prediction_date,
    )
    print("[V5.1 Candidate Ranker] Selected blend params")
    print(json.dumps(result.selected_blend_params, ensure_ascii=False, indent=2))
    print()
    print("[V5.1 Candidate Ranker] TopN metrics")
    print(format_metric_table(result.topn_metrics))
    print()
    print("[V5.1 Candidate Ranker] Ranker metrics")
    print(format_metric_table(result.ranker_metrics))
    print()
    print("[V5.1 Candidate Ranker] V4.2/V5 comparison")
    print(format_metric_table(result.comparison_metrics))
    print()
    print("[V5.1 Candidate Ranker] Blend grid")
    print(format_metric_table(result.blend_grid.sort_values("objective", ascending=False).head(10)))
    if result.prediction_path is not None:
        print()
        print("[V5.1 Candidate Ranker] Latest predictions")
        print(format_candidate_ranker_prediction_table(result.latest_predictions, limit=20))
        print(f"\nSaved V5.1 predictions to {result.prediction_path}")
    print(f"\nSaved V5.1 candidate ranker model to {result.model_path}")
    print(f"Saved V5.1 candidate ranker metadata to {result.metadata_path}")
    print(f"Saved V5.1 candidate ranker reports to {candidate_ranker_report_dir(project_root)}")


def _run_validate_model_walkforward(
    *,
    storage: Storage,
    config,
    project_root: Path,
    model: str,
    start_date: date | None,
    end_date: date | None,
    limit: int | None,
    windows: int,
    train_days: int,
    valid_days: int,
    test_days: int,
    min_train_days: int,
    max_iter: int,
    top_n_list: tuple[int, ...],
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    result = validate_model_walkforward(
        storage=storage,
        config=config,
        project_root=project_root,
        model=model,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        windows=windows,
        train_days=train_days,
        valid_days=valid_days,
        test_days=test_days,
        min_train_days=min_train_days,
        max_iter=max_iter,
        top_n_list=top_n_list,
    )
    print("[Walk-Forward Validation] Windows")
    window_columns = [
        "window_id",
        "status",
        "train_start",
        "train_end",
        "valid_start",
        "valid_end",
        "test_start",
        "test_end",
        "top20_win_rate",
        "top20_avg_return_20d",
        "top20_stop_loss_rate_20d",
        "top20_bad_risk_rate",
        "top20_coverage",
    ]
    visible_windows = result.windows.loc[:, [column for column in window_columns if column in result.windows.columns]]
    print(format_metric_table(visible_windows))
    print()
    print("[Walk-Forward Validation] TopN metrics")
    metric_columns = [
        "window_id",
        "model_version",
        "top_n",
        "test_start",
        "test_end",
        "test_days",
        "allowed_days",
        "coverage",
        "win_rate",
        "avg_return_20d",
        "median_return_20d",
        "take_profit_rate_20d",
        "stop_loss_rate_20d",
        "bad_risk_rate",
    ]
    visible_metrics = result.topn_metrics.loc[
        :, [column for column in metric_columns if column in result.topn_metrics.columns]
    ]
    print(format_metric_table(visible_metrics))
    print()
    print("[Walk-Forward Validation] Summary")
    summary_columns = [
        "model_version",
        "top_n",
        "windows",
        "coverage_mean",
        "win_rate_mean",
        "win_rate_min",
        "avg_return_20d_mean",
        "avg_return_20d_min",
        "take_profit_rate_20d_mean",
        "stop_loss_rate_20d_mean",
        "bad_risk_rate_mean",
        "window_pass_rate",
        "pass_all_top20_thresholds",
    ]
    visible_summary = result.summary.loc[:, [column for column in summary_columns if column in result.summary.columns]]
    print(format_metric_table(visible_summary))
    print(f"\nSaved walk-forward windows to {result.windows_path}")
    print(f"Saved walk-forward TopN metrics to {result.topn_metrics_path}")
    print(f"Saved walk-forward summary to {result.summary_path}")
    print(f"Saved walk-forward config to {result.config_path}")
    print(f"Saved walk-forward reports to {model_walkforward_report_dir(project_root)}")


def _run_predict_candidate_ranker(
    *,
    storage: Storage,
    config,
    project_root: Path,
    trade_date: date,
    top_n: int,
    output: str | None,
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    predictions = predict_candidate_ranker(
        storage=storage,
        config=config,
        project_root=project_root,
        trade_date=trade_date,
    )
    if predictions.empty:
        raise RuntimeError(f"No V5.1 candidate ranker predictions generated for {trade_date.isoformat()}")
    output_path = (
        Path(output)
        if output
        else candidate_ranker_report_dir(project_root) / f"predictions_{trade_date.isoformat()}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(format_candidate_ranker_prediction_table(predictions, limit=top_n))
    print(f"\nSaved V5.1 candidate ranking to {output_path}")


def _run_predict_model(
    *,
    storage: Storage,
    config,
    project_root: Path,
    trade_date: date,
    top_n: int,
    output: str | None,
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    predictions = predict_opportunity_ranker(
        storage=storage,
        config=config,
        project_root=project_root,
        trade_date=trade_date,
        rank_source="v4",
    )
    if predictions.empty:
        raise RuntimeError(f"No predict_model predictions generated for {trade_date.isoformat()}")
    output_path = save_predict_model_predictions(
        predictions,
        project_root=project_root,
        trade_date=trade_date,
        output=output,
        model_version="v42_gate_v4_rank",
    )
    print(format_opportunity_ranker_prediction_table(predictions, limit=top_n))
    print(f"\nSaved predict_model ranking to {output_path}")


def _run_build_event_labels(
    *,
    storage: Storage,
    config,
    project_root: Path,
    start_date: date,
    end_date: date,
    limit: int | None,
    stop_atr: float,
    take_atr: float,
    max_holding_days: int,
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    report_dir = event_risk_ranker_report_dir(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    labels, features, skipped = build_event_risk_ranker_dataset(
        storage=storage,
        config=config,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        label_config=EventLabelConfig(
            stop_atr_mult=stop_atr,
            take_atr_mult=take_atr,
            max_holding_days=max_holding_days,
        ),
        progress_callback=lambda current, total: _log_scan_progress("event-labels", current, total),
    )
    labels_path = report_dir / f"event_labels_{start_date.isoformat()}_{end_date.isoformat()}.csv"
    features_path = report_dir / f"event_features_{start_date.isoformat()}_{end_date.isoformat()}.csv"
    skipped_path = report_dir / "skipped_events.csv"
    labels.to_csv(labels_path, index=False, encoding="utf-8-sig")
    features.to_csv(features_path, index=False, encoding="utf-8-sig")
    skipped.to_csv(skipped_path, index=False, encoding="utf-8-sig")
    print("Event labels built.")
    print(f"labels={len(labels)} features={len(features)} skipped={len(skipped)}")
    print(f"Labels: {labels_path}")
    print(f"Features: {features_path}")
    print(f"Skipped events: {skipped_path}")


def _run_train_event_risk_ranker(
    *,
    storage: Storage,
    config,
    project_root: Path,
    start_date: date,
    end_date: date,
    limit: int | None,
    max_iter: int,
    top_n_list: tuple[int, ...],
    stop_atr_grid: tuple[float, ...],
    take_atr_grid: tuple[float, ...],
    holding_days_grid: tuple[int, ...],
    prediction_date: date | None,
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    result = train_event_risk_ranker_model(
        storage=storage,
        config=config,
        project_root=project_root,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        max_iter=max_iter,
        top_n_list=top_n_list,
        stop_atr_grid=stop_atr_grid,
        take_atr_grid=take_atr_grid,
        holding_days_grid=holding_days_grid,
        prediction_date=prediction_date,
        progress_callback=lambda current, total: _log_scan_progress("event-train", current, total),
    )
    print("Event risk ranker trained.")
    print(f"Labels: {len(result.labels)}")
    print(f"Features: {len(result.features)}")
    print(f"Skipped events: {len(result.skipped_events)}")
    print("\nTopN metrics:")
    print(_format_dataframe(result.topn_metrics, list(result.topn_metrics.columns), top_n=80) if not result.topn_metrics.empty else "No metrics.")
    print(f"\nSaved model: {result.model_path}")
    print(f"Saved metadata: {result.metadata_path}")
    print(f"Saved reports: {result.report_dir}")


def _run_predict_event_risk_ranker(
    *,
    storage: Storage,
    config,
    project_root: Path,
    trade_date: date,
    top_n: int,
    output: str | None,
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    predictions = predict_event_risk_ranker(
        storage=storage,
        config=config,
        project_root=project_root,
        trade_date=trade_date,
        output=output,
    )
    print(format_event_ranker_prediction_table(predictions, limit=top_n))
    output_path = Path(output) if output else event_risk_ranker_predictions_path(project_root, trade_date)
    print(f"\nSaved event risk ranker predictions to {output_path}")


def _run_validate_event_risk_ranker(
    *,
    storage: Storage,
    config,
    project_root: Path,
    start_date: date,
    end_date: date,
    limit: int | None,
    windows: int,
    train_days: int,
    valid_days: int,
    test_days: int,
    max_iter: int,
    top_n_list: tuple[int, ...],
    stop_atr_grid: tuple[float, ...],
    take_atr_grid: tuple[float, ...],
    holding_days_grid: tuple[int, ...],
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    windows_frame, metrics, summary = validate_event_risk_ranker_walkforward(
        storage=storage,
        config=config,
        project_root=project_root,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        windows=windows,
        train_days=train_days,
        valid_days=valid_days,
        test_days=test_days,
        max_iter=max_iter,
        top_n_list=top_n_list,
        stop_atr_grid=stop_atr_grid,
        take_atr_grid=take_atr_grid,
        holding_days_grid=holding_days_grid,
        progress_callback=lambda current, total: _log_scan_progress("event-validate", current, total),
    )
    print("Event risk ranker walk-forward validation complete.")
    print("\nWindows:")
    print(_format_dataframe(windows_frame, list(windows_frame.columns), top_n=50) if not windows_frame.empty else "No valid windows.")
    print("\nSummary:")
    print(_format_dataframe(summary, list(summary.columns), top_n=80) if not summary.empty else "No summary.")
    print(f"\nSaved reports to {event_risk_ranker_report_dir(project_root)}")


def _run_research_tradingview_factor(
    *,
    storage: Storage,
    paths: ProjectPaths,
    start_date: date,
    end_date: date,
    horizons: tuple[int, ...],
    factor_fields: tuple[str, ...],
    rank_fields: tuple[str, ...],
    top_n: int,
    quantiles: int,
    symbols: list[str] | None,
) -> None:
    result = run_tradingview_factor_research(
        storage,
        start_date=start_date,
        end_date=end_date,
        horizons=horizons,
        factor_fields=factor_fields,
        rank_fields=rank_fields,
        top_n=top_n,
        quantiles=quantiles,
        symbols=symbols,
        progress_callback=lambda current, total: _log_scan_progress("TradingView-factor", current, total),
    )
    report_paths = save_tradingview_factor_research_reports(
        paths,
        result=result,
        start_date=start_date,
        end_date=end_date,
        top_n=top_n,
    )
    print("TradingView 单因子研究完成")
    print(f"样本数：{len(result.samples)}")
    print("\nRank IC 汇总：")
    print(
        _format_dataframe(
            result.ic_summary,
            ["factor", "horizon_days", "ic_days", "mean_rank_ic", "positive_ic_rate", "avg_sample_count"],
            top_n=50,
        )
    )
    print("\nTopN 汇总：")
    print(
        _format_dataframe(
            result.topn_summary,
            [
                "rank_field",
                "top_count",
                "horizon_days",
                "trade_days",
                "avg_daily_equal_weight_return",
                "daily_win_rate",
                "avg_stock_return",
                "stock_win_rate",
            ],
            top_n=80,
        )
    )
    print(f"\n样本明细：{report_paths['samples_path']}")
    print(f"IC 汇总：{report_paths['ic_summary_path']}")
    print(f"Top{top_n} 明细：{report_paths['topn_detail_path']}")
    print(f"Top{top_n} 汇总：{report_paths['topn_summary_path']}")


def _load_daily_history_map(storage: Storage, symbols: list[str]) -> dict[str, pd.DataFrame]:
    history: dict[str, pd.DataFrame] = {}
    for symbol in {str(item).zfill(6) for item in symbols}:
        try:
            history[symbol] = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError) as exc:
            if isinstance(exc, DailyBarsReadError):
                logging.warning("Skip cached daily history for %s because it is unreadable: %s", symbol, exc)
            continue
    return history


def _format_dataframe(dataframe: pd.DataFrame, columns: list[str], top_n: int) -> str:
    if dataframe.empty:
        return "No rows matched."
    available = [column for column in columns if column in dataframe.columns]
    if not available:
        return dataframe.head(top_n).to_string(index=False)
    return dataframe.loc[:, available].head(top_n).to_string(index=False)


def format_tradingview_summary(scores: pd.DataFrame, limit: int) -> str:
    if scores.empty:
        return "No TradingView ratings generated."
    rating_date_columns = sorted(column for column in scores.columns if column.startswith("all_rating_20"))
    columns = [
        column
        for column in (
            "symbol",
            "name",
            *rating_date_columns,
            "avg_all_rating_5d",
            "ma_rating",
            "osc_rating",
            "all_rating",
            "all_rating_label",
        )
        if column in scores.columns
    ]
    display = scores.loc[:, columns].head(limit).copy()
    for column in ("ma_rating", "osc_rating", "all_rating", "avg_all_rating_5d", *rating_date_columns):
        if column in display.columns:
            display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    return display.to_string(index=False)


def _selected_patterns(args: argparse.Namespace) -> list[str]:
    selected = [pattern for field, pattern in PATTERN_FLAG_MAP.items() if getattr(args, field)]
    return selected or list(STRATEGY_NAMES)


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
    preferred_order = [
        "trade_date",
        "symbol",
        "name",
        "pattern_id",
        "风险情况",
        "close",
        "old_high_date",
        "old_high_price",
        "days_since_old_high",
        "max_drawdown_since_old_high",
        "distance_to_old_high_pct",
        "recent_high_date",
        "recent_high_price",
        "days_since_recent_high",
        "distance_from_recent_high_pct",
        "extension_above_old_high_pct",
        "ma20_slope_short_pct",
        "ma20_slope_long_pct",
        "ma60_slope_short_pct",
        "ma60_slope_long_pct",
        "pullback_volume_contraction_ratio",
        "main_rise_start_date",
        "main_rise_end_date",
        "main_rise_return_pct",
        "transition_days",
        "duck_peak_date",
        "duck_peak_price",
        "days_since_duck_peak",
        "neck_start_date",
        "neck_return_pct",
        "neck_low_to_peak_return_pct",
        "nostril_cross_date",
        "days_since_nostril_cross",
        "cross_after_pullback_low_days",
        "nostril_cross_ma5_ma10_gap_pct",
        "latest_ma5_ma10_gap_pct",
        "nostril_volume_ma20_ratio",
        "distance_to_duck_peak_pct",
        "peak_tail_avg_volume",
        "pullback_avg_volume",
        "pullback_volume_peak_tail_ratio",
        "pullback_max_single_day_peak_tail_ratio",
        "large_bearish_count",
        "max_bearish_body_pct",
        "max_bearish_volume_ratio",
        "breakout_date",
        "breakout_volume_ratio",
        "breakout_close_position",
        "breakout_upper_shadow_pct",
        "breakout_body_pct",
        "breakout_turnover",
        "breakout_turnover_state",
        "days_after_breakout",
        "post_breakout_max_high_extension_pct",
        "platform_window_days",
        "platform_range_pct",
        "platform_volume_contraction_ratio",
        "platform_range_contraction_ratio",
        "platform_low_lift_pct",
        "platform_max_bearish_body_pct",
        "platform_max_bearish_volume_ratio",
        "distance_to_platform_high_pct",
        "ma20_touch_date",
        "ma20_touch_distance",
        "distance_to_ma20",
        "pattern6_branch",
        "anchor_date",
        "anchor_close",
        "support_price",
        "anchor_volume_ratio_prev",
        "anchor_volume_ratio_ma20",
        "launch_confirm_high_date",
        "launch_confirm_high_price",
        "launch_confirm_return_pct",
        "peak_date",
        "peak_price",
        "anchor_to_peak_return_pct",
        "limit_up_like_count",
        "pullback_low_date",
        "pullback_low_price",
        "peak_to_pullback_drawdown_pct",
        "pullback_volume_ratio_to_anchor",
        "pullback_front_half_avg_volume",
        "pullback_back_half_avg_volume",
        "pullback_back_half_volume_ratio",
        "rise_tail_avg_volume",
        "pullback_max_volume",
        "pullback_max_rise_tail_volume_ratio",
        "support_touch_date",
        "breakdown_date",
        "breakdown_volume_ratio_to_anchor",
        "reclaim_date",
        "days_to_reclaim",
        "post_reclaim_days",
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


def _append_predict_model_risk_summary(exported: pd.DataFrame, model_predictions: pd.DataFrame) -> pd.DataFrame:
    if exported.empty or "symbol" not in exported.columns:
        return exported
    result = exported.copy()
    prediction = model_predictions.copy()
    if prediction.empty or "symbol" not in prediction.columns:
        result["风险情况"] = "模型风险未知"
        return _move_column_after(result, "风险情况", "pattern_id")

    prediction["_normalized_symbol"] = prediction["symbol"].map(_normalize_exported_symbol)
    if "risk_score" in prediction.columns:
        prediction["_risk_sort_score"] = pd.to_numeric(prediction["risk_score"], errors="coerce")
        prediction = prediction.sort_values("_risk_sort_score", ascending=True)
    prediction = prediction.drop_duplicates(subset=["_normalized_symbol"], keep="first")
    prediction = prediction.drop(columns=["_risk_sort_score"], errors="ignore")
    keep_columns = [
        "_normalized_symbol",
        "risk_tier",
        "risk_gate_reason",
        "risk_score",
        "risk_candidate_action",
        "risk_action",
        "action",
        "trade_permission",
    ]
    prediction = prediction.loc[:, [column for column in keep_columns if column in prediction.columns]].copy()
    prediction["风险情况"] = prediction.apply(_format_pattern_risk_summary, axis=1)

    result["_normalized_symbol"] = result["symbol"].map(_normalize_exported_symbol)
    result = result.drop(columns=["风险情况"], errors="ignore")
    result = result.merge(
        prediction.loc[:, ["_normalized_symbol", "风险情况"]],
        on="_normalized_symbol",
        how="left",
    )
    result["风险情况"] = result["风险情况"].fillna("模型风险未知")
    result = result.drop(columns=["_normalized_symbol"], errors="ignore")
    return _move_column_after(result, "风险情况", "pattern_id")


def _format_pattern_risk_summary(row: pd.Series) -> str:
    risk_tier = str(row.get("risk_tier", "")).strip()
    risk_reason = str(row.get("risk_gate_reason", "")).strip()
    risk_score = pd.to_numeric(pd.Series([row.get("risk_score")]), errors="coerce").iloc[0]
    low_risk = _is_predict_model_low_risk_row(row)
    prefix = "低风险" if low_risk else "风险排除"
    parts = [prefix]
    if risk_tier and risk_tier.lower() != "nan":
        parts.append(f"tier={risk_tier}")
    if pd.notna(risk_score):
        parts.append(f"score={float(risk_score):.3f}")
    if risk_reason and risk_reason.lower() != "nan":
        parts.append(f"reason={risk_reason}")
    return " | ".join(parts)


def _is_predict_model_low_risk_row(row: pd.Series) -> bool:
    final_action = str(row.get("final_action", "")).strip().lower()
    action = str(row.get("action", "")).strip().lower()
    risk_candidate_action = str(row.get("risk_candidate_action", "")).strip().lower()
    risk_action = str(row.get("risk_action", "")).strip().lower()
    risk_tier = str(row.get("risk_tier", "")).strip().lower()
    if final_action == "avoid" or action == "avoid" or risk_tier == "high":
        return False
    if risk_candidate_action in {"candidate", "pass", "low_risk"}:
        return True
    if risk_action in {"pass", "candidate", "low_risk"}:
        return True
    if risk_tier in {"low", "medium", "中", "低"}:
        return True
    return action == "candidate"


def _move_column_after(frame: pd.DataFrame, column: str, after: str) -> pd.DataFrame:
    if column not in frame.columns or after not in frame.columns:
        return frame
    columns = [item for item in frame.columns if item != column]
    index = columns.index(after) + 1
    columns.insert(index, column)
    return frame.loc[:, columns].copy()


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
        except (FileNotFoundError, DailyBarsReadError) as exc:
            if isinstance(exc, DailyBarsReadError):
                logging.warning("Skip TradingView for %s because cached daily bars are unreadable: %s", symbol, exc)
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
        except (FileNotFoundError, DailyBarsReadError) as exc:
            if isinstance(exc, DailyBarsReadError):
                logging.warning("Skip MACD for %s because cached daily bars are unreadable: %s", symbol, exc)
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
        except (FileNotFoundError, DailyBarsReadError) as exc:
            if isinstance(exc, DailyBarsReadError):
                logging.warning("Skip ATR for %s because cached daily bars are unreadable: %s", symbol, exc)
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
        except (FileNotFoundError, DailyBarsReadError) as exc:
            if isinstance(exc, DailyBarsReadError):
                logging.warning("Skip cached daily history for %s because it is unreadable: %s", symbol, exc)
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


def _parse_optional_date(value: str | None) -> date | None:
    if value is None:
        return None
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


def _parse_str_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one text value")
    return items


def _parse_optional_symbol_list(value: str | None) -> list[str] | None:
    if value is None or not str(value).strip():
        return None
    return [item.zfill(6) for item in _parse_str_list(value)]


def _infer_report_date_from_path(path: Path) -> date:
    stem = path.stem
    for token in reversed(stem.split("_")):
        try:
            return datetime.fromisoformat(token).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot infer report date from input path: {path}")


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


def _command_needs_network(command_name: str) -> bool:
    return command_name in {"update", "intraday-screening"}


def _create_update_data_provider(data_interface: str):
    normalized = str(data_interface or "baostock").strip().lower()
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
