from __future__ import annotations

import argparse
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from time import perf_counter

import pandas as pd

from .config import load_config
from .data_sources import create_data_provider
from .daily_screening import run_daily_screening
from .macd_divergence import summarize_recent_macd_divergence
from .features import build_feature_frame
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
from .universe import build_main_board_universe
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
    "pattern1": "type1",
    "pattern2": "type2",
    "pattern3": "type3",
    "pattern4": "type4",
}
PATTERN_LABEL_MAP = {
    "type1": "1",
    "type2": "2",
    "type3": "3",
    "type4": "4",
}
PROGRESS_LOG_INTERVAL = 100


def build_parser() -> argparse.ArgumentParser:
    _localize_argparse()
    parser = argparse.ArgumentParser(
        description="A 股主板技术分析命令行工具",
        epilog=(
            "第一次使用建议顺序：\n"
            "  1. mystock update --start-date 20240101\n"
            "     更新主板股票池并拉取本地日线数据。\n"
            "  2. mystock pattern\n"
            "     扫描本地全部股票，识别 1 到 4 号模式并生成 CSV。\n"
            "  3. mystock plot 603588\n"
            "     查看单只股票近两年的 K 线和成交量图。\n\n"
            "常见示例：\n"
            "  mystock update --start-date 20240101 --skip-existing\n"
            "  mystock update 603588 --start-date 20240101\n"
            "  mystock pattern --1 --4\n"
            "  mystock report --date 2026-04-10\n"
            "  mystock tradingview --date 2026-04-10\n"
            "  mystock divergence --date 2026-04-10\n"
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
            "  mystock update --start-date 20240101 --skip-existing\n"
            "  mystock update 603588 --start-date 20240101\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    update.add_argument("symbol", nargs="?", help="可选的 6 位股票代码，只更新该股票")
    update.add_argument("--start-date", default="20230101", help="开始日期，格式 YYYYMMDD")
    update.add_argument("--end-date", default=datetime.today().strftime("%Y%m%d"), help="结束日期，格式 YYYYMMDD")
    update.add_argument("--limit", type=int, default=None, help="仅更新前 N 只股票，便于小范围测试")
    update.add_argument("--skip-existing", action="store_true", help="跳过本地已有缓存的股票")

    pattern = subparsers.add_parser(
        "pattern",
        help="识别本地日线数据中的 1 到 4 号模式",
        description=(
            "扫描本地缓存的全部股票日线数据，识别模式 1 到 4。\n"
            "默认识别全部模式；如果传入 --1 --2 --3 --4 中的任意组合，则只识别指定模式。"
        ),
        epilog=(
            "常见示例：\n"
            "  mystock pattern\n"
            "  mystock pattern --1\n"
            "  mystock pattern --2 --4\n"
            "  mystock pattern --as-of 2026-04-10 --output reports/my_patterns.csv\n"
            "  mystock pattern --plot-all\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    pattern.add_argument("--1", dest="pattern1", action="store_true", help="只识别模式 1")
    pattern.add_argument("--2", dest="pattern2", action="store_true", help="只识别模式 2")
    pattern.add_argument("--3", dest="pattern3", action="store_true", help="只识别模式 3")
    pattern.add_argument("--4", dest="pattern4", action="store_true", help="只识别模式 4")
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

    divergence = subparsers.add_parser(
        "divergence",
        help="识别指定日期最近 15 个交易日内的 MACD 顶背离/底背离",
        description="读取本地主板日线数据，输出全市场 TradingView 评分和 MACD 背离识别汇总。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    divergence.add_argument("--date", required=True, help="识别日期，格式 YYYY-MM-DD")
    divergence.add_argument("--top-n", type=int, default=20, help="终端展示前 N 行")
    divergence.add_argument("--output", default=None, help="可选的 CSV 输出路径")

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
        help="按交易日执行每日筛选，并把结果插入选股.md 顶部",
        description="自动判断是否为交易日，串行执行 update/tradingview/pattern，再生成当日选股 Markdown。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    daily_screening.add_argument("--date", default=None, help="目标日期，格式 YYYY-MM-DD，默认今天")
    daily_screening.add_argument("--start-date", default="20240101", help="更新数据的起始日期，格式 YYYYMMDD")
    daily_screening.add_argument("--picks-file", default="选股.md", help="选股结果 Markdown 文件名")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    project_root = Path(args.project_root).resolve()
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
                args.skip_existing,
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

    if args.command == "divergence":
        _run_divergence(
            storage=storage,
            config=config,
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
            picks_filename=args.picks_file,
        )
        print(result.message)
        if result.report_path:
            print(f"报告文件：{result.report_path}")
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
    skip_existing: bool,
) -> None:
    if symbol:
        normalized_symbol = str(symbol).zfill(6)
        bars = provider.get_daily_bars(normalized_symbol, start_date=start_date, end_date=end_date, adjust=adjust)
        target = storage.save_daily_bars(normalized_symbol, bars)
        logging.info("Cached %s rows for %s to %s", len(bars), normalized_symbol, target)
        return

    universe = _refresh_or_load_universe(storage, provider, exclude_st)
    symbols = universe["symbol"].tolist()
    if limit is not None:
        symbols = symbols[:limit]

    success_count = 0
    skipped_count = 0
    failed_symbols: list[str] = []

    for index, item_symbol in enumerate(symbols, start=1):
        if skip_existing and storage.has_daily_bars(item_symbol):
            skipped_count += 1
            logging.info("[%s/%s] skipped existing cache for %s", index, len(symbols), item_symbol)
            continue

        try:
            bars = provider.get_daily_bars(item_symbol, start_date=start_date, end_date=end_date, adjust=adjust)
            storage.save_daily_bars(item_symbol, bars)
            success_count += 1
            logging.info("[%s/%s] cached %s rows for %s", index, len(symbols), len(bars), item_symbol)
        except Exception as exc:
            failed_symbols.append(item_symbol)
            logging.warning("[%s/%s] failed to fetch %s: %s", index, len(symbols), item_symbol, exc)

    logging.info(
        "Daily update finished: success=%s skipped=%s failed=%s",
        success_count,
        skipped_count,
        len(failed_symbols),
    )
    if failed_symbols:
        logging.warning("Failed symbols sample: %s", ", ".join(failed_symbols[:20]))


def _run_pattern(
    storage: Storage,
    provider_name: str,
    config,
    as_of: date,
    selected_patterns: list[str],
    limit: int | None,
    output: str | None,
    plot_all: bool,
) -> None:
    _ensure_universe(storage, provider_name, config.universe.exclude_st)
    screener = Screener(storage, config)
    results = screener.run(as_of=as_of, selected_strategies=selected_patterns)
    exported = _prepare_pattern_results(results)
    exported = _append_recent_tradingview_scores(storage, exported, as_of=as_of, lookback_days=5)
    exported = _append_recent_macd_divergence(storage, exported, as_of=as_of)

    output_path = Path(output) if output else _default_pattern_output_path(
        storage,
        as_of=as_of,
        selected_patterns=selected_patterns,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(output_path, index=False, encoding="utf-8-sig")

    if exported.empty:
        logging.info("No patterns matched for %s", as_of.isoformat())
        logging.info("Saved empty pattern report to %s", output_path)
        print(f"No patterns matched. Saved empty CSV to {output_path}")
        return

    multi_pattern_summary = format_multi_pattern_summary(exported)
    if multi_pattern_summary:
        print(multi_pattern_summary)
        print()
    print(format_report(exported, limit=limit or config.screening.output_limit))
    if plot_all:
        plots_dir = _plot_pattern_matches(storage, config, as_of, results)
        print(f"\n已生成图形目录: {plots_dir}")
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
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    summary, daily_frames = _build_tradingview_snapshots(storage, trade_date=trade_date, lookback_days=5)
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


def _run_divergence(
    storage: Storage,
    config,
    trade_date: date,
    top_n: int,
    output: str | None,
) -> None:
    _ensure_universe(storage, config.provider, config.universe.exclude_st)
    summary = _build_macd_divergence_summary(storage, trade_date=trade_date)
    if summary.empty:
        raise RuntimeError(f"No MACD divergence summary could be generated for {trade_date.isoformat()}")

    output_path = Path(output) if output else storage.paths.reports_dir / "divergence" / f"macd_divergence_{trade_date.isoformat()}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")

    display_columns = [
        "symbol",
        "name",
        "avg_all_rating_5d",
        "all_rating_label",
        "macd_top_divergence_15d",
        "macd_bottom_divergence_15d",
        "macd_top_divergence_signal_date",
        "macd_bottom_divergence_signal_date",
    ]
    available = [column for column in display_columns if column in summary.columns]
    print(summary.loc[:, available].head(top_n).to_string(index=False))
    print(f"\nSaved MACD divergence summary to {output_path}")


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
    preferred_order = [
        "trade_date",
        "symbol",
        "name",
        "pattern_id",
        "close",
        "old_high_date",
        "distance_to_old_high_pct",
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
) -> pd.DataFrame:
    if exported.empty or "symbol" not in exported.columns:
        return exported

    summary, _ = _load_or_build_tradingview_summary(storage, trade_date=as_of, lookback_days=lookback_days)
    if summary.empty:
        return exported

    rating_date_columns = sorted(column for column in summary.columns if column.startswith("all_rating_20"))
    if len(rating_date_columns) != lookback_days:
        return exported

    merge_columns = ["symbol", *rating_date_columns, "avg_all_rating_5d", "all_rating_label"]
    tradingview = summary.loc[:, [column for column in merge_columns if column in summary.columns]].copy()
    tradingview["symbol"] = tradingview["symbol"].map(_normalize_exported_symbol)
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


def _append_recent_macd_divergence(
    storage: Storage,
    exported: pd.DataFrame,
    *,
    as_of: date,
) -> pd.DataFrame:
    if exported.empty or "symbol" not in exported.columns:
        return exported

    divergence = _load_or_build_macd_divergence_summary(storage, trade_date=as_of)
    if divergence.empty:
        return exported

    merge_columns = [
        "symbol",
        "macd_top_divergence_15d",
        "macd_bottom_divergence_15d",
    ]
    divergence = divergence.loc[:, [column for column in merge_columns if column in divergence.columns]].copy()
    divergence["symbol"] = divergence["symbol"].map(_normalize_exported_symbol)

    enriched = exported.copy()
    enriched["_normalized_symbol"] = enriched["symbol"].map(_normalize_exported_symbol)
    enriched = enriched.merge(
        divergence,
        how="left",
        left_on="_normalized_symbol",
        right_on="symbol",
    )
    enriched = enriched.drop(columns=["_normalized_symbol", "symbol_y"], errors="ignore")
    enriched = enriched.rename(columns={"symbol_x": "symbol"})

    for column in ("macd_top_divergence_15d", "macd_bottom_divergence_15d"):
        if column in enriched.columns:
            enriched[column] = enriched[column].fillna(False).astype(bool)
    return enriched


def _load_or_build_macd_divergence_summary(storage: Storage, trade_date: date) -> pd.DataFrame:
    default_path = storage.paths.reports_dir / "divergence" / f"macd_divergence_{trade_date.isoformat()}.csv"
    if default_path.exists():
        return pd.read_csv(default_path)
    return _build_macd_divergence_summary(storage, trade_date=trade_date)


def _load_or_build_tradingview_summary(
    storage: Storage,
    *,
    trade_date: date,
    lookback_days: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
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
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    universe = storage.load_universe()
    target_dates = _resolve_recent_trading_dates(storage, as_of=trade_date, lookback_days=lookback_days)
    if len(target_dates) != lookback_days:
        return pd.DataFrame(), []

    summary_rows: list[dict[str, object]] = []
    daily_rows: dict[str, list[dict[str, object]]] = {}
    target_set = set(target_dates)
    instruments = universe.to_dict("records")
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


def _build_macd_divergence_summary(storage: Storage, trade_date: date) -> pd.DataFrame:
    tradingview_summary, _ = _load_or_build_tradingview_summary(storage, trade_date=trade_date, lookback_days=5)
    if tradingview_summary.empty:
        return pd.DataFrame()

    divergence_rows: list[dict[str, object]] = []
    total_rows = len(tradingview_summary)
    logging.info("MACD divergence scan started for %s: %s symbols", trade_date.isoformat(), total_rows)
    for index, (_, row) in enumerate(tradingview_summary.iterrows(), start=1):
        symbol = _normalize_exported_symbol(row["symbol"])
        try:
            bars = storage.load_daily_bars(symbol)
        except FileNotFoundError:
            _log_scan_progress("MACD divergence", index, total_rows)
            continue

        cutoff = bars[pd.to_datetime(bars["trade_date"]).dt.date <= trade_date].reset_index(drop=True)
        if cutoff.empty:
            _log_scan_progress("MACD divergence", index, total_rows)
            continue

        divergence_row = summarize_recent_macd_divergence(cutoff)
        divergence_row["symbol"] = _format_symbol_for_excel(symbol)
        divergence_rows.append(divergence_row)
        _log_scan_progress("MACD divergence", index, total_rows)

    if not divergence_rows:
        return pd.DataFrame()

    divergence = pd.DataFrame(divergence_rows)
    summary = tradingview_summary.merge(divergence, how="left", on="symbol")
    for column in ("macd_top_divergence_15d", "macd_bottom_divergence_15d"):
        if column in summary.columns:
            summary[column] = summary[column].fillna(False).astype(bool)
    return summary


def _log_scan_progress(stage_name: str, current: int, total: int) -> None:
    if total <= 0:
        return
    if current == 1 or current == total or current % PROGRESS_LOG_INTERVAL == 0:
        logging.info("%s progress: %s/%s", stage_name, current, total)


def _resolve_recent_trading_dates(storage: Storage, as_of: date, lookback_days: int) -> list[date]:
    universe = storage.load_universe()
    candidate_dates: set[date] = set()
    for instrument in universe.to_dict("records"):
        symbol = str(instrument["symbol"])
        try:
            bars = storage.load_daily_bars(symbol)
        except FileNotFoundError:
            continue
        trade_dates = pd.to_datetime(bars["trade_date"]).dt.date
        recent_dates = sorted(item for item in trade_dates.unique() if item <= as_of)
        candidate_dates.update(recent_dates[-lookback_days:])

    return sorted(candidate_dates)[-lookback_days:]


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


def _configure_network(network: NetworkConfig) -> None:
    if network.http_proxy:
        os.environ["HTTP_PROXY"] = network.http_proxy
        os.environ["http_proxy"] = network.http_proxy
    if network.https_proxy:
        os.environ["HTTPS_PROXY"] = network.https_proxy
        os.environ["https_proxy"] = network.https_proxy
    if network.no_proxy:
        os.environ["NO_PROXY"] = network.no_proxy
        os.environ["no_proxy"] = network.no_proxy

    if network.http_proxy or network.https_proxy:
        logging.info(
            "Configured proxy: http=%s https=%s",
            network.http_proxy or "-",
            network.https_proxy or "-",
        )


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
