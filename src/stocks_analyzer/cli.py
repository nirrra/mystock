from __future__ import annotations

import argparse
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import load_config
from .data_sources import create_data_provider
from .models import NetworkConfig
from .paths import ProjectPaths
from .plotting import default_start_date, filter_by_date, load_or_fetch_daily, plot_candles_and_volume
from .reporting import format_multi_pattern_summary, format_report
from .screener import Screener, parse_as_of
from .storage import Storage
from .strategies import STRATEGY_NAMES
from .universe import build_main_board_universe


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

    universe = _refresh_universe(storage, provider, exclude_st)
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

    output_path = Path(output) if output else storage.paths.reports_dir / _default_pattern_filename(as_of, selected_patterns)
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
    report_path = storage.paths.reports_dir / f"patterns_all_{trade_date.isoformat()}.csv"
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


def _default_pattern_filename(as_of: date, selected_patterns: list[str]) -> str:
    if len(selected_patterns) == len(STRATEGY_NAMES):
        label = "all"
    else:
        pattern_labels = [PATTERN_LABEL_MAP[item] for item in selected_patterns]
        label = "-".join(pattern_labels)
    return f"patterns_{label}_{as_of.isoformat()}.csv"


def _format_symbol_for_excel(value: object) -> str:
    symbol = str(value).zfill(6)
    return f'="{symbol}"'


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
    target = storage.save_universe(universe)
    logging.info("Saved %s symbols to %s", len(universe), target)
    return universe


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
