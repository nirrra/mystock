from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocks_analyzer.config import load_config
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.plotting import default_start_date, filter_by_date, load_or_fetch_daily, plot_candles_and_volume
from stocks_analyzer.storage import Storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot candlestick and volume chart for one stock.")
    parser.add_argument("symbol", help="6-digit stock code, e.g. 603588")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--start-date", default=default_start_date(), help="YYYYMMDD, default is 2 years ago")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"), help="YYYYMMDD, default is today")
    parser.add_argument("--output", default=None, help="Optional PNG output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    project_root = ROOT
    config = load_config(project_root / args.config)
    paths = ProjectPaths(project_root, config.storage)
    storage = Storage(paths)

    symbol = str(args.symbol).zfill(6)
    dataframe = load_or_fetch_daily(
        storage=storage,
        provider_name=config.provider,
        symbol=symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        adjust=config.adjustment,
    )
    filtered = filter_by_date(dataframe, args.start_date, args.end_date)
    if filtered.empty:
        raise RuntimeError(f"No data available for {symbol} in {args.start_date} to {args.end_date}")

    output_path = Path(args.output) if args.output else paths.reports_dir / "plots" / f"{symbol}_2y.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_candles_and_volume(filtered, symbol, output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
