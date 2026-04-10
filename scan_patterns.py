from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocks_analyzer.pattern_scan import PatternScanConfig, scan_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan all cached daily bars for old-high pattern setups.")
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data" / "daily"),
        help="Directory containing per-symbol parquet files.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "pattern_scan.csv"),
        help="CSV output path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = scan_directory(Path(args.data_dir), PatternScanConfig())
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")

    if results.empty:
        print(f"No matches found. Saved empty result to {output_path}")
        return

    matched = results[results["pattern_type"] != "error"].copy()
    print(f"Saved {len(results)} rows to {output_path}")
    if not matched.empty:
        summary = matched["pattern_type"].value_counts().to_dict()
        print(f"Summary: {summary}")
        display_columns = [
            "symbol",
            "pattern_type",
            "current_date",
            "old_high_date",
            "old_high_price",
            "current_close",
            "distance_to_old_high_pct",
            "breakout_date",
        ]
        available = [column for column in display_columns if column in matched.columns]
        print(matched.loc[:, available].head(20).to_string(index=False))
    else:
        print("Only error rows were generated. Check the CSV for details.")


if __name__ == "__main__":
    main()
