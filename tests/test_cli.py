from stocks_analyzer.cli import PATTERN_LABEL_MAP, _prepare_pattern_results, build_parser
import pandas as pd


def test_build_parser_accepts_update_with_symbol() -> None:
    parser = build_parser()
    args = parser.parse_args(["update", "603588", "--start-date", "20240101"])

    assert args.command == "update"
    assert args.symbol == "603588"
    assert args.start_date == "20240101"


def test_build_parser_accepts_pattern_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["pattern", "--1", "--4", "--plot-all", "--as-of", "2026-04-10"])

    assert args.command == "pattern"
    assert args.pattern1 is True
    assert args.pattern4 is True
    assert args.plot_all is True
    assert args.as_of == "2026-04-10"


def test_build_parser_accepts_plot_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["plot", "603588", "--start-date", "20240101"])

    assert args.command == "plot"
    assert args.symbol == "603588"
    assert args.start_date == "20240101"


def test_prepare_pattern_results_maps_internal_type_to_pattern_id() -> None:
    results = pd.DataFrame(
        [
            {
                "trade_date": "2026-04-10",
                "symbol": "600000",
                "name": "测试股份",
                "strategy_name": "type1",
                "close": 10.0,
                "reason": "demo",
            }
        ]
    )

    prepared = _prepare_pattern_results(results)

    assert prepared["pattern_id"].tolist() == [PATTERN_LABEL_MAP["type1"]]
    assert "strategy_name" not in prepared.columns
