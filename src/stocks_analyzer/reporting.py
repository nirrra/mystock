from __future__ import annotations

import pandas as pd


DISPLAY_COLUMNS = [
    "trade_date",
    "symbol",
    "name",
    "pattern_id",
    "strategy_name",
    "close",
    "old_high_date",
    "old_high_price",
    "distance_to_old_high_pct",
    "extension_above_old_high_pct",
    "ma20_slope_short_pct",
    "ma20_slope_long_pct",
    "ma60_slope_short_pct",
    "ma60_slope_long_pct",
    "pullback_volume_contraction_ratio",
    "breakout_date",
    "breakout_volume_ratio",
    "breakout_close_position",
    "breakout_upper_shadow_pct",
    "breakout_body_pct",
    "breakout_turnover",
    "breakout_turnover_state",
    "post_breakout_max_high_extension_pct",
    "platform_volume_contraction_ratio",
    "platform_range_contraction_ratio",
    "platform_low_lift_pct",
    "platform_max_bearish_body_pct",
    "platform_max_bearish_volume_ratio",
    "pullback_max_rise_tail_volume_ratio",
    "volume_ratio_20",
    "reason",
]


def format_report(dataframe: pd.DataFrame, limit: int) -> str:
    if dataframe.empty:
        return "No candidates matched the selected type."

    display = dataframe.copy()
    if "symbol" in display.columns:
        display["symbol"] = display["symbol"].map(_display_symbol)
    for column in (
        "return_15d",
        "distance_to_old_high_pct",
        "extension_above_old_high_pct",
        "breakout_close_position",
        "breakout_upper_shadow_pct",
        "breakout_body_pct",
        "post_breakout_max_high_extension_pct",
        "ma20_slope_short_pct",
        "ma20_slope_long_pct",
        "ma60_slope_short_pct",
        "ma60_slope_long_pct",
        "platform_low_lift_pct",
        "platform_max_bearish_body_pct",
    ):
        if column in display.columns:
            display[column] = display[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "nan")
    for column in (
        "close",
        "old_high_price",
        "volume_ratio_20",
        "breakout_volume_ratio",
        "breakout_turnover",
        "pullback_volume_contraction_ratio",
        "platform_volume_contraction_ratio",
        "platform_range_contraction_ratio",
        "platform_max_bearish_volume_ratio",
        "pullback_max_rise_tail_volume_ratio",
    ):
        if column in display.columns:
            display[column] = display[column].map(lambda value: f"{value:.2f}" if pd.notna(value) else "nan")

    available = [column for column in DISPLAY_COLUMNS if column in display.columns]
    return display.loc[:, available].head(limit).to_string(index=False)


def format_multi_pattern_summary(dataframe: pd.DataFrame) -> str:
    if dataframe.empty or "pattern_id" not in dataframe.columns or "symbol" not in dataframe.columns:
        return ""

    summary_lines: list[str] = []
    grouped = dataframe.groupby(["symbol", "name"], dropna=False)["pattern_id"].apply(
        lambda values: sorted({str(item) for item in values})
    )
    repeated = grouped[grouped.map(len) > 1]
    if repeated.empty:
        return ""

    summary_lines.append("多模式命中:")
    for (symbol, name), pattern_ids in repeated.items():
        display_symbol = _display_symbol(symbol)
        display_name = "" if pd.isna(name) else str(name)
        summary_lines.append(f"  {display_symbol} {display_name}: {', '.join(pattern_ids)}")
    return "\n".join(summary_lines)


def _display_symbol(value: object) -> str:
    text = str(value)
    if text.startswith('="') and text.endswith('"'):
        return text[2:-1]
    return text
