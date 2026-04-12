from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .indicators import add_indicators


@dataclass(slots=True)
class MacdDivergenceConfig:
    lookback_days: int = 15
    pivot_left_bars: int = 2
    pivot_right_bars: int = 2


def summarize_recent_macd_divergence(
    dataframe: pd.DataFrame,
    config: MacdDivergenceConfig | None = None,
) -> dict[str, object]:
    divergence_config = config or MacdDivergenceConfig()
    df = _prepare_frame(dataframe)
    default_result = {
        "macd_top_divergence_15d": False,
        "macd_bottom_divergence_15d": False,
        "macd_top_divergence_signal_date": None,
        "macd_bottom_divergence_signal_date": None,
    }
    if len(df) < divergence_config.pivot_left_bars + divergence_config.pivot_right_bars + 3:
        return default_result

    high_pivots = _collect_price_pivots(df, "high", divergence_config)
    low_pivots = _collect_price_pivots(df, "low", divergence_config)
    recent_start_index = max(0, len(df) - divergence_config.lookback_days)

    top_signals = _collect_divergence_signals(high_pivots, recent_start_index, mode="top")
    bottom_signals = _collect_divergence_signals(low_pivots, recent_start_index, mode="bottom")

    latest_top = top_signals[-1] if top_signals else None
    latest_bottom = bottom_signals[-1] if bottom_signals else None
    return {
        "macd_top_divergence_15d": latest_top is not None,
        "macd_bottom_divergence_15d": latest_bottom is not None,
        "macd_top_divergence_signal_date": latest_top["signal_date"] if latest_top else None,
        "macd_bottom_divergence_signal_date": latest_bottom["signal_date"] if latest_bottom else None,
    }


def _prepare_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    required = {"macd_dif", "macd_dea", "macd_hist"}
    if required.issubset(dataframe.columns):
        return dataframe.copy().sort_values("trade_date").reset_index(drop=True)
    return add_indicators(dataframe).sort_values("trade_date").reset_index(drop=True)


def _collect_price_pivots(
    df: pd.DataFrame,
    price_column: str,
    config: MacdDivergenceConfig,
) -> list[dict[str, object]]:
    pivots: list[dict[str, object]] = []
    prices = df[price_column].astype(float).tolist()
    dif_values = df["macd_dif"].astype(float).tolist()
    left = config.pivot_left_bars
    right = config.pivot_right_bars

    for index in range(left, len(df) - right):
        dif_value = dif_values[index]
        if pd.isna(dif_value):
            continue

        window = prices[index - left : index + right + 1]
        center = prices[index]
        neighbors = window[:left] + window[left + 1 :]
        if price_column == "high":
            is_pivot = center == max(window) and all(center > item for item in neighbors)
        else:
            is_pivot = center == min(window) and all(center < item for item in neighbors)
        if not is_pivot:
            continue

        confirm_index = index + right
        pivots.append(
            {
                "pivot_index": index,
                "confirm_index": confirm_index,
                "price_value": center,
                "macd_dif": float(dif_value),
                "pivot_date": pd.Timestamp(df.iloc[index]["trade_date"]).date().isoformat(),
                "signal_date": pd.Timestamp(df.iloc[confirm_index]["trade_date"]).date().isoformat(),
            }
        )

    return pivots


def _collect_divergence_signals(
    pivots: list[dict[str, object]],
    recent_start_index: int,
    *,
    mode: str,
) -> list[dict[str, object]]:
    signals: list[dict[str, object]] = []
    for previous, current in zip(pivots, pivots[1:]):
        if int(current["confirm_index"]) < recent_start_index:
            continue

        previous_price = float(previous["price_value"])
        current_price = float(current["price_value"])
        previous_dif = float(previous["macd_dif"])
        current_dif = float(current["macd_dif"])

        if mode == "top":
            matched = current_price > previous_price and current_dif < previous_dif
        else:
            matched = current_price < previous_price and current_dif > previous_dif
        if not matched:
            continue

        signals.append(
            {
                "first_pivot_date": previous["pivot_date"],
                "second_pivot_date": current["pivot_date"],
                "signal_date": current["signal_date"],
            }
        )

    return signals
