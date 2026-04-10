from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

from .data_sources import create_data_provider
from .storage import Storage


def default_start_date() -> str:
    return (date.today() - timedelta(days=365 * 2)).strftime("%Y%m%d")


def normalize_date(value: str) -> str:
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    return value[:10]


def load_or_fetch_daily(
    storage: Storage,
    provider_name: str,
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
) -> pd.DataFrame:
    needs_refresh = True
    try:
        cached = storage.load_daily_bars(symbol)
        filtered = filter_by_date(cached, start_date, end_date)
        if not filtered.empty:
            cache_min = cached["trade_date"].min().date()
            cache_max = cached["trade_date"].max().date()
            requested_start = pd.Timestamp(normalize_date(start_date)).date()
            requested_end = pd.Timestamp(normalize_date(end_date)).date()
            needs_refresh = cache_min > requested_start or cache_max < requested_end
        if not needs_refresh:
            return cached
    except FileNotFoundError:
        cached = None

    provider = create_data_provider(provider_name)
    try:
        fresh = provider.get_daily_bars(symbol, start_date=start_date, end_date=end_date, adjust=adjust)
    finally:
        provider.close()

    storage.save_daily_bars(symbol, fresh)
    return fresh


def filter_by_date(dataframe: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(normalize_date(start_date))
    end_ts = pd.Timestamp(normalize_date(end_date))
    filtered = dataframe.copy()
    filtered["trade_date"] = pd.to_datetime(filtered["trade_date"])
    return filtered[(filtered["trade_date"] >= start_ts) & (filtered["trade_date"] <= end_ts)].sort_values("trade_date")


def plot_candles_and_volume(dataframe: pd.DataFrame, symbol: str, output_path: Path) -> None:
    df = dataframe.copy().sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    x_values = mdates.date2num(df["trade_date"].to_numpy())

    fig, (ax_price, ax_volume) = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    candle_width = 0.6
    colors: list[str] = []
    for x, row in zip(x_values, df.itertuples(index=False), strict=True):
        up = row.close >= row.open
        color = "#d62728" if up else "#2ca02c"
        colors.append(color)

        ax_price.vlines(x, row.low, row.high, color=color, linewidth=1)
        body_bottom = min(row.open, row.close)
        body_height = abs(row.close - row.open)
        if body_height == 0:
            body_height = max(row.close * 0.001, 0.001)
        ax_price.add_patch(
            Rectangle(
                (x - candle_width / 2, body_bottom),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=1,
            )
        )

    ax_volume.bar(x_values, df["volume"], width=candle_width, color=colors, align="center")

    ax_price.set_title(f"{symbol} Historical Candlestick (K-line) and Volume")
    ax_price.set_ylabel("Price")
    ax_volume.set_ylabel("Volume")
    ax_volume.set_xlabel("Date")

    ax_price.grid(True, linestyle="--", alpha=0.25)
    ax_volume.grid(True, linestyle="--", alpha=0.25)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax_volume.xaxis.set_major_locator(locator)
    ax_volume.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
