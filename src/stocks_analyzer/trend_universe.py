from __future__ import annotations

from datetime import date

import pandas as pd

from .features import build_feature_frame
from .models import AppConfig, TrendUniverseConfig
from .storage import Storage


def build_symbol_trend_frame(
    daily_bars: pd.DataFrame,
    *,
    symbol: str,
    name: str,
    config: TrendUniverseConfig,
) -> pd.DataFrame:
    frame = build_feature_frame(daily_bars)
    frame = frame.copy().sort_values("trade_date").reset_index(drop=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["symbol"] = str(symbol).zfill(6)
    frame["name"] = name

    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    amount = frame["amount"].astype(float)

    short_col = _ensure_ma(frame, close, config.ma_short_window)
    medium_col = _ensure_ma(frame, close, config.ma_medium_window)
    long_col = _ensure_ma(frame, close, config.ma_long_window)

    strength_lookback = int(config.strength_lookback_days)
    quality_lookback = int(config.quality_lookback_days)
    slope_lookback = int(config.slope_lookback_days)
    high_lookback = int(config.high_lookback_days)

    frame["trend_return_strength_lookback"] = close.pct_change(strength_lookback)
    frame["trend_return_20d"] = close.pct_change(20)
    frame["trend_return_120d"] = close.pct_change(120)
    frame["trend_drawdown_quality_lookback"] = 1 - close.div(close.rolling(quality_lookback).max())
    frame["trend_distance_to_high_pct"] = close.div(close.rolling(high_lookback).max()) - 1
    frame["trend_up_ratio_20"] = frame["up_day"].rolling(20).mean()
    frame["trend_up_ratio_strength_lookback"] = frame["up_day"].rolling(strength_lookback).mean()
    absolute_path = frame["return_1d"].abs().rolling(strength_lookback).sum()
    frame["trend_efficiency_strength_lookback"] = close.pct_change(strength_lookback).abs().div(
        absolute_path.replace(0.0, pd.NA)
    )
    frame["trend_amount_stability_20"] = frame["amount_ma_20"].div(amount.rolling(20).std().replace(0.0, pd.NA))
    frame["trend_range_quality_lookback"] = high.rolling(quality_lookback).max().div(low.rolling(quality_lookback).min()) - 1

    frame["trend_ma_short_rising"] = frame[short_col] > frame[short_col].shift(slope_lookback)
    frame["trend_ma_medium_rising"] = frame[medium_col] > frame[medium_col].shift(slope_lookback)
    frame["trend_ma_alignment"] = (frame[short_col] > frame[medium_col]) & (frame[medium_col] > frame[long_col])
    frame["trend_price_above_short"] = close >= frame[short_col]
    frame["trend_price_above_medium"] = close >= frame[medium_col]
    frame["trend_price_location_ok"] = frame["trend_price_above_short"] & frame["trend_price_above_medium"]
    frame["trend_strength_ok"] = frame["trend_return_strength_lookback"] >= config.min_return_strength_lookback
    frame["trend_quality_ok"] = (
        frame["trend_drawdown_quality_lookback"] <= config.max_drawdown_quality_lookback
    ) & frame["volatility_20d"].notna()
    frame["trend_liquidity_ok"] = frame["amount_ma_20"] >= config.min_avg_amount_20d

    minimum_history = max(
        config.min_history_days,
        config.ma_long_window + slope_lookback,
        strength_lookback + 1,
        quality_lookback + 1,
        high_lookback + 1,
    )
    frame["trend_has_min_history"] = pd.Series(range(len(frame))) >= (minimum_history - 1)

    frame["in_trend_universe"] = (
        frame["trend_has_min_history"]
        & frame["trend_ma_alignment"]
        & frame["trend_ma_short_rising"]
        & frame["trend_ma_medium_rising"]
        & frame["trend_price_location_ok"]
        & frame["trend_strength_ok"]
        & frame["trend_quality_ok"]
        & frame["trend_liquidity_ok"]
    )

    frame["trend_direction_score"] = (
        40.0 * frame["trend_ma_alignment"].astype(float)
        + 25.0 * frame["trend_ma_short_rising"].astype(float)
        + 15.0 * frame["trend_ma_medium_rising"].astype(float)
        + 10.0 * frame["trend_price_above_short"].astype(float)
        + 10.0 * frame["trend_price_above_medium"].astype(float)
    )
    frame["trend_strength_score"] = 100.0 * (
        0.45 * _clip01(frame["trend_return_20d"] / 0.15)
        + 0.35 * _clip01(frame["trend_return_strength_lookback"] / max(config.min_return_strength_lookback * 2.0, 1e-6))
        + 0.20 * _clip01((0.10 + frame["trend_distance_to_high_pct"]) / 0.10)
    )
    frame["trend_quality_score"] = 100.0 * (
        0.40 * _clip01(1 - frame["trend_drawdown_quality_lookback"] / max(config.max_drawdown_quality_lookback, 1e-6))
        + 0.20 * _clip01(1 - frame["volatility_20d"] / 0.20)
        + 0.20 * _clip01(frame["trend_up_ratio_strength_lookback"])
        + 0.20 * _clip01(frame["trend_efficiency_strength_lookback"])
    )
    frame["trend_liquidity_score"] = 100.0 * (
        0.70 * _clip01(frame["amount_ma_20"] / max(config.min_avg_amount_20d * 3.0, 1.0))
        + 0.30 * _clip01(frame["trend_amount_stability_20"] / 5.0)
    )
    frame["trend_score"] = (
        0.35 * frame["trend_direction_score"]
        + 0.30 * frame["trend_strength_score"]
        + 0.20 * frame["trend_quality_score"]
        + 0.15 * frame["trend_liquidity_score"]
    ).round(4)

    preferred_columns = [
        "trade_date",
        "symbol",
        "name",
        "close",
        "amount_ma_20",
        short_col,
        medium_col,
        long_col,
        "trend_return_20d",
        "trend_return_strength_lookback",
        "trend_return_120d",
        "trend_drawdown_quality_lookback",
        "trend_distance_to_high_pct",
        "volatility_20d",
        "trend_up_ratio_strength_lookback",
        "trend_efficiency_strength_lookback",
        "trend_amount_stability_20",
        "trend_direction_score",
        "trend_strength_score",
        "trend_quality_score",
        "trend_liquidity_score",
        "trend_score",
        "trend_ma_alignment",
        "trend_ma_short_rising",
        "trend_ma_medium_rising",
        "trend_price_location_ok",
        "trend_strength_ok",
        "trend_quality_ok",
        "trend_liquidity_ok",
        "in_trend_universe",
    ]
    available_columns = [column for column in preferred_columns if column in frame.columns]
    remaining_columns = [column for column in frame.columns if column not in available_columns]
    return frame.loc[:, available_columns + remaining_columns]


def scan_trend_universe(
    storage: Storage,
    config: AppConfig,
    *,
    as_of: date,
    symbols: list[str] | None = None,
    include_all: bool = False,
) -> pd.DataFrame:
    universe = _load_instruments(storage, symbols=symbols)
    rows: list[dict[str, object]] = []
    for instrument in universe:
        symbol = str(instrument["symbol"]).zfill(6)
        try:
            daily_bars = storage.load_daily_bars(symbol)
        except FileNotFoundError:
            continue

        trend_frame = build_symbol_trend_frame(
            daily_bars,
            symbol=symbol,
            name=str(instrument.get("name", "")),
            config=config.trend_universe,
        )
        latest = trend_frame[trend_frame["trade_date"].dt.date <= as_of].tail(1)
        if latest.empty:
            continue
        record = latest.iloc[-1].to_dict()
        if not include_all and not bool(record.get("in_trend_universe", False)):
            continue
        rows.append(record)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result = result.sort_values(["in_trend_universe", "trend_score", "symbol"], ascending=[False, False, True]).reset_index(
        drop=True
    )
    return result


def _load_instruments(storage: Storage, symbols: list[str] | None = None) -> list[dict[str, object]]:
    universe = storage.load_universe().copy()
    universe["symbol"] = universe["symbol"].astype(str).str.zfill(6)
    if symbols:
        symbol_set = {str(symbol).zfill(6) for symbol in symbols}
        universe = universe[universe["symbol"].isin(symbol_set)].reset_index(drop=True)
    return universe.to_dict("records")


def _ensure_ma(frame: pd.DataFrame, close: pd.Series, window: int) -> str:
    column = f"ma_{int(window)}"
    if column not in frame.columns:
        frame[column] = close.rolling(window).mean()
    return column


def _clip01(values: pd.Series) -> pd.Series:
    return values.astype(float).clip(lower=0.0, upper=1.0).fillna(0.0)
