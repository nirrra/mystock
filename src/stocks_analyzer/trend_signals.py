from __future__ import annotations

from datetime import date

import pandas as pd

from .models import AppConfig, TrendBreakoutConfig, TrendPullbackConfig
from .storage import Storage
from .trend_universe import build_symbol_trend_frame


SIGNAL_PRIORITY = {
    "pullback": 0,
    "breakout": 1,
}
SIGNAL_COLUMNS = [
    "trade_date",
    "symbol",
    "name",
    "signal_type",
    "close",
    "trend_score",
    "entry_score",
    "platform_window_days",
    "platform_high",
    "platform_low",
    "platform_range_pct",
    "distance_to_breakout_pct",
    "distance_to_new_high_pct",
    "volume_ratio_20",
    "volume_contraction_ratio",
    "distance_to_ma20",
    "distance_to_ma60",
    "drawdown_from_recent_high",
    "trend_direction_score",
    "trend_strength_score",
    "trend_quality_score",
    "trend_liquidity_score",
    "portfolio_rank_score",
    "trigger_reason",
]


def generate_symbol_trend_signals(
    daily_bars: pd.DataFrame,
    *,
    symbol: str,
    name: str,
    config: AppConfig,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    trend_frame = build_symbol_trend_frame(
        daily_bars,
        symbol=symbol,
        name=name,
        config=config.trend_universe,
    )
    return generate_trend_signals_from_frame(
        trend_frame,
        config=config,
        start_date=start_date,
        end_date=end_date,
    )


def generate_trend_signals_from_frame(
    trend_frame: pd.DataFrame,
    *,
    config: AppConfig,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(len(trend_frame)):
        trade_date = pd.Timestamp(trend_frame.iloc[index]["trade_date"]).date()
        if start_date is not None and trade_date < start_date:
            continue
        if end_date is not None and trade_date > end_date:
            continue

        breakout = _evaluate_breakout_signal(trend_frame, index, config.trend_signals.breakout)
        if breakout is not None:
            rows.append(breakout)

        pullback = _evaluate_pullback_signal(trend_frame, index, config.trend_signals.pullback)
        if pullback is not None:
            rows.append(pullback)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).reindex(columns=SIGNAL_COLUMNS)
    return dedupe_trend_signals(result)


def scan_trend_signals(
    storage: Storage,
    config: AppConfig,
    *,
    trade_date: date | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    if trade_date is not None:
        start_date = trade_date
        end_date = trade_date

    universe = storage.load_universe().copy()
    universe["symbol"] = universe["symbol"].astype(str).str.zfill(6)
    if symbols:
        symbol_set = {str(symbol).zfill(6) for symbol in symbols}
        universe = universe[universe["symbol"].isin(symbol_set)].reset_index(drop=True)

    signal_rows: list[dict[str, object]] = []
    for instrument in universe.to_dict("records"):
        symbol = str(instrument["symbol"]).zfill(6)
        try:
            bars = storage.load_daily_bars(symbol)
        except FileNotFoundError:
            continue
        signal_frame = generate_symbol_trend_signals(
            bars,
            symbol=symbol,
            name=str(instrument.get("name", "")),
            config=config,
            start_date=start_date,
            end_date=end_date,
        )
        if signal_frame.empty:
            continue
        signal_rows.extend(signal_frame.to_dict("records"))

    if not signal_rows:
        return pd.DataFrame()

    result = pd.DataFrame(signal_rows).reindex(columns=SIGNAL_COLUMNS)
    result = result.sort_values(["trade_date", "entry_score", "trend_score", "symbol"], ascending=[True, False, False, True])
    return result.reset_index(drop=True)


def dedupe_trend_signals(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()

    frame = signals.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["_signal_priority"] = frame["signal_type"].map(SIGNAL_PRIORITY).fillna(9)
    frame = frame.sort_values(
        ["trade_date", "symbol", "entry_score", "_signal_priority"],
        ascending=[True, True, False, True],
    )
    frame = frame.drop_duplicates(subset=["trade_date", "symbol"], keep="first")
    frame = frame.drop(columns=["_signal_priority"])
    return frame.reset_index(drop=True)


def _evaluate_breakout_signal(
    frame: pd.DataFrame,
    index: int,
    config: TrendBreakoutConfig,
) -> dict[str, object] | None:
    row = frame.iloc[index]
    if not bool(row.get("in_trend_universe", False)):
        return None
    if index < config.platform_max_window_days:
        return None
    if pd.isna(row.get("volume_ratio_20")):
        return None

    platform = _find_platform_window(frame, index, config)
    if platform is None:
        return None

    close = float(row["close"])
    platform_high = float(platform["platform_high"])
    breakout_distance = 0.0 if platform_high <= 0 else (close - platform_high) / platform_high
    prior_high = float(frame["high"].iloc[max(0, index - config.new_high_lookback_days) : index].max())
    near_new_high = prior_high > 0 and close >= prior_high * (1 - config.new_high_tolerance_pct)
    if breakout_distance < 0 and not near_new_high:
        return None
    if breakout_distance > config.breakout_max_distance_pct:
        return None

    volume_ratio_20 = float(row["volume_ratio_20"])
    if volume_ratio_20 < config.breakout_volume_ratio_min:
        return None

    trend_score = float(row["trend_score"])
    tightness_score = max(0.0, 1 - platform["platform_range_pct"] / max(config.platform_range_max, 1e-6))
    volume_score = min(volume_ratio_20 / max(config.breakout_volume_ratio_min * 1.5, 1e-6), 1.5) / 1.5
    breakout_score = max(0.0, 1 - max(breakout_distance, 0.0) / max(config.breakout_max_distance_pct, 1e-6))
    entry_score = round(0.55 * trend_score + 20.0 * tightness_score + 15.0 * volume_score + 10.0 * breakout_score, 4)

    return {
        "trade_date": pd.Timestamp(row["trade_date"]),
        "symbol": str(row["symbol"]).zfill(6),
        "name": row["name"],
        "signal_type": "breakout",
        "close": close,
        "trend_score": trend_score,
        "entry_score": entry_score,
        "platform_window_days": int(platform["window_days"]),
        "platform_high": round(platform_high, 4),
        "platform_low": round(float(platform["platform_low"]), 4),
        "platform_range_pct": round(float(platform["platform_range_pct"]), 4),
        "distance_to_breakout_pct": round(breakout_distance, 4),
        "distance_to_new_high_pct": round(0.0 if prior_high <= 0 else close / prior_high - 1, 4),
        "volume_ratio_20": round(volume_ratio_20, 4),
        "volume_contraction_ratio": None,
        "distance_to_ma20": round(float(row.get("distance_to_ma20", pd.NA)), 4)
        if pd.notna(row.get("distance_to_ma20"))
        else None,
        "distance_to_ma60": round(float(row.get("distance_to_ma60", pd.NA)), 4)
        if pd.notna(row.get("distance_to_ma60"))
        else None,
        "drawdown_from_recent_high": None,
        "trend_direction_score": round(float(row["trend_direction_score"]), 4),
        "trend_strength_score": round(float(row["trend_strength_score"]), 4),
        "trend_quality_score": round(float(row["trend_quality_score"]), 4),
        "trend_liquidity_score": round(float(row["trend_liquidity_score"]), 4),
        "portfolio_rank_score": None,
        "trigger_reason": (
            f"breakout window={platform['window_days']} range={platform['platform_range_pct']:.2%} "
            f"distance={breakout_distance:.2%} vol20={volume_ratio_20:.2f}"
        ),
    }


def _evaluate_pullback_signal(
    frame: pd.DataFrame,
    index: int,
    config: TrendPullbackConfig,
) -> dict[str, object] | None:
    row = frame.iloc[index]
    if not bool(row.get("in_trend_universe", False)):
        return None
    if index < config.recent_high_lookback_days:
        return None
    if pd.isna(row.get("ma_20")) or pd.isna(row.get("ma_60")):
        return None

    close = float(row["close"])
    recent_high = float(frame["high"].iloc[index - config.recent_high_lookback_days : index].max())
    if recent_high <= 0:
        return None

    drawdown_from_recent_high = 1 - close / recent_high
    if drawdown_from_recent_high < 0 or drawdown_from_recent_high > config.max_drawdown_from_recent_high:
        return None

    distance_to_ma20 = close / float(row["ma_20"]) - 1
    distance_to_ma60 = close / float(row["ma_60"]) - 1
    near_ma20 = abs(distance_to_ma20) <= config.proximity_to_ma20
    near_ma60 = abs(distance_to_ma60) <= config.proximity_to_ma60
    if not near_ma20 and not near_ma60:
        return None

    volume_ma_5 = row.get("volume_ma_5")
    volume_ma_20 = row.get("volume_ma_20")
    if pd.isna(volume_ma_5) or pd.isna(volume_ma_20) or float(volume_ma_20) <= 0:
        return None
    volume_contraction_ratio = float(volume_ma_5) / float(volume_ma_20)
    if volume_contraction_ratio > config.volume_contraction_max:
        return None

    rebound_ok = _safe_float(row.get("return_1d"), 0.0) >= config.rebound_min_return_1d
    lower_shadow_ok = _safe_float(row.get("lower_shadow_pct"), 0.0) >= config.lower_shadow_min
    if not rebound_ok and not lower_shadow_ok:
        return None

    trend_score = float(row["trend_score"])
    proximity = min(abs(distance_to_ma20), abs(distance_to_ma60))
    proximity_limit = config.proximity_to_ma20 if near_ma20 else config.proximity_to_ma60
    proximity_score = max(0.0, 1 - proximity / max(proximity_limit, 1e-6))
    volume_score = max(0.0, 1 - volume_contraction_ratio / max(config.volume_contraction_max, 1e-6))
    rebound_score = 1.0 if rebound_ok else 0.6
    entry_score = round(0.55 * trend_score + 20.0 * proximity_score + 15.0 * volume_score + 10.0 * rebound_score, 4)

    return {
        "trade_date": pd.Timestamp(row["trade_date"]),
        "symbol": str(row["symbol"]).zfill(6),
        "name": row["name"],
        "signal_type": "pullback",
        "close": close,
        "trend_score": trend_score,
        "entry_score": entry_score,
        "platform_window_days": None,
        "platform_high": None,
        "platform_low": None,
        "platform_range_pct": None,
        "distance_to_breakout_pct": None,
        "distance_to_new_high_pct": round(close / recent_high - 1, 4),
        "volume_ratio_20": round(_safe_float(row.get("volume_ratio_20"), 0.0), 4) if pd.notna(row.get("volume_ratio_20")) else None,
        "volume_contraction_ratio": round(volume_contraction_ratio, 4),
        "distance_to_ma20": round(distance_to_ma20, 4),
        "distance_to_ma60": round(distance_to_ma60, 4),
        "drawdown_from_recent_high": round(drawdown_from_recent_high, 4),
        "trend_direction_score": round(float(row["trend_direction_score"]), 4),
        "trend_strength_score": round(float(row["trend_strength_score"]), 4),
        "trend_quality_score": round(float(row["trend_quality_score"]), 4),
        "trend_liquidity_score": round(float(row["trend_liquidity_score"]), 4),
        "portfolio_rank_score": None,
        "trigger_reason": (
            f"pullback drawdown={drawdown_from_recent_high:.2%} "
            f"dist_ma20={distance_to_ma20:.2%} dist_ma60={distance_to_ma60:.2%} "
            f"vol5_20={volume_contraction_ratio:.2f}"
        ),
    }


def _find_platform_window(frame: pd.DataFrame, index: int, config: TrendBreakoutConfig) -> dict[str, float] | None:
    best: dict[str, float] | None = None
    for window_days in range(config.platform_min_window_days, config.platform_max_window_days + 1):
        start_index = index - window_days
        if start_index < 0:
            continue
        platform_slice = frame.iloc[start_index:index]
        if len(platform_slice) != window_days:
            continue

        platform_high = float(platform_slice["high"].max())
        platform_low = float(platform_slice["low"].min())
        if platform_high <= 0:
            continue
        platform_range_pct = (platform_high - platform_low) / platform_high
        if platform_range_pct > config.platform_range_max:
            continue

        candidate = {
            "window_days": float(window_days),
            "platform_high": platform_high,
            "platform_low": platform_low,
            "platform_range_pct": platform_range_pct,
        }
        if best is None or candidate["platform_range_pct"] < best["platform_range_pct"]:
            best = candidate
    return best


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return float(default)
    return float(value)
