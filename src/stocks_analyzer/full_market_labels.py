from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .storage import DailyBarsReadError, Storage


TAIL_RISK_FEATURE_COLUMNS = (
    "log_return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "volatility_60d",
    "downside_volatility_20d",
    "max_drawdown_20d",
    "distance_to_ma20",
    "distance_to_ma60",
    "volume_ratio_5d_20d",
    "amount_log",
    "intraday_range_pct",
)


def build_tail_risk_frame(
    bars: pd.DataFrame,
    *,
    symbol: str,
    name: str = "",
    lookback_days: int = 100,
    quantile: float = 0.05,
    horizon_days: int = 1,
) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    frame = bars.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()

    for column in ("open", "high", "low", "close", "volume", "amount"):
        values = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
        frame[column] = pd.to_numeric(values, errors="coerce")
    close = frame["close"].where(frame["close"].gt(0))
    log_return = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    tail_threshold = log_return.shift(1).rolling(lookback_days, min_periods=lookback_days).quantile(quantile)
    tail_event = log_return.lt(tail_threshold)

    result = pd.DataFrame(
        {
            "trade_date": frame["trade_date"],
            "symbol": str(symbol).zfill(6),
            "name": name,
            "log_return_1d": log_return,
            "tail_threshold_past": tail_threshold,
            "tail_event_today": tail_event.astype("float"),
            "risk_label": tail_event.shift(-horizon_days).astype("float"),
            "forward_log_return": log_return.shift(-horizon_days),
        }
    )
    result["return_5d"] = close.pct_change(5, fill_method=None)
    result["return_20d"] = close.pct_change(20, fill_method=None)
    result["return_60d"] = close.pct_change(60, fill_method=None)
    result["volatility_20d"] = log_return.rolling(20, min_periods=20).std()
    result["volatility_60d"] = log_return.rolling(60, min_periods=60).std()
    downside = log_return.where(log_return < 0, 0.0)
    result["downside_volatility_20d"] = downside.rolling(20, min_periods=20).std()
    rolling_max = close.rolling(20, min_periods=20).max()
    result["max_drawdown_20d"] = close.div(rolling_max).sub(1.0)
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    result["distance_to_ma20"] = close.div(ma20).sub(1.0)
    result["distance_to_ma60"] = close.div(ma60).sub(1.0)
    volume = frame["volume"]
    result["volume_ratio_5d_20d"] = volume.rolling(5, min_periods=5).mean().div(volume.rolling(20, min_periods=20).mean())
    result["amount_log"] = np.log1p(frame["amount"].where(frame["amount"].ge(0)))
    result["intraday_range_pct"] = frame["high"].sub(frame["low"]).div(close)
    result["future_max_drawdown_5d"] = _future_min_return(close, horizon=5)
    result["future_return_5d"] = close.shift(-5).div(close).sub(1.0)
    return result


def build_tail_risk_panel(
    *,
    storage: Storage,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    lookback_days: int = 100,
    quantile: float = 0.05,
    horizon_days: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = storage.load_universe().copy()
    if limit is not None:
        universe = universe.head(max(int(limit), 0)).copy()
    rows: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []
    for instrument in universe.to_dict("records"):
        symbol = str(instrument.get("symbol", "")).zfill(6)
        name = str(instrument.get("name", ""))
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError) as exc:
            skipped.append({"symbol": symbol, "name": name, "reason": type(exc).__name__})
            continue
        frame = build_tail_risk_frame(
            bars,
            symbol=symbol,
            name=name,
            lookback_days=lookback_days,
            quantile=quantile,
            horizon_days=horizon_days,
        )
        if frame.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "empty_tail_risk_frame"})
            continue
        if start_date is not None:
            frame = frame[frame["trade_date"].dt.date >= start_date]
        if end_date is not None:
            frame = frame[frame["trade_date"].dt.date <= end_date]
        frame = frame.dropna(subset=["risk_label", *TAIL_RISK_FEATURE_COLUMNS]).copy()
        if frame.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "no_labeled_rows"})
            continue
        rows.append(frame)
    dataset = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not dataset.empty:
        dataset = dataset.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return dataset, pd.DataFrame(skipped)


def _future_min_return(close: pd.Series, *, horizon: int) -> pd.Series:
    values = []
    for offset in range(1, horizon + 1):
        values.append(close.shift(-offset).div(close).sub(1.0))
    if not values:
        return pd.Series(np.nan, index=close.index)
    return pd.concat(values, axis=1).min(axis=1)
