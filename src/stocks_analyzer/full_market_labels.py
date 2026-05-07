from __future__ import annotations

from datetime import date
import logging

import numpy as np
import pandas as pd

from .storage import DailyBarsReadError, Storage


PROGRESS_LOG_INTERVAL = 500
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
BARRIER_RISK_FEATURE_COLUMNS = TAIL_RISK_FEATURE_COLUMNS


def build_barrier_risk_frame(
    bars: pd.DataFrame,
    *,
    symbol: str,
    name: str = "",
    horizon_days: int = 20,
    downside_atr_mult: float = 1.0,
    upside_atr_mult: float | None = 2.0,
    downside_pct: float | None = None,
    upside_pct: float | None = None,
    label_variant: str = "barrier_down_first",
) -> pd.DataFrame:
    if label_variant not in {"barrier_down_first", "max_drawdown_exceed"}:
        raise ValueError(f"Unsupported barrier label variant: {label_variant}")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    frame = _prepare_price_frame(bars)
    if frame.empty:
        return pd.DataFrame()

    close = frame["close"].where(frame["close"].gt(0))
    high = frame["high"]
    low = frame["low"]
    open_ = frame["open"]
    atr14 = _atr14(frame)
    features = _base_feature_frame(frame, symbol=symbol, name=name)
    features["atr14"] = atr14

    outcomes: list[str | float] = []
    risk_labels: list[float] = []
    entry_dates: list[pd.Timestamp | float] = []
    entry_prices: list[float] = []
    exit_dates: list[pd.Timestamp | float] = []
    exit_prices: list[float] = []
    down_prices: list[float] = []
    up_prices: list[float] = []
    realized_returns: list[float] = []
    max_drawdowns: list[float] = []

    for index in range(len(frame)):
        entry_index = index + 1
        end_index = min(index + horizon_days, len(frame) - 1)
        entry_price = float(open_.iloc[entry_index]) if entry_index < len(frame) else np.nan
        atr_value = float(atr14.iloc[index]) if pd.notna(atr14.iloc[index]) else np.nan
        if entry_index >= len(frame) or end_index < entry_index or not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(atr_value) or atr_value <= 0:
            _append_empty_barrier(
                outcomes,
                risk_labels,
                entry_dates,
                entry_prices,
                exit_dates,
                exit_prices,
                down_prices,
                up_prices,
                realized_returns,
                max_drawdowns,
            )
            continue

        downside_amount = entry_price * float(downside_pct) if downside_pct is not None else atr_value * float(downside_atr_mult)
        upside_amount = None
        if upside_pct is not None:
            upside_amount = entry_price * float(upside_pct)
        elif upside_atr_mult is not None:
            upside_amount = atr_value * float(upside_atr_mult)
        down_price = entry_price - downside_amount
        up_price = entry_price + upside_amount if upside_amount is not None else np.nan

        outcome = "timeout"
        exit_index = end_index
        exit_price = float(close.iloc[end_index]) if pd.notna(close.iloc[end_index]) else np.nan
        window_high = high.iloc[entry_index : end_index + 1]
        window_low = low.iloc[entry_index : end_index + 1]
        for bar_index in range(entry_index, end_index + 1):
            low_touched = pd.notna(low.iloc[bar_index]) and float(low.iloc[bar_index]) <= down_price
            high_touched = upside_amount is not None and pd.notna(high.iloc[bar_index]) and float(high.iloc[bar_index]) >= up_price
            if low_touched and high_touched:
                outcome = "down_first"
                exit_index = bar_index
                exit_price = down_price
                break
            if low_touched:
                outcome = "down_first"
                exit_index = bar_index
                exit_price = down_price
                break
            if high_touched:
                outcome = "up_first"
                exit_index = bar_index
                exit_price = up_price
                break

        future_min_return = float(window_low.min() / entry_price - 1.0) if not window_low.dropna().empty else np.nan
        down_exceed = bool(np.isfinite(future_min_return) and future_min_return <= -downside_amount / entry_price)
        risk_label = 1.0 if (outcome == "down_first" if label_variant == "barrier_down_first" else down_exceed) else 0.0

        outcomes.append(outcome)
        risk_labels.append(risk_label)
        entry_dates.append(frame["trade_date"].iloc[entry_index])
        entry_prices.append(entry_price)
        exit_dates.append(frame["trade_date"].iloc[exit_index])
        exit_prices.append(exit_price)
        down_prices.append(down_price)
        up_prices.append(up_price)
        realized_returns.append(exit_price / entry_price - 1.0 if np.isfinite(exit_price) else np.nan)
        max_drawdowns.append(future_min_return)

    features["entry_date"] = entry_dates
    features["entry_price"] = entry_prices
    features["downside_barrier_price"] = down_prices
    features["upside_barrier_price"] = up_prices
    features["barrier_outcome"] = outcomes
    features["risk_label"] = risk_labels
    features["exit_date"] = exit_dates
    features["exit_price"] = exit_prices
    features["barrier_realized_return"] = realized_returns
    features["barrier_max_drawdown"] = max_drawdowns
    exit_series = pd.Series(exit_prices, index=features.index).where(lambda values: values.gt(0))
    entry_series = pd.Series(entry_prices, index=features.index).where(lambda values: values.gt(0))
    features["forward_log_return"] = np.log(exit_series / entry_series).replace([np.inf, -np.inf], np.nan)
    return features


def build_tail_risk_frame(
    bars: pd.DataFrame,
    *,
    symbol: str,
    name: str = "",
    lookback_days: int = 100,
    quantile: float = 0.05,
    horizon_days: int = 1,
) -> pd.DataFrame:
    frame = _prepare_price_frame(bars)
    if frame.empty:
        return pd.DataFrame()
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
    instruments = universe.to_dict("records")
    total_symbols = len(instruments)
    for index, instrument in enumerate(instruments, start=1):
        _log_progress("Tail-risk panel build", index, total_symbols)
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
    logging.info(
        "Tail-risk panel build complete: symbols=%s rows=%s skipped=%s",
        total_symbols,
        len(dataset),
        len(skipped),
    )
    return dataset, pd.DataFrame(skipped)


def build_barrier_risk_panel(
    *,
    storage: Storage,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    horizon_days: int = 20,
    downside_atr_mult: float = 1.0,
    upside_atr_mult: float | None = 2.0,
    downside_pct: float | None = None,
    upside_pct: float | None = None,
    label_variant: str = "barrier_down_first",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = storage.load_universe().copy()
    if limit is not None:
        universe = universe.head(max(int(limit), 0)).copy()
    rows: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []
    instruments = universe.to_dict("records")
    total_symbols = len(instruments)
    for index, instrument in enumerate(instruments, start=1):
        _log_progress("Barrier-risk panel build", index, total_symbols)
        symbol = str(instrument.get("symbol", "")).zfill(6)
        name = str(instrument.get("name", ""))
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError) as exc:
            skipped.append({"symbol": symbol, "name": name, "reason": type(exc).__name__})
            continue
        frame = build_barrier_risk_frame(
            bars,
            symbol=symbol,
            name=name,
            horizon_days=horizon_days,
            downside_atr_mult=downside_atr_mult,
            upside_atr_mult=upside_atr_mult,
            downside_pct=downside_pct,
            upside_pct=upside_pct,
            label_variant=label_variant,
        )
        if frame.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "empty_barrier_risk_frame"})
            continue
        if start_date is not None:
            frame = frame[frame["trade_date"].dt.date >= start_date]
        if end_date is not None:
            frame = frame[frame["trade_date"].dt.date <= end_date]
        frame = frame.dropna(subset=["risk_label", *BARRIER_RISK_FEATURE_COLUMNS, "entry_price", "barrier_max_drawdown"]).copy()
        if frame.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "no_labeled_rows"})
            continue
        rows.append(frame)
    dataset = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not dataset.empty:
        dataset = dataset.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    logging.info(
        "Barrier-risk panel build complete: symbols=%s rows=%s skipped=%s",
        total_symbols,
        len(dataset),
        len(skipped),
    )
    return dataset, pd.DataFrame(skipped)


def summarize_barrier_label_distribution(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    rows = []
    for outcome, group in dataset.groupby("barrier_outcome", dropna=False):
        rows.append(
            {
                "barrier_outcome": str(outcome),
                "rows": int(len(group)),
                "row_rate": float(len(group) / len(dataset)),
                "risk_label_rate": float(pd.to_numeric(group["risk_label"], errors="coerce").mean()),
                "avg_barrier_realized_return": float(pd.to_numeric(group["barrier_realized_return"], errors="coerce").mean()),
                "avg_barrier_max_drawdown": float(pd.to_numeric(group["barrier_max_drawdown"], errors="coerce").mean()),
            }
        )
    rows.append(
        {
            "barrier_outcome": "ALL",
            "rows": int(len(dataset)),
            "row_rate": 1.0,
            "risk_label_rate": float(pd.to_numeric(dataset["risk_label"], errors="coerce").mean()),
            "avg_barrier_realized_return": float(pd.to_numeric(dataset["barrier_realized_return"], errors="coerce").mean()),
            "avg_barrier_max_drawdown": float(pd.to_numeric(dataset["barrier_max_drawdown"], errors="coerce").mean()),
        }
    )
    return pd.DataFrame(rows)


def _prepare_price_frame(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    frame = bars.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        values = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
        frame[column] = pd.to_numeric(values, errors="coerce")
    return frame


def _base_feature_frame(frame: pd.DataFrame, *, symbol: str, name: str) -> pd.DataFrame:
    close = frame["close"].where(frame["close"].gt(0))
    log_return = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    result = pd.DataFrame(
        {
            "trade_date": frame["trade_date"],
            "symbol": str(symbol).zfill(6),
            "name": name,
            "log_return_1d": log_return,
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


def _atr14(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"].sub(frame["low"]),
            frame["high"].sub(previous_close).abs(),
            frame["low"].sub(previous_close).abs(),
        ],
        axis=1,
    )
    true_range = ranges.max(axis=1)
    return true_range.rolling(14, min_periods=14).mean()


def _append_empty_barrier(
    outcomes: list[str | float],
    risk_labels: list[float],
    entry_dates: list[pd.Timestamp | float],
    entry_prices: list[float],
    exit_dates: list[pd.Timestamp | float],
    exit_prices: list[float],
    down_prices: list[float],
    up_prices: list[float],
    realized_returns: list[float],
    max_drawdowns: list[float],
) -> None:
    outcomes.append(np.nan)
    risk_labels.append(np.nan)
    entry_dates.append(np.nan)
    entry_prices.append(np.nan)
    exit_dates.append(np.nan)
    exit_prices.append(np.nan)
    down_prices.append(np.nan)
    up_prices.append(np.nan)
    realized_returns.append(np.nan)
    max_drawdowns.append(np.nan)


def _future_min_return(close: pd.Series, *, horizon: int) -> pd.Series:
    values = []
    for offset in range(1, horizon + 1):
        values.append(close.shift(-offset).div(close).sub(1.0))
    if not values:
        return pd.Series(np.nan, index=close.index)
    return pd.concat(values, axis=1).min(axis=1)


def _log_progress(stage_name: str, current: int, total: int) -> None:
    if total <= 0:
        return
    if current == 1 or current % PROGRESS_LOG_INTERVAL == 0 or current == total:
        logging.info("%s progress: %s/%s", stage_name, current, total)
