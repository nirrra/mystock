from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .full_market_labels import build_mlfin_barrier_risk_frame
from .storage import DailyBarsReadError, Storage


ALPHA158_WINDOWS = (5, 10, 20, 30, 60)
PROGRESS_LOG_INTERVAL = 500


@dataclass(slots=True)
class Alpha158PanelResult:
    dataset: pd.DataFrame
    skipped: pd.DataFrame
    feature_columns: tuple[str, ...]
    feature_audit: pd.DataFrame


@dataclass(slots=True)
class Alpha158ReturnPanelResult:
    dataset: pd.DataFrame
    skipped: pd.DataFrame
    feature_columns: tuple[str, ...]
    feature_audit: pd.DataFrame


def build_alpha158_feature_frame(
    bars: pd.DataFrame,
    *,
    symbol: str,
    name: str = "",
    windows: tuple[int, ...] = ALPHA158_WINDOWS,
) -> pd.DataFrame:
    frame = _prepare_price_frame(bars)
    if frame.empty:
        return pd.DataFrame()
    open_ = frame["open"].where(frame["open"].gt(0))
    high = frame["high"].where(frame["high"].gt(0))
    low = frame["low"].where(frame["low"].gt(0))
    close = frame["close"].where(frame["close"].gt(0))
    volume = frame["volume"].where(frame["volume"].ge(0))
    amount = frame["amount"].where(frame["amount"].ge(0))
    vwap = amount.div(volume.replace(0, np.nan)).where(lambda values: values.gt(0), close)
    log_volume = np.log1p(volume)
    close_ref1 = close.shift(1)
    volume_ref1 = volume.shift(1)
    close_diff = close.sub(close_ref1)
    volume_diff = volume.sub(volume_ref1)

    columns: dict[str, Any] = {
        "trade_date": frame["trade_date"],
        "symbol": pd.Series([str(symbol).zfill(6)] * len(frame), index=frame.index),
        "name": pd.Series([name] * len(frame), index=frame.index),
    }
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    day_range = high.sub(low).replace(0, np.nan)
    columns["KMID"] = close.sub(open_).div(open_)
    columns["KLEN"] = high.sub(low).div(open_)
    columns["KMID2"] = close.sub(open_).div(day_range)
    columns["KUP"] = high.sub(body_high).div(open_)
    columns["KUP2"] = high.sub(body_high).div(day_range)
    columns["KLOW"] = body_low.sub(low).div(open_)
    columns["KLOW2"] = body_low.sub(low).div(day_range)
    columns["KSFT"] = close.mul(2).sub(high).sub(low).div(open_)
    columns["KSFT2"] = close.mul(2).sub(high).sub(low).div(day_range)
    columns["OPEN0"] = open_.div(close)
    columns["HIGH0"] = high.div(close)
    columns["LOW0"] = low.div(close)
    columns["VWAP0"] = vwap.div(close)

    for window in windows:
        rolling_close = close.rolling(window, min_periods=1)
        rolling_high = high.rolling(window, min_periods=1)
        rolling_low = low.rolling(window, min_periods=1)
        rolling_volume = volume.rolling(window, min_periods=1)
        columns[f"ROC{window}"] = close.shift(window).div(close)
        columns[f"MA{window}"] = rolling_close.mean().div(close)
        columns[f"STD{window}"] = rolling_close.std().div(close)
        beta, rsqr, resi = _rolling_trend_features(close, window=window)
        columns[f"BETA{window}"] = beta
        columns[f"RSQR{window}"] = rsqr
        columns[f"RESI{window}"] = resi
        columns[f"MAX{window}"] = rolling_high.max().div(close)
        columns[f"MIN{window}"] = rolling_low.min().div(close)
        columns[f"QTLU{window}"] = rolling_close.quantile(0.8).div(close)
        columns[f"QTLD{window}"] = rolling_close.quantile(0.2).div(close)
        columns[f"RANK{window}"] = rolling_close.rank(pct=True)
        low_min = rolling_low.min()
        high_max = rolling_high.max()
        columns[f"RSV{window}"] = close.sub(low_min).div(high_max.sub(low_min).add(1e-12))
        imax = rolling_high.apply(_idxmax_one_based, raw=True).div(window)
        imin = rolling_low.apply(_idxmin_one_based, raw=True).div(window)
        columns[f"IMAX{window}"] = imax
        columns[f"IMIN{window}"] = imin
        columns[f"IMXD{window}"] = imax.sub(imin)
        columns[f"CORR{window}"] = close.rolling(window, min_periods=1).corr(log_volume)
        price_ratio = close.div(close_ref1)
        volume_ratio_log = np.log(volume.div(volume_ref1).add(1.0))
        columns[f"CORD{window}"] = price_ratio.rolling(window, min_periods=1).corr(volume_ratio_log)
        cntp = close.gt(close_ref1).rolling(window, min_periods=1).mean()
        cntn = close.lt(close_ref1).rolling(window, min_periods=1).mean()
        columns[f"CNTP{window}"] = cntp
        columns[f"CNTN{window}"] = cntn
        columns[f"CNTD{window}"] = cntp.sub(cntn)
        gain = close_diff.clip(lower=0.0)
        loss = close_ref1.sub(close).clip(lower=0.0)
        abs_price_change = close_diff.abs()
        gain_sum = gain.rolling(window, min_periods=1).sum()
        loss_sum = loss.rolling(window, min_periods=1).sum()
        abs_price_sum = abs_price_change.rolling(window, min_periods=1).sum().add(1e-12)
        columns[f"SUMP{window}"] = gain_sum.div(abs_price_sum)
        columns[f"SUMN{window}"] = loss_sum.div(abs_price_sum)
        columns[f"SUMD{window}"] = gain_sum.sub(loss_sum).div(abs_price_sum)
        columns[f"VMA{window}"] = rolling_volume.mean().div(volume.add(1e-12))
        columns[f"VSTD{window}"] = rolling_volume.std().div(volume.add(1e-12))
        weighted_abs_return_volume = close.div(close_ref1).sub(1.0).abs().mul(volume)
        columns[f"WVMA{window}"] = weighted_abs_return_volume.rolling(window, min_periods=1).std().div(
            weighted_abs_return_volume.rolling(window, min_periods=1).mean().add(1e-12)
        )
        volume_gain = volume_diff.clip(lower=0.0)
        volume_loss = volume_ref1.sub(volume).clip(lower=0.0)
        abs_volume_sum = volume_diff.abs().rolling(window, min_periods=1).sum().add(1e-12)
        vsump = volume_gain.rolling(window, min_periods=1).sum().div(abs_volume_sum)
        vsumn = volume_loss.rolling(window, min_periods=1).sum().div(abs_volume_sum)
        columns[f"VSUMP{window}"] = vsump
        columns[f"VSUMN{window}"] = vsumn
        columns[f"VSUMD{window}"] = vsump.sub(vsumn)

    columns["future_return_5d"] = close.shift(-5).div(close).sub(1.0)
    columns["future_max_drawdown_5d"] = _future_min_return(close, horizon=5)
    result = pd.DataFrame(columns).replace([np.inf, -np.inf], np.nan)
    return _downcast_alpha158_numeric(result)


def build_alpha158_latest_feature_frame(
    bars: pd.DataFrame,
    *,
    symbol: str,
    name: str = "",
    windows: tuple[int, ...] = ALPHA158_WINDOWS,
    lookback_bars: int = 120,
) -> pd.DataFrame:
    """Build only the latest Alpha158 feature row from a recent price window."""
    frame = _prepare_price_frame(bars)
    if frame.empty:
        return pd.DataFrame()
    max_window = max(windows) if windows else 1
    tail_length = max(int(lookback_bars), max_window + 1)
    recent = frame.tail(tail_length).copy()
    features = build_alpha158_feature_frame(recent, symbol=symbol, name=name, windows=windows)
    if features.empty:
        return features
    return features.tail(1).reset_index(drop=True)


def build_alpha158_return_frame(
    bars: pd.DataFrame,
    *,
    symbol: str,
    name: str = "",
    windows: tuple[int, ...] = ALPHA158_WINDOWS,
) -> pd.DataFrame:
    frame = build_alpha158_feature_frame(bars, symbol=symbol, name=name, windows=windows)
    if frame.empty:
        return frame
    close = _prepare_price_frame(bars)["close"].where(lambda values: values.gt(0))
    label = close.shift(-2).div(close.shift(-1)).sub(1.0).astype("float32").rename("LABEL0_raw")
    frame = pd.concat([frame, label], axis=1)
    return _downcast_alpha158_numeric(frame.replace([np.inf, -np.inf], np.nan))


def alpha158_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    excluded = {
        "trade_date",
        "symbol",
        "name",
        "risk_label",
        "barrier_outcome",
        "barrier_bin",
        "entry_date",
        "entry_price",
        "exit_date",
        "exit_price",
        "barrier_realized_return",
        "barrier_max_drawdown",
        "forward_log_return",
        "future_return_5d",
        "future_max_drawdown_5d",
        "LABEL0",
        "LABEL0_raw",
        "mlfin_target",
        "mlfin_cusum_threshold",
        "pt_mult",
        "sl_mult",
    }
    return tuple(column for column in frame.columns if column not in excluded)


def build_alpha158_risk_panel(
    *,
    storage: Storage,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    horizon_days: int = 5,
    volatility_lookback: int = 100,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    min_ret: float = 0.005,
) -> Alpha158PanelResult:
    universe = storage.load_universe().copy()
    if limit is not None:
        universe = universe.head(max(int(limit), 0)).copy()
    rows: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []
    instruments = universe.to_dict("records")
    total_symbols = len(instruments)
    for index, instrument in enumerate(instruments, start=1):
        _log_progress("Alpha158 risk panel build", index, total_symbols)
        symbol = str(instrument.get("symbol", "")).zfill(6)
        name = str(instrument.get("name", ""))
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError) as exc:
            skipped.append({"symbol": symbol, "name": name, "reason": type(exc).__name__})
            continue
        labels = build_mlfin_barrier_risk_frame(
            bars,
            symbol=symbol,
            name=name,
            vertical_barrier_days=horizon_days,
            volatility_lookback=volatility_lookback,
            pt_mult=pt_mult,
            sl_mult=sl_mult,
            min_ret=min_ret,
        )
        if labels.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "empty_mlfin_labels"})
            continue
        features = build_alpha158_feature_frame(bars, symbol=symbol, name=name)
        if features.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "empty_alpha158_features"})
            continue
        keep_label_columns = [
            "trade_date",
            "symbol",
            "risk_label",
            "barrier_outcome",
            "barrier_bin",
            "entry_date",
            "entry_price",
            "exit_date",
            "exit_price",
            "barrier_realized_return",
            "barrier_max_drawdown",
            "forward_log_return",
            "mlfin_target",
        ]
        merged = features.merge(labels.loc[:, keep_label_columns], on=["trade_date", "symbol"], how="inner")
        if start_date is not None:
            merged = merged[merged["trade_date"].dt.date >= start_date]
        if end_date is not None:
            merged = merged[merged["trade_date"].dt.date <= end_date]
        merged = merged.loc[merged[["risk_label", "future_return_5d", "future_max_drawdown_5d"]].notna().all(axis=1)].reset_index(drop=True)
        if merged.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "no_labeled_feature_rows"})
            continue
        merged = _downcast_alpha158_numeric(merged)
        rows.append(merged)
    dataset = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not dataset.empty:
        dataset = _sort_alpha158_dataset(dataset)
    feature_columns = alpha158_feature_columns(dataset) if not dataset.empty else tuple()
    all_missing_features = [column for column in feature_columns if dataset[column].isna().all()] if feature_columns else []
    if all_missing_features:
        dataset = dataset.drop(columns=all_missing_features)
        feature_columns = alpha158_feature_columns(dataset)
    feature_audit = build_alpha158_feature_audit(dataset, feature_columns=feature_columns)
    logging.info(
        "Alpha158 risk panel build complete: symbols=%s rows=%s features=%s skipped=%s",
        total_symbols,
        len(dataset),
        len(feature_columns),
        len(skipped),
    )
    return Alpha158PanelResult(dataset=dataset, skipped=pd.DataFrame(skipped), feature_columns=feature_columns, feature_audit=feature_audit)


def build_alpha158_return_panel(
    *,
    storage: Storage,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
) -> Alpha158ReturnPanelResult:
    universe = storage.load_universe().copy()
    if limit is not None:
        universe = universe.head(max(int(limit), 0)).copy()
    rows: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []
    instruments = universe.to_dict("records")
    total_symbols = len(instruments)
    for index, instrument in enumerate(instruments, start=1):
        _log_progress("Alpha158 return panel build", index, total_symbols)
        symbol = str(instrument.get("symbol", "")).zfill(6)
        name = str(instrument.get("name", ""))
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError) as exc:
            skipped.append({"symbol": symbol, "name": name, "reason": type(exc).__name__})
            continue
        frame = build_alpha158_return_frame(bars, symbol=symbol, name=name)
        if frame.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "empty_alpha158_return_frame"})
            continue
        if start_date is not None:
            frame = frame[frame["trade_date"].dt.date >= start_date]
        if end_date is not None:
            frame = frame[frame["trade_date"].dt.date <= end_date]
        frame = frame.loc[frame["LABEL0_raw"].notna()].reset_index(drop=True)
        if frame.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "no_return_label_rows"})
            continue
        frame = _downcast_alpha158_numeric(frame)
        rows.append(frame)
    dataset = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not dataset.empty:
        dataset = _sort_alpha158_dataset(dataset)
        dataset["LABEL0"] = _cross_sectional_zscore(dataset, value_column="LABEL0_raw").astype("float32")
        dataset = dataset.loc[dataset["LABEL0"].notna()].reset_index(drop=True)
    feature_columns = alpha158_feature_columns(dataset) if not dataset.empty else tuple()
    all_missing_features = [column for column in feature_columns if dataset[column].isna().all()] if feature_columns else []
    if all_missing_features:
        dataset = dataset.drop(columns=all_missing_features)
        feature_columns = alpha158_feature_columns(dataset)
    feature_audit = build_alpha158_feature_audit(dataset, feature_columns=feature_columns)
    logging.info(
        "Alpha158 return panel build complete: symbols=%s rows=%s features=%s skipped=%s",
        total_symbols,
        len(dataset),
        len(feature_columns),
        len(skipped),
    )
    return Alpha158ReturnPanelResult(
        dataset=dataset,
        skipped=pd.DataFrame(skipped),
        feature_columns=feature_columns,
        feature_audit=feature_audit,
    )


def build_alpha158_feature_audit(dataset: pd.DataFrame, *, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    rows = []
    for column in feature_columns:
        values = pd.to_numeric(dataset[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "missing_rate": float(values.isna().mean()),
                "mean": float(values.mean()) if values.notna().any() else np.nan,
                "std": float(values.std()) if values.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _downcast_alpha158_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    excluded = {"trade_date", "symbol", "name", "barrier_outcome", "barrier_bin", "entry_date", "exit_date", "atr_volatility_regime"}
    for column in result.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("float32")
    return result


def _sort_alpha158_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return dataset
    # Full-market Alpha158 can exceed six million rows. Sorting that frame forces
    # a large full-column copy, so keep append order for large panels; downstream
    # code already slices by date and does not require physical sort order.
    if len(dataset) > 1_000_000:
        return dataset.reset_index(drop=True)
    return dataset.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


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


def _rolling_trend_features(close: pd.Series, *, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    y = close.astype(float)
    x = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    count = y.rolling(window, min_periods=1).count()
    sum_x = x.rolling(window, min_periods=1).sum()
    sum_y = y.rolling(window, min_periods=1).sum()
    sum_x2 = x.pow(2).rolling(window, min_periods=1).sum()
    sum_y2 = y.pow(2).rolling(window, min_periods=1).sum()
    sum_xy = x.mul(y).rolling(window, min_periods=1).sum()
    denominator = count * sum_x2 - sum_x.pow(2)
    beta = (count * sum_xy - sum_x * sum_y).div(denominator.replace(0, np.nan))
    intercept = sum_y.sub(beta.mul(sum_x)).div(count.replace(0, np.nan))
    fitted = beta.mul(x).add(intercept)
    residual = y.sub(fitted).div(y.replace(0, np.nan))
    corr_num = count * sum_xy - sum_x * sum_y
    corr_den = ((count * sum_x2 - sum_x.pow(2)) * (count * sum_y2 - sum_y.pow(2))).pow(0.5)
    rsqr = corr_num.div(corr_den.replace(0, np.nan)).pow(2)
    return beta.div(y.replace(0, np.nan)), rsqr, residual


def _weighted_volume_mean(volume: pd.Series, ret: pd.Series, *, window: int) -> pd.Series:
    weights = ret.abs()
    numerator = volume.mul(weights).rolling(window, min_periods=window).sum()
    denominator = weights.rolling(window, min_periods=window).sum()
    return numerator.div(denominator.replace(0, np.nan)).div(volume.replace(0, np.nan))


def _idxmax_one_based(values: np.ndarray) -> float:
    if len(values) == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanargmax(values) + 1)


def _idxmin_one_based(values: np.ndarray) -> float:
    if len(values) == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanargmin(values) + 1)


def _cross_sectional_zscore(frame: pd.DataFrame, *, value_column: str) -> pd.Series:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    means = values.groupby(frame["trade_date"]).transform("mean")
    stds = values.groupby(frame["trade_date"]).transform("std")
    return values.sub(means).div(stds.replace(0, np.nan))


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
