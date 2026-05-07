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
    ret = close.pct_change(fill_method=None)

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
        rolling_close = close.rolling(window, min_periods=window)
        rolling_ret = ret.rolling(window, min_periods=window)
        rolling_volume = volume.rolling(window, min_periods=window)
        rolling_log_volume = log_volume.rolling(window, min_periods=window)
        columns[f"ROC{window}"] = close.div(close.shift(window)).sub(1.0)
        columns[f"MA{window}"] = rolling_close.mean().div(close)
        columns[f"STD{window}"] = rolling_ret.std()
        beta, rsqr, resi = _rolling_trend_features(close, window=window)
        columns[f"BETA{window}"] = beta
        columns[f"RSQR{window}"] = rsqr
        columns[f"RESI{window}"] = resi
        columns[f"MAX{window}"] = rolling_close.max().div(close)
        columns[f"MIN{window}"] = rolling_close.min().div(close)
        columns[f"QTLU{window}"] = rolling_close.quantile(0.8).div(close)
        columns[f"QTLD{window}"] = rolling_close.quantile(0.2).div(close)
        columns[f"RANK{window}"] = rolling_close.rank(pct=True)
        columns[f"RSV{window}"] = close.sub(rolling_close.min()).div(rolling_close.max().sub(rolling_close.min()).replace(0, np.nan))
        imax = rolling_close.apply(_argmax_ratio, raw=True)
        imin = rolling_close.apply(_argmin_ratio, raw=True)
        columns[f"IMAX{window}"] = imax
        columns[f"IMIN{window}"] = imin
        columns[f"IMXD{window}"] = imax.sub(imin)
        columns[f"CORR{window}"] = close.rolling(window, min_periods=window).corr(log_volume)
        columns[f"CORD{window}"] = close.diff().rolling(window, min_periods=window).corr(log_volume.diff())
        cntp = ret.gt(0).rolling(window, min_periods=window).mean()
        cntn = ret.lt(0).rolling(window, min_periods=window).mean()
        columns[f"CNTP{window}"] = cntp
        columns[f"CNTN{window}"] = cntn
        columns[f"CNTD{window}"] = cntp.sub(cntn)
        positive = ret.where(ret > 0, 0.0)
        negative = ret.where(ret < 0, 0.0).abs()
        sump = positive.rolling(window, min_periods=window).sum()
        sumn = negative.rolling(window, min_periods=window).sum()
        columns[f"SUMP{window}"] = sump
        columns[f"SUMN{window}"] = sumn
        columns[f"SUMD{window}"] = sump.sub(sumn)
        columns[f"VMA{window}"] = rolling_volume.mean().div(volume.replace(0, np.nan))
        columns[f"VSTD{window}"] = rolling_log_volume.std()
        columns[f"WVMA{window}"] = _weighted_volume_mean(volume, ret, window=window)
        volume_sum = rolling_volume.sum().replace(0, np.nan)
        vsump = volume.where(ret > 0, 0.0).rolling(window, min_periods=window).sum().div(volume_sum)
        vsumn = volume.where(ret < 0, 0.0).rolling(window, min_periods=window).sum().div(volume_sum)
        columns[f"VSUMP{window}"] = vsump
        columns[f"VSUMN{window}"] = vsumn
        columns[f"VSUMD{window}"] = vsump.sub(vsumn)

    columns["future_return_5d"] = close.shift(-5).div(close).sub(1.0)
    columns["future_max_drawdown_5d"] = _future_min_return(close, horizon=5)
    return pd.DataFrame(columns).replace([np.inf, -np.inf], np.nan)


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
        merged = merged.dropna(subset=["risk_label", "future_return_5d", "future_max_drawdown_5d"]).copy()
        if merged.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "no_labeled_feature_rows"})
            continue
        rows.append(merged)
    dataset = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not dataset.empty:
        dataset = dataset.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
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
    sum_x = x.rolling(window, min_periods=window).sum()
    sum_y = y.rolling(window, min_periods=window).sum()
    sum_x2 = x.pow(2).rolling(window, min_periods=window).sum()
    sum_y2 = y.pow(2).rolling(window, min_periods=window).sum()
    sum_xy = x.mul(y).rolling(window, min_periods=window).sum()
    denominator = window * sum_x2 - sum_x.pow(2)
    beta = (window * sum_xy - sum_x * sum_y).div(denominator.replace(0, np.nan))
    intercept = sum_y.sub(beta.mul(sum_x)).div(window)
    fitted = beta.mul(x).add(intercept)
    residual = y.sub(fitted).div(y.replace(0, np.nan))
    corr_num = window * sum_xy - sum_x * sum_y
    corr_den = ((window * sum_x2 - sum_x.pow(2)) * (window * sum_y2 - sum_y.pow(2))).pow(0.5)
    rsqr = corr_num.div(corr_den.replace(0, np.nan)).pow(2)
    return beta.div(y.replace(0, np.nan)), rsqr, residual


def _weighted_volume_mean(volume: pd.Series, ret: pd.Series, *, window: int) -> pd.Series:
    weights = ret.abs()
    numerator = volume.mul(weights).rolling(window, min_periods=window).sum()
    denominator = weights.rolling(window, min_periods=window).sum()
    return numerator.div(denominator.replace(0, np.nan)).div(volume.replace(0, np.nan))


def _argmax_ratio(values: np.ndarray) -> float:
    if len(values) == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanargmax(values) / max(len(values) - 1, 1))


def _argmin_ratio(values: np.ndarray) -> float:
    if len(values) == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanargmin(values) / max(len(values) - 1, 1))


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
