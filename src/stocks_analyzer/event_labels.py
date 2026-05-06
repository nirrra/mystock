from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from .indicators import add_indicators


DEFAULT_STOP_ATR_GRID = (1.0, 1.2, 1.5)
DEFAULT_TAKE_ATR_GRID = (2.0, 2.5, 3.0)
DEFAULT_HOLDING_DAYS_GRID = (10, 20, 40)
DEFAULT_MIN_HISTORY_DAYS = 120


@dataclass(frozen=True, slots=True)
class EventLabelConfig:
    stop_atr_mult: float = 1.2
    take_atr_mult: float = 2.5
    max_holding_days: int = 20
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS


def build_event_labels(
    signals: pd.DataFrame,
    daily_history_by_symbol: dict[str, pd.DataFrame],
    *,
    config: EventLabelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_config = config or EventLabelConfig()
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame()

    prepared_history = {
        str(symbol).zfill(6): _prepare_history(history)
        for symbol, history in daily_history_by_symbol.items()
        if not history.empty
    }
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    last_kept_index: dict[tuple[str, str], int] = {}

    prepared_signals = signals.copy()
    prepared_signals["symbol"] = prepared_signals["symbol"].astype(str).str.zfill(6)
    prepared_signals["signal_date"] = pd.to_datetime(
        prepared_signals.get("trade_date", prepared_signals.get("signal_date")), errors="coerce"
    ).dt.date
    prepared_signals["pattern_id"] = prepared_signals.get("pattern_id", "").astype(str)
    prepared_signals = prepared_signals.sort_values(["signal_date", "symbol", "pattern_id"]).reset_index(drop=True)

    for signal in prepared_signals.to_dict("records"):
        symbol = str(signal["symbol"]).zfill(6)
        pattern_id = str(signal.get("pattern_id", ""))
        signal_date = signal.get("signal_date")
        base_skip = _base_event_record(signal, symbol=symbol, pattern_id=pattern_id, signal_date=signal_date)

        if not isinstance(signal_date, date):
            skipped.append({**base_skip, "skip_reason": "invalid_signal_date"})
            continue

        history = prepared_history.get(symbol)
        if history is None:
            skipped.append({**base_skip, "skip_reason": "missing_daily_history"})
            continue

        signal_index = history.attrs["index_by_date"].get(signal_date)
        if signal_index is None:
            skipped.append({**base_skip, "skip_reason": "signal_date_not_found"})
            continue

        if signal_index + 1 < label_config.min_history_days:
            skipped.append({**base_skip, "skip_reason": "insufficient_history"})
            continue

        cooldown_key = (symbol, pattern_id)
        previous_index = last_kept_index.get(cooldown_key)
        if previous_index is not None and signal_index - previous_index < label_config.max_holding_days:
            skipped.append({**base_skip, "skip_reason": "cooldown_skipped", "cooldown_skipped": True})
            continue

        labeled = _label_one_event(signal, history, signal_index, label_config)
        if labeled.get("skip_reason"):
            skipped.append(labeled)
            continue

        rows.append(labeled)
        last_kept_index[cooldown_key] = signal_index

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def build_prediction_events(
    signals: pd.DataFrame,
    daily_history_by_symbol: dict[str, pd.DataFrame],
    *,
    config: EventLabelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_config = config or EventLabelConfig()
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame()

    prepared_history = {
        str(symbol).zfill(6): _prepare_history(history)
        for symbol, history in daily_history_by_symbol.items()
        if not history.empty
    }
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    prepared_signals = signals.copy()
    prepared_signals["symbol"] = prepared_signals["symbol"].astype(str).str.zfill(6)
    prepared_signals["signal_date"] = pd.to_datetime(
        prepared_signals.get("trade_date", prepared_signals.get("signal_date")), errors="coerce"
    ).dt.date
    prepared_signals["pattern_id"] = prepared_signals.get("pattern_id", "").astype(str)

    for signal in prepared_signals.to_dict("records"):
        symbol = str(signal["symbol"]).zfill(6)
        pattern_id = str(signal.get("pattern_id", ""))
        signal_date = signal.get("signal_date")
        base = _base_event_record(signal, symbol=symbol, pattern_id=pattern_id, signal_date=signal_date)
        if not isinstance(signal_date, date):
            skipped.append({**base, "skip_reason": "invalid_signal_date"})
            continue
        history = prepared_history.get(symbol)
        if history is None:
            skipped.append({**base, "skip_reason": "missing_daily_history"})
            continue
        signal_index = history.attrs["index_by_date"].get(signal_date)
        if signal_index is None:
            skipped.append({**base, "skip_reason": "signal_date_not_found"})
            continue
        if signal_index + 1 < label_config.min_history_days:
            skipped.append({**base, "skip_reason": "insufficient_history"})
            continue

        signal_row = history.iloc[signal_index]
        atr14 = _float_or_none(signal_row.get("atr_14"))
        close = _float_or_none(signal_row.get("close"))
        if atr14 is None or atr14 <= 0 or close is None or close <= 0:
            skipped.append({**base, "skip_reason": "invalid_atr_or_risk"})
            continue
        stop_loss_price = close - label_config.stop_atr_mult * atr14
        take_profit_price = close + label_config.take_atr_mult * atr14
        rows.append(
            {
                **base,
                "event_id": f"{signal_date:%Y%m%d}_{symbol}_{pattern_id}",
                "entry_date": pd.NaT,
                "entry_price": round(float(close), 4),
                "atr14_signal": round(float(atr14), 4),
                "stop_loss_price": round(float(stop_loss_price), 4),
                "take_profit_price": round(float(take_profit_price), 4),
                "initial_risk": round(float(close - stop_loss_price), 4),
                "max_holding_days": int(label_config.max_holding_days),
                "cooldown_skipped": False,
                "sample_weight": 1.0,
                "skip_reason": "",
                "trigger_reason": signal.get("trigger_reason", signal.get("reason", "")),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def add_rank_labels(labels: pd.DataFrame, *, min_events_per_day: int = 5) -> pd.DataFrame:
    if labels.empty:
        return labels.copy()
    result = labels.copy()
    result["rank_train_eligible"] = False
    result["rank_pct"] = pd.NA
    result["rank_grade"] = pd.NA
    result["rank_target"] = pd.NA

    for _, group in result.groupby("signal_date", sort=False):
        eligible = group["rank_value"].notna()
        if int(eligible.sum()) < min_events_per_day:
            continue
        values = pd.to_numeric(group.loc[eligible, "rank_value"], errors="coerce")
        pct = values.rank(method="average", pct=True)
        grades = pd.cut(
            pct,
            bins=[0.0, 0.15, 0.35, 0.70, 0.90, 1.0000001],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True,
            right=False,
        ).astype("int64")
        result.loc[pct.index, "rank_pct"] = pct
        result.loc[pct.index, "rank_grade"] = grades
        result.loc[pct.index, "rank_target"] = grades.astype(float) / 4.0
        result.loc[pct.index, "rank_train_eligible"] = True

    return result


def _label_one_event(signal: dict[str, Any], history: pd.DataFrame, signal_index: int, config: EventLabelConfig) -> dict[str, Any]:
    symbol = str(signal["symbol"]).zfill(6)
    pattern_id = str(signal.get("pattern_id", ""))
    signal_date = pd.Timestamp(signal["signal_date"]).date()
    event_id = f"{signal_date:%Y%m%d}_{symbol}_{pattern_id}"
    signal_row = history.iloc[signal_index]
    entry_index = signal_index + 1
    exit_index = entry_index + config.max_holding_days - 1
    base = _base_event_record(signal, symbol=symbol, pattern_id=pattern_id, signal_date=signal_date)
    base.update(
        {
            "event_id": event_id,
            "stop_atr_mult": float(config.stop_atr_mult),
            "take_atr_mult": float(config.take_atr_mult),
            "max_holding_days": int(config.max_holding_days),
            "cooldown_skipped": False,
        }
    )

    if exit_index >= len(history):
        return {**base, "skip_reason": "insufficient_forward_bars"}

    atr14 = _float_or_none(signal_row.get("atr_14"))
    if atr14 is None or atr14 <= 0:
        return {**base, "skip_reason": "invalid_atr_or_risk"}

    entry_row = history.iloc[entry_index]
    entry_price = _float_or_none(entry_row.get("open"))
    if entry_price is None or entry_price <= 0:
        return {**base, "skip_reason": "invalid_entry_price"}

    if _is_locked_limit_up(history, entry_index):
        return {
            **base,
            "entry_date": pd.Timestamp(entry_row["trade_date"]),
            "entry_price": float(entry_price),
            "barrier_outcome": "entry_unfillable",
            "skip_reason": "entry_unfillable_limit_up",
        }

    stop_loss_price = entry_price - float(config.stop_atr_mult) * atr14
    take_profit_price = entry_price + float(config.take_atr_mult) * atr14
    initial_risk = entry_price - stop_loss_price
    if initial_risk <= 0:
        return {**base, "skip_reason": "invalid_atr_or_risk"}

    holding = history.iloc[entry_index : exit_index + 1].reset_index(drop=True)
    outcome = "timeout"
    exit_offset = len(holding) - 1
    exit_price = _float_or_none(holding.iloc[-1].get("close")) or entry_price

    for offset, row in enumerate(holding.to_dict("records")):
        high = float(row["high"])
        low = float(row["low"])
        hit_take = high >= take_profit_price
        hit_stop = low <= stop_loss_price
        if hit_stop and hit_take:
            outcome = "stop_loss_first"
            exit_offset = offset
            absolute_index = entry_index + offset
            exit_price = _stop_exit_price(history, absolute_index, stop_loss_price)
            break
        if hit_stop:
            outcome = "stop_loss_first"
            exit_offset = offset
            absolute_index = entry_index + offset
            exit_price = _stop_exit_price(history, absolute_index, stop_loss_price)
            break
        if hit_take:
            outcome = "take_profit_first"
            exit_offset = offset
            exit_price = take_profit_price
            break

    exit_row = holding.iloc[exit_offset]
    exit_date = pd.Timestamp(exit_row["trade_date"])
    holding_days = exit_offset + 1
    window = holding.iloc[:holding_days]
    min_low = float(pd.to_numeric(window["low"], errors="coerce").min())
    max_high = float(pd.to_numeric(window["high"], errors="coerce").max())
    realized_r = (float(exit_price) - entry_price) / initial_risk
    max_drawdown_r = max(0.0, entry_price - min_low) / initial_risk
    max_upside_r = max(0.0, max_high - entry_price) / initial_risk
    holding_days_penalty = min(1.0, holding_days / float(config.max_holding_days))
    rank_value = realized_r - 0.4 * max_drawdown_r - 0.2 * holding_days_penalty
    risk_label = int(outcome == "stop_loss_first" or realized_r <= -0.8)

    return {
        **base,
        "event_id": event_id,
        "entry_date": pd.Timestamp(entry_row["trade_date"]),
        "entry_price": round(float(entry_price), 4),
        "atr14_signal": round(float(atr14), 4),
        "stop_loss_price": round(float(stop_loss_price), 4),
        "take_profit_price": round(float(take_profit_price), 4),
        "initial_risk": round(float(initial_risk), 4),
        "max_holding_days": int(config.max_holding_days),
        "barrier_outcome": outcome,
        "exit_date": exit_date,
        "exit_price": round(float(exit_price), 4),
        "realized_R": float(realized_r),
        "expected_R_label": float(max(-1.2, min(3.0, realized_r))),
        "risk_label": risk_label,
        "max_drawdown_R": float(max_drawdown_r),
        "max_upside_R": float(max_upside_r),
        "holding_days": int(holding_days),
        "holding_days_penalty": float(holding_days_penalty),
        "rank_value": float(rank_value),
        "cooldown_skipped": False,
        "sample_weight": 1.0,
        "skip_reason": "",
        "trigger_reason": signal.get("trigger_reason", signal.get("reason", "")),
    }


def _base_event_record(signal: dict[str, Any], *, symbol: str, pattern_id: str, signal_date: object) -> dict[str, Any]:
    event_id = f"{pd.Timestamp(signal_date).strftime('%Y%m%d')}_{symbol}_{pattern_id}" if isinstance(signal_date, date) else ""
    return {
        "event_id": event_id,
        "symbol": symbol,
        "name": signal.get("name", ""),
        "pattern_id": pattern_id,
        "signal_date": pd.Timestamp(signal_date) if isinstance(signal_date, date) else pd.NaT,
        "strategy_name": signal.get("strategy_name", signal.get("signal_type", "")),
        "trigger_reason": signal.get("trigger_reason", signal.get("reason", "")),
        "cooldown_skipped": False,
    }


def _prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    prepared = add_indicators(history) if "atr_14" not in history.columns else history.copy()
    prepared = prepared.sort_values("trade_date").reset_index(drop=True)
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"])
    prepared.attrs["index_by_date"] = {item.date(): index for index, item in enumerate(prepared["trade_date"])}
    return prepared


def _float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_locked_limit_up(history: pd.DataFrame, index: int) -> bool:
    if index <= 0 or index >= len(history):
        return False
    row = history.iloc[index]
    previous_close = _float_or_none(history.iloc[index - 1].get("close"))
    close = _float_or_none(row.get("close"))
    if previous_close is None or close is None or previous_close <= 0:
        return False
    one_price = len({round(float(row[column]), 4) for column in ("open", "high", "low", "close")}) == 1
    return bool(one_price and close / previous_close - 1 >= 0.095)


def _is_locked_limit_down(history: pd.DataFrame, index: int) -> bool:
    if index <= 0 or index >= len(history):
        return False
    row = history.iloc[index]
    previous_close = _float_or_none(history.iloc[index - 1].get("close"))
    close = _float_or_none(row.get("close"))
    if previous_close is None or close is None or previous_close <= 0:
        return False
    one_price = len({round(float(row[column]), 4) for column in ("open", "high", "low", "close")}) == 1
    return bool(one_price and close / previous_close - 1 <= -0.095)


def _stop_exit_price(history: pd.DataFrame, index: int, stop_loss_price: float) -> float:
    if _is_locked_limit_down(history, index):
        close = _float_or_none(history.iloc[index].get("close"))
        return min(float(stop_loss_price), float(close if close is not None else stop_loss_price))
    return float(stop_loss_price)
