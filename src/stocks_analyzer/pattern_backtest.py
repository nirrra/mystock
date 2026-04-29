from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
import random

import pandas as pd

from .indicators import add_indicators
from .models import AppConfig
from .storage import DailyBarsReadError, Storage
from .strategies import STRATEGY_NAMES, evaluate_strategies, required_history_days


PATTERN_ID_BY_STRATEGY = {
    "volume_top_pre_breakout": "1",
    "volume_top_breakout": "2",
    "volume_top_follow_through": "3",
    "duck_nostril_cross": "4",
    "trend_pullback": "5",
    "double_volume_support_rebound": "6",
}


def scan_pattern_backtest_signals(
    storage: Storage,
    config: AppConfig,
    *,
    start_date: date,
    end_date: date,
    selected_strategies: Sequence[str] | None = None,
    sampled_trade_dates: Sequence[date] | None = None,
    cooldown_trading_days: int = 5,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    strategies = list(selected_strategies or STRATEGY_NAMES)
    minimum_history = required_history_days(config, strategies)
    universe = storage.load_universe().copy()
    universe["symbol"] = universe["symbol"].astype(str).str.zfill(6)
    instruments = universe.to_dict("records")
    rows: list[dict[str, object]] = []
    sampled_trade_date_set = {item for item in sampled_trade_dates} if sampled_trade_dates is not None else None

    for instrument_index, instrument in enumerate(instruments, start=1):
        symbol = str(instrument["symbol"]).zfill(6)
        try:
            daily_bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError):
            if progress_callback is not None:
                progress_callback(instrument_index, len(instruments))
            continue

        history = add_indicators(daily_bars).sort_values("trade_date").reset_index(drop=True)
        if history.empty:
            if progress_callback is not None:
                progress_callback(instrument_index, len(instruments))
            continue

        history["trade_date"] = pd.to_datetime(history["trade_date"])
        trade_days = history["trade_date"].dt.date
        candidate_indices = [
            index
            for index, trade_day in enumerate(trade_days.tolist())
            if start_date <= trade_day <= end_date
            and (sampled_trade_date_set is None or trade_day in sampled_trade_date_set)
            and index + 1 >= minimum_history
        ]
        last_kept_index_by_strategy: dict[str, int] = {}

        for history_index in candidate_indices:
            cutoff = history.iloc[: history_index + 1].reset_index(drop=True)
            latest = cutoff.iloc[-1]
            if pd.isna(latest["amount_ma_20"]) or latest["amount_ma_20"] < config.universe.min_avg_amount_20d:
                continue

            matches = evaluate_strategies(cutoff, instrument, config, strategies)
            for match in matches:
                strategy_name = str(match.get("strategy_name", ""))
                last_kept_index = last_kept_index_by_strategy.get(strategy_name)
                if last_kept_index is not None and history_index - last_kept_index < cooldown_trading_days:
                    continue

                prepared = dict(match)
                prepared["symbol"] = symbol
                prepared["pattern_id"] = PATTERN_ID_BY_STRATEGY.get(strategy_name)
                prepared["signal_type"] = strategy_name
                prepared["trigger_reason"] = prepared.get("reason", "")
                prepared["trend_score"] = 0.0
                prepared["entry_score"] = 0.0
                rows.append(prepared)
                last_kept_index_by_strategy[strategy_name] = history_index

        if progress_callback is not None:
            progress_callback(instrument_index, len(instruments))

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    return result.sort_values(["trade_date", "pattern_id", "symbol"]).reset_index(drop=True)


def sample_pattern_backtest_trade_dates(
    storage: Storage,
    *,
    start_date: date,
    end_date: date,
    sample_size: int,
    seed: int | None = None,
) -> list[date]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    available_dates: set[date] = set()
    universe = storage.load_universe().copy()
    universe["symbol"] = universe["symbol"].astype(str).str.zfill(6)
    for symbol in universe["symbol"].tolist():
        try:
            daily_bars = storage.load_daily_bars(str(symbol))
        except (FileNotFoundError, DailyBarsReadError):
            continue

        if daily_bars.empty or "trade_date" not in daily_bars.columns:
            continue

        trade_dates = pd.to_datetime(daily_bars["trade_date"], errors="coerce").dropna().dt.date
        available_dates.update(item for item in trade_dates.tolist() if start_date <= item <= end_date)

    sorted_dates = sorted(available_dates)
    if not sorted_dates:
        return []
    if sample_size >= len(sorted_dates):
        return sorted_dates

    rng = random.Random(seed)
    return sorted(rng.sample(sorted_dates, sample_size))


def build_pattern_forward_price_frame(
    signals: pd.DataFrame,
    daily_history_by_symbol: dict[str, pd.DataFrame],
    *,
    forward_days: int = 40,
    entry_timing: str = "next_open",
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    if forward_days <= 0:
        raise ValueError("forward_days must be positive")

    prepared_history = _prepare_daily_history_map(daily_history_by_symbol, with_indicators=True)
    rows: list[dict[str, object]] = []
    for signal in signals.to_dict("records"):
        symbol = str(signal["symbol"]).zfill(6)
        history = prepared_history.get(symbol)
        if history is None:
            continue

        signal_date = pd.Timestamp(signal["trade_date"]).date()
        signal_index = history["index_by_date"].get(signal_date)
        if signal_index is None:
            continue

        if entry_timing == "next_open":
            entry_index = signal_index + 1
            entry_price_column = "open"
        else:
            entry_index = signal_index
            entry_price_column = "close"
        exit_index = entry_index + forward_days - 1
        if entry_index >= len(history["frame"]) or exit_index >= len(history["frame"]):
            continue

        entry_row = history["frame"].iloc[entry_index]
        entry_price = float(entry_row[entry_price_column])
        if entry_price <= 0:
            continue

        pattern_id = str(signal.get("pattern_id") or PATTERN_ID_BY_STRATEGY.get(str(signal.get("signal_type", ""))) or "")
        sample_id = f"{signal_date:%Y%m%d}_{symbol}_{pattern_id}"
        forward_slice = history["frame"].iloc[entry_index : exit_index + 1].reset_index(drop=True)
        for offset, row in enumerate(forward_slice.to_dict("records"), start=1):
            rows.append(
                {
                    "sample_id": sample_id,
                    "symbol": symbol,
                    "name": signal.get("name", ""),
                    "pattern_id": pattern_id,
                    "strategy_name": signal.get("signal_type"),
                    "signal_date": pd.Timestamp(signal_date),
                    "entry_date": pd.Timestamp(entry_row["trade_date"]),
                    "entry_price": round(entry_price, 4),
                    "entry_timing": entry_timing,
                    "forward_day": int(offset),
                    "trade_date": pd.Timestamp(row["trade_date"]),
                    "open": round(float(row["open"]), 4),
                    "high": round(float(row["high"]), 4),
                    "low": round(float(row["low"]), 4),
                    "close": round(float(row["close"]), 4),
                    "ma_20": round(float(row["ma_20"]), 4) if pd.notna(row.get("ma_20")) else None,
                    "close_below_ma20": bool(float(row["close"]) < float(row["ma_20"])) if pd.notna(row.get("ma_20")) else None,
                    "volume": row.get("volume"),
                    "amount": row.get("amount"),
                    "open_return_pct": round(float(row["open"]) / entry_price - 1, 4),
                    "high_return_pct": round(float(row["high"]) / entry_price - 1, 4),
                    "low_return_pct": round(float(row["low"]) / entry_price - 1, 4),
                    "close_return_pct": round(float(row["close"]) / entry_price - 1, 4),
                    "trigger_reason": signal.get("trigger_reason", signal.get("reason", "")),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["entry_date", "sample_id", "forward_day"]).reset_index(drop=True)


def summarize_pattern_backtest(backtest_results: pd.DataFrame) -> pd.DataFrame:
    if backtest_results.empty:
        return pd.DataFrame()

    frame = backtest_results.copy()
    if "pattern_id" not in frame.columns:
        frame["pattern_id"] = frame["signal_type"].map(PATTERN_ID_BY_STRATEGY)

    rows: list[dict[str, object]] = []
    grouped = frame.groupby(["pattern_id", "signal_type", "holding_days"], dropna=False)
    for (pattern_id, signal_type, holding_days), subset in grouped:
        rows.append(
            {
                "pattern_id": str(pattern_id),
                "strategy_name": signal_type,
                "holding_days": int(holding_days),
                "signal_count": int(len(subset)),
                "win_rate": round(float((subset["return_pct"] > 0).mean()), 4),
                "avg_return_pct": round(float(subset["return_pct"].mean()), 4),
                "median_return_pct": round(float(subset["return_pct"].median()), 4),
                "return_std_pct": round(float(subset["return_pct"].std(ddof=0)), 4),
                "avg_max_upside_pct": round(float(subset["max_upside_pct"].mean()), 4),
                "avg_max_drawdown_pct": round(float(subset["max_drawdown_pct"].mean()), 4),
                "entry_note": subset["entry_note"].iloc[0] if "entry_note" in subset.columns else None,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["pattern_id", "holding_days"]).reset_index(drop=True)


def _prepare_daily_history_map(daily_history_by_symbol: dict[str, pd.DataFrame], *, with_indicators: bool = False) -> dict[str, dict[str, object]]:
    prepared: dict[str, dict[str, object]] = {}
    for symbol, frame in daily_history_by_symbol.items():
        normalized_symbol = str(symbol).zfill(6)
        history = add_indicators(frame) if with_indicators and "ma_20" not in frame.columns else frame.copy()
        history = history.sort_values("trade_date").reset_index(drop=True)
        history["trade_date"] = pd.to_datetime(history["trade_date"])
        history["trade_day"] = history["trade_date"].dt.date
        prepared[normalized_symbol] = {
            "frame": history,
            "index_by_date": {trade_day: index for index, trade_day in enumerate(history["trade_day"].tolist())},
        }
    return prepared
