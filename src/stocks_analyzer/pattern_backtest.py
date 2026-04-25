from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date

import pandas as pd

from .indicators import add_indicators
from .models import AppConfig
from .storage import DailyBarsReadError, Storage
from .strategies import STRATEGY_NAMES, evaluate_strategies, required_history_days


PATTERN_ID_BY_STRATEGY = {
    "volume_top_pre_breakout": "1",
    "volume_top_breakout": "2",
    "volume_top_follow_through": "3",
    "platform_breakout": "4",
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
    cooldown_trading_days: int = 5,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    strategies = list(selected_strategies or STRATEGY_NAMES)
    minimum_history = required_history_days(config, strategies)
    universe = storage.load_universe().copy()
    universe["symbol"] = universe["symbol"].astype(str).str.zfill(6)
    instruments = universe.to_dict("records")
    rows: list[dict[str, object]] = []

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
            if start_date <= trade_day <= end_date and index + 1 >= minimum_history
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
