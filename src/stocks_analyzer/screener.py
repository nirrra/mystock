from __future__ import annotations

from datetime import date, datetime
from typing import Callable

import pandas as pd

from .indicators import add_indicators
from .models import AppConfig
from .storage import Storage
from .strategies import STRATEGY_NAMES, evaluate_strategies, required_history_days


class Screener:
    def __init__(self, storage: Storage, config: AppConfig) -> None:
        self.storage = storage
        self.config = config

    def run(
        self,
        as_of: date,
        selected_strategies: list[str] | None = None,
        symbols: list[str] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> pd.DataFrame:
        strategies = selected_strategies or list(STRATEGY_NAMES)
        universe = self.storage.load_universe()
        if symbols:
            symbol_set = {str(symbol).zfill(6) for symbol in symbols}
            universe = universe[universe["symbol"].astype(str).isin(symbol_set)].reset_index(drop=True)
        minimum_history = required_history_days(self.config, strategies)
        results: list[dict[str, object]] = []
        instruments = universe.to_dict("records")
        total_instruments = len(instruments)

        for index, instrument in enumerate(instruments, start=1):
            symbol = instrument["symbol"]
            try:
                daily_bars = self.storage.load_daily_bars(symbol)
            except FileNotFoundError:
                if progress_callback is not None:
                    progress_callback(index, total_instruments)
                continue

            indicator_frame = add_indicators(daily_bars)
            cutoff = indicator_frame[indicator_frame["trade_date"].dt.date <= as_of].reset_index(drop=True)
            if cutoff.empty or len(cutoff) < minimum_history:
                if progress_callback is not None:
                    progress_callback(index, total_instruments)
                continue

            latest = cutoff.iloc[-1]
            if pd.isna(latest["amount_ma_20"]) or latest["amount_ma_20"] < self.config.universe.min_avg_amount_20d:
                if progress_callback is not None:
                    progress_callback(index, total_instruments)
                continue

            results.extend(evaluate_strategies(cutoff, instrument, self.config, strategies))
            if progress_callback is not None:
                progress_callback(index, total_instruments)

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results).sort_values(["strategy_name", "symbol"]).reset_index(drop=True)


def parse_as_of(value: str | None) -> date:
    if value is None:
        return date.today()
    return datetime.fromisoformat(value).date()
