from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    @abstractmethod
    def get_instruments(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_daily_bars(self, symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_index_daily_bars(self, index_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise RuntimeError(f"{type(self).__name__} does not implement index daily bars.")

    @abstractmethod
    def get_intraday_bars(
        self,
        symbol: str,
        start_datetime: str,
        end_datetime: str,
        period: str,
        adjust: str,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def close(self) -> None:
        return None
