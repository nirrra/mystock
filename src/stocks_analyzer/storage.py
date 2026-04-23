from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .paths import ProjectPaths


class DailyBarsReadError(RuntimeError):
    """Raised when a cached daily-bars parquet file exists but cannot be read."""


class Storage:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self.paths.ensure()

    def save_universe(self, dataframe: pd.DataFrame) -> Path:
        dataframe.to_parquet(self.paths.universe_path, index=False)
        return self.paths.universe_path

    def load_universe(self) -> pd.DataFrame:
        if not self.paths.universe_path.exists():
            raise FileNotFoundError(f"Universe file not found: {self.paths.universe_path}")
        return pd.read_parquet(self.paths.universe_path)

    def save_daily_bars(self, symbol: str, dataframe: pd.DataFrame) -> Path:
        target = self.paths.daily_dir / f"{symbol}.parquet"
        dataframe.to_parquet(target, index=False)
        return target

    def load_daily_bars(self, symbol: str) -> pd.DataFrame:
        target = self.paths.daily_dir / f"{symbol}.parquet"
        if not target.exists():
            raise FileNotFoundError(f"Daily bars not found for {symbol}: {target}")
        try:
            return pd.read_parquet(target)
        except Exception as exc:
            raise DailyBarsReadError(f"Daily bars are unreadable for {symbol}: {target}: {exc}") from exc

    def has_daily_bars(self, symbol: str) -> bool:
        target = self.paths.daily_dir / f"{symbol}.parquet"
        return target.exists()

    def save_signals(self, trade_date: date, dataframe: pd.DataFrame) -> Path:
        target = self.paths.signals_dir / f"signals_{trade_date.isoformat()}.parquet"
        dataframe.to_parquet(target, index=False)
        return target

    def load_signals(self, trade_date: date) -> pd.DataFrame:
        target = self.paths.signals_dir / f"signals_{trade_date.isoformat()}.parquet"
        if not target.exists():
            raise FileNotFoundError(f"Signals not found for {trade_date.isoformat()}: {target}")
        return pd.read_parquet(target)

    def save_report(self, trade_date: date, dataframe: pd.DataFrame) -> Path:
        target = self.paths.reports_dir / f"report_{trade_date.isoformat()}.csv"
        dataframe.to_csv(target, index=False, encoding="utf-8-sig")
        return target
