from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class StorageConfig:
    base_dir: Path
    universe_file: str
    signals_dir: str
    reports_dir: str
    daily_dir: str


@dataclass(slots=True)
class UniverseConfig:
    exclude_st: bool
    min_history_days: int
    min_avg_amount_20d: float


@dataclass(slots=True)
class Type1Config:
    min_old_high_gap_days: int
    max_old_high_gap_days: int
    min_drawdown_pct: float
    near_high_threshold_pct: float
    breakout_lookback_days: int
    breakout_pullback_min_distance_pct: float
    breakout_pullback_max_distance_pct: float
    peak_window_days: int
    volume_window_min_days: int
    volume_window_max_days: int
    volume_median_multiplier: float


@dataclass(slots=True)
class Type2Config:
    trend_lookback_days: int
    min_return_trend_lookback: float
    ma60_rising_lookback: int
    platform_window_days: int
    platform_range_max: float
    platform_upper_half_min_days: int
    breakout_volume_ratio_min: float
    breakout_lookback_days: int
    breakout_min_distance_pct: float
    breakout_max_distance_pct: float


@dataclass(slots=True)
class Type3Config:
    trend_lookback_days: int
    min_return_trend_lookback: float
    ma_rising_lookback: int
    proximity_to_ma20: float
    max_drawdown_15d: float
    volume_contraction_max: float


@dataclass(slots=True)
class Type4Config:
    strong_lookback_days: int
    min_return_strong_lookback: float
    strong_day_return_min: float
    consolidation_min_days: int
    consolidation_max_days: int
    consolidation_range_max: float
    restart_breakout_days: int


@dataclass(slots=True)
class HistoryMomentumFilterConfig:
    lookback_days: int
    window_days: int
    min_return: float


@dataclass(slots=True)
class ScreeningConfig:
    output_limit: int


@dataclass(slots=True)
class ProbabilityConfig:
    horizon_days: int
    min_future_return: float
    max_future_drawdown: float
    min_history_days: int
    top_n_list: tuple[int, ...]


@dataclass(slots=True)
class NetworkConfig:
    http_proxy: str | None
    https_proxy: str | None
    no_proxy: str | None


@dataclass(slots=True)
class AppConfig:
    provider: str
    adjustment: str
    network: NetworkConfig
    storage: StorageConfig
    universe: UniverseConfig
    history_momentum_filter: HistoryMomentumFilterConfig
    screening: ScreeningConfig
    probability: ProbabilityConfig
    type1: Type1Config
    type2: Type2Config
    type3: Type3Config
    type4: Type4Config
