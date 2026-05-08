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
    min_drawdown_pct: float
    peak_window_days: int
    breakout_volume_high_lookback_days: int
    breakout_min_close_position: float
    breakout_max_upper_shadow_pct: float
    breakout_min_body_pct: float
    near_high_threshold_pct: float
    pre_breakout_volume_ratio_max: float
    require_break_below_ma60: bool = True


@dataclass(slots=True)
class Type2Config:
    min_old_high_gap_days: int
    min_drawdown_pct: float
    peak_window_days: int
    breakout_volume_high_lookback_days: int
    breakout_min_close_position: float
    breakout_max_upper_shadow_pct: float
    breakout_min_body_pct: float
    post_breakout_max_days: int
    post_breakout_max_high_extension_pct: float
    post_breakout_ma20_break_tolerance_pct: float
    require_break_below_ma60: bool = True


@dataclass(slots=True)
class Type3Config:
    min_old_high_gap_days: int
    min_drawdown_pct: float
    peak_window_days: int
    breakout_volume_high_lookback_days: int
    breakout_min_close_position: float
    breakout_max_upper_shadow_pct: float
    breakout_min_body_pct: float
    post_breakout_max_days: int
    post_breakout_max_high_extension_pct: float
    post_breakout_ma20_break_tolerance_pct: float
    require_break_below_ma60: bool = True


@dataclass(slots=True)
class Type4Config:
    max_peak_scan_days: int
    min_peak_age_days: int
    max_peak_age_days: int
    peak_left_window_days: int
    peak_right_window_days: int
    neck_lookback_days: int
    neck_min_return: float
    neck_low_to_peak_min_return: float
    require_peak_above_ma60: bool
    peak_close_above_ma20_min_pct: float
    pullback_min_days: int
    pullback_max_days: int
    peak_to_pullback_min_drawdown_pct: float
    pullback_low_ma60_tolerance_pct: float
    forbid_effective_break_ma60: bool
    pullback_volume_max_peak_tail_ratio: float
    pullback_back_half_volume_ratio: float
    nostril_day_volume_ma20_max_ratio: float
    pullback_max_single_day_volume_peak_tail_ratio: float
    require_prior_ma5_below_ma10: bool
    prior_ma5_below_ma10_min_days: int
    require_cross_after_pullback_low: bool
    cross_lookback_days: int
    cross_confirm_gap_min_pct: float
    cross_confirm_gap_max_pct: float
    require_post_cross_ma5_above_ma10: bool
    latest_close_ma10_tolerance_pct: float
    current_below_peak_min_pct: float
    current_above_ma20_min_pct: float
    current_above_ma60_min_pct: float
    max_today_return_pct: float
    large_bearish_body_min_pct: float
    large_bearish_volume_ratio_min: float
    max_large_bearish_count: int


@dataclass(slots=True)
class Type5Config:
    recent_high_lookback_days: int
    high_pre_lookback_days: int
    high_peak_window_days: int
    ma20_touch_lookback_days: int
    ma20_touch_abs_tolerance: float
    ma20_touch_pct_tolerance: float
    ma20_reclaim_min_pct: float
    ma_slope_short_lookback_days: int
    ma_slope_long_lookback_days: int
    pullback_volume_contraction_max: float


@dataclass(slots=True)
class Type6Config:
    max_anchor_scan_days: int
    min_anchor_age_days: int
    anchor_min_return: float
    anchor_prev_volume_multiplier: float
    anchor_ma_volume_multiplier: float
    launch_confirm_days: int
    launch_min_high_return: float
    launch_limit_up_return: float
    launch_limit_up_min_count: int
    peak_to_pullback_min_drawdown_pct: float
    pullback_max_days: int
    pullback_volume_max_anchor_ratio: float
    pullback_volume_split_min_days: int
    pullback_back_half_volume_ratio: float
    pullback_max_rise_tail_volume_ratio: float
    support_tolerance_pct: float
    support_close_range_pct: float
    support_touch_lookback_days: int
    support_break_tolerance_pct: float
    break_reclaim_lookback_days: int
    break_below_pct: float
    breakdown_volume_max_anchor_ratio: float
    max_reclaim_days: int
    post_reclaim_max_sideways_days: int
    post_reclaim_range_max: float


@dataclass(slots=True)
class HistoryMomentumFilterConfig:
    lookback_days: int
    window_days: int
    min_return: float


@dataclass(slots=True)
class ScreeningConfig:
    output_limit: int


@dataclass(slots=True)
class NetworkConfig:
    http_proxy: str | None
    https_proxy: str | None
    socks5_proxy: str | None
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
    type1: Type1Config
    type2: Type2Config
    type3: Type3Config
    type4: Type4Config
    type5: Type5Config
    type6: Type6Config
