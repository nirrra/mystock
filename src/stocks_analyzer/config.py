from __future__ import annotations

from pathlib import Path

import yaml

from .models import (
    AppConfig,
    HistoryMomentumFilterConfig,
    NetworkConfig,
    PickTrendWatchlistConfig,
    ScreeningConfig,
    StorageConfig,
    TrendBacktestConfig,
    TrendBreakoutConfig,
    TrendEntryRulesConfig,
    TrendIndicatorWeightsConfig,
    TrendPullbackConfig,
    TrendSignalEntryRulesConfig,
    TrendSignalsConfig,
    TrendUniverseConfig,
    Type1Config,
    Type2Config,
    Type3Config,
    Type4Config,
    Type5Config,
    Type6Config,
    UniverseConfig,
    WatchlistTrendFilterConfig,
)


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    storage = raw["storage"]
    universe = raw["universe"]
    screening = raw["screening"]
    watchlist_trend_filter = raw.get("watchlist_trend_filter", {})
    pick_trend_watchlist = raw.get("pick_trend_watchlist", {})
    trend_universe = raw.get("trend_universe", {})
    trend_signals = raw.get("trend_signals", {})
    trend_breakout = trend_signals.get("breakout", {})
    trend_pullback = trend_signals.get("pullback", {})
    trend_indicator_weights = raw.get("trend_indicator_weights", {})
    trend_entry_rules = raw.get("trend_entry_rules", {})
    trend_entry_rules_breakout = trend_entry_rules.get("breakout", {})
    trend_entry_rules_pullback = trend_entry_rules.get("pullback", {})
    trend_backtest = raw.get("trend_backtest", {})
    strategies = raw["strategies"]
    network = raw.get("network", {})
    history_momentum_filter = raw.get("history_momentum_filter", {})

    return AppConfig(
        provider=raw["provider"],
        intraday_provider=raw.get("intraday_provider", raw["provider"]),
        adjustment=raw["adjustment"],
        network=NetworkConfig(
            http_proxy=network.get("http_proxy"),
            https_proxy=network.get("https_proxy"),
            socks5_proxy=network.get("socks5_proxy"),
            no_proxy=network.get("no_proxy"),
        ),
        storage=StorageConfig(
            base_dir=Path(storage["base_dir"]),
            universe_file=storage["universe_file"],
            signals_dir=storage["signals_dir"],
            reports_dir=storage["reports_dir"],
            daily_dir=storage["daily_dir"],
        ),
        universe=UniverseConfig(
            exclude_st=bool(universe["exclude_st"]),
            min_history_days=int(universe["min_history_days"]),
            min_avg_amount_20d=float(universe["min_avg_amount_20d"]),
        ),
        history_momentum_filter=HistoryMomentumFilterConfig(
            lookback_days=int(history_momentum_filter.get("lookback_days", 200)),
            window_days=int(history_momentum_filter.get("window_days", 5)),
            min_return=float(history_momentum_filter.get("min_return", 0.10)),
        ),
        screening=ScreeningConfig(output_limit=int(screening["output_limit"])),
        watchlist_trend_filter=WatchlistTrendFilterConfig(
            enabled=bool(watchlist_trend_filter.get("enabled", True)),
            buy_score_min=float(watchlist_trend_filter.get("buy_score_min", 0.0)),
            price_action_score_min=float(watchlist_trend_filter.get("price_action_score_min", 0.0)),
        ),
        pick_trend_watchlist=PickTrendWatchlistConfig(
            buy_score_min=float(pick_trend_watchlist.get("buy_score_min", 70.0)),
            price_action_score_min=float(pick_trend_watchlist.get("price_action_score_min", 55.0)),
        ),
        trend_universe=TrendUniverseConfig(
            min_history_days=int(trend_universe.get("min_history_days", 180)),
            ma_short_window=int(trend_universe.get("ma_short_window", 20)),
            ma_medium_window=int(trend_universe.get("ma_medium_window", 60)),
            ma_long_window=int(trend_universe.get("ma_long_window", 120)),
            slope_lookback_days=int(trend_universe.get("slope_lookback_days", 10)),
            strength_lookback_days=int(trend_universe.get("strength_lookback_days", 60)),
            quality_lookback_days=int(trend_universe.get("quality_lookback_days", 60)),
            min_return_strength_lookback=float(trend_universe.get("min_return_strength_lookback", 0.15)),
            max_drawdown_quality_lookback=float(trend_universe.get("max_drawdown_quality_lookback", 0.18)),
            min_avg_amount_20d=float(trend_universe.get("min_avg_amount_20d", universe["min_avg_amount_20d"])),
            high_lookback_days=int(trend_universe.get("high_lookback_days", 120)),
        ),
        trend_signals=TrendSignalsConfig(
            breakout=TrendBreakoutConfig(
                platform_min_window_days=int(trend_breakout.get("platform_min_window_days", 20)),
                platform_max_window_days=int(trend_breakout.get("platform_max_window_days", 40)),
                platform_range_max=float(trend_breakout.get("platform_range_max", 0.18)),
                breakout_volume_ratio_min=float(trend_breakout.get("breakout_volume_ratio_min", 1.2)),
                breakout_max_distance_pct=float(trend_breakout.get("breakout_max_distance_pct", 0.05)),
                new_high_lookback_days=int(trend_breakout.get("new_high_lookback_days", 120)),
                new_high_tolerance_pct=float(trend_breakout.get("new_high_tolerance_pct", 0.02)),
            ),
            pullback=TrendPullbackConfig(
                recent_high_lookback_days=int(trend_pullback.get("recent_high_lookback_days", 20)),
                max_drawdown_from_recent_high=float(trend_pullback.get("max_drawdown_from_recent_high", 0.12)),
                proximity_to_ma20=float(trend_pullback.get("proximity_to_ma20", 0.03)),
                proximity_to_ma60=float(trend_pullback.get("proximity_to_ma60", 0.05)),
                volume_contraction_max=float(trend_pullback.get("volume_contraction_max", 0.95)),
                rebound_min_return_1d=float(trend_pullback.get("rebound_min_return_1d", 0.0)),
                lower_shadow_min=float(trend_pullback.get("lower_shadow_min", 0.2)),
            ),
        ),
        trend_indicator_weights=TrendIndicatorWeightsConfig(
            trend_base_weight=float(trend_indicator_weights.get("trend_base_weight", 0.35)),
            price_action_weight=float(trend_indicator_weights.get("price_action_weight", 0.30)),
            macd_weight=float(trend_indicator_weights.get("macd_weight", 0.18)),
            volume_weight=float(trend_indicator_weights.get("volume_weight", 0.10)),
            volume_price_divergence_weight=float(trend_indicator_weights.get("volume_price_divergence_weight", 0.10)),
            boll_weight=float(trend_indicator_weights.get("boll_weight", 0.10)),
            rsi_weight=float(trend_indicator_weights.get("rsi_weight", 0.08)),
            kdj_weight=float(trend_indicator_weights.get("kdj_weight", 0.08)),
            atr_weight=float(trend_indicator_weights.get("atr_weight", 0.06)),
        ),
        trend_entry_rules=TrendEntryRulesConfig(
            buy_score_min=float(trend_entry_rules.get("buy_score_min", 80.0)),
            trend_base_score_min=float(trend_entry_rules.get("trend_base_score_min", 65.0)),
            price_action_score_min=float(trend_entry_rules.get("price_action_score_min", 60.0)),
            macd_score_min=float(trend_entry_rules.get("macd_score_min", 35.0)),
            positive_indicator_count_min=int(trend_entry_rules.get("positive_indicator_count_min", 3)),
            breakout=TrendSignalEntryRulesConfig(
                buy_score_min=_optional_float(trend_entry_rules_breakout.get("buy_score_min")),
                trend_base_score_min=_optional_float(trend_entry_rules_breakout.get("trend_base_score_min")),
                price_action_score_min=_optional_float(trend_entry_rules_breakout.get("price_action_score_min")),
                macd_score_min=_optional_float(trend_entry_rules_breakout.get("macd_score_min")),
                positive_indicator_count_min=_optional_int(trend_entry_rules_breakout.get("positive_indicator_count_min")),
            ),
            pullback=TrendSignalEntryRulesConfig(
                buy_score_min=_optional_float(trend_entry_rules_pullback.get("buy_score_min")),
                trend_base_score_min=_optional_float(trend_entry_rules_pullback.get("trend_base_score_min")),
                price_action_score_min=_optional_float(trend_entry_rules_pullback.get("price_action_score_min")),
                macd_score_min=_optional_float(trend_entry_rules_pullback.get("macd_score_min")),
                positive_indicator_count_min=_optional_int(trend_entry_rules_pullback.get("positive_indicator_count_min")),
            ),
        ),
        trend_backtest=TrendBacktestConfig(
            holding_days=tuple(int(item) for item in trend_backtest.get("holding_days", [5, 10, 20, 40])),
            portfolio_top_n=tuple(int(item) for item in trend_backtest.get("portfolio_top_n", [3, 5, 10])),
            entry_score_weight=float(trend_backtest.get("entry_score_weight", 0.6)),
            trend_score_weight=float(trend_backtest.get("trend_score_weight", 0.4)),
            entry_timing=str(trend_backtest.get("entry_timing", "same_close")),
        ),
        type1=Type1Config(**strategies["type1"]),
        type2=Type2Config(**strategies["type2"]),
        type3=Type3Config(**strategies["type3"]),
        type4=Type4Config(**strategies["type4"]),
        type5=Type5Config(**strategies["type5"]),
        type6=Type6Config(**strategies["type6"]),
    )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)
