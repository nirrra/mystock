from __future__ import annotations

from pathlib import Path

import yaml

from .models import (
    AppConfig,
    HistoryMomentumFilterConfig,
    NetworkConfig,
    ProbabilityConfig,
    ScreeningConfig,
    StorageConfig,
    Type1Config,
    Type2Config,
    Type3Config,
    Type4Config,
    UniverseConfig,
)


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    storage = raw["storage"]
    universe = raw["universe"]
    screening = raw["screening"]
    probability = raw.get("probability", {})
    strategies = raw["strategies"]
    network = raw.get("network", {})
    history_momentum_filter = raw.get("history_momentum_filter", {})

    return AppConfig(
        provider=raw["provider"],
        adjustment=raw["adjustment"],
        network=NetworkConfig(
            http_proxy=network.get("http_proxy"),
            https_proxy=network.get("https_proxy"),
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
        probability=ProbabilityConfig(
            horizon_days=int(probability.get("horizon_days", 20)),
            min_future_return=float(probability.get("min_future_return", 0.03)),
            max_future_drawdown=float(probability.get("max_future_drawdown", 0.08)),
            min_history_days=int(probability.get("min_history_days", 200)),
            top_n_list=tuple(int(item) for item in probability.get("top_n_list", [10, 20, 50])),
        ),
        type1=Type1Config(**strategies["type1"]),
        type2=Type2Config(**strategies["type2"]),
        type3=Type3Config(**strategies["type3"]),
        type4=Type4Config(**strategies["type4"]),
    )
