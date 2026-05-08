from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .full_market_return import alpha158_qlib_return_predictions_path, predict_alpha158_qlib_return
from .full_market_risk import barrier_risk_predictions_path, predict_barrier_risk, predict_tail_risk, tail_risk_predictions_path
from .full_market_trade_day import (
    _latest_trade_day_prediction_row,
    _score_trade_day_model,
    build_trade_day_feature_frame,
    load_trade_day_gate_model_artifact,
    trade_day_gate_prediction_path,
)
from .phase_display import add_phase5_score_100
from .screener import Screener
from .storage import DailyBarsReadError, Storage
from .strategies import STRATEGY_NAMES


DEFAULT_BACKTEST_STRATEGIES = (
    "full_market_random",
    "phase1_filter_only",
    "phase2_filter_only",
    "phase4_top",
    "phase5_group_only",
    "phase7_gate_only",
    "phase1_filter_phase4_top",
    "phase2_filter_phase4_top",
    "phase1_phase2_filter_phase4_top",
    "patterns_only",
    "patterns_phase1_phase2_filter",
    "patterns_phase4_sort",
    "current_watchlist_without_phase7",
    "current_watchlist_with_phase7",
)

PATTERN_ID_BY_STRATEGY = {
    "volume_top_pre_breakout": "1",
    "volume_top_breakout": "2",
    "volume_top_follow_through": "3",
    "duck_nostril_cross": "4",
    "trend_pullback": "5",
    "double_volume_support_rebound": "6",
}
PATTERN_PRIORITY = {"5": 6.0, "1": 5.0, "6": 4.0, "3": 3.0, "2": 2.0, "4": 1.0}


@dataclass(slots=True)
class DailyScreeningBacktestResult:
    trades: pd.DataFrame
    daily_portfolio: pd.DataFrame
    summary: pd.DataFrame
    comparison: pd.DataFrame
    output_dir: Path
    trades_path: Path
    daily_portfolio_path: Path
    summary_path: Path
    comparison_path: Path


def backtest_daily_screening_components(
    *,
    storage: Storage,
    project_root: Path,
    config: Any,
    start_date: date,
    end_date: date,
    strategies: tuple[str, ...] = DEFAULT_BACKTEST_STRATEGIES,
    horizons: tuple[int, ...] = (5, 10, 20, 60),
    top_n: int = 20,
    phase1_filter_rate: float = 0.2,
    phase2_filter_rate: float = 0.2,
    phase4_top_n: int = 20,
    stop_loss_pct: float = 0.08,
    take_profit_pct: float = 0.15,
    max_signal_days: int = 30,
    symbol_limit: int | None = 500,
    output_dir: Path | None = None,
    progress: bool = False,
    use_cache: bool = True,
) -> DailyScreeningBacktestResult:
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")
    if not horizons:
        raise ValueError("horizons must not be empty")
    unknown = sorted(set(strategies) - set(DEFAULT_BACKTEST_STRATEGIES))
    if unknown:
        raise ValueError(f"Unsupported backtest strategies: {unknown}")

    output_root = output_dir or project_root / "reports" / "daily_screening_smoke_backtest"
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = output_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    universe = _load_backtest_universe(storage, symbol_limit=symbol_limit)
    signal_dates = _select_signal_dates(
        storage=storage,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        max_horizon=max(horizons),
        max_signal_days=max_signal_days,
    )
    if not signal_dates:
        raise RuntimeError("No signal dates have enough forward data for the requested backtest.")

    phase5 = _load_phase5_measures(project_root)
    phase7_predictions = _prepare_phase7_predictions(
        storage=storage,
        project_root=project_root,
        signal_dates=signal_dates,
        cache_dir=cache_dir,
        use_cache=use_cache,
        progress=progress,
    )
    price_cache: dict[str, pd.DataFrame] = {}
    trade_parts: list[pd.DataFrame] = []
    candidate_count_rows: list[dict[str, object]] = []

    total_dates = len(signal_dates)
    for date_index, signal_date in enumerate(signal_dates, start=1):
        if progress:
            print(f"Backtest signal date {date_index}/{total_dates}: {signal_date.isoformat()}", flush=True)
        snapshots = _load_daily_snapshots(
            storage=storage,
            project_root=project_root,
            config=config,
            universe=universe,
            signal_date=signal_date,
            phase5_measures=phase5,
            phase7_predictions=phase7_predictions,
            cache_dir=cache_dir,
            symbol_limit=symbol_limit,
            phase1_filter_rate=phase1_filter_rate,
            phase2_filter_rate=phase2_filter_rate,
            use_cache=use_cache,
        )
        for strategy in strategies:
            candidates = _select_strategy_candidates(
                strategy=strategy,
                signal_date=signal_date,
                universe=universe,
                snapshots=snapshots,
                top_n=top_n,
                phase4_top_n=phase4_top_n,
            )
            candidate_count_rows.append(
                {
                    "strategy": strategy,
                    "signal_date": signal_date.isoformat(),
                    "candidate_count": int(len(candidates)),
                    "trade_permission": snapshots.phase7_permission,
                }
            )
            if candidates.empty:
                continue
            trades = _simulate_candidate_trades(
                storage=storage,
                candidates=candidates,
                signal_date=signal_date,
                horizons=horizons,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                price_cache=price_cache,
            )
            if not trades.empty:
                trade_parts.append(trades)

    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    candidate_counts = pd.DataFrame(candidate_count_rows)
    daily_portfolio = _build_daily_portfolio(trades, candidate_counts)
    summary = _build_backtest_summary(
        trades=trades,
        daily_portfolio=daily_portfolio,
        candidate_counts=candidate_counts,
        strategies=strategies,
        horizons=horizons,
        signal_days=len(signal_dates),
    )
    comparison = _build_comparison(summary)

    trades_path = output_root / "trades.csv"
    daily_path = output_root / "daily_portfolio.csv"
    summary_path = output_root / "summary.csv"
    comparison_path = output_root / "comparison.csv"
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    daily_portfolio.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    return DailyScreeningBacktestResult(
        trades=trades,
        daily_portfolio=daily_portfolio,
        summary=summary,
        comparison=comparison,
        output_dir=output_root,
        trades_path=trades_path,
        daily_portfolio_path=daily_path,
        summary_path=summary_path,
        comparison_path=comparison_path,
    )


@dataclass(slots=True)
class _DailySnapshots:
    phase1: pd.DataFrame
    phase2: pd.DataFrame
    phase4: pd.DataFrame
    phase5: pd.DataFrame
    patterns: pd.DataFrame
    phase7_permission: str


def _load_backtest_universe(storage: Storage, *, symbol_limit: int | None) -> pd.DataFrame:
    universe = storage.load_universe().copy()
    if "symbol" not in universe.columns:
        raise RuntimeError("Universe is missing symbol column.")
    universe["symbol"] = universe["symbol"].astype(str).str.zfill(6)
    if "name" not in universe.columns:
        universe["name"] = ""
    if symbol_limit is not None:
        universe = universe.head(max(int(symbol_limit), 0)).copy()
    return universe.loc[:, ["symbol", "name"]].drop_duplicates("symbol").reset_index(drop=True)


def _select_signal_dates(
    *,
    storage: Storage,
    universe: pd.DataFrame,
    start_date: date,
    end_date: date,
    max_horizon: int,
    max_signal_days: int,
) -> list[date]:
    collected: set[date] = set()
    for symbol in universe["symbol"].astype(str).head(80):
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError):
            continue
        dates = pd.to_datetime(bars.get("trade_date"), errors="coerce").dropna().dt.date.tolist()
        collected.update(dates)
        if len(collected) > max_signal_days + max_horizon + 260:
            break
    if not collected:
        return []
    all_dates = sorted(collected)
    latest_signal = all_dates[-(max_horizon + 1)] if len(all_dates) > max_horizon else all_dates[-1]
    candidates = [item for item in all_dates if start_date <= item <= end_date and item <= latest_signal]
    if max_signal_days > 0 and len(candidates) > max_signal_days:
        candidates = candidates[-max_signal_days:]
    return candidates


def _load_daily_snapshots(
    *,
    storage: Storage,
    project_root: Path,
    config: Any,
    universe: pd.DataFrame,
    signal_date: date,
    phase5_measures: pd.DataFrame,
    phase7_predictions: dict[date, pd.DataFrame],
    cache_dir: Path,
    symbol_limit: int | None,
    phase1_filter_rate: float,
    phase2_filter_rate: float,
    use_cache: bool,
) -> _DailySnapshots:
    phase1 = _load_or_predict_phase1(storage, project_root, signal_date, cache_dir, symbol_limit=symbol_limit, use_cache=use_cache)
    phase2 = _load_or_predict_phase2(storage, project_root, signal_date, cache_dir, symbol_limit=symbol_limit, use_cache=use_cache)
    phase4 = _load_or_predict_phase4(storage, project_root, signal_date, cache_dir, symbol_limit=symbol_limit, use_cache=use_cache)
    phase7 = phase7_predictions.get(signal_date, pd.DataFrame())
    patterns = _load_or_scan_patterns(
        storage=storage,
        config=config,
        universe=universe,
        signal_date=signal_date,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )
    phase1 = _prepare_phase1(phase1, filter_rate=phase1_filter_rate)
    phase2 = _prepare_phase2(phase2, filter_rate=phase2_filter_rate)
    phase4 = _prepare_phase4(phase4)
    phase5 = _prepare_phase5_for_date(phase5_measures, signal_date)
    permission = "unknown"
    if not phase7.empty and "trade_permission" in phase7.columns:
        permission = str(phase7.iloc[-1]["trade_permission"]).strip().lower()
    return _DailySnapshots(
        phase1=phase1,
        phase2=phase2,
        phase4=phase4,
        phase5=phase5,
        patterns=patterns,
        phase7_permission=permission,
    )


def _load_or_predict_phase1(
    storage: Storage,
    project_root: Path,
    signal_date: date,
    cache_dir: Path,
    *,
    symbol_limit: int | None,
    use_cache: bool,
) -> pd.DataFrame:
    path = cache_dir / tail_risk_predictions_path(project_root, signal_date).name
    if use_cache and path.exists():
        return pd.read_csv(path)
    result = predict_tail_risk(storage=storage, project_root=project_root, trade_date=signal_date, output=path, limit=symbol_limit)
    return result.predictions


def _load_or_predict_phase2(
    storage: Storage,
    project_root: Path,
    signal_date: date,
    cache_dir: Path,
    *,
    symbol_limit: int | None,
    use_cache: bool,
) -> pd.DataFrame:
    path = cache_dir / barrier_risk_predictions_path(project_root, signal_date).name
    if use_cache and path.exists():
        return pd.read_csv(path)
    result = predict_barrier_risk(storage=storage, project_root=project_root, trade_date=signal_date, output=path, limit=symbol_limit)
    return result.predictions


def _load_or_predict_phase4(
    storage: Storage,
    project_root: Path,
    signal_date: date,
    cache_dir: Path,
    *,
    symbol_limit: int | None,
    use_cache: bool,
) -> pd.DataFrame:
    path = cache_dir / alpha158_qlib_return_predictions_path(project_root, signal_date).name
    if use_cache and path.exists():
        return pd.read_csv(path)
    result = predict_alpha158_qlib_return(storage=storage, project_root=project_root, trade_date=signal_date, output=path, limit=symbol_limit)
    return result.predictions


def _prepare_phase7_predictions(
    *,
    storage: Storage,
    project_root: Path,
    signal_dates: list[date],
    cache_dir: Path,
    use_cache: bool,
    progress: bool,
) -> dict[date, pd.DataFrame]:
    predictions: dict[date, pd.DataFrame] = {}
    missing_dates: list[date] = []
    for signal_date in signal_dates:
        path = cache_dir / trade_day_gate_prediction_path(project_root, signal_date).name
        if use_cache and path.exists():
            predictions[signal_date] = pd.read_csv(path)
        else:
            missing_dates.append(signal_date)
    if not missing_dates:
        return predictions

    if progress:
        print(f"Building Phase7 market features once for {len(missing_dates)} uncached signal dates.", flush=True)
    artifact = load_trade_day_gate_model_artifact(project_root)
    train_config = artifact.get("train_config", {})
    features = build_trade_day_feature_frame(
        storage=storage,
        start_date=None,
        end_date=max(missing_dates),
        limit=train_config.get("limit"),
        min_stock_count=int(train_config.get("min_stock_count", 500)),
    )
    for signal_date in missing_dates:
        prediction = _score_phase7_prediction_from_features(features, artifact=artifact, trade_date=signal_date)
        path = cache_dir / trade_day_gate_prediction_path(project_root, signal_date).name
        path.parent.mkdir(parents=True, exist_ok=True)
        prediction.to_csv(path, index=False, encoding="utf-8-sig")
        predictions[signal_date] = prediction
    return predictions


def _score_phase7_prediction_from_features(features: pd.DataFrame, *, artifact: dict[str, Any], trade_date: date) -> pd.DataFrame:
    feature_columns = tuple(artifact["feature_columns"])
    row = _latest_trade_day_prediction_row(features, trade_date)
    if row.empty:
        raise RuntimeError(f"No trade-day gate feature row on or before {trade_date.isoformat()}")
    row = row.tail(1).dropna(subset=list(feature_columns)).copy()
    if row.empty:
        raise RuntimeError(f"Trade-day gate feature row has missing model features for {trade_date.isoformat()}")
    score = float(
        _score_trade_day_model(
            row,
            model=artifact.get("model"),
            feature_columns=feature_columns,
            model_name=str(artifact["model_name"]),
        )[0]
    )
    threshold = float(artifact["selected_threshold"])
    permission = "allow" if score < threshold else "no_trade"
    record: dict[str, Any] = {
        "trade_date": trade_date.isoformat(),
        "feature_trade_date": pd.Timestamp(row.iloc[0]["trade_date"]).date().isoformat(),
        "buy_day_risk_score": score,
        "selected_threshold": threshold,
        "trade_permission": permission,
        "suggested_action": "candidate_allowed" if permission == "allow" else "observation_only",
        "reason": "buy_day_risk_score_below_threshold" if permission == "allow" else "buy_day_risk_score_ge_threshold",
        "model_name": artifact["model_name"],
        "model_version": artifact["model_version"],
    }
    for column in feature_columns:
        record[column] = float(row.iloc[0][column])
    return pd.DataFrame([record])


def _load_or_scan_patterns(
    *,
    storage: Storage,
    config: Any,
    universe: pd.DataFrame,
    signal_date: date,
    cache_dir: Path,
    use_cache: bool,
) -> pd.DataFrame:
    path = cache_dir / f"patterns_all_{signal_date.isoformat()}.csv"
    if use_cache and path.exists():
        return pd.read_csv(path)
    screener = Screener(storage, config)
    symbols = universe["symbol"].astype(str).tolist()
    frame = screener.run(as_of=signal_date, selected_strategies=list(STRATEGY_NAMES), symbols=symbols)
    if frame.empty:
        frame = pd.DataFrame(columns=["trade_date", "symbol", "name", "strategy_name", "pattern_id", "reason"])
    else:
        frame = frame.copy()
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        frame["pattern_id"] = frame["strategy_name"].map(PATTERN_ID_BY_STRATEGY).fillna("")
        frame = frame.drop_duplicates(["symbol", "pattern_id"], keep="first").reset_index(drop=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def _load_phase5_measures(project_root: Path) -> pd.DataFrame:
    path = project_root / "reports" / "full_market_model" / "mcd_crash_annual_measures.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _prepare_phase1(frame: pd.DataFrame, *, filter_rate: float) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns or "risk_score" not in frame.columns:
        return pd.DataFrame(columns=["symbol", "name", "phase1_risk_score", "phase1_pass"])
    result = frame.copy()
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    result["phase1_risk_score"] = pd.to_numeric(result["risk_score"], errors="coerce")
    result = result.dropna(subset=["phase1_risk_score"]).sort_values(["phase1_risk_score", "symbol"], ascending=[False, True])
    removed = max(1, int(math.ceil(len(result) * float(filter_rate)))) if len(result) else 0
    result["phase1_pass"] = True
    if removed:
        result.loc[result.index[:removed], "phase1_pass"] = False
    keep = ["symbol", "phase1_risk_score", "phase1_pass"]
    if "name" in result.columns:
        keep.insert(1, "name")
    return result.loc[:, keep].drop_duplicates("symbol", keep="first")


def _prepare_phase2(frame: pd.DataFrame, *, filter_rate: float) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns or "barrier_risk_score" not in frame.columns:
        return pd.DataFrame(columns=["symbol", "phase2_barrier_risk_score", "phase2_pass"])
    result = frame.copy()
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    result["phase2_barrier_risk_score"] = pd.to_numeric(result["barrier_risk_score"], errors="coerce")
    result = result.dropna(subset=["phase2_barrier_risk_score"]).sort_values(
        ["phase2_barrier_risk_score", "symbol"], ascending=[False, True]
    )
    removed = max(1, int(math.ceil(len(result) * float(filter_rate)))) if len(result) else 0
    result["phase2_pass"] = True
    if removed:
        result.loc[result.index[:removed], "phase2_pass"] = False
    keep = ["symbol", "phase2_barrier_risk_score", "phase2_pass"]
    if "is_cusum_event" in result.columns:
        result["phase2_is_cusum_event"] = result["is_cusum_event"]
        keep.append("phase2_is_cusum_event")
    return result.loc[:, keep].drop_duplicates("symbol", keep="first")


def _prepare_phase4(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns or "return_score" not in frame.columns:
        return pd.DataFrame(columns=["symbol", "phase4_return_score"])
    result = frame.copy()
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    result["phase4_return_score"] = pd.to_numeric(result["return_score"], errors="coerce")
    result = result.dropna(subset=["phase4_return_score"]).sort_values(["phase4_return_score", "symbol"], ascending=[False, True])
    result["phase4_rank"] = range(1, len(result) + 1)
    keep = ["symbol", "phase4_return_score", "phase4_rank"]
    if "name" in result.columns:
        result["phase4_name"] = result["name"]
        keep.append("phase4_name")
    return result.loc[:, keep].drop_duplicates("symbol", keep="first")


def _prepare_phase5_for_date(frame: pd.DataFrame, signal_date: date) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns or "year" not in frame.columns:
        return pd.DataFrame(columns=["symbol", "phase5_score_100"])
    result = frame.copy()
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    result["year"] = pd.to_numeric(result["year"], errors="coerce")
    result = result.dropna(subset=["year"])
    result = result[result["year"].astype(int).le(signal_date.year)]
    if result.empty:
        return pd.DataFrame(columns=["symbol", "phase5_score_100"])
    result = result.sort_values(["symbol", "year"]).drop_duplicates("symbol", keep="last")
    for column in ("NEGOUTLIER", "CRASH", "CRASH_count", "NCSKEW", "DUVOL", "RET", "SIGMA", "MINRET"):
        if column in result.columns:
            result[f"phase5_{column}"] = result[column]
    result = add_phase5_score_100(result)
    keep = ["symbol", "phase5_score_100"]
    keep.extend(
        [
            column
            for column in (
                "phase5_NEGOUTLIER",
                "phase5_CRASH",
                "phase5_CRASH_count",
                "phase5_NCSKEW",
                "phase5_DUVOL",
                "phase5_RET",
                "phase5_SIGMA",
                "phase5_MINRET",
            )
            if column in result.columns
        ]
    )
    return result.loc[:, keep]


def _select_strategy_candidates(
    *,
    strategy: str,
    signal_date: date,
    universe: pd.DataFrame,
    snapshots: _DailySnapshots,
    top_n: int,
    phase4_top_n: int,
) -> pd.DataFrame:
    base = _base_daily_frame(universe, snapshots, signal_date)
    if strategy == "full_market_random":
        selected = _sort_random(base, signal_date)
    elif strategy == "phase1_filter_only":
        selected = _sort_random(base[_mask_true(base, "phase1_pass")], signal_date)
    elif strategy == "phase2_filter_only":
        selected = _sort_random(base[_mask_true(base, "phase2_pass")], signal_date)
    elif strategy == "phase4_top":
        selected = base.sort_values(["phase4_return_score", "symbol"], ascending=[False, True], na_position="last")
    elif strategy == "phase5_group_only":
        selected = base.dropna(subset=["phase5_score_100"]).sort_values(
            ["phase5_score_100", "symbol"], ascending=[False, True], na_position="last"
        )
    elif strategy == "phase7_gate_only":
        selected = _sort_random(base, signal_date) if snapshots.phase7_permission == "allow" else base.iloc[0:0]
    elif strategy == "phase1_filter_phase4_top":
        selected = base[_mask_true(base, "phase1_pass")].sort_values(
            ["phase4_return_score", "symbol"], ascending=[False, True], na_position="last"
        )
    elif strategy == "phase2_filter_phase4_top":
        selected = base[_mask_true(base, "phase2_pass")].sort_values(
            ["phase4_return_score", "symbol"], ascending=[False, True], na_position="last"
        )
    elif strategy == "phase1_phase2_filter_phase4_top":
        selected = base[_mask_true(base, "phase1_pass") & _mask_true(base, "phase2_pass")].sort_values(
            ["phase4_return_score", "symbol"], ascending=[False, True], na_position="last"
        )
    elif strategy == "patterns_only":
        selected = _pattern_base(base).sort_values(["pattern_priority", "symbol"], ascending=[False, True])
    elif strategy == "patterns_phase1_phase2_filter":
        selected = _pattern_base(base)
        selected = selected[_mask_true(selected, "phase1_pass") & _mask_true(selected, "phase2_pass")].sort_values(
            ["pattern_priority", "symbol"], ascending=[False, True]
        )
    elif strategy == "patterns_phase4_sort":
        selected = _pattern_base(base).sort_values(["phase4_return_score", "pattern_priority", "symbol"], ascending=[False, False, True])
    elif strategy in {"current_watchlist_without_phase7", "current_watchlist_with_phase7"}:
        if strategy == "current_watchlist_with_phase7" and snapshots.phase7_permission != "allow":
            selected = base.iloc[0:0]
        else:
            selected = _current_watchlist_like(base, phase4_top_n=phase4_top_n)
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"Unsupported strategy: {strategy}")
    selected = selected.head(max(int(top_n), 0)).copy()
    if selected.empty:
        return selected
    selected["strategy"] = strategy
    selected["signal_date"] = signal_date.isoformat()
    selected["selected_rank"] = range(1, len(selected) + 1)
    selected["trade_permission"] = snapshots.phase7_permission
    return selected


def _base_daily_frame(universe: pd.DataFrame, snapshots: _DailySnapshots, signal_date: date) -> pd.DataFrame:
    base = universe.copy()
    base["symbol"] = base["symbol"].astype(str).str.zfill(6)
    base = base.merge(snapshots.phase1, on="symbol", how="left", suffixes=("", "_phase1"))
    if "name_phase1" in base.columns:
        base["name"] = base["name"].where(base["name"].astype(str).str.strip().ne(""), base["name_phase1"])
    base = base.merge(snapshots.phase2, on="symbol", how="left")
    base = base.merge(snapshots.phase4, on="symbol", how="left")
    if "phase4_name" in base.columns:
        base["name"] = base["name"].where(base["name"].astype(str).str.strip().ne(""), base["phase4_name"])
    if not snapshots.phase5.empty:
        base = base.merge(snapshots.phase5, on="symbol", how="left")
    patterns = _pattern_summary(snapshots.patterns)
    base = base.merge(patterns, on="symbol", how="left")
    base["pattern_match"] = base["pattern_ids"].notna()
    base["random_score"] = base["symbol"].map(lambda symbol: _stable_random_score(signal_date, symbol))
    return base


def _pattern_summary(patterns: pd.DataFrame) -> pd.DataFrame:
    if patterns.empty or "symbol" not in patterns.columns:
        return pd.DataFrame(columns=["symbol", "pattern_ids", "pattern_priority"])
    frame = patterns.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["pattern_id"] = frame.get("pattern_id", pd.Series("", index=frame.index)).astype(str)
    grouped = frame.groupby("symbol", sort=False)["pattern_id"].apply(lambda values: ",".join(sorted(set(values), key=str))).reset_index()
    grouped = grouped.rename(columns={"pattern_id": "pattern_ids"})
    grouped["pattern_priority"] = grouped["pattern_ids"].map(
        lambda text: max((PATTERN_PRIORITY.get(item, 0.0) for item in str(text).split(",") if item), default=0.0)
    )
    return grouped


def _pattern_base(base: pd.DataFrame) -> pd.DataFrame:
    return base[_mask_true(base, "pattern_match")].copy()


def _current_watchlist_like(base: pd.DataFrame, *, phase4_top_n: int) -> pd.DataFrame:
    passed = base[_mask_true(base, "phase1_pass") & _mask_true(base, "phase2_pass")].copy()
    pattern_part = passed[_mask_true(passed, "pattern_match")].copy()
    pattern_part["_source_order"] = 0
    phase4_part = passed.sort_values(["phase4_return_score", "symbol"], ascending=[False, True], na_position="last").head(
        max(int(phase4_top_n), 0)
    )
    phase4_part = phase4_part[~phase4_part["symbol"].isin(pattern_part["symbol"])].copy()
    phase4_part["_source_order"] = 1
    combined = pd.concat([pattern_part, phase4_part], ignore_index=True) if not phase4_part.empty else pattern_part
    if combined.empty:
        return combined
    return combined.sort_values(["_source_order", "phase4_return_score", "symbol"], ascending=[True, False, True], na_position="last")


def _sort_random(frame: pd.DataFrame, signal_date: date) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    if "random_score" not in result.columns:
        result["random_score"] = result["symbol"].map(lambda symbol: _stable_random_score(signal_date, symbol))
    return result.sort_values(["random_score", "symbol"], ascending=[False, True])


def _mask_true(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].eq(True)


def _stable_random_score(signal_date: date, symbol: str) -> float:
    digest = hashlib.blake2b(f"{signal_date.isoformat()}:{str(symbol).zfill(6)}".encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def _simulate_candidate_trades(
    *,
    storage: Storage,
    candidates: pd.DataFrame,
    signal_date: date,
    horizons: tuple[int, ...],
    stop_loss_pct: float,
    take_profit_pct: float,
    price_cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in candidates.to_dict("records"):
        symbol = str(candidate["symbol"]).zfill(6)
        bars = _load_price_bars(storage, symbol, price_cache)
        if bars.empty:
            continue
        for horizon in horizons:
            outcome = simulate_forward_trade(
                bars,
                signal_date=signal_date,
                horizon_days=int(horizon),
                stop_loss_pct=float(stop_loss_pct),
                take_profit_pct=float(take_profit_pct),
            )
            if outcome is None:
                continue
            row = {
                "strategy": candidate.get("strategy"),
                "signal_date": signal_date.isoformat(),
                "symbol": symbol,
                "name": candidate.get("name", ""),
                "selected_rank": candidate.get("selected_rank"),
                "source": candidate.get("_source_order", ""),
                "pattern_match": bool(candidate.get("pattern_match", False)),
                "pattern_ids": candidate.get("pattern_ids", ""),
                "trade_permission": candidate.get("trade_permission", ""),
                "phase1_risk_score": candidate.get("phase1_risk_score"),
                "phase1_pass": candidate.get("phase1_pass"),
                "phase2_barrier_risk_score": candidate.get("phase2_barrier_risk_score"),
                "phase2_pass": candidate.get("phase2_pass"),
                "phase2_is_cusum_event": candidate.get("phase2_is_cusum_event"),
                "phase4_return_score": candidate.get("phase4_return_score"),
                "phase4_rank": candidate.get("phase4_rank"),
                "phase5_score_100": candidate.get("phase5_score_100"),
            }
            row.update(outcome)
            rows.append(row)
    return pd.DataFrame(rows)


def _load_price_bars(storage: Storage, symbol: str, price_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if symbol in price_cache:
        return price_cache[symbol]
    try:
        bars = storage.load_daily_bars(symbol)
    except (FileNotFoundError, DailyBarsReadError):
        bars = pd.DataFrame()
    if not bars.empty:
        bars = bars.copy()
        bars["trade_date"] = pd.to_datetime(bars["trade_date"], errors="coerce")
        bars = bars.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    price_cache[symbol] = bars
    return bars


def simulate_forward_trade(
    bars: pd.DataFrame,
    *,
    signal_date: date,
    horizon_days: int,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> dict[str, object] | None:
    if bars.empty or horizon_days <= 0:
        return None
    frame = bars.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    future = frame[frame["trade_date"].dt.date.gt(signal_date)].head(horizon_days).copy()
    if len(future) < horizon_days:
        return None
    required = {"open", "high", "low", "close"}
    if not required.issubset(future.columns):
        return None
    entry = float(future.iloc[0]["open"])
    if not math.isfinite(entry) or entry <= 0:
        return None
    stop_price = entry * (1.0 - float(stop_loss_pct))
    take_price = entry * (1.0 + float(take_profit_pct))
    exit_reason = "timeout"
    exit_date = pd.Timestamp(future.iloc[-1]["trade_date"]).date()
    exit_price = float(future.iloc[-1]["close"])
    holding_days = int(horizon_days)
    for offset, row in enumerate(future.itertuples(index=False), start=1):
        low = float(getattr(row, "low"))
        high = float(getattr(row, "high"))
        row_date = pd.Timestamp(getattr(row, "trade_date")).date()
        if low <= stop_price and high >= take_price:
            exit_reason = "stop_loss_first"
            exit_date = row_date
            exit_price = stop_price
            holding_days = offset
            break
        if low <= stop_price:
            exit_reason = "stop_loss"
            exit_date = row_date
            exit_price = stop_price
            holding_days = offset
            break
        if high >= take_price:
            exit_reason = "take_profit"
            exit_date = row_date
            exit_price = take_price
            holding_days = offset
            break
    raw_exit_price = float(future.iloc[-1]["close"])
    lows = pd.to_numeric(future["low"], errors="coerce")
    highs = pd.to_numeric(future["high"], errors="coerce")
    return {
        "horizon": int(horizon_days),
        "entry_date": pd.Timestamp(future.iloc[0]["trade_date"]).date().isoformat(),
        "entry_price": entry,
        "exit_date": exit_date.isoformat(),
        "exit_reason": exit_reason,
        "exit_price": exit_price,
        "holding_days": holding_days,
        "raw_return": raw_exit_price / entry - 1.0,
        "barrier_return": exit_price / entry - 1.0,
        "max_drawdown": lows.min() / entry - 1.0,
        "max_profit": highs.max() / entry - 1.0,
        "stop_loss_hit": exit_reason in {"stop_loss", "stop_loss_first"},
        "take_profit_hit": exit_reason == "take_profit",
    }


def _build_daily_portfolio(trades: pd.DataFrame, candidate_counts: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["strategy", "signal_date", "horizon", "trade_count", "raw_return", "barrier_return", "max_drawdown"])
    grouped = trades.groupby(["strategy", "signal_date", "horizon"], dropna=False)
    daily = grouped.agg(
        trade_count=("symbol", "count"),
        raw_return=("raw_return", "mean"),
        barrier_return=("barrier_return", "mean"),
        max_drawdown=("max_drawdown", "mean"),
        stop_loss_rate=("stop_loss_hit", "mean"),
        take_profit_rate=("take_profit_hit", "mean"),
    ).reset_index()
    if not candidate_counts.empty:
        daily = daily.merge(candidate_counts, on=["strategy", "signal_date"], how="left")
    return daily


def _build_backtest_summary(
    *,
    trades: pd.DataFrame,
    daily_portfolio: pd.DataFrame,
    candidate_counts: pd.DataFrame,
    strategies: tuple[str, ...],
    horizons: tuple[int, ...],
    signal_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy in strategies:
        strategy_counts = candidate_counts[candidate_counts["strategy"].eq(strategy)] if not candidate_counts.empty else pd.DataFrame()
        avg_candidates = float(strategy_counts["candidate_count"].mean()) if not strategy_counts.empty else 0.0
        no_candidate_rate = float(strategy_counts["candidate_count"].eq(0).mean()) if not strategy_counts.empty else 1.0
        for horizon in horizons:
            subset = trades[trades["strategy"].eq(strategy) & trades["horizon"].eq(horizon)] if not trades.empty else pd.DataFrame()
            daily = (
                daily_portfolio[daily_portfolio["strategy"].eq(strategy) & daily_portfolio["horizon"].eq(horizon)]
                if not daily_portfolio.empty
                else pd.DataFrame()
            )
            rows.append(
                {
                    "strategy": strategy,
                    "horizon": int(horizon),
                    "signal_days": int(signal_days),
                    "days_with_trades": int(daily["signal_date"].nunique()) if not daily.empty else 0,
                    "avg_candidates_per_day": avg_candidates,
                    "no_candidate_day_rate": no_candidate_rate,
                    "trade_count": int(len(subset)),
                    "avg_raw_return": _mean_or_nan(subset, "raw_return"),
                    "median_raw_return": _median_or_nan(subset, "raw_return"),
                    "raw_win_rate": _rate_gt_zero(subset, "raw_return"),
                    "avg_barrier_return": _mean_or_nan(subset, "barrier_return"),
                    "median_barrier_return": _median_or_nan(subset, "barrier_return"),
                    "barrier_win_rate": _rate_gt_zero(subset, "barrier_return"),
                    "avg_max_drawdown": _mean_or_nan(subset, "max_drawdown"),
                    "avg_max_profit": _mean_or_nan(subset, "max_profit"),
                    "return_5pct_quantile": _quantile_or_nan(subset, "raw_return", 0.05),
                    "loss_worse_5pct_rate": _rate_le(subset, "raw_return", -0.05),
                    "loss_worse_8pct_rate": _rate_le(subset, "raw_return", -0.08),
                    "stop_loss_rate": _mean_or_nan(subset, "stop_loss_hit"),
                    "take_profit_rate": _mean_or_nan(subset, "take_profit_hit"),
                    "avg_holding_days": _mean_or_nan(subset, "holding_days"),
                    "portfolio_compound_barrier_return": _compound_return(daily, "barrier_return"),
                    "portfolio_max_drawdown": _portfolio_max_drawdown(daily, "barrier_return"),
                    "portfolio_sharpe_like": _sharpe_like(daily, "barrier_return"),
                }
            )
    return pd.DataFrame(rows)


def _build_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    columns = [
        "strategy",
        "horizon",
        "trade_count",
        "avg_barrier_return",
        "barrier_win_rate",
        "avg_raw_return",
        "raw_win_rate",
        "avg_max_drawdown",
        "stop_loss_rate",
        "take_profit_rate",
        "portfolio_compound_barrier_return",
    ]
    available = [column for column in columns if column in summary.columns]
    return summary.loc[:, available].sort_values(["horizon", "avg_barrier_return"], ascending=[True, False])


def _mean_or_nan(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _median_or_nan(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    return float(pd.to_numeric(frame[column], errors="coerce").median())


def _quantile_or_nan(frame: pd.DataFrame, column: str, q: float) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    return float(pd.to_numeric(frame[column], errors="coerce").quantile(q))


def _rate_gt_zero(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.gt(0).mean()) if not values.empty else math.nan


def _rate_le(frame: pd.DataFrame, column: str, threshold: float) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.le(threshold).mean()) if not values.empty else math.nan


def _compound_return(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return math.nan
    return float((1.0 + values).prod() - 1.0)


def _portfolio_max_drawdown(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    values = pd.to_numeric(frame.sort_values("signal_date")[column], errors="coerce").dropna()
    if values.empty:
        return math.nan
    equity = (1.0 + values).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _sharpe_like(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if len(values) < 2:
        return math.nan
    std = float(values.std(ddof=1))
    if std == 0 or not math.isfinite(std):
        return math.nan
    return float(values.mean() / std * math.sqrt(252))


def format_backtest_summary(summary: pd.DataFrame, *, top_n: int = 80) -> str:
    if summary.empty:
        return "No backtest summary rows."
    columns = [
        "strategy",
        "horizon",
        "trade_count",
        "avg_barrier_return",
        "barrier_win_rate",
        "avg_raw_return",
        "raw_win_rate",
        "avg_max_drawdown",
        "stop_loss_rate",
        "take_profit_rate",
        "no_candidate_day_rate",
    ]
    frame = summary.loc[:, [column for column in columns if column in summary.columns]].copy()
    for column in frame.columns:
        if column not in {"strategy", "horizon", "trade_count"}:
            frame[column] = frame[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return frame.head(max(int(top_n), 0)).to_string(index=False)
