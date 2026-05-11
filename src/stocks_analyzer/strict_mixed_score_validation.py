from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import logging
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .full_market_alpha158 import alpha158_feature_columns, build_alpha158_return_panel
from .full_market_labels import (
    BARRIER_RISK_FEATURE_COLUMNS,
    TAIL_RISK_FEATURE_COLUMNS,
    build_barrier_risk_panel,
    build_tail_risk_panel,
)
from .full_market_return import LGBMRegressor, QLIB_ALPHA158_LGBM_PARAMS
from .full_market_risk import LGBMClassifier, TAIL_RISK_LGBM_CLASSIFIER_PARAMS, _predict_risk_proba, _tail_risk_models
from .full_market_trade_day import (
    _available_feature_columns,
    _fit_trade_day_model,
    _score_trade_day_model,
    add_trade_day_labels,
    build_trade_day_feature_frame,
)
from .indicators import add_indicators
from .lgbm_utils import fit_lgbm_with_device, normalized_lgbm_device
from .phase_display import add_phase5_score_100, score_series_100
from .storage import DailyBarsReadError, Storage
from .synthetic_market import build_synthetic_market_index


DEFAULT_STRICT_MIXED_STRATEGIES = (
    "random_top20",
    "phase4_top20",
    "phase4_top20_p12_ge30",
    "phase4_top20_p12_ge40",
    "mixed_010_top20",
    "mixed_top20",
    "centered_risk_top20",
    "all90",
    "mixed_010_top20_phase7_allow",
    "mixed_top20_phase7_allow",
    "all90_phase7_allow",
    "mixed_010_top20_phase5_safe",
    "mixed_top20_phase5_safe",
    "all90_phase5_safe",
)


@dataclass(slots=True)
class StrictMixedScoreValidationResult:
    windows: pd.DataFrame
    oos_panel: pd.DataFrame
    strategy_trades: pd.DataFrame
    strategy_summary: pd.DataFrame
    score_deciles: pd.DataFrame
    benchmark: pd.DataFrame
    benchmark_comparison: pd.DataFrame
    selected_forward_paths: pd.DataFrame
    output_dir: Path
    windows_path: Path
    oos_panel_path: Path
    strategy_trades_path: Path
    strategy_summary_path: Path
    score_deciles_path: Path
    benchmark_path: Path
    benchmark_comparison_path: Path
    selected_forward_paths_path: Path | None


def validate_strict_mixed_score(
    *,
    storage: Storage,
    project_root: Path,
    start_date: date,
    test_start_date: date,
    end_date: date,
    horizons: tuple[int, ...] = (5, 10, 20, 60),
    test_window_days: int = 60,
    step_days: int = 60,
    embargo_days: int = 60,
    min_train_days: int = 900,
    limit: int | None = None,
    max_windows: int | None = None,
    output_dir: Path | None = None,
    strategies: tuple[str, ...] = DEFAULT_STRICT_MIXED_STRATEGIES,
    top_n: int = 20,
    phase1_min_score: float = 40.0,
    phase2_min_score: float = 40.0,
    all90_min_score: float = 90.0,
    phase5_safe_min_score: float = 40.0,
    stop_loss_pct: float = 0.08,
    take_profit_pct: float = 0.15,
    phase1_model_name: str = "logistic_regression",
    phase2_model_name: str = "lightgbm_classifier",
    phase7_model_name: str = "naive_bayes",
    min_training_rows: int = 200,
    min_stock_count: int = 500,
    include_phase5: bool = True,
    include_phase7: bool = True,
    include_atr: bool = True,
    use_cache: bool = True,
    progress: bool = False,
    save_selected_forward_paths: bool = False,
    path_max_horizon: int | None = None,
    path_output_format: str = "parquet",
    path_strategies: tuple[str, ...] | None = None,
    lgbm_device: str = "cpu",
    lgbm_n_jobs: int | None = 1,
    lgbm_gpu_platform_id: int | None = None,
    lgbm_gpu_device_id: int | None = None,
) -> StrictMixedScoreValidationResult:
    if start_date > test_start_date:
        raise ValueError("start_date must be <= test_start_date")
    if test_start_date > end_date:
        raise ValueError("test_start_date must be <= end_date")
    if any(int(horizon) <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive")
    unknown = sorted(set(strategies) - set(DEFAULT_STRICT_MIXED_STRATEGIES))
    if unknown:
        raise ValueError(f"Unsupported strict validation strategies: {unknown}")
    normalized_path_format = str(path_output_format).strip().lower()
    if normalized_path_format not in {"parquet", "csv"}:
        raise ValueError("path_output_format must be either 'parquet' or 'csv'")
    active_lgbm_device = normalized_lgbm_device(lgbm_device)

    output_root = output_dir or project_root / "reports" / "strict_mixed_score_validation"
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = output_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    effective_min_stock_count = _effective_min_stock_count(limit=limit, min_stock_count=min_stock_count)
    started_at = perf_counter()

    logging.info("Strict OOS panel build started")
    tail_panel = _load_or_build_frame_cache(
        cache_dir / "tail_risk_panel.parquet",
        use_cache=use_cache,
        build=lambda: build_tail_risk_panel(storage=storage, start_date=start_date, end_date=end_date, limit=limit)[0],
    )
    _print_strict_progress(progress, started_at, f"Tail-risk panel ready: rows={len(tail_panel)}")
    barrier_panel = _load_or_build_frame_cache(
        cache_dir / "barrier_risk_panel.parquet",
        use_cache=use_cache,
        build=lambda: build_barrier_risk_panel(storage=storage, start_date=start_date, end_date=end_date, limit=limit)[0],
    )
    _print_strict_progress(progress, started_at, f"Barrier-risk panel ready: rows={len(barrier_panel)}")
    alpha_panel = _load_or_build_frame_cache(
        cache_dir / "alpha158_return_panel.parquet",
        use_cache=use_cache,
        build=lambda: build_alpha158_return_panel(storage=storage, start_date=start_date, end_date=end_date, limit=limit).dataset,
    )
    _print_strict_progress(progress, started_at, f"Alpha158 return panel ready: rows={len(alpha_panel)}")
    phase7_dataset = pd.DataFrame()
    phase7_feature_columns: tuple[str, ...] = tuple()
    if include_phase7:
        phase7_features = _load_or_build_frame_cache(
            cache_dir / "trade_day_features.parquet",
            use_cache=use_cache,
            build=lambda: build_trade_day_feature_frame(
                storage=storage,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                min_stock_count=effective_min_stock_count,
            ),
        )
        _print_strict_progress(progress, started_at, f"Phase7 feature panel ready: rows={len(phase7_features)}")
        phase7_feature_columns = _available_feature_columns(phase7_features)
        phase7_dataset = add_trade_day_labels(
            phase7_features,
            horizon_days=10,
            drawdown_threshold=-0.02,
            return_threshold=-0.01,
            market_source_column="synthetic_equal_weight_index",
        ).dropna(subset=["bad_buy_day", *phase7_feature_columns])

    for frame in (tail_panel, barrier_panel, alpha_panel, phase7_dataset):
        if not frame.empty and "trade_date" in frame.columns:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")

    trade_dates = _validation_trade_dates(
        alpha_panel,
        start_date=start_date,
        test_start_date=test_start_date,
        end_date=end_date,
    )
    windows = build_anchored_oos_windows(
        trade_dates,
        train_start=start_date,
        test_start=test_start_date,
        test_end=end_date,
        min_train_days=min_train_days,
        test_window_days=test_window_days,
        step_days=step_days,
        embargo_days=embargo_days,
        max_windows=max_windows,
    )
    if windows.empty:
        raise RuntimeError("No strict OOS windows could be built with the requested split.")
    _print_strict_progress(progress, started_at, f"Strict OOS windows ready: windows={len(windows)}")

    scored_parts: list[pd.DataFrame] = []
    window_rows: list[dict[str, Any]] = []
    alpha_features = alpha158_feature_columns(alpha_panel)
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is required for strict Phase4 OOS validation.")

    window_loop_started_at = perf_counter()
    for window_index, window in enumerate(windows.to_dict("records"), start=1):
        window_cache_path = _oos_window_cache_path(cache_dir, int(window["window_id"]))
        if progress:
            print(
                "Strict OOS window "
                f"{window_index}/{len(windows)}: train {window['train_start']}->{window['train_end']}, "
                f"test {window['test_start']}->{window['test_end']} "
                f"(elapsed={_format_duration(perf_counter() - started_at)})"
            )
        train_start = date.fromisoformat(str(window["train_start"]))
        train_end = date.fromisoformat(str(window["train_end"]))
        test_start = date.fromisoformat(str(window["test_start"]))
        test_end = date.fromisoformat(str(window["test_end"]))
        if use_cache and window_cache_path.exists():
            scored = pd.read_parquet(window_cache_path)
            if progress:
                print(f"  reused cached window: {window_cache_path.name} rows={len(scored)}")
        else:
            scored, used_lgbm_device = _score_oos_window(
                window_id=int(window["window_id"]),
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                tail_panel=tail_panel,
                barrier_panel=barrier_panel,
                alpha_panel=alpha_panel,
                alpha_features=alpha_features,
                phase7_dataset=phase7_dataset,
                phase7_feature_columns=phase7_feature_columns,
                phase1_model_name=phase1_model_name,
                phase2_model_name=phase2_model_name,
                phase7_model_name=phase7_model_name,
                min_training_rows=min_training_rows,
                include_phase7=include_phase7,
                lgbm_device=active_lgbm_device,
                lgbm_n_jobs=lgbm_n_jobs,
                lgbm_gpu_platform_id=lgbm_gpu_platform_id,
                lgbm_gpu_device_id=lgbm_gpu_device_id,
            )
            if used_lgbm_device != active_lgbm_device:
                _print_strict_progress(
                    progress,
                    started_at,
                    f"LightGBM device fallback: requested={active_lgbm_device} used={used_lgbm_device}",
                )
                active_lgbm_device = used_lgbm_device
            scored.to_parquet(window_cache_path, index=False)
            if progress:
                print(f"  saved window cache: {window_cache_path.name} rows={len(scored)} lgbm_device={used_lgbm_device}")
        if not scored.empty:
            scored_parts.append(scored)
        window_rows.append({**window, "oos_rows": int(len(scored))})
        if progress:
            avg_window_seconds = (perf_counter() - window_loop_started_at) / max(window_index, 1)
            remaining_seconds = avg_window_seconds * max(len(windows) - window_index, 0)
            print(
                "  progress "
                f"{window_index}/{len(windows)} elapsed={_format_duration(perf_counter() - started_at)} "
                f"avg/window={_format_duration(avg_window_seconds)} eta={_format_duration(remaining_seconds)}"
            )

    oos_panel = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    if oos_panel.empty:
        raise RuntimeError("Strict OOS scoring produced no rows.")
    oos_panel = _add_oos_scores(oos_panel)
    _print_strict_progress(progress, started_at, f"OOS panel scored: rows={len(oos_panel)}")
    signal_dates = sorted(pd.to_datetime(oos_panel["signal_date"], errors="coerce").dropna().dt.date.unique())
    phase5 = _load_phase5_measures(project_root) if include_phase5 else pd.DataFrame()
    if not phase5.empty:
        oos_panel = _attach_phase5(oos_panel, phase5)
        _print_strict_progress(progress, started_at, "Phase5 attached")
    else:
        oos_panel["phase5_score_100"] = pd.NA
    if include_atr:
        atr_panel = _build_atr_panel(storage=storage, symbols=tuple(oos_panel["symbol"].dropna().astype(str).unique()), signal_dates=signal_dates)
        oos_panel = oos_panel.merge(atr_panel, on=["signal_date", "symbol"], how="left")
        _print_strict_progress(progress, started_at, f"ATR attached: rows={len(atr_panel)}")
    oos_panel = _attach_forward_outcomes(
        storage=storage,
        panel=oos_panel,
        horizons=horizons,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )
    _print_strict_progress(progress, started_at, "Forward outcomes attached")

    strategy_trades, candidate_counts = build_strict_strategy_trades(
        oos_panel,
        strategies=strategies,
        horizons=horizons,
        top_n=top_n,
        phase1_min_score=phase1_min_score,
        phase2_min_score=phase2_min_score,
        all90_min_score=all90_min_score,
        phase5_safe_min_score=phase5_safe_min_score,
    )
    _print_strict_progress(progress, started_at, f"Strategy trades built: rows={len(strategy_trades)}")
    selected_forward_paths = pd.DataFrame()
    selected_forward_paths_path: Path | None = None
    if save_selected_forward_paths:
        selected_forward_paths = build_selected_forward_paths(
            storage=storage,
            strategy_trades=strategy_trades,
            max_horizon=int(path_max_horizon or max(int(item) for item in horizons)),
            strategies=path_strategies,
            progress=progress,
            started_at=started_at,
        )
        suffix = "parquet" if normalized_path_format == "parquet" else "csv"
        selected_forward_paths_path = output_root / f"selected_forward_paths.{suffix}"
        if normalized_path_format == "parquet":
            selected_forward_paths.to_parquet(selected_forward_paths_path, index=False)
        else:
            selected_forward_paths.to_csv(selected_forward_paths_path, index=False, encoding="utf-8-sig")
        _print_strict_progress(
            progress,
            started_at,
            f"Selected forward paths saved: rows={len(selected_forward_paths)} path={selected_forward_paths_path}",
        )
    strategy_summary = summarize_strict_strategy_trades(
        strategy_trades,
        candidate_counts=candidate_counts,
        signal_days=len(signal_dates),
    )
    score_deciles = build_score_decile_report(oos_panel, horizons=horizons)
    benchmark = _build_synthetic_benchmark(
        storage=storage,
        project_root=project_root,
        output_dir=output_root,
        signal_dates=signal_dates,
        horizons=horizons,
        limit=limit,
        min_stock_count=effective_min_stock_count,
    )
    _print_strict_progress(progress, started_at, f"Benchmark built: rows={len(benchmark)}")
    benchmark_comparison = build_strict_benchmark_comparison(strategy_summary, benchmark)

    windows = pd.DataFrame(window_rows)
    windows_path = output_root / "windows.csv"
    oos_panel_path = output_root / "oos_panel.csv"
    trades_path = output_root / "strategy_trades.csv"
    summary_path = output_root / "strategy_summary.csv"
    deciles_path = output_root / "score_decile_report.csv"
    benchmark_path = output_root / "benchmark.csv"
    benchmark_comparison_path = output_root / "benchmark_comparison.csv"
    windows.to_csv(windows_path, index=False, encoding="utf-8-sig")
    oos_panel.to_csv(oos_panel_path, index=False, encoding="utf-8-sig")
    strategy_trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    strategy_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    score_deciles.to_csv(deciles_path, index=False, encoding="utf-8-sig")
    benchmark.to_csv(benchmark_path, index=False, encoding="utf-8-sig")
    benchmark_comparison.to_csv(benchmark_comparison_path, index=False, encoding="utf-8-sig")
    return StrictMixedScoreValidationResult(
        windows=windows,
        oos_panel=oos_panel,
        strategy_trades=strategy_trades,
        strategy_summary=strategy_summary,
        score_deciles=score_deciles,
        benchmark=benchmark,
        benchmark_comparison=benchmark_comparison,
        selected_forward_paths=selected_forward_paths,
        output_dir=output_root,
        windows_path=windows_path,
        oos_panel_path=oos_panel_path,
        strategy_trades_path=trades_path,
        strategy_summary_path=summary_path,
        score_deciles_path=deciles_path,
        benchmark_path=benchmark_path,
        benchmark_comparison_path=benchmark_comparison_path,
        selected_forward_paths_path=selected_forward_paths_path,
    )


def build_anchored_oos_windows(
    trade_dates: pd.Series | list[date],
    *,
    train_start: date,
    test_start: date,
    test_end: date,
    min_train_days: int,
    test_window_days: int,
    step_days: int,
    embargo_days: int,
    max_windows: int | None = None,
) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(pd.Series(trade_dates), errors="coerce").dropna().dt.date.unique()).sort_values().reset_index(drop=True)
    dates = dates[(dates >= train_start) & (dates <= test_end)].reset_index(drop=True)
    if dates.empty:
        return pd.DataFrame()
    test_indices = dates[dates >= test_start].index.tolist()
    rows: list[dict[str, Any]] = []
    cursor = test_indices[0] if test_indices else len(dates)
    window_id = 1
    while cursor < len(dates):
        train_end_index = cursor - int(embargo_days) - 1
        if train_end_index < 0:
            cursor += int(step_days)
            continue
        train_dates = dates[(dates >= train_start) & (dates <= dates.iloc[train_end_index])]
        if len(train_dates) < int(min_train_days):
            cursor += int(step_days)
            continue
        test_end_index = min(cursor + int(test_window_days) - 1, len(dates) - 1)
        rows.append(
            {
                "window_id": window_id,
                "train_start": train_start.isoformat(),
                "train_end": dates.iloc[train_end_index].isoformat(),
                "embargo_days": int(embargo_days),
                "test_start": dates.iloc[cursor].isoformat(),
                "test_end": dates.iloc[test_end_index].isoformat(),
                "train_days": int(len(train_dates)),
                "test_days": int(test_end_index - cursor + 1),
            }
        )
        window_id += 1
        if max_windows is not None and len(rows) >= int(max_windows):
            break
        cursor += int(step_days)
    return pd.DataFrame(rows)


def build_strict_strategy_trades(
    panel: pd.DataFrame,
    *,
    strategies: tuple[str, ...],
    horizons: tuple[int, ...],
    top_n: int,
    phase1_min_score: float,
    phase2_min_score: float,
    all90_min_score: float,
    phase5_safe_min_score: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    for signal_date, day in panel.groupby("signal_date", sort=True):
        for strategy in strategies:
            selected = _select_strict_strategy(
                day,
                strategy=strategy,
                signal_date=date.fromisoformat(str(signal_date)),
                top_n=top_n,
                phase1_min_score=phase1_min_score,
                phase2_min_score=phase2_min_score,
                all90_min_score=all90_min_score,
                phase5_safe_min_score=phase5_safe_min_score,
            )
            count_rows.append({"strategy": strategy, "signal_date": signal_date, "candidate_count": int(len(selected))})
            for selected_rank, candidate in enumerate(selected.to_dict("records"), start=1):
                for horizon in horizons:
                    raw_return = candidate.get(f"return_{int(horizon)}d")
                    if pd.isna(raw_return):
                        continue
                    rows.append(
                        {
                            "strategy": strategy,
                            "signal_date": signal_date,
                            "symbol": candidate.get("symbol"),
                            "name": candidate.get("name", ""),
                            "selected_rank": selected_rank,
                            "horizon": int(horizon),
                            "phase1_score_100": candidate.get("phase1_score_100"),
                            "phase2_score_100": candidate.get("phase2_score_100"),
                            "phase4_score_100": candidate.get("phase4_score_100"),
                            "mixed_010_score": candidate.get("mixed_010_score"),
                            "mixed_score": candidate.get("mixed_score"),
                            "phase1_center_score": candidate.get("phase1_center_score"),
                            "phase2_center_score": candidate.get("phase2_center_score"),
                            "centered_risk_score": candidate.get("centered_risk_score"),
                            "all90_flag": candidate.get("all90_flag"),
                            "phase5_score_100": candidate.get("phase5_score_100"),
                            "phase7_trade_permission": candidate.get("phase7_trade_permission", ""),
                            "entry_date": candidate.get("entry_date"),
                            "entry_open": candidate.get("entry_open"),
                            "raw_return": raw_return,
                            "barrier_return": candidate.get(f"barrier_return_{int(horizon)}d"),
                            "max_profit": candidate.get(f"max_profit_{int(horizon)}d"),
                            "max_drawdown": candidate.get(f"max_drawdown_{int(horizon)}d"),
                            "raw_R": candidate.get(f"return_R_{int(horizon)}d"),
                            "barrier_R": candidate.get(f"barrier_R_{int(horizon)}d"),
                            "max_profit_R": candidate.get(f"max_profit_R_{int(horizon)}d"),
                            "max_drawdown_R": candidate.get(f"max_drawdown_R_{int(horizon)}d"),
                            "exit_reason": candidate.get(f"exit_reason_{int(horizon)}d"),
                            "stop_loss_hit": candidate.get(f"stop_loss_hit_{int(horizon)}d"),
                            "take_profit_hit": candidate.get(f"take_profit_hit_{int(horizon)}d"),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(count_rows)


def build_selected_forward_paths(
    *,
    storage: Storage,
    strategy_trades: pd.DataFrame,
    max_horizon: int,
    strategies: tuple[str, ...] | None = None,
    progress: bool = False,
    started_at: float | None = None,
) -> pd.DataFrame:
    if strategy_trades.empty or max_horizon <= 0:
        return pd.DataFrame()
    trades = strategy_trades.copy()
    if strategies:
        wanted = {str(item) for item in strategies}
        trades = trades[trades["strategy"].astype(str).isin(wanted)].copy()
    if trades.empty:
        return pd.DataFrame()
    key_columns = ["strategy", "signal_date", "symbol", "selected_rank"]
    metadata_columns = [
        "name",
        "phase1_score_100",
        "phase2_score_100",
        "phase4_score_100",
        "mixed_010_score",
        "mixed_score",
        "phase1_center_score",
        "phase2_center_score",
        "centered_risk_score",
        "all90_flag",
        "phase5_score_100",
        "phase7_trade_permission",
        "entry_date",
        "entry_open",
    ]
    available_metadata = [column for column in metadata_columns if column in trades.columns]
    selected = trades.drop_duplicates(key_columns, keep="first").loc[:, key_columns + available_metadata].copy()
    selected["symbol"] = selected["symbol"].astype(str).str.zfill(6)
    selected["signal_date"] = pd.to_datetime(selected["signal_date"], errors="coerce").dt.date.astype(str)
    rows: list[pd.DataFrame] = []
    total = len(selected)
    for index, trade in enumerate(selected.to_dict("records"), start=1):
        if progress and (index == 1 or index == total or index % 5000 == 0):
            elapsed = _format_duration(perf_counter() - started_at) if started_at is not None else "0s"
            print(f"Selected forward paths progress: {index}/{total} elapsed={elapsed}")
        symbol = str(trade["symbol"]).zfill(6)
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError):
            continue
        path = _selected_forward_path_frame(
            bars,
            trade=trade,
            max_horizon=int(max_horizon),
        )
        if not path.empty:
            rows.append(path)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _selected_forward_path_frame(bars: pd.DataFrame, *, trade: dict[str, Any], max_horizon: int) -> pd.DataFrame:
    signal_date = date.fromisoformat(str(trade["signal_date"]))
    frame = bars.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    future = frame[frame["trade_date"].dt.date.gt(signal_date)].head(max_horizon).copy()
    if future.empty:
        return pd.DataFrame()
    for column in ("open", "high", "low", "close"):
        future[column] = pd.to_numeric(future[column], errors="coerce")
    future = future.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if future.empty:
        return pd.DataFrame()
    entry_open = _safe_float(trade.get("entry_open"))
    if entry_open is None:
        entry_open = _safe_float(future.iloc[0]["open"])
    if entry_open is None or entry_open <= 0:
        return pd.DataFrame()
    result = future.loc[:, ["trade_date", "open", "high", "low", "close"]].copy()
    result["trade_date"] = result["trade_date"].dt.date.astype(str)
    result.insert(0, "day_offset", range(1, len(result) + 1))
    for column in (
        "strategy",
        "signal_date",
        "symbol",
        "name",
        "selected_rank",
        "phase1_score_100",
        "phase2_score_100",
        "phase4_score_100",
        "mixed_010_score",
        "mixed_score",
        "phase1_center_score",
        "phase2_center_score",
        "centered_risk_score",
        "all90_flag",
        "phase5_score_100",
        "phase7_trade_permission",
        "entry_date",
    ):
        if column in trade:
            result[column] = trade.get(column)
    result["entry_open"] = entry_open
    result["max_saved_horizon"] = int(max_horizon)
    preferred = [
        "strategy",
        "signal_date",
        "symbol",
        "name",
        "selected_rank",
        "day_offset",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "entry_date",
        "entry_open",
        "max_saved_horizon",
        "phase1_score_100",
        "phase2_score_100",
        "phase4_score_100",
        "mixed_010_score",
        "mixed_score",
        "phase1_center_score",
        "phase2_center_score",
        "centered_risk_score",
        "all90_flag",
        "phase5_score_100",
        "phase7_trade_permission",
    ]
    return result.loc[:, [column for column in preferred if column in result.columns]]


def summarize_strict_strategy_trades(
    trades: pd.DataFrame,
    *,
    candidate_counts: pd.DataFrame,
    signal_days: int,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    grouped = trades.groupby(["strategy", "horizon"], dropna=False)
    summary = grouped.agg(
        trade_count=("symbol", "count"),
        signal_days_with_trades=("signal_date", "nunique"),
        avg_raw_return=("raw_return", "mean"),
        raw_win_rate=("raw_return", lambda values: float(pd.to_numeric(values, errors="coerce").gt(0).mean())),
        avg_barrier_return=("barrier_return", "mean"),
        barrier_win_rate=("barrier_return", lambda values: float(pd.to_numeric(values, errors="coerce").gt(0).mean())),
        avg_max_profit=("max_profit", "mean"),
        avg_max_drawdown=("max_drawdown", "mean"),
        avg_raw_R=("raw_R", "mean"),
        avg_barrier_R=("barrier_R", "mean"),
        avg_max_profit_R=("max_profit_R", "mean"),
        avg_max_drawdown_R=("max_drawdown_R", "mean"),
        stop_loss_rate=("stop_loss_hit", "mean"),
        take_profit_rate=("take_profit_hit", "mean"),
    ).reset_index()
    gains = trades[pd.to_numeric(trades["raw_return"], errors="coerce").gt(0)].groupby(["strategy", "horizon"])["raw_return"].mean().rename("avg_gain")
    losses = trades[pd.to_numeric(trades["raw_return"], errors="coerce").lt(0)].groupby(["strategy", "horizon"])["raw_return"].mean().rename("avg_loss")
    summary = summary.merge(gains, on=["strategy", "horizon"], how="left").merge(losses, on=["strategy", "horizon"], how="left")
    summary["payoff_ratio"] = summary["avg_gain"] / summary["avg_loss"].abs()
    summary["expectancy"] = summary["raw_win_rate"] * summary["avg_gain"].fillna(0.0) + (1.0 - summary["raw_win_rate"]) * summary["avg_loss"].fillna(0.0)
    if not candidate_counts.empty:
        no_candidate = candidate_counts.groupby("strategy")["candidate_count"].apply(lambda values: float(pd.to_numeric(values, errors="coerce").eq(0).mean()))
        summary = summary.merge(no_candidate.rename("no_candidate_day_rate"), on="strategy", how="left")
    else:
        summary["no_candidate_day_rate"] = math.nan
    summary["signal_days"] = int(signal_days)
    return summary.sort_values(["horizon", "avg_raw_return"], ascending=[True, False]).reset_index(drop=True)


def build_score_decile_report(panel: pd.DataFrame, *, horizons: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    score_columns = [
        "phase1_score_100",
        "phase2_score_100",
        "phase4_score_100",
        "mixed_010_score",
        "mixed_score",
        "centered_risk_score",
        "phase5_score_100",
    ]
    for score_column in score_columns:
        if score_column not in panel.columns:
            continue
        frame = panel.dropna(subset=[score_column]).copy()
        if frame.empty:
            continue
        frame["score_decile"] = frame.groupby("signal_date", group_keys=False)[score_column].transform(_daily_decile)
        for horizon in horizons:
            return_column = f"return_{int(horizon)}d"
            profit_column = f"max_profit_{int(horizon)}d"
            drawdown_column = f"max_drawdown_{int(horizon)}d"
            if return_column not in frame.columns:
                continue
            grouped = frame.dropna(subset=["score_decile", return_column]).groupby("score_decile", dropna=False)
            for decile, group in grouped:
                rows.append(
                    {
                        "score_column": score_column,
                        "horizon": int(horizon),
                        "score_decile": int(decile),
                        "rows": int(len(group)),
                        "avg_return": float(pd.to_numeric(group[return_column], errors="coerce").mean()),
                        "win_rate": float(pd.to_numeric(group[return_column], errors="coerce").gt(0).mean()),
                        "avg_max_profit": float(pd.to_numeric(group[profit_column], errors="coerce").mean()) if profit_column in group else math.nan,
                        "avg_max_drawdown": float(pd.to_numeric(group[drawdown_column], errors="coerce").mean()) if drawdown_column in group else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_strict_benchmark_comparison(summary: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or benchmark.empty:
        return pd.DataFrame()
    bench = benchmark.groupby("horizon", dropna=False).agg(
        benchmark_days=("signal_date", "nunique"),
        benchmark_avg_raw_return=("raw_return", "mean"),
        benchmark_avg_max_profit=("max_profit", "mean"),
        benchmark_avg_max_drawdown=("max_drawdown", "mean"),
    ).reset_index()
    result = summary.merge(bench, on="horizon", how="left")
    result["excess_avg_raw_return"] = result["avg_raw_return"] - result["benchmark_avg_raw_return"]
    result["excess_avg_max_profit"] = result["avg_max_profit"] - result["benchmark_avg_max_profit"]
    result["drawdown_advantage"] = result["avg_max_drawdown"] - result["benchmark_avg_max_drawdown"]
    return result


def format_strict_strategy_summary(summary: pd.DataFrame, *, top_n: int = 80) -> str:
    if summary.empty:
        return "No strict validation summary rows."
    columns = [
        "strategy",
        "horizon",
        "trade_count",
        "avg_raw_return",
        "raw_win_rate",
        "avg_gain",
        "avg_loss",
        "payoff_ratio",
        "avg_raw_R",
        "avg_max_profit",
        "avg_max_drawdown",
        "stop_loss_rate",
        "take_profit_rate",
        "no_candidate_day_rate",
    ]
    available = [column for column in columns if column in summary.columns]
    return summary.loc[:, available].head(top_n).to_string(index=False)


def _score_oos_window(
    *,
    window_id: int,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    tail_panel: pd.DataFrame,
    barrier_panel: pd.DataFrame,
    alpha_panel: pd.DataFrame,
    alpha_features: tuple[str, ...],
    phase7_dataset: pd.DataFrame,
    phase7_feature_columns: tuple[str, ...],
    phase1_model_name: str,
    phase2_model_name: str,
    phase7_model_name: str,
    min_training_rows: int,
    include_phase7: bool,
    lgbm_device: str,
    lgbm_n_jobs: int | None,
    lgbm_gpu_platform_id: int | None,
    lgbm_gpu_device_id: int | None,
) -> tuple[pd.DataFrame, str]:
    tail_train = _date_slice(tail_panel, start=train_start, end=train_end).dropna(subset=["risk_label", *TAIL_RISK_FEATURE_COLUMNS])
    tail_test = _date_slice(tail_panel, start=test_start, end=test_end).dropna(subset=list(TAIL_RISK_FEATURE_COLUMNS))
    barrier_train = _date_slice(barrier_panel, start=train_start, end=train_end).dropna(subset=["risk_label", *BARRIER_RISK_FEATURE_COLUMNS])
    barrier_test = _date_slice(barrier_panel, start=test_start, end=test_end).dropna(subset=list(BARRIER_RISK_FEATURE_COLUMNS))
    alpha_train = _date_slice(alpha_panel, start=train_start, end=train_end).dropna(subset=["LABEL0", *alpha_features])
    alpha_test = _date_slice(alpha_panel, start=test_start, end=test_end).dropna(subset=list(alpha_features))
    if len(tail_train) < min_training_rows or len(barrier_train) < min_training_rows or len(alpha_train) < min_training_rows:
        logging.warning("Skip OOS window %s due to insufficient training rows", window_id)
        return pd.DataFrame(), normalized_lgbm_device(lgbm_device)
    if tail_train["risk_label"].astype(int).nunique() < 2 or barrier_train["risk_label"].astype(int).nunique() < 2:
        logging.warning("Skip OOS window %s due to single-class risk labels", window_id)
        return pd.DataFrame(), normalized_lgbm_device(lgbm_device)
    phase1_model, phase1_device = _fit_strict_risk_model(
        model_name=phase1_model_name,
        X=tail_train.loc[:, TAIL_RISK_FEATURE_COLUMNS],
        y=tail_train["risk_label"].astype(int),
        lgbm_device=lgbm_device,
        lgbm_n_jobs=lgbm_n_jobs,
        lgbm_gpu_platform_id=lgbm_gpu_platform_id,
        lgbm_gpu_device_id=lgbm_gpu_device_id,
        fit_label=f"Strict OOS window {window_id} Phase1",
    )
    phase2_model, phase2_device = _fit_strict_risk_model(
        model_name=phase2_model_name,
        X=barrier_train.loc[:, BARRIER_RISK_FEATURE_COLUMNS],
        y=barrier_train["risk_label"].astype(int),
        lgbm_device=lgbm_device,
        lgbm_n_jobs=lgbm_n_jobs,
        lgbm_gpu_platform_id=lgbm_gpu_platform_id,
        lgbm_gpu_device_id=lgbm_gpu_device_id,
        fit_label=f"Strict OOS window {window_id} Phase2",
    )
    phase4_model, phase4_device = fit_lgbm_with_device(
        LGBMRegressor,
        QLIB_ALPHA158_LGBM_PARAMS,
        alpha_train.loc[:, alpha_features],
        alpha_train["LABEL0"],
        device=lgbm_device,
        n_jobs=lgbm_n_jobs,
        gpu_platform_id=lgbm_gpu_platform_id,
        gpu_device_id=lgbm_gpu_device_id,
        fallback_to_cpu=True,
        fit_label=f"Strict OOS window {window_id} Phase4",
    )
    used_lgbm_device = "cpu" if "cpu" in {phase1_device, phase2_device, phase4_device} else phase4_device

    phase1 = tail_test.loc[:, ["trade_date", "symbol", "name"]].copy()
    phase1["phase1_risk_score"] = _predict_risk_proba(phase1_model, tail_test.loc[:, TAIL_RISK_FEATURE_COLUMNS])
    phase2 = barrier_test.loc[:, ["trade_date", "symbol"]].copy()
    phase2["phase2_barrier_risk_score"] = _predict_risk_proba(phase2_model, barrier_test.loc[:, BARRIER_RISK_FEATURE_COLUMNS])
    if "is_cusum_event" in barrier_test.columns:
        phase2["phase2_is_cusum_event"] = barrier_test["is_cusum_event"].astype(int).values
    phase4 = alpha_test.loc[:, ["trade_date", "symbol", "name"]].copy()
    phase4["phase4_return_score"] = phase4_model.predict(alpha_test.loc[:, alpha_features])
    base = phase4.merge(phase1.drop(columns=["name"], errors="ignore"), on=["trade_date", "symbol"], how="inner")
    base = base.merge(phase2, on=["trade_date", "symbol"], how="inner")
    base["signal_date"] = pd.to_datetime(base["trade_date"], errors="coerce").dt.date.astype(str)
    base["window_id"] = int(window_id)
    base["train_start"] = train_start.isoformat()
    base["train_end"] = train_end.isoformat()
    base["test_start"] = test_start.isoformat()
    base["test_end"] = test_end.isoformat()

    if include_phase7 and not phase7_dataset.empty and phase7_feature_columns:
        phase7 = _score_phase7_oos(
            phase7_dataset=phase7_dataset,
            feature_columns=phase7_feature_columns,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            model_name=phase7_model_name,
        )
        base = base.merge(phase7, on="signal_date", how="left")
    else:
        base["phase7_buy_day_risk_score"] = pd.NA
        base["phase7_selected_threshold"] = pd.NA
        base["phase7_trade_permission"] = "unknown"
    return base, used_lgbm_device


def _fit_strict_risk_model(
    *,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    lgbm_device: str,
    lgbm_n_jobs: int | None,
    lgbm_gpu_platform_id: int | None,
    lgbm_gpu_device_id: int | None,
    fit_label: str,
) -> tuple[Any, str]:
    if model_name == "lightgbm_classifier":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is required for lightgbm_classifier.")
        return fit_lgbm_with_device(
            LGBMClassifier,
            TAIL_RISK_LGBM_CLASSIFIER_PARAMS,
            X,
            y,
            device=lgbm_device,
            n_jobs=lgbm_n_jobs,
            gpu_platform_id=lgbm_gpu_platform_id,
            gpu_device_id=lgbm_gpu_device_id,
            fallback_to_cpu=True,
            fit_label=fit_label,
        )
    model = _tail_risk_models((model_name,), n_jobs=lgbm_n_jobs)[model_name]
    return model.fit(X, y), normalized_lgbm_device(lgbm_device)


def _score_phase7_oos(
    *,
    phase7_dataset: pd.DataFrame,
    feature_columns: tuple[str, ...],
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    model_name: str,
) -> pd.DataFrame:
    train = _date_slice(phase7_dataset, start=train_start, end=train_end).dropna(subset=["bad_buy_day", *feature_columns])
    test = _date_slice(phase7_dataset, start=test_start, end=test_end).dropna(subset=list(feature_columns))
    if train.empty or test.empty or train["bad_buy_day"].astype(int).nunique() < 2:
        return pd.DataFrame(columns=["signal_date", "phase7_buy_day_risk_score", "phase7_selected_threshold", "phase7_trade_permission"])
    model = _fit_trade_day_model(train, feature_columns=feature_columns, model_name=model_name)
    train_score = _score_trade_day_model(train, model=model, feature_columns=feature_columns, model_name=model_name)
    threshold = float(pd.Series(train_score).quantile(0.8))
    score = _score_trade_day_model(test, model=model, feature_columns=feature_columns, model_name=model_name)
    result = test.loc[:, ["trade_date"]].copy()
    result["signal_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.date.astype(str)
    result["phase7_buy_day_risk_score"] = score
    result["phase7_selected_threshold"] = threshold
    result["phase7_trade_permission"] = np.where(result["phase7_buy_day_risk_score"].lt(threshold), "allow", "no_trade")
    return result.drop(columns=["trade_date"]).drop_duplicates("signal_date", keep="last")


def _add_oos_scores(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    result["phase1_score_100"] = result.groupby("signal_date", group_keys=False)["phase1_risk_score"].apply(
        lambda values: score_series_100(values, higher_is_better=False)
    )
    result["phase2_score_100"] = result.groupby("signal_date", group_keys=False)["phase2_barrier_risk_score"].apply(
        lambda values: score_series_100(values, higher_is_better=False)
    )
    result["phase4_score_100"] = result.groupby("signal_date", group_keys=False)["phase4_return_score"].apply(
        lambda values: score_series_100(values, higher_is_better=True)
    )
    result["mixed_score"] = (
        pd.to_numeric(result["phase4_score_100"], errors="coerce")
        + 0.2 * pd.to_numeric(result["phase1_score_100"], errors="coerce")
        + 0.2 * pd.to_numeric(result["phase2_score_100"], errors="coerce")
    ).round(4)
    result["mixed_010_score"] = (
        pd.to_numeric(result["phase4_score_100"], errors="coerce")
        + 0.1 * pd.to_numeric(result["phase1_score_100"], errors="coerce")
        + 0.1 * pd.to_numeric(result["phase2_score_100"], errors="coerce")
    ).round(4)
    phase1_score = pd.to_numeric(result["phase1_score_100"], errors="coerce")
    phase2_score = pd.to_numeric(result["phase2_score_100"], errors="coerce")
    result["phase1_center_score"] = (100.0 - 2.0 * (phase1_score - 80.0).abs()).clip(lower=0.0, upper=100.0).round(4)
    result["phase2_center_score"] = (100.0 - 2.0 * (phase2_score - 80.0).abs()).clip(lower=0.0, upper=100.0).round(4)
    result["centered_risk_score"] = (
        pd.to_numeric(result["phase4_score_100"], errors="coerce")
        + 0.08 * pd.to_numeric(result["phase1_center_score"], errors="coerce")
        + 0.12 * pd.to_numeric(result["phase2_center_score"], errors="coerce")
    ).round(4)
    result["all90_flag"] = (
        pd.to_numeric(result["phase1_score_100"], errors="coerce").ge(90.0)
        & pd.to_numeric(result["phase2_score_100"], errors="coerce").ge(90.0)
        & pd.to_numeric(result["phase4_score_100"], errors="coerce").ge(90.0)
    )
    return result


def _select_strict_strategy(
    day: pd.DataFrame,
    *,
    strategy: str,
    signal_date: date,
    top_n: int,
    phase1_min_score: float,
    phase2_min_score: float,
    all90_min_score: float,
    phase5_safe_min_score: float,
) -> pd.DataFrame:
    base = day.copy()
    if strategy == "random_top20":
        base["random_score"] = base["symbol"].map(lambda symbol: _stable_random_score(signal_date, symbol))
        return base.sort_values(["random_score", "symbol"], ascending=[False, True]).head(max(int(top_n), 0))
    if strategy == "phase4_top20":
        return base.dropna(subset=["phase4_score_100"]).sort_values(["phase4_score_100", "symbol"], ascending=[False, True]).head(max(int(top_n), 0))
    if strategy.startswith("phase4_top20_p12_ge"):
        threshold = float(str(strategy).rsplit("ge", 1)[-1])
        selected = base[
            pd.to_numeric(base["phase1_score_100"], errors="coerce").ge(threshold)
            & pd.to_numeric(base["phase2_score_100"], errors="coerce").ge(threshold)
        ].copy()
        return selected.dropna(subset=["phase4_score_100"]).sort_values(["phase4_score_100", "symbol"], ascending=[False, True]).head(max(int(top_n), 0))
    if strategy.startswith("mixed_010_top20"):
        selected = base[
            pd.to_numeric(base["phase1_score_100"], errors="coerce").ge(float(phase1_min_score))
            & pd.to_numeric(base["phase2_score_100"], errors="coerce").ge(float(phase2_min_score))
            & pd.to_numeric(base["mixed_010_score"], errors="coerce").notna()
        ].copy()
        if "phase7_allow" in strategy:
            selected = selected[selected.get("phase7_trade_permission", "").astype(str).str.lower().eq("allow")]
        if "phase5_safe" in strategy:
            selected = selected[pd.to_numeric(selected.get("phase5_score_100"), errors="coerce").ge(float(phase5_safe_min_score))]
        return selected.sort_values(
            ["mixed_010_score", "phase4_score_100", "phase1_score_100", "phase2_score_100", "symbol"],
            ascending=[False, False, False, False, True],
        ).head(max(int(top_n), 0))
    if strategy.startswith("mixed_top20"):
        selected = base[
            pd.to_numeric(base["phase1_score_100"], errors="coerce").ge(float(phase1_min_score))
            & pd.to_numeric(base["phase2_score_100"], errors="coerce").ge(float(phase2_min_score))
            & pd.to_numeric(base["mixed_score"], errors="coerce").notna()
        ].copy()
        if "phase7_allow" in strategy:
            selected = selected[selected.get("phase7_trade_permission", "").astype(str).str.lower().eq("allow")]
        if "phase5_safe" in strategy:
            selected = selected[pd.to_numeric(selected.get("phase5_score_100"), errors="coerce").ge(float(phase5_safe_min_score))]
        return selected.sort_values(
            ["mixed_score", "phase4_score_100", "phase1_score_100", "phase2_score_100", "symbol"],
            ascending=[False, False, False, False, True],
        ).head(max(int(top_n), 0))
    if strategy == "centered_risk_top20":
        selected = base[
            pd.to_numeric(base["phase1_score_100"], errors="coerce").ge(float(phase1_min_score))
            & pd.to_numeric(base["phase2_score_100"], errors="coerce").ge(max(float(phase2_min_score), 50.0))
            & pd.to_numeric(base["phase4_score_100"], errors="coerce").ge(70.0)
            & pd.to_numeric(base["centered_risk_score"], errors="coerce").notna()
        ].copy()
        return selected.sort_values(
            ["centered_risk_score", "phase4_score_100", "phase1_center_score", "phase2_center_score", "symbol"],
            ascending=[False, False, False, False, True],
        ).head(max(int(top_n), 0))
    if strategy.startswith("all90"):
        selected = base[
            pd.to_numeric(base["phase1_score_100"], errors="coerce").ge(float(all90_min_score))
            & pd.to_numeric(base["phase2_score_100"], errors="coerce").ge(float(all90_min_score))
            & pd.to_numeric(base["phase4_score_100"], errors="coerce").ge(float(all90_min_score))
        ].copy()
        if "phase7_allow" in strategy:
            selected = selected[selected.get("phase7_trade_permission", "").astype(str).str.lower().eq("allow")]
        if "phase5_safe" in strategy:
            selected = selected[pd.to_numeric(selected.get("phase5_score_100"), errors="coerce").ge(float(phase5_safe_min_score))]
        return selected.sort_values(
            ["mixed_score", "phase4_score_100", "phase1_score_100", "phase2_score_100", "symbol"],
            ascending=[False, False, False, False, True],
        )
    raise ValueError(f"Unsupported strategy: {strategy}")


def _attach_forward_outcomes(
    *,
    storage: Storage,
    panel: pd.DataFrame,
    horizons: tuple[int, ...],
    stop_loss_pct: float,
    take_profit_pct: float,
) -> pd.DataFrame:
    if panel.empty:
        return panel.copy()
    parts: list[pd.DataFrame] = []
    for symbol, group in panel.groupby("symbol", sort=False):
        try:
            bars = storage.load_daily_bars(str(symbol).zfill(6))
        except (FileNotFoundError, DailyBarsReadError):
            parts.append(group.copy())
            continue
        outcomes = _forward_outcome_frame(
            bars,
            signal_dates=tuple(pd.to_datetime(group["signal_date"], errors="coerce").dropna().dt.date.unique()),
            horizons=horizons,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        merged = group.merge(outcomes, on="signal_date", how="left")
        parts.append(merged)
    return pd.concat(parts, ignore_index=True) if parts else panel.copy()


def _forward_outcome_frame(
    bars: pd.DataFrame,
    *,
    signal_dates: tuple[date, ...],
    horizons: tuple[int, ...],
    stop_loss_pct: float,
    take_profit_pct: float,
) -> pd.DataFrame:
    frame = bars.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["signal_date"] = frame["trade_date"].dt.date.astype(str)
    result = frame.loc[:, ["signal_date"]].copy()
    entry_open = frame["open"].shift(-1)
    result["entry_open"] = entry_open
    result["entry_date"] = frame["trade_date"].shift(-1).dt.date.astype(str)
    stop_price = entry_open * (1.0 - float(stop_loss_pct))
    take_price = entry_open * (1.0 + float(take_profit_pct))
    for horizon in horizons:
        shifts_high = [frame["high"].shift(-offset) for offset in range(1, int(horizon) + 1)]
        shifts_low = [frame["low"].shift(-offset) for offset in range(1, int(horizon) + 1)]
        high_matrix = pd.concat(shifts_high, axis=1)
        low_matrix = pd.concat(shifts_low, axis=1)
        exit_close = frame["close"].shift(-int(horizon))
        result[f"return_{int(horizon)}d"] = exit_close / entry_open - 1.0
        result[f"max_profit_{int(horizon)}d"] = high_matrix.max(axis=1, skipna=False) / entry_open - 1.0
        result[f"max_drawdown_{int(horizon)}d"] = low_matrix.min(axis=1, skipna=False) / entry_open - 1.0
        exit_price = exit_close.copy()
        exit_reason = pd.Series("timeout", index=frame.index, dtype="object")
        hit = pd.Series(False, index=frame.index)
        for offset in range(1, int(horizon) + 1):
            low = frame["low"].shift(-offset)
            high = frame["high"].shift(-offset)
            both = (~hit) & low.le(stop_price) & high.ge(take_price)
            stop = (~hit) & low.le(stop_price)
            take = (~hit) & high.ge(take_price)
            exit_price.loc[both] = stop_price.loc[both]
            exit_reason.loc[both] = "stop_loss_first"
            hit.loc[both] = True
            stop = stop & (~hit)
            exit_price.loc[stop] = stop_price.loc[stop]
            exit_reason.loc[stop] = "stop_loss"
            hit.loc[stop] = True
            take = take & (~hit)
            exit_price.loc[take] = take_price.loc[take]
            exit_reason.loc[take] = "take_profit"
            hit.loc[take] = True
        result[f"barrier_return_{int(horizon)}d"] = exit_price / entry_open - 1.0
        result[f"return_R_{int(horizon)}d"] = result[f"return_{int(horizon)}d"] / float(stop_loss_pct)
        result[f"barrier_R_{int(horizon)}d"] = result[f"barrier_return_{int(horizon)}d"] / float(stop_loss_pct)
        result[f"max_profit_R_{int(horizon)}d"] = result[f"max_profit_{int(horizon)}d"] / float(stop_loss_pct)
        result[f"max_drawdown_R_{int(horizon)}d"] = result[f"max_drawdown_{int(horizon)}d"] / float(stop_loss_pct)
        result[f"exit_reason_{int(horizon)}d"] = exit_reason
        result[f"stop_loss_hit_{int(horizon)}d"] = exit_reason.isin({"stop_loss", "stop_loss_first"})
        result[f"take_profit_hit_{int(horizon)}d"] = exit_reason.eq("take_profit")
    wanted = {item.isoformat() for item in signal_dates}
    return result[result["signal_date"].isin(wanted)].copy()


def _build_atr_panel(*, storage: Storage, symbols: tuple[str, ...], signal_dates: list[date]) -> pd.DataFrame:
    wanted = {item.isoformat() for item in signal_dates}
    rows: list[pd.DataFrame] = []
    for symbol in symbols:
        try:
            bars = storage.load_daily_bars(str(symbol).zfill(6))
        except (FileNotFoundError, DailyBarsReadError):
            continue
        frame = add_indicators(bars)
        frame["signal_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date.astype(str)
        subset = frame[frame["signal_date"].isin(wanted)].copy()
        if subset.empty:
            continue
        subset["symbol"] = str(symbol).zfill(6)
        rows.append(subset.loc[:, ["signal_date", "symbol", "atr_14", "atr_pct_14", "atr_volatility_regime"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["signal_date", "symbol", "atr_14", "atr_pct_14", "atr_volatility_regime"])


def _load_phase5_measures(project_root: Path) -> pd.DataFrame:
    path = project_root / "reports" / "full_market_model" / "mcd_crash_annual_measures.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty or "symbol" not in frame.columns or "year" not in frame.columns:
        return pd.DataFrame()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    return frame


def _attach_phase5(panel: pd.DataFrame, phase5: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for signal_date, group in panel.groupby("signal_date", sort=False):
        year = date.fromisoformat(str(signal_date)).year
        visible = phase5[pd.to_numeric(phase5["year"], errors="coerce").lt(year)].copy()
        if visible.empty:
            part = group.copy()
            part["phase5_score_100"] = pd.NA
            rows.append(part)
            continue
        raw_phase5_columns = ("NEGOUTLIER", "CRASH", "CRASH_count", "NCSKEW", "DUVOL", "RET", "SIGMA", "MINRET")
        visible = visible.rename(columns={column: f"phase5_{column}" for column in raw_phase5_columns if column in visible.columns})
        visible = add_phase5_score_100(visible)
        latest = visible.sort_values(["year", "symbol"]).drop_duplicates("symbol", keep="last")
        keep = [
            "symbol",
            "year",
            "phase5_score_100",
            "phase5_NEGOUTLIER",
            "phase5_CRASH",
            "phase5_CRASH_count",
            "phase5_NCSKEW",
            "phase5_DUVOL",
            "phase5_RET",
            "phase5_SIGMA",
            "phase5_MINRET",
        ]
        latest = latest.loc[:, [column for column in keep if column in latest.columns]].rename(columns={"year": "phase5_year"})
        rows.append(group.merge(latest, on="symbol", how="left"))
    return pd.concat(rows, ignore_index=True) if rows else panel.copy()


def _build_synthetic_benchmark(
    *,
    storage: Storage,
    project_root: Path,
    output_dir: Path,
    signal_dates: list[date],
    horizons: tuple[int, ...],
    limit: int | None,
    min_stock_count: int,
) -> pd.DataFrame:
    if not signal_dates:
        return pd.DataFrame()
    result = build_synthetic_market_index(
        storage=storage,
        project_root=project_root,
        start_date=min(signal_dates).isoformat(),
        end_date=None,
        limit=limit,
        min_stock_count=min_stock_count if limit is None or limit >= min_stock_count else max(1, int(limit * 0.8)),
        output=output_dir / "synthetic_market_benchmark.csv",
    )
    market = result.frame.copy()
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce")
    market = market.dropna(subset=["trade_date", "synthetic_equal_weight_index"]).sort_values("trade_date")
    rows: list[dict[str, Any]] = []
    for signal_date in signal_dates:
        future = market[market["trade_date"].dt.date.gt(signal_date)].copy()
        for horizon in horizons:
            subset = future.head(int(horizon))
            if len(subset) < int(horizon):
                continue
            values = pd.to_numeric(subset["synthetic_equal_weight_index"], errors="coerce")
            entry = float(values.iloc[0])
            rows.append(
                {
                    "signal_date": signal_date.isoformat(),
                    "horizon": int(horizon),
                    "raw_return": float(values.iloc[-1]) / entry - 1.0,
                    "max_profit": float(values.max()) / entry - 1.0,
                    "max_drawdown": float(values.min()) / entry - 1.0,
                }
            )
    return pd.DataFrame(rows)


def _load_or_build_frame_cache(path: Path, *, use_cache: bool, build) -> pd.DataFrame:
    if use_cache and path.exists():
        return pd.read_parquet(path)
    frame = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def _print_strict_progress(progress: bool, started_at: float, message: str) -> None:
    if progress:
        print(f"{message} (elapsed={_format_duration(perf_counter() - started_at)})", flush=True)


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _effective_min_stock_count(*, limit: int | None, min_stock_count: int) -> int:
    if limit is None:
        return int(min_stock_count)
    return max(1, min(int(min_stock_count), int(math.floor(int(limit) * 0.8))))


def _oos_window_cache_path(cache_dir: Path, window_id: int) -> Path:
    return cache_dir / f"oos_window_{int(window_id):03d}.parquet"


def _validation_trade_dates(panel: pd.DataFrame, *, start_date: date, test_start_date: date, end_date: date) -> pd.Series:
    if panel.empty or "trade_date" not in panel.columns:
        return pd.Series(dtype="object")
    dates = pd.to_datetime(panel["trade_date"], errors="coerce").dropna().dt.date
    return pd.Series(sorted(date_value for date_value in dates.unique() if start_date <= date_value <= end_date and date_value >= start_date))


def _date_slice(frame: pd.DataFrame, *, start: date, end: date) -> pd.DataFrame:
    if frame.empty or "trade_date" not in frame.columns:
        return pd.DataFrame()
    dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    return frame[(dates >= start) & (dates <= end)].copy()


def _daily_decile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = numeric.dropna()
    if valid.empty:
        return result
    if len(valid) < 10:
        rank = valid.rank(method="first")
        decile = np.floor((rank - 1.0) / len(valid) * 10.0).astype(int).clip(0, 9)
    else:
        decile = pd.qcut(valid.rank(method="first"), 10, labels=False, duplicates="drop")
    result.loc[valid.index] = pd.Series(decile, index=valid.index).astype("Int64")
    return result


def _stable_random_score(signal_date: date, symbol: str) -> float:
    digest = hashlib.blake2b(f"{signal_date.isoformat()}:{str(symbol).zfill(6)}".encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)
