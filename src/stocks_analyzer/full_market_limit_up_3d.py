from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # LightGBM is intentionally kept consistent with Phase4.
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - depends on optional local package.
    LGBMRegressor = None

from .full_market_alpha158 import (
    build_alpha158_feature_audit,
    build_alpha158_feature_frame,
    build_alpha158_latest_feature_frame,
)
from .full_market_panel import full_market_report_dir
from .full_market_return import QLIB_ALPHA158_LGBM_PARAMS
from .lgbm_utils import fit_lgbm_with_device
from .phase_display import score_series_100
from .storage import DailyBarsReadError, Storage


LIMIT_UP_3D_MODEL_VERSION = "limit_up_3d_opportunity_phase8_v1"
LIMIT_UP_3D_LABEL_COLUMNS = {
    "today_limit_up",
    "today_high_return_vs_prev_close",
    "today_close_return_vs_prev_close",
    "limit_up_hit_3d",
    "down_3d",
    "trap_3d",
    "future_return_3d",
    "future_max_high_return_3d",
    "future_max_drawdown_3d",
    "phase8_target",
}
LIMIT_UP_3D_LGBM_PARAMS: dict[str, Any] = {
    **QLIB_ALPHA158_LGBM_PARAMS,
    "objective": "regression",
}
PROGRESS_LOG_INTERVAL = 500


@dataclass(slots=True)
class LimitUp3DPanelResult:
    dataset: pd.DataFrame
    skipped: pd.DataFrame
    feature_columns: tuple[str, ...]
    feature_audit: pd.DataFrame


@dataclass(slots=True)
class LimitUp3DTrainResult:
    model_path: Path
    metadata_path: Path
    train_rows: int
    train_start: str
    train_end: str
    feature_columns: tuple[str, ...]
    used_lgbm_device: str


@dataclass(slots=True)
class LimitUp3DPredictionResult:
    predictions: pd.DataFrame
    skipped: pd.DataFrame
    output_path: Path
    artifact_path: Path


@dataclass(slots=True)
class LimitUp3DValidationResult:
    windows: pd.DataFrame
    scored: pd.DataFrame
    summary: pd.DataFrame
    deciles: pd.DataFrame
    report_dir: Path
    windows_path: Path
    scored_path: Path
    summary_path: Path
    deciles_path: Path
    config_path: Path


def build_limit_up_3d_frame(
    bars: pd.DataFrame,
    *,
    symbol: str,
    name: str = "",
    limit_up_threshold: float = 0.099,
    down_threshold: float = -0.05,
    down_penalty: float = 1.2,
    trap_penalty: float = 1.5,
    hit_reward: float = 1.0,
) -> pd.DataFrame:
    features = build_alpha158_feature_frame(bars, symbol=symbol, name=name)
    prices = _prepare_price_frame(bars)
    if features.empty or prices.empty:
        return pd.DataFrame()

    close = prices["close"].where(prices["close"].gt(0))
    high = prices["high"].where(prices["high"].gt(0))
    previous_close = close.shift(1)
    today_high_return = high.div(previous_close).sub(1.0)
    today_close_return = close.div(previous_close).sub(1.0)
    today_limit_up = today_high_return.gt(float(limit_up_threshold))

    future_limit_returns = []
    future_drawdowns = []
    for offset in range(1, 4):
        future_limit_returns.append(high.shift(-offset).div(close.shift(-(offset - 1))).sub(1.0))
        future_drawdowns.append(prices["low"].shift(-offset).div(close).sub(1.0))
    limit_up_hit = pd.concat(future_limit_returns, axis=1).gt(float(limit_up_threshold)).any(axis=1)
    future_return_3d = close.shift(-3).div(close).sub(1.0)
    down_3d = future_return_3d.lt(float(down_threshold))
    trap_3d = limit_up_hit & down_3d
    target = (
        float(hit_reward) * limit_up_hit.astype(float)
        - float(down_penalty) * down_3d.astype(float)
        - float(trap_penalty) * trap_3d.astype(float)
    )

    labels = pd.DataFrame(
        {
            "today_limit_up": today_limit_up.astype("float32"),
            "today_high_return_vs_prev_close": today_high_return,
            "today_close_return_vs_prev_close": today_close_return,
            "limit_up_hit_3d": limit_up_hit.astype("float32"),
            "down_3d": down_3d.astype("float32"),
            "trap_3d": trap_3d.astype("float32"),
            "future_return_3d": future_return_3d,
            "future_max_high_return_3d": pd.concat(future_limit_returns, axis=1).max(axis=1),
            "future_max_drawdown_3d": pd.concat(future_drawdowns, axis=1).min(axis=1),
            "phase8_target": target,
        },
        index=prices.index,
    ).replace([np.inf, -np.inf], np.nan)
    frame = pd.concat([features.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
    frame = frame.loc[frame["future_return_3d"].notna()].copy()
    frame = frame.loc[~frame["today_limit_up"].fillna(0).astype(bool)].reset_index(drop=True)
    return _downcast_phase8_numeric(frame)


def build_limit_up_3d_panel(
    *,
    storage: Storage,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    limit_up_threshold: float = 0.099,
    down_threshold: float = -0.05,
    down_penalty: float = 1.2,
    trap_penalty: float = 1.5,
    hit_reward: float = 1.0,
) -> LimitUp3DPanelResult:
    universe = storage.load_universe().copy()
    if limit is not None:
        universe = universe.head(max(int(limit), 0)).copy()
    rows: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []
    instruments = universe.to_dict("records")
    total_symbols = len(instruments)
    for index, instrument in enumerate(instruments, start=1):
        _log_progress("Limit-up 3D panel build", index, total_symbols)
        symbol = str(instrument.get("symbol", "")).zfill(6)
        name = str(instrument.get("name", ""))
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError) as exc:
            skipped.append({"symbol": symbol, "name": name, "reason": type(exc).__name__})
            continue
        frame = build_limit_up_3d_frame(
            bars,
            symbol=symbol,
            name=name,
            limit_up_threshold=limit_up_threshold,
            down_threshold=down_threshold,
            down_penalty=down_penalty,
            trap_penalty=trap_penalty,
            hit_reward=hit_reward,
        )
        if frame.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "empty_limit_up_3d_frame"})
            continue
        if start_date is not None:
            frame = frame[frame["trade_date"].dt.date >= start_date]
        if end_date is not None:
            frame = frame[frame["trade_date"].dt.date <= end_date]
        frame = frame.loc[frame["phase8_target"].notna()].reset_index(drop=True)
        if frame.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "no_labeled_feature_rows"})
            continue
        rows.append(_downcast_phase8_numeric(frame))
    dataset = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not dataset.empty:
        dataset = dataset.sort_values(["trade_date", "symbol"]).reset_index(drop=True) if len(dataset) <= 1_000_000 else dataset.reset_index(drop=True)
    feature_columns = limit_up_3d_feature_columns(dataset) if not dataset.empty else tuple()
    all_missing_features = [column for column in feature_columns if dataset[column].isna().all()] if feature_columns else []
    if all_missing_features:
        dataset = dataset.drop(columns=all_missing_features)
        feature_columns = limit_up_3d_feature_columns(dataset)
    feature_audit = build_alpha158_feature_audit(dataset, feature_columns=feature_columns)
    logging.info(
        "Limit-up 3D panel build complete: symbols=%s rows=%s features=%s skipped=%s",
        total_symbols,
        len(dataset),
        len(feature_columns),
        len(skipped),
    )
    return LimitUp3DPanelResult(
        dataset=dataset,
        skipped=pd.DataFrame(skipped),
        feature_columns=feature_columns,
        feature_audit=feature_audit,
    )


def limit_up_3d_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    excluded = {
        "trade_date",
        "symbol",
        "name",
        "future_return_5d",
        "future_max_drawdown_5d",
        *LIMIT_UP_3D_LABEL_COLUMNS,
    }
    return tuple(column for column in frame.columns if column not in excluded)


def train_limit_up_3d_opportunity_model(
    *,
    storage: Storage,
    project_root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    limit_up_threshold: float = 0.099,
    down_threshold: float = -0.05,
    down_penalty: float = 1.2,
    trap_penalty: float = 1.5,
    hit_reward: float = 1.0,
    min_training_rows: int = 200,
    lgbm_device: str = "cpu",
    lgbm_n_jobs: int | None = 1,
    lgbm_gpu_platform_id: int | None = None,
    lgbm_gpu_device_id: int | None = None,
) -> LimitUp3DTrainResult:
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is required for Phase8 limit-up 3D deployment training.")
    logging.info("Phase8 limit-up 3D deployment panel build started")
    panel = build_limit_up_3d_panel(
        storage=storage,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        limit_up_threshold=limit_up_threshold,
        down_threshold=down_threshold,
        down_penalty=down_penalty,
        trap_penalty=trap_penalty,
        hit_reward=hit_reward,
    )
    dataset = panel.dataset
    if len(dataset) < min_training_rows:
        raise RuntimeError(f"Insufficient Phase8 limit-up 3D training rows: {len(dataset)}")
    if not panel.feature_columns:
        raise RuntimeError("Phase8 limit-up 3D training has no feature columns.")

    logging.info(
        "Phase8 limit-up 3D deployment model fit started: rows=%s features=%s",
        len(dataset),
        len(panel.feature_columns),
    )
    fitted, used_device = fit_lgbm_with_device(
        LGBMRegressor,
        LIMIT_UP_3D_LGBM_PARAMS,
        dataset.loc[:, panel.feature_columns],
        dataset["phase8_target"],
        device=lgbm_device,
        n_jobs=lgbm_n_jobs,
        gpu_platform_id=lgbm_gpu_platform_id,
        gpu_device_id=lgbm_gpu_device_id,
        fit_label="Phase8 limit-up 3D",
    )

    trade_dates = pd.to_datetime(dataset["trade_date"], errors="coerce").dropna()
    train_start = trade_dates.min().date().isoformat()
    train_end = trade_dates.max().date().isoformat()
    artifact = {
        "model_version": LIMIT_UP_3D_MODEL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": "lightgbm_regressor",
        "model": fitted,
        "feature_columns": tuple(panel.feature_columns),
        "reference": "Local Phase8: Qlib Alpha158 + LightGBM regression with 3-day limit-up opportunity target",
        "handler": "Qlib Alpha158 compatible local features",
        "label": "1*limit_up_hit_3d - down_penalty*down_3d - trap_penalty*(limit_up_hit_3d & down_3d)",
        "label_config": {
            "limit_up_threshold": float(limit_up_threshold),
            "down_threshold": float(down_threshold),
            "down_penalty": float(down_penalty),
            "trap_penalty": float(trap_penalty),
            "hit_reward": float(hit_reward),
            "today_limit_up_rows": "excluded from training and prediction ranking",
        },
        "lightgbm_params": LIMIT_UP_3D_LGBM_PARAMS,
        "used_lgbm_device": used_device,
        "train_config": {
            "start_date": start_date.isoformat() if start_date else "",
            "end_date": end_date.isoformat() if end_date else "",
            "limit": limit,
            "min_training_rows": int(min_training_rows),
        },
        "train_rows": int(len(dataset)),
        "train_start": train_start,
        "train_end": train_end,
        "feature_count": int(len(panel.feature_columns)),
        "skipped_symbols": int(len(panel.skipped)),
    }
    model_path, metadata_path = save_limit_up_3d_model_artifact(project_root, artifact)
    logging.info("Phase8 limit-up 3D deployment model saved: %s", model_path)
    return LimitUp3DTrainResult(
        model_path=model_path,
        metadata_path=metadata_path,
        train_rows=int(len(dataset)),
        train_start=train_start,
        train_end=train_end,
        feature_columns=tuple(panel.feature_columns),
        used_lgbm_device=used_device,
    )


def predict_limit_up_3d_opportunity(
    *,
    storage: Storage,
    project_root: Path,
    trade_date: date,
    output: Path | None = None,
    limit: int | None = None,
    latest_only: bool = False,
    feature_lookback_bars: int = 120,
    include_features: bool = True,
    prediction_scope: str = "full_market_daily",
) -> LimitUp3DPredictionResult:
    artifact = load_limit_up_3d_model_artifact(project_root)
    feature_columns = tuple(artifact["feature_columns"])
    limit_up_threshold = float(artifact.get("label_config", {}).get("limit_up_threshold", 0.099))
    universe = storage.load_universe().copy()
    if limit is not None:
        universe = universe.head(max(int(limit), 0)).copy()

    rows: list[dict[str, Any]] = []
    feature_rows: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []
    instruments = universe.to_dict("records")
    total_symbols = len(instruments)
    for index, instrument in enumerate(instruments, start=1):
        _log_progress("Phase8 limit-up 3D prediction", index, total_symbols)
        symbol = str(instrument.get("symbol", "")).zfill(6)
        name = str(instrument.get("name", ""))
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError) as exc:
            skipped.append({"symbol": symbol, "name": name, "reason": type(exc).__name__})
            continue
        prepared = _prepare_price_frame(bars)
        prepared = prepared[prepared["trade_date"].dt.date <= trade_date].copy()
        if prepared.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "no_bars_on_or_before_trade_date"})
            continue
        if latest_only:
            frame = build_alpha158_latest_feature_frame(
                prepared,
                symbol=symbol,
                name=name,
                lookback_bars=feature_lookback_bars,
            )
        else:
            frame = build_alpha158_feature_frame(prepared, symbol=symbol, name=name)
        if frame.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "empty_alpha158_feature_frame"})
            continue
        row = _latest_prediction_row(frame, trade_date)
        if row.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "no_feature_date_on_or_before_trade_date"})
            continue
        row = row.dropna(subset=list(feature_columns))
        if row.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "no_feature_row"})
            continue
        latest_price_row = prepared.tail(1).iloc[0]
        previous_close = prepared["close"].shift(1).tail(1).iloc[0] if len(prepared) >= 2 else np.nan
        today_high_return = _safe_return(latest_price_row.get("high"), previous_close)
        today_close_return = _safe_return(latest_price_row.get("close"), previous_close)
        today_limit_up = bool(pd.notna(today_high_return) and float(today_high_return) > limit_up_threshold)
        feature_trade_date = pd.Timestamp(row.iloc[0]["trade_date"]).date().isoformat()
        record: dict[str, Any] = {
            "trade_date": trade_date.isoformat(),
            "feature_trade_date": feature_trade_date,
            "symbol": symbol,
            "name": name,
            "today_limit_up_excluded": today_limit_up,
            "today_high_return_vs_prev_close": today_high_return,
            "today_close_return_vs_prev_close": today_close_return,
            "prediction_scope": prediction_scope,
            "model_name": artifact["model_name"],
            "model_version": artifact["model_version"],
        }
        feature_row = row.loc[:, feature_columns].copy()
        feature_rows.append(feature_row)
        if include_features:
            for column in feature_columns:
                record[column] = float(row.iloc[0][column])
        rows.append(record)

    predictions = pd.DataFrame(rows)
    if rows and feature_rows:
        scores = artifact["model"].predict(pd.concat(feature_rows, ignore_index=True))
        predictions["phase8_raw_score"] = [float(score) for score in scores]
        tradable = ~predictions["today_limit_up_excluded"].fillna(False).astype(bool)
        predictions["phase8_score_100"] = pd.NA
        predictions.loc[tradable, "phase8_score_100"] = score_series_100(
            predictions.loc[tradable, "phase8_raw_score"],
            higher_is_better=True,
        )
        predictions.loc[~tradable, "phase8_score_100"] = 0.0
        predictions["phase8_rank"] = pd.NA
        ranked_index = predictions.loc[tradable].sort_values(["phase8_raw_score", "symbol"], ascending=[False, True]).index
        predictions.loc[ranked_index, "phase8_rank"] = range(1, len(ranked_index) + 1)
    if not predictions.empty:
        preferred_columns = [
            "trade_date",
            "feature_trade_date",
            "symbol",
            "name",
            "phase8_score_100",
            "phase8_raw_score",
            "phase8_rank",
            "today_limit_up_excluded",
            "today_high_return_vs_prev_close",
            "today_close_return_vs_prev_close",
            "prediction_scope",
            "model_name",
            "model_version",
            *[column for column in feature_columns if column in predictions.columns],
        ]
        preferred_columns.extend(column for column in predictions.columns if column not in preferred_columns)
        predictions = predictions.loc[:, preferred_columns]
        predictions = predictions.sort_values(
            ["today_limit_up_excluded", "phase8_score_100", "symbol"],
            ascending=[True, False, True],
            na_position="last",
        ).reset_index(drop=True)
    skipped_frame = pd.DataFrame(skipped)
    output_path = output if output is not None else limit_up_3d_predictions_path(project_root, trade_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
    skipped_frame.to_csv(output_path.with_name(f"{output_path.stem}_skipped.csv"), index=False, encoding="utf-8-sig")
    logging.info("Phase8 limit-up 3D predictions saved: rows=%s skipped=%s output=%s", len(predictions), len(skipped_frame), output_path)
    return LimitUp3DPredictionResult(
        predictions=predictions,
        skipped=skipped_frame,
        output_path=output_path,
        artifact_path=limit_up_3d_model_path(project_root),
    )


def validate_limit_up_3d_opportunity(
    *,
    storage: Storage,
    project_root: Path,
    start_date: date | None = None,
    test_start_date: date,
    end_date: date | None = None,
    limit: int | None = None,
    top_ns: tuple[int, ...] = (5, 10, 20, 50),
    test_window_days: int = 60,
    step_days: int = 60,
    embargo_days: int = 3,
    min_train_days: int = 900,
    max_windows: int | None = None,
    limit_up_threshold: float = 0.099,
    down_threshold: float = -0.05,
    down_penalty: float = 1.2,
    trap_penalty: float = 1.5,
    hit_reward: float = 1.0,
    min_training_rows: int = 200,
    output_dir: Path | None = None,
    lgbm_device: str = "cpu",
    lgbm_n_jobs: int | None = 1,
    lgbm_gpu_platform_id: int | None = None,
    lgbm_gpu_device_id: int | None = None,
    progress: bool = False,
) -> LimitUp3DValidationResult:
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is required for Phase8 limit-up 3D validation.")
    logging.info("Phase8 limit-up 3D validation panel build started")
    panel = build_limit_up_3d_panel(
        storage=storage,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        limit_up_threshold=limit_up_threshold,
        down_threshold=down_threshold,
        down_penalty=down_penalty,
        trap_penalty=trap_penalty,
        hit_reward=hit_reward,
    )
    dataset = panel.dataset
    if dataset.empty:
        raise RuntimeError("Phase8 limit-up 3D validation has no labeled rows.")
    if not panel.feature_columns:
        raise RuntimeError("Phase8 limit-up 3D validation has no feature columns.")
    windows = build_limit_up_3d_walkforward_windows(
        dataset,
        test_start_date=test_start_date,
        test_window_days=test_window_days,
        step_days=step_days,
        embargo_days=embargo_days,
        min_train_days=min_train_days,
        max_windows=max_windows,
    )
    if windows.empty:
        raise RuntimeError("No Phase8 validation windows can be built from the requested date range.")

    scored_parts: list[pd.DataFrame] = []
    window_rows: list[dict[str, object]] = []
    for window_index, window in windows.reset_index(drop=True).iterrows():
        window_id = str(window["window_id"])
        train = _date_slice(dataset, start=window["train_start"], end=window["train_end"])
        test = _date_slice(dataset, start=window["test_start"], end=window["test_end"])
        if progress:
            print(
                f"Phase8 OOS window {window_index + 1}/{len(windows)}: "
                f"train {window['train_start']}->{window['train_end']}, "
                f"test {window['test_start']}->{window['test_end']}"
            )
        if len(train) < min_training_rows or test.empty:
            window_rows.append({**window.to_dict(), "train_rows": int(len(train)), "test_rows": int(len(test)), "status": "skipped"})
            continue
        model, used_device = fit_lgbm_with_device(
            LGBMRegressor,
            LIMIT_UP_3D_LGBM_PARAMS,
            train.loc[:, panel.feature_columns],
            train["phase8_target"],
            device=lgbm_device,
            n_jobs=lgbm_n_jobs,
            gpu_platform_id=lgbm_gpu_platform_id,
            gpu_device_id=lgbm_gpu_device_id,
            fit_label=f"Phase8 limit-up 3D {window_id}",
        )
        scored = test.loc[
            :,
            [
                "trade_date",
                "symbol",
                "name",
                "phase8_target",
                "limit_up_hit_3d",
                "down_3d",
                "trap_3d",
                "future_return_3d",
                "future_max_high_return_3d",
                "future_max_drawdown_3d",
            ],
        ].copy()
        scored["window_id"] = window_id
        scored["phase8_raw_score"] = model.predict(test.loc[:, panel.feature_columns])
        scored["phase8_score_100"] = scored.groupby("trade_date", group_keys=False)["phase8_raw_score"].apply(
            lambda values: score_series_100(values, higher_is_better=True)
        )
        scored_parts.append(scored)
        window_rows.append(
            {
                **window.to_dict(),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "used_lgbm_device": used_device,
                "status": "ok",
            }
        )
    scored_frame = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    if scored_frame.empty:
        raise RuntimeError("Phase8 validation produced no scored rows.")
    summary = summarize_limit_up_3d_strategies(scored_frame, top_ns=top_ns)
    deciles = build_limit_up_3d_decile_report(scored_frame)

    report_dir = output_dir if output_dir is not None else full_market_report_dir(project_root) / "limit_up_3d_opportunity_validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    windows_path = report_dir / "windows.csv"
    scored_path = report_dir / "scored.csv"
    summary_path = report_dir / "summary.csv"
    deciles_path = report_dir / "decile_report.csv"
    config_path = report_dir / "config.json"
    windows_frame = pd.DataFrame(window_rows)
    windows_frame.to_csv(windows_path, index=False, encoding="utf-8-sig")
    scored_frame.to_csv(scored_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    deciles.to_csv(deciles_path, index=False, encoding="utf-8-sig")
    config_path.write_text(
        json.dumps(
            {
                "model_version": LIMIT_UP_3D_MODEL_VERSION,
                "label_config": {
                    "limit_up_threshold": float(limit_up_threshold),
                    "down_threshold": float(down_threshold),
                    "down_penalty": float(down_penalty),
                    "trap_penalty": float(trap_penalty),
                    "hit_reward": float(hit_reward),
                    "today_limit_up_rows": "excluded before training and validation ranking",
                },
                "start_date": start_date.isoformat() if start_date else "",
                "test_start_date": test_start_date.isoformat(),
                "end_date": end_date.isoformat() if end_date else "",
                "limit": limit,
                "top_ns": list(top_ns),
                "test_window_days": int(test_window_days),
                "step_days": int(step_days),
                "embargo_days": int(embargo_days),
                "min_train_days": int(min_train_days),
                "max_windows": max_windows,
                "min_training_rows": int(min_training_rows),
                "feature_count": int(len(panel.feature_columns)),
                "skipped_symbols": int(len(panel.skipped)),
                "lightgbm_params": LIMIT_UP_3D_LGBM_PARAMS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return LimitUp3DValidationResult(
        windows=windows_frame,
        scored=scored_frame,
        summary=summary,
        deciles=deciles,
        report_dir=report_dir,
        windows_path=windows_path,
        scored_path=scored_path,
        summary_path=summary_path,
        deciles_path=deciles_path,
        config_path=config_path,
    )


def build_limit_up_3d_walkforward_windows(
    dataset: pd.DataFrame,
    *,
    test_start_date: date,
    test_window_days: int,
    step_days: int,
    embargo_days: int,
    min_train_days: int,
    max_windows: int | None = None,
) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(dataset["trade_date"], errors="coerce").dropna().dt.date.unique())
    if not dates:
        return pd.DataFrame()
    test_start_pos = next((idx for idx, item in enumerate(dates) if item >= test_start_date), None)
    if test_start_pos is None:
        return pd.DataFrame()
    rows = []
    current = test_start_pos
    window_index = 1
    while current < len(dates):
        if max_windows is not None and len(rows) >= int(max_windows):
            break
        test_end_pos = min(current + max(int(test_window_days), 1) - 1, len(dates) - 1)
        train_end_pos = current - max(int(embargo_days), 0) - 1
        if train_end_pos >= 0:
            train_start_pos = 0
            train_days = train_end_pos - train_start_pos + 1
            if train_days >= int(min_train_days):
                rows.append(
                    {
                        "window_id": f"wf_{window_index:03d}",
                        "train_start": dates[train_start_pos],
                        "train_end": dates[train_end_pos],
                        "test_start": dates[current],
                        "test_end": dates[test_end_pos],
                        "train_days": int(train_days),
                        "test_days": int(test_end_pos - current + 1),
                    }
                )
                window_index += 1
        current += max(int(step_days), 1)
    return pd.DataFrame(rows)


def summarize_limit_up_3d_strategies(scored: pd.DataFrame, *, top_ns: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for top_n in top_ns:
        selected = _select_top_n_by_day(scored, top_n=top_n)
        rows.append(_summarize_selection(selected, strategy=f"phase8_top{top_n}", top_n=top_n, source_days=scored["trade_date"].nunique()))
        random_selected = _select_random_top_n_by_day(scored, top_n=top_n)
        rows.append(_summarize_selection(random_selected, strategy=f"random_top{top_n}", top_n=top_n, source_days=scored["trade_date"].nunique()))
    rows.append(_summarize_selection(scored, strategy="all_tradable", top_n=0, source_days=scored["trade_date"].nunique()))
    return pd.DataFrame(rows)


def build_limit_up_3d_decile_report(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    frame = scored.copy()
    frame["score_pct"] = frame.groupby("trade_date")["phase8_raw_score"].rank(pct=True, method="first")
    frame["score_decile"] = np.ceil(frame["score_pct"].mul(10)).sub(1).clip(0, 9).astype(int)
    rows = []
    for decile, group in frame.groupby("score_decile", sort=True):
        rows.append(_summarize_selection(group, strategy=f"decile_{int(decile)}", top_n=0, source_days=group["trade_date"].nunique()))
    return pd.DataFrame(rows)


def limit_up_3d_model_dir(project_root: Path) -> Path:
    return project_root / "data" / "ml" / "full_market_limit_up_3d"


def limit_up_3d_model_path(project_root: Path) -> Path:
    return limit_up_3d_model_dir(project_root) / "limit_up_3d_opportunity_model.pkl"


def limit_up_3d_metadata_path(project_root: Path) -> Path:
    return limit_up_3d_model_dir(project_root) / "limit_up_3d_opportunity_model_metadata.json"


def save_limit_up_3d_model_artifact(project_root: Path, artifact: dict[str, Any]) -> tuple[Path, Path]:
    model_dir = limit_up_3d_model_dir(project_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = limit_up_3d_model_path(project_root)
    metadata_path = limit_up_3d_metadata_path(project_root)
    with model_path.open("wb") as file:
        pickle.dump(artifact, file)
    metadata = {key: value for key, value in artifact.items() if key != "model"}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_path, metadata_path


def load_limit_up_3d_model_artifact(project_root: Path) -> dict[str, Any]:
    model_path = limit_up_3d_model_path(project_root)
    if not model_path.exists():
        raise FileNotFoundError(f"Phase8 limit-up 3D model artifact not found: {model_path}")
    with model_path.open("rb") as file:
        return pickle.load(file)


def limit_up_3d_predictions_path(project_root: Path, trade_date: date) -> Path:
    return full_market_report_dir(project_root) / f"limit_up_3d_opportunity_predictions_{trade_date.isoformat()}.csv"


def format_limit_up_3d_prediction_table(predictions: pd.DataFrame, *, top_n: int = 20) -> str:
    if predictions.empty:
        return "No Phase8 limit-up 3D predictions."
    columns = [
        "trade_date",
        "feature_trade_date",
        "symbol",
        "name",
        "phase8_score_100",
        "phase8_raw_score",
        "phase8_rank",
        "today_limit_up_excluded",
        "today_close_return_vs_prev_close",
    ]
    available = [column for column in columns if column in predictions.columns]
    frame = predictions.loc[:, available].head(max(int(top_n), 0)).copy()
    for column in ("phase8_score_100", "phase8_raw_score", "today_close_return_vs_prev_close"):
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
    return frame.to_string(index=False)


def _summarize_selection(selected: pd.DataFrame, *, strategy: str, top_n: int, source_days: int) -> dict[str, object]:
    hit = pd.to_numeric(selected.get("limit_up_hit_3d"), errors="coerce")
    down = pd.to_numeric(selected.get("down_3d"), errors="coerce")
    trap = pd.to_numeric(selected.get("trap_3d"), errors="coerce")
    target = pd.to_numeric(selected.get("phase8_target"), errors="coerce")
    future_return = pd.to_numeric(selected.get("future_return_3d"), errors="coerce")
    max_high = pd.to_numeric(selected.get("future_max_high_return_3d"), errors="coerce")
    max_drawdown = pd.to_numeric(selected.get("future_max_drawdown_3d"), errors="coerce")
    active_days = int(selected["trade_date"].nunique()) if not selected.empty and "trade_date" in selected.columns else 0
    return {
        "strategy": strategy,
        "top_n": int(top_n),
        "days": int(source_days),
        "active_days": active_days,
        "trade_count": int(len(selected)),
        "no_candidate_day_rate": float(1.0 - active_days / source_days) if source_days else np.nan,
        "hit_3d_rate": _safe_mean(hit),
        "down_3d_rate": _safe_mean(down),
        "trap_3d_rate": _safe_mean(trap),
        "avg_phase8_target": _safe_mean(target),
        "avg_future_return_3d": _safe_mean(future_return),
        "median_future_return_3d": _safe_median(future_return),
        "win_3d_rate": float(future_return.gt(0).mean()) if future_return.notna().any() else np.nan,
        "avg_max_high_return_3d": _safe_mean(max_high),
        "avg_max_drawdown_3d": _safe_mean(max_drawdown),
    }


def _select_top_n_by_day(scored: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    parts = []
    for _, group in scored.groupby("trade_date", sort=True):
        selected = group.dropna(subset=["phase8_raw_score"]).sort_values(["phase8_raw_score", "symbol"], ascending=[False, True]).head(max(int(top_n), 0))
        if not selected.empty:
            parts.append(selected)
    return pd.concat(parts, ignore_index=True) if parts else scored.head(0).copy()


def _select_random_top_n_by_day(scored: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    parts = []
    for trade_date, group in scored.groupby("trade_date", sort=True):
        n = min(max(int(top_n), 0), len(group))
        if n <= 0:
            continue
        seed = int(pd.Timestamp(trade_date).strftime("%Y%m%d")) + n * 7919
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True) if parts else scored.head(0).copy()


def _date_slice(frame: pd.DataFrame, *, start: date | None, end: date | None) -> pd.DataFrame:
    result = frame
    if start is not None:
        result = result[result["trade_date"].dt.date >= start]
    if end is not None:
        result = result[result["trade_date"].dt.date <= end]
    return result.copy()


def _latest_prediction_row(frame: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    valid = frame[frame["trade_date"].dt.date <= trade_date].copy()
    if valid.empty:
        return pd.DataFrame()
    return valid.tail(1).copy()


def _prepare_price_frame(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    frame = bars.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        values = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
        frame[column] = pd.to_numeric(values, errors="coerce")
    return frame


def _downcast_phase8_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    excluded = {"trade_date", "symbol", "name"}
    for column in result.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("float32")
    return result


def _safe_return(value: object, base: object) -> float:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    numeric_base = pd.to_numeric(pd.Series([base]), errors="coerce").iloc[0]
    if pd.isna(numeric_value) or pd.isna(numeric_base) or float(numeric_base) <= 0:
        return float("nan")
    return float(numeric_value) / float(numeric_base) - 1.0


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else np.nan


def _safe_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else np.nan


def _log_progress(label: str, index: int, total: int) -> None:
    if index == 1 or index == total or index % PROGRESS_LOG_INTERVAL == 0:
        logging.info("%s progress: %s/%s", label, index, total)
