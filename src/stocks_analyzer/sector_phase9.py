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

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - depends on optional local package.
    LGBMClassifier = None

from .lgbm_utils import fit_lgbm_with_device
from .sector_membership import sector_performance_dir
from .sector_pullback_metrics import (
    _apply_buy_score_caps,
    _available_daily_symbols,
    _build_sector_info,
    _build_sector_return_frame,
    _linear_score,
    _load_stock_return_history,
    _prepare_membership,
    _score_pullback_depth,
    _score_pullback_timing,
    _score_rebound_confirmation,
    _score_risk_control,
    _score_stabilization,
)


SECTOR_PHASE9_MODEL_VERSION = "sector_phase9_20d_close_buy_score_v1"
SECTOR_PHASE9_IDENTIFIER_COLUMNS = {
    "trade_date",
    "sector_key",
    "sector_type",
    "sector_name",
    "sector_label",
}
SECTOR_PHASE9_LABEL_COLUMNS = {
    "future_return_20d_close",
    "future_max_return_20d",
    "future_min_return_20d",
    "phase9_label",
}
SECTOR_PHASE9_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": -1,
}


@dataclass(slots=True)
class SectorPhase9PanelResult:
    dataset: pd.DataFrame
    feature_columns: tuple[str, ...]
    symbol_count: int
    sector_count: int


@dataclass(slots=True)
class SectorPhase9TrainResult:
    model_path: Path
    metadata_path: Path
    train_rows: int
    train_start: str
    train_end: str
    feature_columns: tuple[str, ...]
    used_lgbm_device: str


@dataclass(slots=True)
class SectorPhase9PredictionResult:
    predictions: pd.DataFrame
    output_path: Path
    artifact_path: Path


@dataclass(slots=True)
class SectorPhase9ValidationResult:
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


@dataclass(slots=True)
class SectorRuleBuyScoreValidationResult:
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


def build_sector_phase9_panel(
    *,
    project_root: Path,
    trade_date: date | None = None,
    daily_dir: Path | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    history_days: int = 3200,
    min_members: int = 5,
    horizon_days: int = 20,
    return_threshold: float = 0.05,
    min_feature_history_days: int = 60,
    include_unlabeled: bool = False,
    progress: bool = False,
) -> SectorPhase9PanelResult:
    if history_days <= 0:
        raise ValueError("history_days must be positive")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if min_feature_history_days < 0:
        raise ValueError("min_feature_history_days must be non-negative")

    daily_root = daily_dir if daily_dir is not None else project_root / "data" / "daily"
    resolved_end = trade_date or end_date
    members = _prepare_membership(project_root=project_root, min_members=min_members)
    if members.empty:
        return SectorPhase9PanelResult(
            dataset=pd.DataFrame(),
            feature_columns=tuple(),
            symbol_count=0,
            sector_count=0,
        )

    symbols = sorted(set(members["symbol"].unique()) | set(_available_daily_symbols(daily_root)))
    stock_returns = _load_stock_return_history(
        daily_root=daily_root,
        symbols=symbols,
        trade_date=resolved_end,
        history_days=history_days,
        progress=progress,
    )
    if stock_returns.empty:
        return SectorPhase9PanelResult(
            dataset=pd.DataFrame(),
            feature_columns=tuple(),
            symbol_count=0,
            sector_count=members["sector_key"].nunique(),
        )

    if end_date is not None:
        stock_returns = stock_returns[stock_returns["trade_date"].dt.date <= end_date].copy()
    if stock_returns.empty:
        return SectorPhase9PanelResult(
            dataset=pd.DataFrame(),
            feature_columns=tuple(),
            symbol_count=0,
            sector_count=members["sector_key"].nunique(),
        )

    all_dates = pd.Index(sorted(stock_returns["trade_date"].unique()))
    benchmark_returns = stock_returns.groupby("trade_date", sort=True)["return_pct"].mean().reindex(all_dates).fillna(0.0)
    benchmark_index = (1.0 + benchmark_returns / 100.0).cumprod() * 100.0
    sector_returns = _build_sector_return_frame(stock_returns=stock_returns, members=members)
    sector_info = _build_sector_info(members)
    if sector_returns.empty or sector_info.empty:
        return SectorPhase9PanelResult(
            dataset=pd.DataFrame(),
            feature_columns=tuple(),
            symbol_count=stock_returns["symbol"].nunique(),
            sector_count=members["sector_key"].nunique(),
        )

    rows: list[pd.DataFrame] = []
    grouped = sector_returns.groupby("sector_key", sort=True)
    total = len(grouped)
    for index, (sector_key, group) in enumerate(grouped, start=1):
        if progress and (index == 1 or index % 100 == 0 or index == total):
            logging.info("Sector Phase9 panel build progress: %s/%s", index, total)
        if sector_key not in sector_info.index:
            continue
        info = sector_info.loc[sector_key]
        series = group.set_index("trade_date").sort_index()
        sector_ret = series["sector_return_pct"].reindex(all_dates).fillna(0.0).astype(float)
        sector_amount = series["total_amount"].reindex(all_dates).fillna(0.0).astype(float)
        valid_count = series["valid_count"].reindex(all_dates).fillna(0.0).astype(float)
        sector_index = (1.0 + sector_ret / 100.0).cumprod() * 100.0
        frame = _build_sector_phase9_feature_frame(
            sector_key=str(sector_key),
            sector_type=str(info["sector_type"]),
            sector_name=str(info["sector_name"]),
            sector_label=str(info["sector_label"]),
            member_count=int(info["member_count"]),
            sector_index=sector_index,
            sector_ret=sector_ret,
            sector_amount=sector_amount,
            valid_count=valid_count,
            benchmark_index=benchmark_index,
            horizon_days=horizon_days,
            return_threshold=return_threshold,
            min_feature_history_days=min_feature_history_days,
            include_unlabeled=include_unlabeled,
        )
        if not frame.empty:
            rows.append(frame)

    dataset = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if dataset.empty:
        return SectorPhase9PanelResult(
            dataset=dataset,
            feature_columns=tuple(),
            symbol_count=stock_returns["symbol"].nunique(),
            sector_count=members["sector_key"].nunique(),
        )
    if start_date is not None:
        dataset = dataset[dataset["trade_date"].dt.date >= start_date].copy()
    if end_date is not None:
        dataset = dataset[dataset["trade_date"].dt.date <= end_date].copy()
    if not include_unlabeled:
        dataset = dataset.dropna(subset=["phase9_label", "future_return_20d_close"]).copy()
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.sort_values(["trade_date", "sector_type", "sector_name"], kind="stable").reset_index(drop=True)
    feature_columns = sector_phase9_feature_columns(dataset)
    all_missing = [column for column in feature_columns if dataset[column].isna().all()]
    if all_missing:
        dataset = dataset.drop(columns=all_missing)
        feature_columns = sector_phase9_feature_columns(dataset)
    dataset = _downcast_sector_phase9_numeric(dataset)
    logging.info(
        "Sector Phase9 panel build complete: sectors=%s symbols=%s rows=%s features=%s",
        members["sector_key"].nunique(),
        stock_returns["symbol"].nunique(),
        len(dataset),
        len(feature_columns),
    )
    return SectorPhase9PanelResult(
        dataset=dataset,
        feature_columns=tuple(feature_columns),
        symbol_count=stock_returns["symbol"].nunique(),
        sector_count=members["sector_key"].nunique(),
    )


def sector_phase9_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    if frame.empty:
        return tuple()
    excluded = SECTOR_PHASE9_IDENTIFIER_COLUMNS | SECTOR_PHASE9_LABEL_COLUMNS
    columns = []
    for column in frame.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return tuple(columns)


def validate_sector_phase9_buy_score(
    *,
    project_root: Path,
    start_date: date | None = None,
    test_start_date: date,
    end_date: date | None = None,
    history_days: int = 3200,
    min_members: int = 5,
    horizon_days: int = 20,
    return_threshold: float = 0.05,
    top_ns: tuple[int, ...] = (5, 10, 20, 50),
    test_window_days: int = 60,
    step_days: int = 60,
    embargo_days: int = 20,
    min_train_days: int = 900,
    min_training_rows: int = 200,
    max_windows: int | None = None,
    output_dir: Path | None = None,
    lgbm_device: str = "cpu",
    lgbm_n_jobs: int | None = 1,
    lgbm_gpu_platform_id: int | None = None,
    lgbm_gpu_device_id: int | None = None,
    progress: bool = False,
) -> SectorPhase9ValidationResult:
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm is required for Sector Phase9 validation.")
    logging.info("Sector Phase9 validation panel build started")
    panel = build_sector_phase9_panel(
        project_root=project_root,
        start_date=start_date,
        end_date=end_date,
        history_days=history_days,
        min_members=min_members,
        horizon_days=horizon_days,
        return_threshold=return_threshold,
        include_unlabeled=False,
        progress=progress,
    )
    dataset = panel.dataset
    if dataset.empty:
        raise RuntimeError("Sector Phase9 validation has no labeled rows.")
    if not panel.feature_columns:
        raise RuntimeError("Sector Phase9 validation has no feature columns.")

    windows = build_sector_phase9_walkforward_windows(
        dataset,
        test_start_date=test_start_date,
        test_window_days=test_window_days,
        step_days=step_days,
        embargo_days=embargo_days,
        min_train_days=min_train_days,
        max_windows=max_windows,
    )
    if windows.empty:
        raise RuntimeError("No Sector Phase9 validation windows can be built from the requested date range.")

    scored_parts: list[pd.DataFrame] = []
    window_rows: list[dict[str, object]] = []
    for window_index, window in windows.reset_index(drop=True).iterrows():
        window_id = str(window["window_id"])
        train = _date_slice(dataset, start=window["train_start"], end=window["train_end"])
        test = _date_slice(dataset, start=window["test_start"], end=window["test_end"])
        if progress:
            print(
                f"Sector Phase9 OOS window {window_index + 1}/{len(windows)}: "
                f"train {window['train_start']}->{window['train_end']}, "
                f"test {window['test_start']}->{window['test_end']}"
            )
        label_count = train["phase9_label"].nunique(dropna=True)
        if len(train) < min_training_rows or test.empty or label_count < 2:
            window_rows.append(
                {
                    **window.to_dict(),
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "status": "skipped_one_class_or_too_small" if label_count < 2 else "skipped",
                }
            )
            continue
        model, used_device = fit_lgbm_with_device(
            LGBMClassifier,
            SECTOR_PHASE9_LGBM_PARAMS,
            train.loc[:, panel.feature_columns],
            train["phase9_label"].astype(int),
            device=lgbm_device,
            n_jobs=lgbm_n_jobs,
            gpu_platform_id=lgbm_gpu_platform_id,
            gpu_device_id=lgbm_gpu_device_id,
            fit_label=f"Sector Phase9 {window_id}",
        )
        scored = test.loc[
            :,
            [
                "trade_date",
                "sector_key",
                "sector_type",
                "sector_name",
                "sector_label",
                "member_count",
                "valid_ratio",
                "phase9_label",
                "future_return_20d_close",
                "future_max_return_20d",
                "future_min_return_20d",
            ],
        ].copy()
        scored["window_id"] = window_id
        scored["phase9_probability"] = _predict_positive_probability(model, test.loc[:, panel.feature_columns])
        scored["phase9_score_100"] = scored["phase9_probability"].mul(100.0)
        scored_parts.append(scored)
        window_rows.append(
            {
                **window.to_dict(),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "positive_rate_train": float(train["phase9_label"].mean()),
                "used_lgbm_device": used_device,
                "status": "ok",
            }
        )
    scored_frame = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    if scored_frame.empty:
        raise RuntimeError("Sector Phase9 validation produced no scored rows.")

    summary = summarize_sector_phase9_strategies(scored_frame, top_ns=top_ns)
    deciles = build_sector_phase9_decile_report(scored_frame)

    report_dir = output_dir if output_dir is not None else sector_performance_dir(project_root) / "phase9_buy_score_validation"
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
                "model_version": SECTOR_PHASE9_MODEL_VERSION,
                "label": f"20-trading-day close return >= {return_threshold:.4f}",
                "start_date": start_date.isoformat() if start_date else "",
                "test_start_date": test_start_date.isoformat(),
                "end_date": end_date.isoformat() if end_date else "",
                "history_days": int(history_days),
                "min_members": int(min_members),
                "horizon_days": int(horizon_days),
                "return_threshold": float(return_threshold),
                "top_ns": list(top_ns),
                "test_window_days": int(test_window_days),
                "step_days": int(step_days),
                "embargo_days": int(embargo_days),
                "min_train_days": int(min_train_days),
                "min_training_rows": int(min_training_rows),
                "max_windows": max_windows,
                "feature_count": int(len(panel.feature_columns)),
                "lightgbm_params": SECTOR_PHASE9_LGBM_PARAMS,
                "membership_note": "Uses the current local sector membership mapping for historical sector reconstruction.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return SectorPhase9ValidationResult(
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


def validate_sector_rule_buy_score(
    *,
    project_root: Path,
    start_date: date | None = None,
    test_start_date: date,
    end_date: date | None = None,
    history_days: int = 3200,
    min_members: int = 5,
    horizon_days: int = 20,
    return_threshold: float = 0.05,
    top_ns: tuple[int, ...] = (5, 10, 20, 50),
    test_window_days: int = 60,
    step_days: int = 60,
    embargo_days: int = 20,
    min_train_days: int = 900,
    max_windows: int | None = None,
    output_dir: Path | None = None,
    progress: bool = False,
) -> SectorRuleBuyScoreValidationResult:
    logging.info("Sector rule buy-score validation panel build started")
    panel = build_sector_phase9_panel(
        project_root=project_root,
        start_date=start_date,
        end_date=end_date,
        history_days=history_days,
        min_members=min_members,
        horizon_days=horizon_days,
        return_threshold=return_threshold,
        include_unlabeled=False,
        progress=progress,
    )
    dataset = add_sector_rule_buy_score_columns(panel.dataset)
    if dataset.empty:
        raise RuntimeError("Sector rule buy-score validation has no labeled rows.")
    windows = build_sector_phase9_walkforward_windows(
        dataset,
        test_start_date=test_start_date,
        test_window_days=test_window_days,
        step_days=step_days,
        embargo_days=embargo_days,
        min_train_days=min_train_days,
        max_windows=max_windows,
    )
    if windows.empty:
        raise RuntimeError("No Sector rule buy-score validation windows can be built from the requested date range.")

    scored_parts: list[pd.DataFrame] = []
    window_rows: list[dict[str, object]] = []
    for window_index, window in windows.reset_index(drop=True).iterrows():
        train = _date_slice(dataset, start=window["train_start"], end=window["train_end"])
        test = _date_slice(dataset, start=window["test_start"], end=window["test_end"])
        if progress:
            print(
                f"Sector rule buy-score window {window_index + 1}/{len(windows)}: "
                f"train {window['train_start']}->{window['train_end']}, "
                f"test {window['test_start']}->{window['test_end']}"
            )
        if test.empty:
            window_rows.append({**window.to_dict(), "train_rows": int(len(train)), "test_rows": 0, "status": "skipped"})
            continue
        scored = test.loc[
            :,
            [
                "trade_date",
                "sector_key",
                "sector_type",
                "sector_name",
                "sector_label",
                "member_count",
                "valid_ratio",
                "phase9_label",
                "future_return_20d_close",
                "future_max_return_20d",
                "future_min_return_20d",
                "rule_buy_score",
                "rule_pullback_depth_score",
                "rule_pullback_timing_score",
                "rule_stabilization_score",
                "rule_rebound_confirmation_score",
                "rule_risk_control_score",
            ],
        ].copy()
        scored["window_id"] = str(window["window_id"])
        scored_parts.append(scored)
        window_rows.append(
            {
                **window.to_dict(),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "status": "ok",
            }
        )

    scored_frame = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    if scored_frame.empty:
        raise RuntimeError("Sector rule buy-score validation produced no scored rows.")
    summary = summarize_sector_score_strategies(scored_frame, score_column="rule_buy_score", score_prefix="rule_buy_score", top_ns=top_ns)
    deciles = build_sector_score_decile_report(scored_frame, score_column="rule_buy_score")

    report_dir = output_dir if output_dir is not None else sector_performance_dir(project_root) / "rule_buy_score_validation"
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
                "label": f"20-trading-day close return >= {return_threshold:.4f}",
                "rule": "Existing pullback/stabilization buy_score formula, computed with trailing rolling peaks to avoid future leakage.",
                "start_date": start_date.isoformat() if start_date else "",
                "test_start_date": test_start_date.isoformat(),
                "end_date": end_date.isoformat() if end_date else "",
                "history_days": int(history_days),
                "min_members": int(min_members),
                "horizon_days": int(horizon_days),
                "return_threshold": float(return_threshold),
                "top_ns": list(top_ns),
                "test_window_days": int(test_window_days),
                "step_days": int(step_days),
                "embargo_days": int(embargo_days),
                "min_train_days": int(min_train_days),
                "max_windows": max_windows,
                "membership_note": "Uses the current local sector membership mapping for historical sector reconstruction.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return SectorRuleBuyScoreValidationResult(
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


def train_sector_phase9_buy_score_model(
    *,
    project_root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    history_days: int = 3200,
    min_members: int = 5,
    horizon_days: int = 20,
    return_threshold: float = 0.05,
    min_training_rows: int = 200,
    lgbm_device: str = "cpu",
    lgbm_n_jobs: int | None = 1,
    lgbm_gpu_platform_id: int | None = None,
    lgbm_gpu_device_id: int | None = None,
    progress: bool = False,
) -> SectorPhase9TrainResult:
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm is required for Sector Phase9 training.")
    panel = build_sector_phase9_panel(
        project_root=project_root,
        start_date=start_date,
        end_date=end_date,
        history_days=history_days,
        min_members=min_members,
        horizon_days=horizon_days,
        return_threshold=return_threshold,
        include_unlabeled=False,
        progress=progress,
    )
    dataset = panel.dataset
    if len(dataset) < min_training_rows:
        raise RuntimeError(f"Insufficient Sector Phase9 training rows: {len(dataset)}")
    if dataset["phase9_label"].nunique(dropna=True) < 2:
        raise RuntimeError("Sector Phase9 training labels contain only one class.")
    if not panel.feature_columns:
        raise RuntimeError("Sector Phase9 training has no feature columns.")

    logging.info("Sector Phase9 deployment model fit started: rows=%s features=%s", len(dataset), len(panel.feature_columns))
    fitted, used_device = fit_lgbm_with_device(
        LGBMClassifier,
        SECTOR_PHASE9_LGBM_PARAMS,
        dataset.loc[:, panel.feature_columns],
        dataset["phase9_label"].astype(int),
        device=lgbm_device,
        n_jobs=lgbm_n_jobs,
        gpu_platform_id=lgbm_gpu_platform_id,
        gpu_device_id=lgbm_gpu_device_id,
        fit_label="Sector Phase9",
    )

    trade_dates = pd.to_datetime(dataset["trade_date"], errors="coerce").dropna()
    train_start = trade_dates.min().date().isoformat()
    train_end = trade_dates.max().date().isoformat()
    artifact = {
        "model_version": SECTOR_PHASE9_MODEL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": "lightgbm_classifier",
        "model": fitted,
        "feature_columns": tuple(panel.feature_columns),
        "label": f"phase9_label = 1 if sector close return at t+{horizon_days} >= {return_threshold:.4f}",
        "label_config": {
            "horizon_days": int(horizon_days),
            "return_threshold": float(return_threshold),
            "target_price": "20th future trading day's sector close index",
        },
        "lightgbm_params": SECTOR_PHASE9_LGBM_PARAMS,
        "used_lgbm_device": used_device,
        "train_config": {
            "start_date": start_date.isoformat() if start_date else "",
            "end_date": end_date.isoformat() if end_date else "",
            "history_days": int(history_days),
            "min_members": int(min_members),
            "min_training_rows": int(min_training_rows),
        },
        "train_rows": int(len(dataset)),
        "train_start": train_start,
        "train_end": train_end,
        "feature_count": int(len(panel.feature_columns)),
        "symbol_count": int(panel.symbol_count),
        "sector_count": int(panel.sector_count),
        "membership_note": "Uses the current local sector membership mapping for historical sector reconstruction.",
    }
    model_path, metadata_path = save_sector_phase9_model_artifact(project_root, artifact)
    logging.info("Sector Phase9 deployment model saved: %s", model_path)
    return SectorPhase9TrainResult(
        model_path=model_path,
        metadata_path=metadata_path,
        train_rows=int(len(dataset)),
        train_start=train_start,
        train_end=train_end,
        feature_columns=tuple(panel.feature_columns),
        used_lgbm_device=used_device,
    )


def predict_sector_phase9_buy_score(
    *,
    project_root: Path,
    trade_date: date,
    output: Path | None = None,
    history_days: int = 3200,
    min_members: int = 5,
    top_n: int = 30,
    progress: bool = False,
) -> SectorPhase9PredictionResult:
    artifact = load_sector_phase9_model_artifact(project_root)
    feature_columns = tuple(artifact["feature_columns"])
    label_config = artifact.get("label_config", {})
    horizon_days = int(label_config.get("horizon_days", 20))
    return_threshold = float(label_config.get("return_threshold", 0.05))
    panel = build_sector_phase9_panel(
        project_root=project_root,
        trade_date=trade_date,
        history_days=history_days,
        min_members=min_members,
        horizon_days=horizon_days,
        return_threshold=return_threshold,
        include_unlabeled=True,
        progress=progress,
    )
    dataset = panel.dataset
    if dataset.empty:
        predictions = pd.DataFrame()
    else:
        latest = _latest_sector_rows(dataset, trade_date=trade_date).copy()
        missing_features = [column for column in feature_columns if column not in latest.columns]
        if missing_features:
            for column in missing_features:
                latest[column] = np.nan
        latest["phase9_probability"] = _predict_positive_probability(artifact["model"], latest.loc[:, feature_columns])
        latest["phase9_score_100"] = latest["phase9_probability"].mul(100.0)
        latest = latest.sort_values(["phase9_probability", "sector_type", "sector_name"], ascending=[False, True, True]).reset_index(drop=True)
        latest["phase9_rank"] = range(1, len(latest) + 1)
        predictions = latest.loc[
            :,
            [
                "trade_date",
                "sector_type",
                "sector_name",
                "sector_label",
                "member_count",
                "valid_ratio",
                "phase9_score_100",
                "phase9_probability",
                "phase9_rank",
                "return_5d",
                "return_20d",
                "drawdown_from_peak_120d_pct",
                "ma5_slope_pct",
                "ma10_slope_pct",
                "ma20_slope_pct",
                "no_new_20d_low_in_5d",
                "no_new_60d_low_in_5d",
                *[column for column in feature_columns if column not in {
                    "member_count",
                    "valid_ratio",
                    "return_5d",
                    "return_20d",
                    "drawdown_from_peak_120d_pct",
                    "ma5_slope_pct",
                    "ma10_slope_pct",
                    "ma20_slope_pct",
                    "no_new_20d_low_in_5d",
                    "no_new_60d_low_in_5d",
                }],
            ],
        ]
    output_path = output if output is not None else sector_phase9_predictions_path(project_root, trade_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
    return SectorPhase9PredictionResult(
        predictions=predictions,
        output_path=output_path,
        artifact_path=sector_phase9_model_path(project_root),
    )


def build_sector_phase9_walkforward_windows(
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


def summarize_sector_phase9_strategies(scored: pd.DataFrame, *, top_ns: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_days = int(scored["trade_date"].nunique()) if not scored.empty else 0
    for top_n in top_ns:
        selected = _select_top_n_by_day(scored, top_n=top_n)
        rows.append(_summarize_sector_phase9_selection(selected, strategy=f"phase9_top{top_n}", top_n=top_n, source_days=source_days))
        random_selected = _select_random_top_n_by_day(scored, top_n=top_n)
        rows.append(_summarize_sector_phase9_selection(random_selected, strategy=f"random_top{top_n}", top_n=top_n, source_days=source_days))
    rows.append(_summarize_sector_phase9_selection(scored, strategy="all_sectors", top_n=0, source_days=source_days))
    return pd.DataFrame(rows)


def summarize_sector_score_strategies(
    scored: pd.DataFrame,
    *,
    score_column: str,
    score_prefix: str,
    top_ns: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_days = int(scored["trade_date"].nunique()) if not scored.empty else 0
    for top_n in top_ns:
        selected = _select_top_n_by_day_for_score(scored, top_n=top_n, score_column=score_column)
        rows.append(_summarize_sector_phase9_selection(selected, strategy=f"{score_prefix}_top{top_n}", top_n=top_n, source_days=source_days))
        random_selected = _select_random_top_n_by_day(scored, top_n=top_n)
        rows.append(_summarize_sector_phase9_selection(random_selected, strategy=f"random_top{top_n}", top_n=top_n, source_days=source_days))
    rows.append(_summarize_sector_phase9_selection(scored, strategy="all_sectors", top_n=0, source_days=source_days))
    return pd.DataFrame(rows)


def build_sector_phase9_decile_report(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    frame = scored.copy()
    frame["score_pct"] = frame.groupby("trade_date")["phase9_probability"].rank(pct=True, method="first")
    frame["score_decile"] = np.ceil(frame["score_pct"].mul(10)).sub(1).clip(0, 9).astype(int)
    rows = []
    for decile, group in frame.groupby("score_decile", sort=True):
        rows.append(_summarize_sector_phase9_selection(group, strategy=f"decile_{int(decile)}", top_n=0, source_days=group["trade_date"].nunique()))
    return pd.DataFrame(rows)


def build_sector_score_decile_report(scored: pd.DataFrame, *, score_column: str) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    frame = scored.copy()
    frame["score_pct"] = frame.groupby("trade_date")[score_column].rank(pct=True, method="first")
    frame["score_decile"] = np.ceil(frame["score_pct"].mul(10)).sub(1).clip(0, 9).astype(int)
    rows = []
    for decile, group in frame.groupby("score_decile", sort=True):
        rows.append(_summarize_sector_phase9_selection(group, strategy=f"decile_{int(decile)}", top_n=0, source_days=group["trade_date"].nunique()))
    return pd.DataFrame(rows)


def sector_phase9_model_dir(project_root: Path) -> Path:
    return project_root / "data" / "ml" / "sector_phase9"


def sector_phase9_model_path(project_root: Path) -> Path:
    return sector_phase9_model_dir(project_root) / "sector_phase9_buy_score_model.pkl"


def sector_phase9_metadata_path(project_root: Path) -> Path:
    return sector_phase9_model_dir(project_root) / "sector_phase9_buy_score_model_metadata.json"


def save_sector_phase9_model_artifact(project_root: Path, artifact: dict[str, Any]) -> tuple[Path, Path]:
    model_dir = sector_phase9_model_dir(project_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = sector_phase9_model_path(project_root)
    metadata_path = sector_phase9_metadata_path(project_root)
    with model_path.open("wb") as file:
        pickle.dump(artifact, file)
    metadata = {key: value for key, value in artifact.items() if key != "model"}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_path, metadata_path


def load_sector_phase9_model_artifact(project_root: Path) -> dict[str, Any]:
    model_path = sector_phase9_model_path(project_root)
    if not model_path.exists():
        raise FileNotFoundError(f"Sector Phase9 model artifact not found: {model_path}")
    with model_path.open("rb") as file:
        return pickle.load(file)


def sector_phase9_predictions_path(project_root: Path, trade_date: date) -> Path:
    return sector_performance_dir(project_root) / f"sector_phase9_buy_score_predictions_{trade_date.isoformat()}.csv"


def format_sector_phase9_prediction_table(predictions: pd.DataFrame, *, top_n: int = 30) -> str:
    if predictions.empty:
        return "No Sector Phase9 predictions."
    columns = [
        "trade_date",
        "sector_type",
        "sector_name",
        "phase9_score_100",
        "phase9_probability",
        "phase9_rank",
        "return_5d",
        "return_20d",
        "drawdown_from_peak_120d_pct",
        "ma5_slope_pct",
        "ma20_slope_pct",
    ]
    available = [column for column in columns if column in predictions.columns]
    frame = predictions.loc[:, available].head(max(int(top_n), 0)).copy()
    for column in frame.columns:
        if column in {"trade_date", "sector_type", "sector_name"}:
            continue
        frame[column] = frame[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
    return frame.to_string(index=False)


def _build_sector_phase9_feature_frame(
    *,
    sector_key: str,
    sector_type: str,
    sector_name: str,
    sector_label: str,
    member_count: int,
    sector_index: pd.Series,
    sector_ret: pd.Series,
    sector_amount: pd.Series,
    valid_count: pd.Series,
    benchmark_index: pd.Series,
    horizon_days: int,
    return_threshold: float,
    min_feature_history_days: int,
    include_unlabeled: bool,
) -> pd.DataFrame:
    index = pd.to_numeric(sector_index, errors="coerce").astype(float)
    ret = pd.to_numeric(sector_ret, errors="coerce").astype(float)
    amount = pd.to_numeric(sector_amount, errors="coerce").astype(float)
    valid = pd.to_numeric(valid_count, errors="coerce").astype(float)
    benchmark = pd.to_numeric(benchmark_index.reindex(index.index), errors="coerce").astype(float)
    frame = pd.DataFrame(index=index.index)
    frame["trade_date"] = pd.to_datetime(index.index)
    frame["sector_key"] = sector_key
    frame["sector_type"] = sector_type
    frame["sector_name"] = sector_name
    frame["sector_label"] = sector_label
    frame["sector_type_code"] = 1.0 if sector_type == "concept" else 0.0
    frame["member_count"] = int(member_count)
    frame["latest_index"] = index
    frame["valid_count"] = valid
    frame["valid_ratio"] = valid.div(float(member_count)).clip(lower=0.0, upper=1.0) if member_count > 0 else np.nan

    for days in (1, 3, 5, 10, 20, 60, 120, 252, 504):
        frame[f"return_{days}d"] = index.pct_change(days).mul(100.0)
        frame[f"benchmark_return_{days}d"] = benchmark.pct_change(days).mul(100.0)
        frame[f"excess_return_{days}d"] = frame[f"return_{days}d"] - frame[f"benchmark_return_{days}d"]

    for window in (5, 10, 20, 60, 120, 240):
        moving_average = index.rolling(window=window, min_periods=window).mean()
        frame[f"close_above_ma{window}"] = index.gt(moving_average).astype("float32")
        frame[f"ma{window}_slope_pct"] = moving_average.div(moving_average.shift(5)).sub(1.0).mul(100.0)
        frame[f"distance_to_ma{window}_pct"] = index.div(moving_average).sub(1.0).mul(100.0)

    rolling_peak_120 = index.rolling(window=120, min_periods=min(20, max(len(index), 1))).max()
    rolling_peak_252 = index.rolling(window=252, min_periods=min(60, max(len(index), 1))).max()
    frame["drawdown_from_peak_120d_pct"] = index.div(rolling_peak_120).sub(1.0).mul(100.0)
    frame["drawdown_from_peak_252d_pct"] = index.div(rolling_peak_252).sub(1.0).mul(100.0)
    frame["days_since_peak_120d"] = _days_since_rolling_peak(index, window=120, min_periods=20)
    frame["days_since_peak_252d"] = _days_since_rolling_peak(index, window=252, min_periods=60)

    rolling_low_20 = index.rolling(window=20, min_periods=20).min()
    rolling_low_60 = index.rolling(window=60, min_periods=60).min()
    frame["distance_to_recent_low_20d_pct"] = index.div(rolling_low_20).sub(1.0).mul(100.0)
    frame["distance_to_recent_low_60d_pct"] = index.div(rolling_low_60).sub(1.0).mul(100.0)
    frame["no_new_20d_low_in_5d"] = _no_new_stage_low_vector(index, recent_days=5, low_window_days=20)
    frame["no_new_60d_low_in_5d"] = _no_new_stage_low_vector(index, recent_days=5, low_window_days=60)
    frame["no_new_20d_low_in_10d"] = _no_new_stage_low_vector(index, recent_days=10, low_window_days=20)

    frame["volatility_5d"] = ret.rolling(window=5, min_periods=5).std()
    frame["volatility_20d"] = ret.rolling(window=20, min_periods=20).std()
    frame["volatility_ratio_5d_20d"] = frame["volatility_5d"].div(frame["volatility_20d"])
    frame["up_ratio_3d"] = ret.gt(0).rolling(window=3, min_periods=3).mean()
    frame["up_ratio_5d"] = ret.gt(0).rolling(window=5, min_periods=5).mean()
    frame["up_ratio_20d"] = ret.gt(0).rolling(window=20, min_periods=20).mean()
    frame["outperform_ratio_20d"] = ret.gt(benchmark.pct_change().mul(100.0)).rolling(window=20, min_periods=20).mean()
    frame["outperform_ratio_60d"] = ret.gt(benchmark.pct_change().mul(100.0)).rolling(window=60, min_periods=60).mean()
    frame["outperform_ratio_252d"] = ret.gt(benchmark.pct_change().mul(100.0)).rolling(window=252, min_periods=120).mean()

    amount_ma5 = amount.rolling(window=5, min_periods=5).mean()
    amount_ma20 = amount.rolling(window=20, min_periods=20).mean()
    amount_ma60 = amount.rolling(window=60, min_periods=60).mean()
    frame["amount_ratio_5d_20d"] = amount_ma5.div(amount_ma20)
    frame["amount_ratio_5d_60d"] = amount_ma5.div(amount_ma60)
    down_amount = amount.where(ret.lt(0), 0.0).rolling(window=10, min_periods=10).mean()
    base_amount = amount.rolling(window=10, min_periods=10).mean()
    frame["down_amount_ratio_10d"] = down_amount.div(base_amount)
    frame["recent_down_speed_5d"] = frame["return_5d"]
    _add_long_mainline_features(frame)

    future_return = index.shift(-horizon_days).div(index).sub(1.0)
    future_returns = [index.shift(-offset).div(index).sub(1.0) for offset in range(1, horizon_days + 1)]
    future_frame = pd.concat(future_returns, axis=1)
    frame["future_return_20d_close"] = future_return
    frame["future_max_return_20d"] = future_frame.max(axis=1)
    frame["future_min_return_20d"] = future_frame.min(axis=1)
    frame["phase9_label"] = future_return.ge(float(return_threshold)).astype("float32")
    frame.loc[future_return.isna(), "phase9_label"] = np.nan

    if min_feature_history_days > 0:
        frame = frame.iloc[min_feature_history_days:].copy()
    if not include_unlabeled:
        frame = frame.dropna(subset=["future_return_20d_close", "phase9_label"]).copy()
    return frame.reset_index(drop=True)


def add_sector_rule_buy_score_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    rows = [
        _compute_rule_buy_score_row(
            sector_type=sector_type,
            drawdown_from_peak_pct=drawdown,
            days_since_peak=days_since_peak,
            no_new_20d_low_in_5d=no_new_20d,
            no_new_60d_low_in_5d=no_new_60d,
            no_new_20d_low_in_10d=no_new_20d_10,
            ma5_slope=ma5,
            ma10_slope=ma10,
            ma20_slope=ma20,
            ma60_slope=ma60,
            close_above_ma5=above5,
            close_above_ma10=above10,
            close_above_ma20=above20,
            volatility_ratio=volatility_ratio,
            return_3d=return_3d,
            return_5d=return_5d,
            excess_return_3d=excess_return_3d,
            up_ratio_3d=up_ratio_3d,
            distance_to_recent_low=distance_to_low,
            down_amount_ratio=down_amount_ratio,
        )
        for (
            sector_type,
            drawdown,
            days_since_peak,
            no_new_20d,
            no_new_60d,
            no_new_20d_10,
            ma5,
            ma10,
            ma20,
            ma60,
            above5,
            above10,
            above20,
            volatility_ratio,
            return_3d,
            return_5d,
            excess_return_3d,
            up_ratio_3d,
            distance_to_low,
            down_amount_ratio,
        ) in zip(
            result.get("sector_type", pd.Series("", index=result.index)),
            result.get("drawdown_from_peak_120d_pct", pd.Series(np.nan, index=result.index)),
            result.get("days_since_peak_120d", pd.Series(np.nan, index=result.index)),
            result.get("no_new_20d_low_in_5d", pd.Series(np.nan, index=result.index)),
            result.get("no_new_60d_low_in_5d", pd.Series(np.nan, index=result.index)),
            result.get("no_new_20d_low_in_10d", pd.Series(np.nan, index=result.index)),
            result.get("ma5_slope_pct", pd.Series(np.nan, index=result.index)),
            result.get("ma10_slope_pct", pd.Series(np.nan, index=result.index)),
            result.get("ma20_slope_pct", pd.Series(np.nan, index=result.index)),
            result.get("ma60_slope_pct", pd.Series(np.nan, index=result.index)),
            result.get("close_above_ma5", pd.Series(np.nan, index=result.index)),
            result.get("close_above_ma10", pd.Series(np.nan, index=result.index)),
            result.get("close_above_ma20", pd.Series(np.nan, index=result.index)),
            result.get("volatility_ratio_5d_20d", pd.Series(np.nan, index=result.index)),
            result.get("return_3d", pd.Series(np.nan, index=result.index)),
            result.get("return_5d", pd.Series(np.nan, index=result.index)),
            result.get("excess_return_3d", pd.Series(np.nan, index=result.index)),
            result.get("up_ratio_3d", pd.Series(np.nan, index=result.index)),
            result.get("distance_to_recent_low_20d_pct", pd.Series(np.nan, index=result.index)),
            result.get("down_amount_ratio_10d", pd.Series(np.nan, index=result.index)),
        )
    ]
    score_frame = pd.DataFrame(rows, index=result.index)
    return pd.concat([result, score_frame], axis=1)


def filter_long_mainline_rows(
    frame: pd.DataFrame,
    *,
    score_threshold: float = 70.0,
    top_pct: float = 0.20,
) -> pd.DataFrame:
    if frame.empty or "long_mainline_score" not in frame.columns:
        return frame.head(0).copy()
    result = frame.copy()
    score = pd.to_numeric(result["long_mainline_score"], errors="coerce")
    keep = score.ge(float(score_threshold))
    if top_pct > 0:
        top_pct_clipped = max(0.0, min(1.0, float(top_pct)))
        rank_pct = score.groupby(result["trade_date"]).rank(pct=True, method="first")
        keep = keep | rank_pct.ge(1.0 - top_pct_clipped)
    result = result.loc[keep.fillna(False)].copy()
    return result.reset_index(drop=True)


def _add_long_mainline_features(frame: pd.DataFrame) -> None:
    excess_2y = pd.to_numeric(frame.get("excess_return_504d"), errors="coerce") if "excess_return_504d" in frame.columns else pd.Series(np.nan, index=frame.index)
    excess_1y = pd.to_numeric(frame.get("excess_return_252d"), errors="coerce") if "excess_return_252d" in frame.columns else pd.Series(np.nan, index=frame.index)
    outperform_1y = pd.to_numeric(frame.get("outperform_ratio_252d"), errors="coerce")
    ma60_slope = pd.to_numeric(frame.get("ma60_slope_pct"), errors="coerce")
    ma120_slope = pd.to_numeric(frame.get("ma120_slope_pct"), errors="coerce")
    distance_ma60 = pd.to_numeric(frame.get("distance_to_ma60_pct"), errors="coerce")
    distance_ma120 = pd.to_numeric(frame.get("distance_to_ma120_pct"), errors="coerce")
    drawdown_252 = pd.to_numeric(frame.get("drawdown_from_peak_252d_pct"), errors="coerce")
    volatility_ratio = pd.to_numeric(frame.get("volatility_ratio_5d_20d"), errors="coerce")

    frame["long_mainline_excess_2y_score"] = excess_2y.map(lambda value: _linear_score_nan(value, low=-10.0, high=60.0))
    frame["long_mainline_excess_1y_score"] = excess_1y.map(lambda value: _linear_score_nan(value, low=-5.0, high=40.0))
    frame["long_mainline_outperform_score"] = outperform_1y.map(lambda value: _linear_score_nan(value, low=0.45, high=0.60))
    trend_raw = (
        0.35 * ma60_slope.map(lambda value: _linear_score_nan(value, low=-2.0, high=2.0))
        + 0.35 * ma120_slope.map(lambda value: _linear_score_nan(value, low=-3.0, high=3.0))
        + 0.15 * distance_ma60.map(lambda value: _linear_score_nan(value, low=-8.0, high=8.0))
        + 0.15 * distance_ma120.map(lambda value: _linear_score_nan(value, low=-12.0, high=12.0))
    )
    frame["long_mainline_trend_score"] = trend_raw
    drawdown_score = drawdown_252.map(_score_long_mainline_drawdown)
    volatility_score = volatility_ratio.map(_score_long_mainline_volatility)
    frame["long_mainline_resilience_score"] = 0.70 * drawdown_score + 0.30 * volatility_score
    frame["long_mainline_score"] = (
        0.25 * frame["long_mainline_excess_2y_score"]
        + 0.20 * frame["long_mainline_excess_1y_score"]
        + 0.20 * frame["long_mainline_outperform_score"]
        + 0.20 * frame["long_mainline_trend_score"]
        + 0.15 * frame["long_mainline_resilience_score"]
    ).clip(lower=0.0, upper=100.0)


def _linear_score_nan(value: object, *, low: float, high: float) -> float:
    if pd.isna(value):
        return 50.0
    if high == low:
        return 50.0
    ratio = (float(value) - low) / (high - low)
    return float(max(0.0, min(100.0, ratio * 100.0)))


def _score_long_mainline_drawdown(value: object) -> float:
    if pd.isna(value):
        return 50.0
    drawdown = float(value)
    if drawdown >= -8.0:
        return 100.0
    if drawdown >= -20.0:
        return _linear_score(drawdown, -20.0, -8.0, 60.0, 100.0)
    if drawdown >= -35.0:
        return _linear_score(drawdown, -35.0, -20.0, 25.0, 60.0)
    return 15.0


def _score_long_mainline_volatility(value: object) -> float:
    if pd.isna(value):
        return 50.0
    ratio = float(value)
    if ratio <= 0.9:
        return 100.0
    if ratio <= 1.3:
        return _linear_score(ratio, 0.9, 1.3, 100.0, 55.0)
    if ratio <= 1.8:
        return _linear_score(ratio, 1.3, 1.8, 55.0, 25.0)
    return 15.0


def _compute_rule_buy_score_row(
    *,
    sector_type: object,
    drawdown_from_peak_pct: object,
    days_since_peak: object,
    no_new_20d_low_in_5d: object,
    no_new_60d_low_in_5d: object,
    no_new_20d_low_in_10d: object,
    ma5_slope: object,
    ma10_slope: object,
    ma20_slope: object,
    ma60_slope: object,
    close_above_ma5: object,
    close_above_ma10: object,
    close_above_ma20: object,
    volatility_ratio: object,
    return_3d: object,
    return_5d: object,
    excess_return_3d: object,
    up_ratio_3d: object,
    distance_to_recent_low: object,
    down_amount_ratio: object,
) -> dict[str, float]:
    drawdown = _as_float_or_na(drawdown_from_peak_pct)
    days = _as_float_or_na(days_since_peak)
    days_int = int(days) if pd.notna(days) else 0
    pullback_depth_score = _score_pullback_depth(sector_type=str(sector_type), drawdown_from_peak_pct=drawdown)
    pullback_timing_score = _score_pullback_timing(days_int)
    stabilization_score = _score_stabilization(
        no_new_20d_low_in_5d=_bool_like(no_new_20d_low_in_5d),
        no_new_60d_low_in_5d=_bool_like(no_new_60d_low_in_5d),
        no_new_20d_low_in_10d=_bool_like(no_new_20d_low_in_10d),
        ma5_slope=_as_float_or_na(ma5_slope),
        ma10_slope=_as_float_or_na(ma10_slope),
        ma20_slope=_as_float_or_na(ma20_slope),
        ma60_slope=_as_float_or_na(ma60_slope),
        close_above_ma5=_bool_like(close_above_ma5),
        close_above_ma10=_bool_like(close_above_ma10),
        close_above_ma20=_bool_like(close_above_ma20),
        volatility_ratio=_as_float_or_na(volatility_ratio),
    )
    rebound_confirmation_score = _score_rebound_confirmation(
        return_3d=_as_float_or_na(return_3d),
        return_5d=_as_float_or_na(return_5d),
        excess_return_3d=_as_float_or_na(excess_return_3d),
        up_ratio_3d=_as_float_or_na(up_ratio_3d),
    )
    risk_control_score = _score_risk_control(
        distance_to_recent_low=_as_float_or_na(distance_to_recent_low),
        return_5d=_as_float_or_na(return_5d),
        down_amount_ratio=_as_float_or_na(down_amount_ratio),
        ma20_slope=_as_float_or_na(ma20_slope),
    )
    raw_score = (
        0.15 * pullback_depth_score
        + 0.10 * pullback_timing_score
        + 0.40 * stabilization_score
        + 0.20 * rebound_confirmation_score
        + 0.15 * risk_control_score
    )
    capped_score, _ = _apply_buy_score_caps(
        score=raw_score,
        no_new_20d_low_in_5d=_bool_like(no_new_20d_low_in_5d),
        no_new_60d_low_in_5d=_bool_like(no_new_60d_low_in_5d),
        drawdown_from_peak_pct=drawdown,
        return_3d=_as_float_or_na(return_3d),
        return_5d=_as_float_or_na(return_5d),
        volatility_ratio=_as_float_or_na(volatility_ratio),
        down_amount_ratio=_as_float_or_na(down_amount_ratio),
        ma5_slope=_as_float_or_na(ma5_slope),
        ma10_slope=_as_float_or_na(ma10_slope),
        ma20_slope=_as_float_or_na(ma20_slope),
    )
    return {
        "rule_buy_score": float(capped_score),
        "rule_pullback_depth_score": float(pullback_depth_score),
        "rule_pullback_timing_score": float(pullback_timing_score),
        "rule_stabilization_score": float(stabilization_score),
        "rule_rebound_confirmation_score": float(rebound_confirmation_score),
        "rule_risk_control_score": float(risk_control_score),
    }


def _as_float_or_na(value: object) -> float:
    if pd.isna(value):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _bool_like(value: object) -> object:
    numeric = _as_float_or_na(value)
    if math.isnan(numeric):
        return pd.NA
    return bool(numeric >= 0.5)


def _days_since_rolling_peak(series: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        segment = values[start : index + 1]
        valid = np.isfinite(segment)
        if valid.sum() < min_periods:
            continue
        valid_indices = np.flatnonzero(valid)
        segment_values = segment[valid_indices]
        if segment_values.size == 0:
            continue
        latest_peak_relative = valid_indices[int(np.nanargmax(segment_values))]
        result[index] = float(index - (start + latest_peak_relative))
    return pd.Series(result, index=series.index)


def _no_new_stage_low_vector(series: pd.Series, *, recent_days: int, low_window_days: int) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    rolling_low = clean.rolling(window=low_window_days, min_periods=low_window_days).min()
    new_low = clean.le(rolling_low.mul(1.002))
    recent_new_low = new_low.rolling(window=recent_days, min_periods=recent_days).max()
    return recent_new_low.eq(0).astype("float32").where(recent_new_low.notna(), np.nan)


def _predict_positive_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        if isinstance(proba, list):
            proba = proba[0]
        array = np.asarray(proba)
        if array.ndim == 2 and array.shape[1] >= 2:
            return array[:, 1].astype(float)
        if array.ndim == 1:
            return array.astype(float)
    predictions = model.predict(features)
    return np.asarray(predictions, dtype=float)


def _latest_sector_rows(dataset: pd.DataFrame, *, trade_date: date) -> pd.DataFrame:
    valid = dataset[pd.to_datetime(dataset["trade_date"], errors="coerce").dt.date <= trade_date].copy()
    if valid.empty:
        return pd.DataFrame()
    valid = valid.sort_values(["sector_key", "trade_date"]).groupby("sector_key", as_index=False, sort=False).tail(1)
    return valid.reset_index(drop=True)


def _select_top_n_by_day(scored: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    parts = []
    for _, group in scored.groupby("trade_date", sort=True):
        selected = group.dropna(subset=["phase9_probability"]).sort_values(
            ["phase9_probability", "sector_type", "sector_name"],
            ascending=[False, True, True],
        ).head(max(int(top_n), 0))
        if not selected.empty:
            parts.append(selected)
    return pd.concat(parts, ignore_index=True) if parts else scored.head(0).copy()


def _select_top_n_by_day_for_score(scored: pd.DataFrame, *, top_n: int, score_column: str) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    parts = []
    for _, group in scored.groupby("trade_date", sort=True):
        selected = group.dropna(subset=[score_column]).sort_values(
            [score_column, "sector_type", "sector_name"],
            ascending=[False, True, True],
        ).head(max(int(top_n), 0))
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
        seed = int(pd.Timestamp(trade_date).strftime("%Y%m%d")) + n * 1009
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True) if parts else scored.head(0).copy()


def _summarize_sector_phase9_selection(selected: pd.DataFrame, *, strategy: str, top_n: int, source_days: int) -> dict[str, object]:
    label = pd.to_numeric(selected.get("phase9_label"), errors="coerce")
    future_return = pd.to_numeric(selected.get("future_return_20d_close"), errors="coerce")
    max_return = pd.to_numeric(selected.get("future_max_return_20d"), errors="coerce")
    min_return = pd.to_numeric(selected.get("future_min_return_20d"), errors="coerce")
    active_days = int(selected["trade_date"].nunique()) if not selected.empty and "trade_date" in selected.columns else 0
    return {
        "strategy": strategy,
        "top_n": int(top_n),
        "days": int(source_days),
        "active_days": active_days,
        "trade_count": int(len(selected)),
        "no_candidate_day_rate": float(1.0 - active_days / source_days) if source_days else np.nan,
        "hit_20d_close_5pct_rate": _safe_mean(label),
        "avg_future_return_20d_close": _safe_mean(future_return),
        "median_future_return_20d_close": _safe_median(future_return),
        "win_20d_close_rate": float(future_return.gt(0).mean()) if future_return.notna().any() else np.nan,
        "avg_future_max_return_20d": _safe_mean(max_return),
        "avg_future_min_return_20d": _safe_mean(min_return),
    }


def _date_slice(frame: pd.DataFrame, *, start: date | None, end: date | None) -> pd.DataFrame:
    result = frame
    if start is not None:
        result = result[result["trade_date"].dt.date >= start]
    if end is not None:
        result = result[result["trade_date"].dt.date <= end]
    return result.copy()


def _downcast_sector_phase9_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    for column in result.columns:
        if column in SECTOR_PHASE9_IDENTIFIER_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("float32")
    return result


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else math.nan


def _safe_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else math.nan
