from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
import json
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - depends on optional local package.
    LGBMClassifier = None

from .full_market_labels import (
    BARRIER_RISK_FEATURE_COLUMNS,
    TAIL_RISK_FEATURE_COLUMNS,
    build_barrier_risk_panel,
    build_tail_risk_frame,
    build_tail_risk_panel,
    summarize_barrier_label_distribution,
)
from .full_market_panel import full_market_report_dir
from .storage import DailyBarsReadError, Storage
from .synthetic_market import build_synthetic_market_index, synthetic_market_path


TAIL_RISK_MODEL_VERSION = "tail_risk_phase1_v1"
NOH_INDEX_FEATURE_COLUMNS = ("log_return_1d",)
PANEL_KNN_MAX_ROWS = 50_000
ALL_TAIL_RISK_MODEL_NAMES = (
    "dummy_prior",
    "logistic_regression",
    "knn",
    "decision_tree",
    "random_forest",
    "linear_discriminant_analysis",
    "naive_bayes",
    "quadratic_discriminant_analysis",
    "adaboost",
    "gradient_boosting",
)
DEFAULT_PANEL_MODEL_NAMES = (
    "dummy_prior",
    "logistic_regression",
    "decision_tree",
    "linear_discriminant_analysis",
    "naive_bayes",
    "quadratic_discriminant_analysis",
)
DEFAULT_BARRIER_MODEL_NAMES = (
    "logistic_regression",
    "lightgbm_classifier",
)


@dataclass(slots=True)
class TailRiskReproductionResult:
    index_reproduction: pd.DataFrame
    dataset: pd.DataFrame
    skipped: pd.DataFrame
    metrics: pd.DataFrame
    deciles: pd.DataFrame
    report_dir: Path
    index_reproduction_path: Path
    index_dataset_path: Path
    dataset_path: Path
    skipped_path: Path
    metrics_path: Path
    deciles_path: Path
    summary_path: Path


@dataclass(slots=True)
class TailRiskWalkforwardResult:
    windows: pd.DataFrame
    metrics: pd.DataFrame
    deciles: pd.DataFrame
    filter_impact: pd.DataFrame
    filter_summary: pd.DataFrame
    summary: pd.DataFrame
    report_dir: Path
    windows_path: Path
    metrics_path: Path
    deciles_path: Path
    filter_impact_path: Path
    filter_summary_path: Path
    summary_path: Path
    config_path: Path


@dataclass(slots=True)
class TailRiskTrainResult:
    model_path: Path
    metadata_path: Path
    model_name: str
    train_rows: int
    train_start: str
    train_end: str
    feature_columns: tuple[str, ...]


@dataclass(slots=True)
class TailRiskPredictionResult:
    predictions: pd.DataFrame
    skipped: pd.DataFrame
    output_path: Path
    artifact_path: Path


@dataclass(slots=True)
class BarrierRiskValidationResult:
    label_distribution: pd.DataFrame
    metrics: pd.DataFrame
    deciles: pd.DataFrame
    filter_impact: pd.DataFrame
    filter_summary: pd.DataFrame
    comparison: pd.DataFrame
    report_dir: Path
    label_distribution_path: Path
    metrics_path: Path
    deciles_path: Path
    filter_impact_path: Path
    filter_summary_path: Path
    comparison_path: Path
    config_path: Path


@dataclass(slots=True)
class BarrierRiskGridResult:
    summary: pd.DataFrame
    label_distribution: pd.DataFrame
    summary_path: Path
    label_distribution_path: Path
    config_path: Path


def train_tail_risk_model(
    *,
    storage: Storage,
    project_root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    model_name: str = "logistic_regression",
    limit: int | None = None,
    lookback_days: int = 100,
    quantile: float = 0.05,
    horizon_days: int = 1,
    min_training_rows: int = 200,
) -> TailRiskTrainResult:
    logging.info("Tail-risk deployment training dataset build started")
    dataset, skipped = build_tail_risk_panel(
        storage=storage,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        lookback_days=lookback_days,
        quantile=quantile,
        horizon_days=horizon_days,
    )
    if len(dataset) < min_training_rows:
        raise RuntimeError(f"Insufficient tail-risk training rows: {len(dataset)}")
    if dataset["risk_label"].astype(int).nunique() < 2:
        raise RuntimeError("Tail-risk training requires both risk and non-risk labels.")

    models = _tail_risk_models((model_name,))
    model = models[model_name]
    X_train = dataset.loc[:, TAIL_RISK_FEATURE_COLUMNS]
    y_train = dataset["risk_label"].astype(int)
    logging.info("Tail-risk deployment model fit started: model=%s rows=%s", model_name, len(dataset))
    fitted = model.fit(X_train, y_train)

    trade_dates = pd.to_datetime(dataset["trade_date"], errors="coerce").dropna()
    train_start = trade_dates.min().date().isoformat()
    train_end = trade_dates.max().date().isoformat()
    artifact = {
        "model_version": TAIL_RISK_MODEL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "model": fitted,
        "feature_columns": tuple(TAIL_RISK_FEATURE_COLUMNS),
        "label_config": {
            "lookback_days": int(lookback_days),
            "quantile": float(quantile),
            "horizon_days": int(horizon_days),
        },
        "train_config": {
            "start_date": start_date.isoformat() if start_date else "",
            "end_date": end_date.isoformat() if end_date else "",
            "limit": limit,
            "min_training_rows": int(min_training_rows),
        },
        "train_rows": int(len(dataset)),
        "train_start": train_start,
        "train_end": train_end,
        "risk_rate": float(y_train.mean()),
        "skipped_symbols": int(len(skipped)),
    }
    model_path, metadata_path = save_tail_risk_model_artifact(project_root, artifact)
    logging.info("Tail-risk deployment model saved: %s", model_path)
    return TailRiskTrainResult(
        model_path=model_path,
        metadata_path=metadata_path,
        model_name=model_name,
        train_rows=int(len(dataset)),
        train_start=train_start,
        train_end=train_end,
        feature_columns=tuple(TAIL_RISK_FEATURE_COLUMNS),
    )


def predict_tail_risk(
    *,
    storage: Storage,
    project_root: Path,
    trade_date: date,
    output: Path | None = None,
    limit: int | None = None,
) -> TailRiskPredictionResult:
    artifact = load_tail_risk_model_artifact(project_root)
    label_config = artifact.get("label_config", {})
    feature_columns = tuple(artifact["feature_columns"])
    universe = storage.load_universe().copy()
    if limit is not None:
        universe = universe.head(max(int(limit), 0)).copy()

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, object]] = []
    instruments = universe.to_dict("records")
    total_symbols = len(instruments)
    for index, instrument in enumerate(instruments, start=1):
        _log_prediction_progress(index, total_symbols)
        symbol = str(instrument.get("symbol", "")).zfill(6)
        name = str(instrument.get("name", ""))
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError) as exc:
            skipped.append({"symbol": symbol, "name": name, "reason": type(exc).__name__})
            continue
        bars = bars.copy()
        bars["trade_date"] = pd.to_datetime(bars["trade_date"], errors="coerce")
        bars = bars.dropna(subset=["trade_date"])
        bars = bars[bars["trade_date"].dt.date <= trade_date].copy()
        frame = build_tail_risk_frame(
            bars,
            symbol=symbol,
            name=name,
            lookback_days=int(label_config.get("lookback_days", 100)),
            quantile=float(label_config.get("quantile", 0.05)),
            horizon_days=int(label_config.get("horizon_days", 1)),
        )
        if frame.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "empty_tail_risk_frame"})
            continue
        row = frame[frame["trade_date"].dt.date.eq(trade_date)]
        if row.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "no_trade_date"})
            continue
        row = row.tail(1).copy()
        row = row.dropna(subset=list(feature_columns))
        if row.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "no_feature_row"})
            continue
        risk_score = float(_predict_risk_proba(artifact["model"], row.loc[:, feature_columns])[0])
        record: dict[str, Any] = {
            "trade_date": trade_date.isoformat(),
            "symbol": symbol,
            "name": name,
            "risk_score": risk_score,
            "model_name": artifact["model_name"],
            "model_version": artifact["model_version"],
        }
        for column in feature_columns:
            record[column] = float(row.iloc[0][column])
        rows.append(record)

    predictions = pd.DataFrame(rows)
    if not predictions.empty:
        predictions = predictions.sort_values(["risk_score", "symbol"], ascending=[False, True]).reset_index(drop=True)
    skipped_frame = pd.DataFrame(skipped)
    output_path = output if output is not None else tail_risk_predictions_path(project_root, trade_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
    skipped_path = output_path.with_name(f"{output_path.stem}_skipped.csv")
    skipped_frame.to_csv(skipped_path, index=False, encoding="utf-8-sig")
    logging.info(
        "Tail-risk predictions saved: rows=%s skipped=%s output=%s",
        len(predictions),
        len(skipped_frame),
        output_path,
    )
    return TailRiskPredictionResult(
        predictions=predictions,
        skipped=skipped_frame,
        output_path=output_path,
        artifact_path=tail_risk_model_path(project_root),
    )


def tail_risk_model_dir(project_root: Path) -> Path:
    return project_root / "data" / "ml" / "full_market_risk"


def tail_risk_model_path(project_root: Path) -> Path:
    return tail_risk_model_dir(project_root) / "tail_risk_model.pkl"


def tail_risk_metadata_path(project_root: Path) -> Path:
    return tail_risk_model_dir(project_root) / "tail_risk_model_metadata.json"


def tail_risk_predictions_path(project_root: Path, trade_date: date) -> Path:
    return full_market_report_dir(project_root) / f"tail_risk_predictions_{trade_date.isoformat()}.csv"


def save_tail_risk_model_artifact(project_root: Path, artifact: dict[str, Any]) -> tuple[Path, Path]:
    model_dir = tail_risk_model_dir(project_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = tail_risk_model_path(project_root)
    metadata_path = tail_risk_metadata_path(project_root)
    with model_path.open("wb") as file:
        pickle.dump(artifact, file)
    metadata = {key: value for key, value in artifact.items() if key != "model"}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_path, metadata_path


def load_tail_risk_model_artifact(project_root: Path) -> dict[str, Any]:
    model_path = tail_risk_model_path(project_root)
    if not model_path.exists():
        raise FileNotFoundError(f"Tail-risk model artifact not found: {model_path}")
    with model_path.open("rb") as file:
        artifact = pickle.load(file)
    return artifact


def format_tail_risk_prediction_table(predictions: pd.DataFrame, *, top_n: int = 20) -> str:
    if predictions.empty:
        return "No tail-risk predictions."
    columns = ["trade_date", "symbol", "name", "risk_score", "model_name"]
    available = [column for column in columns if column in predictions.columns]
    frame = predictions.loc[:, available].head(max(int(top_n), 0)).copy()
    if "risk_score" in frame.columns:
        frame["risk_score"] = frame["risk_score"].map(lambda value: f"{float(value):.6f}")
    return frame.to_string(index=False)


def reproduce_tail_risk(
    *,
    storage: Storage,
    project_root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    train_end: date,
    valid_end: date,
    limit: int | None = None,
    lookback_days: int = 100,
    quantile: float = 0.05,
    horizon_days: int = 1,
    min_training_rows: int = 200,
    allow_short_sample: bool = False,
    index_source_column: str = "synthetic_equal_weight_index",
    run_index: bool = True,
    run_panel: bool = True,
    panel_model_names: tuple[str, ...] = DEFAULT_PANEL_MODEL_NAMES,
) -> TailRiskReproductionResult:
    if not run_index and not run_panel:
        raise ValueError("At least one of run_index or run_panel must be true.")

    report_dir = full_market_report_dir(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    index_reproduction_path = report_dir / "tail_risk_index_reproduction.csv"
    index_dataset_path = report_dir / "tail_risk_index_dataset.csv"
    dataset_path = report_dir / "tail_risk_dataset.csv"
    skipped_path = report_dir / "tail_risk_skipped.csv"
    metrics_path = report_dir / "tail_risk_panel_metrics.csv"
    deciles_path = report_dir / "tail_risk_decile_report.csv"
    summary_path = report_dir / "tail_risk_summary.json"

    index_dataset = pd.DataFrame()
    index_reproduction = pd.DataFrame()
    if run_index:
        logging.info("Tail-risk index dataset build started")
        index_dataset = build_tail_risk_index_dataset(
            storage=storage,
            project_root=project_root,
            start_date=start_date,
            end_date=end_date,
            lookback_days=lookback_days,
            quantile=quantile,
            horizon_days=horizon_days,
            source_column=index_source_column,
            min_stock_count=1 if allow_short_sample else 500,
        )
        logging.info("Tail-risk index dataset rows: %s", len(index_dataset))
        _assert_enough_index_history(index_dataset, lookback_days=lookback_days, allow_short_sample=allow_short_sample)
        logging.info("Tail-risk index model reproduction started")
        index_reproduction = reproduce_tail_risk_index_models(
            index_dataset,
            train_end=train_end,
            valid_end=valid_end,
            min_training_rows=min_training_rows,
        )
        index_dataset.to_csv(index_dataset_path, index=False, encoding="utf-8-sig")
        index_reproduction.to_csv(index_reproduction_path, index=False, encoding="utf-8-sig")
        logging.info("Tail-risk index reports saved: %s, %s", index_reproduction_path, index_dataset_path)

    dataset = pd.DataFrame()
    skipped = pd.DataFrame()
    metrics = pd.DataFrame()
    deciles = pd.DataFrame()
    if run_panel:
        logging.info("Tail-risk panel dataset build started")
        dataset, skipped = build_tail_risk_panel(
            storage=storage,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            lookback_days=lookback_days,
            quantile=quantile,
            horizon_days=horizon_days,
        )
        if not allow_short_sample and dataset["trade_date"].dt.date.nunique() < lookback_days + 252:
            raise RuntimeError(
                "Tail-risk panel reproduction needs more daily history. "
                "Run update from 20150101 or pass allow_short_sample only for smoke tests."
            )

        train = dataset[dataset["trade_date"].dt.date <= train_end].copy()
        valid = dataset[(dataset["trade_date"].dt.date > train_end) & (dataset["trade_date"].dt.date <= valid_end)].copy()
        logging.info("Tail-risk panel split rows: train=%s valid=%s", len(train), len(valid))
        if len(train) < min_training_rows or valid.empty:
            raise RuntimeError(f"Insufficient tail-risk panel split rows: train={len(train)} valid={len(valid)}")

        panel_result = _fit_score_risk_models(
            train=train,
            splits={"valid": valid},
            feature_columns=TAIL_RISK_FEATURE_COLUMNS,
            scope="panel",
            min_training_rows=min_training_rows,
            skip_large_knn=True,
            model_names=panel_model_names,
        )
        metrics = panel_result["metrics"]
        scored_valid = panel_result["scored"]
        deciles = build_risk_decile_report(scored_valid)
        dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")
        skipped.to_csv(skipped_path, index=False, encoding="utf-8-sig")
        metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        deciles.to_csv(deciles_path, index=False, encoding="utf-8-sig")
        logging.info("Tail-risk panel reports saved: %s, %s, %s", dataset_path, metrics_path, deciles_path)

    summary_path.write_text(
        json.dumps(
            {
                "index_rows": int(len(index_dataset)),
                "index_metric_rows": int(len(index_reproduction)),
                "panel_rows": int(len(dataset)),
                "panel_metric_rows": int(len(metrics)),
                "start_date": start_date.isoformat() if start_date else "",
                "end_date": end_date.isoformat() if end_date else "",
                "train_end": train_end.isoformat(),
                "valid_end": valid_end.isoformat(),
                "lookback_days": int(lookback_days),
                "quantile": float(quantile),
                "horizon_days": int(horizon_days),
                "index_source_column": index_source_column,
                "run_index": bool(run_index),
                "run_panel": bool(run_panel),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return TailRiskReproductionResult(
        index_reproduction=index_reproduction,
        dataset=dataset,
        skipped=skipped,
        metrics=metrics,
        deciles=deciles,
        report_dir=report_dir,
        index_reproduction_path=index_reproduction_path,
        index_dataset_path=index_dataset_path,
        dataset_path=dataset_path,
        skipped_path=skipped_path,
        metrics_path=metrics_path,
        deciles_path=deciles_path,
        summary_path=summary_path,
    )


def validate_tail_risk_walkforward(
    *,
    storage: Storage,
    project_root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    train_days: int = 1000,
    valid_days: int = 250,
    step_days: int = 250,
    embargo_days: int | None = None,
    max_windows: int | None = None,
    lookback_days: int = 100,
    quantile: float = 0.05,
    horizon_days: int = 1,
    min_training_rows: int = 200,
    allow_short_sample: bool = False,
    panel_model_names: tuple[str, ...] = DEFAULT_PANEL_MODEL_NAMES,
    filter_rates: tuple[float, ...] = (0.2,),
    return_tolerance: float = 0.001,
) -> TailRiskWalkforwardResult:
    if train_days <= 0 or valid_days <= 0 or step_days <= 0:
        raise ValueError("train_days, valid_days, and step_days must be positive.")
    embargo = int(horizon_days if embargo_days is None else embargo_days)
    logging.info("Tail-risk walk-forward panel dataset build started")
    dataset, skipped = build_tail_risk_panel(
        storage=storage,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        lookback_days=lookback_days,
        quantile=quantile,
        horizon_days=horizon_days,
    )
    if dataset.empty:
        raise RuntimeError("Tail-risk walk-forward has no labeled rows.")
    windows = build_tail_risk_walkforward_windows(
        dataset,
        train_days=train_days,
        valid_days=valid_days,
        step_days=step_days,
        embargo_days=embargo,
        max_windows=max_windows,
    )
    if windows.empty:
        raise RuntimeError(
            "No tail-risk walk-forward windows can be built with the requested train/valid/step days."
        )
    if not allow_short_sample and len(windows) < 3:
        raise RuntimeError(
            f"Tail-risk walk-forward needs at least 3 windows; got {len(windows)}. "
            "Use a longer date range or pass allow_short_sample only for smoke tests."
        )

    metrics_parts: list[pd.DataFrame] = []
    decile_parts: list[pd.DataFrame] = []
    filter_parts: list[pd.DataFrame] = []
    total_windows = len(windows)
    for index, window in enumerate(windows.to_dict("records"), start=1):
        window_id = str(window["window_id"])
        logging.info("Tail-risk walk-forward window %s/%s started: %s", index, total_windows, window_id)
        train = _window_slice(dataset, start=window["train_start"], end=window["train_end"])
        valid = _window_slice(dataset, start=window["valid_start"], end=window["valid_end"])
        logging.info("Tail-risk walk-forward window %s rows: train=%s valid=%s", window_id, len(train), len(valid))
        if len(train) < min_training_rows or valid.empty:
            metrics_parts.append(
                _empty_window_metrics(
                    window=window,
                    model_names=panel_model_names,
                    train_rows=len(train),
                    valid_rows=len(valid),
                    error="insufficient_window_rows",
                )
            )
            continue
        result = _fit_score_risk_models(
            train=train,
            splits={"valid": valid},
            feature_columns=TAIL_RISK_FEATURE_COLUMNS,
            scope="panel_walkforward",
            min_training_rows=min_training_rows,
            skip_large_knn=True,
            model_names=panel_model_names,
        )
        metrics = _attach_window_columns(result["metrics"], window)
        scored = _attach_window_columns(result["scored"], window)
        deciles = build_risk_decile_report(scored)
        filter_impact = build_risk_filter_impact(
            scored,
            filter_rates=filter_rates,
            return_tolerance=return_tolerance,
        )
        metrics_parts.append(metrics)
        decile_parts.append(deciles)
        filter_parts.append(filter_impact)
        logging.info("Tail-risk walk-forward window %s/%s complete: %s", index, total_windows, window_id)

    metrics_frame = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()
    deciles_frame = pd.concat(decile_parts, ignore_index=True) if decile_parts else pd.DataFrame()
    filter_impact_frame = pd.concat(filter_parts, ignore_index=True) if filter_parts else pd.DataFrame()
    filter_summary = summarize_risk_filter_impact(filter_impact_frame)
    summary = summarize_tail_risk_walkforward(metrics_frame, deciles_frame)

    report_dir = full_market_report_dir(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    windows_path = report_dir / "tail_risk_walkforward_windows.csv"
    metrics_path = report_dir / "tail_risk_walkforward_metrics.csv"
    deciles_path = report_dir / "tail_risk_walkforward_decile_report.csv"
    filter_impact_path = report_dir / "tail_risk_filter_impact.csv"
    filter_summary_path = report_dir / "tail_risk_filter_impact_summary.csv"
    summary_path = report_dir / "tail_risk_walkforward_summary.csv"
    config_path = report_dir / "tail_risk_walkforward_config.json"
    windows.to_csv(windows_path, index=False, encoding="utf-8-sig")
    metrics_frame.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    deciles_frame.to_csv(deciles_path, index=False, encoding="utf-8-sig")
    filter_impact_frame.to_csv(filter_impact_path, index=False, encoding="utf-8-sig")
    filter_summary.to_csv(filter_summary_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    config_path.write_text(
        json.dumps(
            {
                "start_date": start_date.isoformat() if start_date else "",
                "end_date": end_date.isoformat() if end_date else "",
                "limit": limit,
                "train_days": int(train_days),
                "valid_days": int(valid_days),
                "step_days": int(step_days),
                "embargo_days": int(embargo),
                "max_windows": max_windows,
                "lookback_days": int(lookback_days),
                "quantile": float(quantile),
                "horizon_days": int(horizon_days),
                "min_training_rows": int(min_training_rows),
                "allow_short_sample": bool(allow_short_sample),
                "panel_model_names": list(panel_model_names),
                "filter_rates": [float(value) for value in filter_rates],
                "return_tolerance": float(return_tolerance),
                "skipped_symbols": int(len(skipped)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logging.info("Tail-risk walk-forward reports saved to %s", report_dir)
    return TailRiskWalkforwardResult(
        windows=windows,
        metrics=metrics_frame,
        deciles=deciles_frame,
        filter_impact=filter_impact_frame,
        filter_summary=filter_summary,
        summary=summary,
        report_dir=report_dir,
        windows_path=windows_path,
        metrics_path=metrics_path,
        deciles_path=deciles_path,
        filter_impact_path=filter_impact_path,
        filter_summary_path=filter_summary_path,
        summary_path=summary_path,
        config_path=config_path,
    )


def validate_barrier_risk_walkforward(
    *,
    storage: Storage,
    project_root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    train_days: int = 1000,
    valid_days: int = 250,
    step_days: int = 250,
    embargo_days: int | None = None,
    max_windows: int | None = None,
    horizon_days: int = 20,
    downside_atr_mult: float = 1.0,
    upside_atr_mult: float | None = 2.0,
    downside_pct: float | None = None,
    upside_pct: float | None = None,
    label_variant: str = "barrier_down_first",
    label_method: str = "a_share_daily",
    volatility_lookback: int = 100,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    min_ret: float = 0.005,
    cusum_threshold: float | None = None,
    cusum_threshold_mult: float = 1.0,
    min_training_rows: int = 200,
    allow_short_sample: bool = False,
    model_names: tuple[str, ...] = DEFAULT_BARRIER_MODEL_NAMES,
    filter_rates: tuple[float, ...] = (0.2,),
    return_tolerance: float = 0.001,
) -> BarrierRiskValidationResult:
    if train_days <= 0 or valid_days <= 0 or step_days <= 0:
        raise ValueError("train_days, valid_days, and step_days must be positive.")
    embargo = int(horizon_days if embargo_days is None else embargo_days)
    logging.info("Barrier-risk walk-forward panel dataset build started")
    dataset, skipped = build_barrier_risk_panel(
        storage=storage,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        horizon_days=horizon_days,
        downside_atr_mult=downside_atr_mult,
        upside_atr_mult=upside_atr_mult,
        downside_pct=downside_pct,
        upside_pct=upside_pct,
        label_variant=label_variant,
        label_method=label_method,
        volatility_lookback=volatility_lookback,
        pt_mult=pt_mult,
        sl_mult=sl_mult,
        min_ret=min_ret,
        cusum_threshold=cusum_threshold,
        cusum_threshold_mult=cusum_threshold_mult,
    )
    if dataset.empty:
        raise RuntimeError("Barrier-risk walk-forward has no labeled rows.")
    windows = build_tail_risk_walkforward_windows(
        dataset,
        train_days=train_days,
        valid_days=valid_days,
        step_days=step_days,
        embargo_days=embargo,
        max_windows=max_windows,
    )
    if windows.empty:
        raise RuntimeError(
            "No barrier-risk walk-forward windows can be built with the requested train/valid/step days."
        )
    if not allow_short_sample and len(windows) < 3:
        raise RuntimeError(
            f"Barrier-risk walk-forward needs at least 3 windows; got {len(windows)}. "
            "Use a longer date range or pass allow_short_sample only for smoke tests."
        )

    label_distribution = summarize_barrier_label_distribution(dataset)
    metrics_parts: list[pd.DataFrame] = []
    decile_parts: list[pd.DataFrame] = []
    filter_parts: list[pd.DataFrame] = []
    total_windows = len(windows)
    for index, window in enumerate(windows.to_dict("records"), start=1):
        window_id = str(window["window_id"])
        logging.info("Barrier-risk walk-forward window %s/%s started: %s", index, total_windows, window_id)
        train = _window_slice(dataset, start=window["train_start"], end=window["train_end"])
        valid = _window_slice(dataset, start=window["valid_start"], end=window["valid_end"])
        logging.info("Barrier-risk walk-forward window %s rows: train=%s valid=%s", window_id, len(train), len(valid))
        if len(train) < min_training_rows or valid.empty:
            metrics_parts.append(
                _empty_window_metrics(
                    window=window,
                    model_names=model_names,
                    train_rows=len(train),
                    valid_rows=len(valid),
                    error="insufficient_window_rows",
                )
            )
            continue
        result = _fit_score_risk_models(
            train=train,
            splits={"valid": valid},
            feature_columns=BARRIER_RISK_FEATURE_COLUMNS,
            scope="barrier_walkforward",
            min_training_rows=min_training_rows,
            skip_large_knn=True,
            model_names=model_names,
        )
        metrics = _attach_window_columns(result["metrics"], window)
        scored = _attach_window_columns(result["scored"], window)
        deciles = build_risk_decile_report(scored)
        filter_impact = build_risk_filter_impact(
            scored,
            filter_rates=filter_rates,
            return_tolerance=return_tolerance,
        )
        metrics_parts.append(metrics)
        decile_parts.append(deciles)
        filter_parts.append(filter_impact)
        logging.info("Barrier-risk walk-forward window %s/%s complete: %s", index, total_windows, window_id)

    metrics_frame = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()
    deciles_frame = pd.concat(decile_parts, ignore_index=True) if decile_parts else pd.DataFrame()
    filter_impact_frame = pd.concat(filter_parts, ignore_index=True) if filter_parts else pd.DataFrame()
    filter_summary = summarize_risk_filter_impact(filter_impact_frame)
    summary = summarize_tail_risk_walkforward(metrics_frame, deciles_frame)
    comparison = build_barrier_vs_tail_comparison(project_root, barrier_summary=summary, barrier_filter_summary=filter_summary)

    report_dir = full_market_report_dir(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    label_distribution_path = report_dir / "barrier_label_distribution.csv"
    metrics_path = report_dir / "barrier_risk_metrics.csv"
    deciles_path = report_dir / "barrier_risk_decile_report.csv"
    filter_impact_path = report_dir / "barrier_risk_filter_impact.csv"
    filter_summary_path = report_dir / "barrier_risk_filter_impact_summary.csv"
    comparison_path = report_dir / "barrier_vs_tail_comparison.csv"
    config_path = report_dir / "barrier_risk_config.json"
    label_distribution.to_csv(label_distribution_path, index=False, encoding="utf-8-sig")
    metrics_frame.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    deciles_frame.to_csv(deciles_path, index=False, encoding="utf-8-sig")
    filter_impact_frame.to_csv(filter_impact_path, index=False, encoding="utf-8-sig")
    filter_summary.to_csv(filter_summary_path, index=False, encoding="utf-8-sig")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    config_path.write_text(
        json.dumps(
            {
                "start_date": start_date.isoformat() if start_date else "",
                "end_date": end_date.isoformat() if end_date else "",
                "limit": limit,
                "train_days": int(train_days),
                "valid_days": int(valid_days),
                "step_days": int(step_days),
                "embargo_days": int(embargo),
                "max_windows": max_windows,
                "horizon_days": int(horizon_days),
                "downside_atr_mult": float(downside_atr_mult),
                "upside_atr_mult": float(upside_atr_mult) if upside_atr_mult is not None else None,
                "downside_pct": float(downside_pct) if downside_pct is not None else None,
                "upside_pct": float(upside_pct) if upside_pct is not None else None,
                "label_variant": label_variant,
                "label_method": label_method,
                "volatility_lookback": int(volatility_lookback),
                "pt_mult": float(pt_mult),
                "sl_mult": float(sl_mult),
                "min_ret": float(min_ret),
                "cusum_threshold": float(cusum_threshold) if cusum_threshold is not None else None,
                "cusum_threshold_mult": float(cusum_threshold_mult),
                "min_training_rows": int(min_training_rows),
                "allow_short_sample": bool(allow_short_sample),
                "model_names": list(model_names),
                "filter_rates": [float(value) for value in filter_rates],
                "return_tolerance": float(return_tolerance),
                "skipped_symbols": int(len(skipped)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logging.info("Barrier-risk walk-forward reports saved to %s", report_dir)
    return BarrierRiskValidationResult(
        label_distribution=label_distribution,
        metrics=metrics_frame,
        deciles=deciles_frame,
        filter_impact=filter_impact_frame,
        filter_summary=filter_summary,
        comparison=comparison,
        report_dir=report_dir,
        label_distribution_path=label_distribution_path,
        metrics_path=metrics_path,
        deciles_path=deciles_path,
        filter_impact_path=filter_impact_path,
        filter_summary_path=filter_summary_path,
        comparison_path=comparison_path,
        config_path=config_path,
    )


def validate_barrier_risk_grid(
    *,
    storage: Storage,
    project_root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    train_days: int = 1000,
    valid_days: int = 250,
    step_days: int = 250,
    max_windows: int | None = None,
    horizon_days_grid: tuple[int, ...] = (5, 10),
    pt_sl_grid: tuple[tuple[float, float], ...] = ((1.0, 1.0), (2.0, 2.0)),
    min_ret_grid: tuple[float, ...] = (0.003, 0.005),
    volatility_lookback: int = 100,
    model_names: tuple[str, ...] = ("lightgbm_classifier",),
    filter_rates: tuple[float, ...] = (0.2,),
    return_tolerance: float = 0.001,
    allow_short_sample: bool = False,
    min_training_rows: int = 200,
) -> BarrierRiskGridResult:
    report_dir = full_market_report_dir(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[pd.DataFrame] = []
    label_rows: list[pd.DataFrame] = []
    configs: list[dict[str, Any]] = []
    total_configs = len(horizon_days_grid) * len(pt_sl_grid) * len(min_ret_grid)
    config_index = 0
    for horizon_days in horizon_days_grid:
        for pt_mult, sl_mult in pt_sl_grid:
            for min_ret in min_ret_grid:
                config_index += 1
                config_id = f"mlfin_h{int(horizon_days)}_pt{pt_mult:g}_sl{sl_mult:g}_minret{min_ret:g}"
                config = {
                    "config_id": config_id,
                    "horizon_days": int(horizon_days),
                    "pt_mult": float(pt_mult),
                    "sl_mult": float(sl_mult),
                    "min_ret": float(min_ret),
                }
                configs.append(config)
                logging.info("Barrier-risk grid config %s/%s started: %s", config_index, total_configs, config_id)
                try:
                    result = validate_barrier_risk_walkforward(
                        storage=storage,
                        project_root=project_root,
                        start_date=start_date,
                        end_date=end_date,
                        limit=limit,
                        train_days=train_days,
                        valid_days=valid_days,
                        step_days=step_days,
                        embargo_days=int(horizon_days),
                        max_windows=max_windows,
                        horizon_days=int(horizon_days),
                        label_method="mlfin_cusum",
                        volatility_lookback=volatility_lookback,
                        pt_mult=float(pt_mult),
                        sl_mult=float(sl_mult),
                        min_ret=float(min_ret),
                        min_training_rows=min_training_rows,
                        allow_short_sample=allow_short_sample,
                        model_names=model_names,
                        filter_rates=filter_rates,
                        return_tolerance=return_tolerance,
                    )
                except Exception as exc:
                    summary_rows.append(
                        pd.DataFrame(
                            [
                                {
                                    **config,
                                    "model_name": "",
                                    "windows": 0,
                                    "successful_windows": 0,
                                    "avg_pr_auc": np.nan,
                                    "avg_pr_auc_baseline": np.nan,
                                    "avg_roc_auc": np.nan,
                                    "filter_pass_rate": np.nan,
                                    "avg_future_return_5d_delta": np.nan,
                                    "avg_future_max_drawdown_5d_delta": np.nan,
                                    "risk_label_rate": np.nan,
                                    "label_rows": 0,
                                    "phase_pass": False,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            ]
                        )
                    )
                    logging.warning("Barrier-risk grid config failed: %s error=%s", config_id, exc)
                    continue
                barrier_rows = result.comparison[result.comparison["risk_target"].astype(str).eq("barrier")].copy()
                if barrier_rows.empty:
                    barrier_rows = pd.DataFrame([{"model_name": model_name} for model_name in model_names])
                all_label = result.label_distribution[result.label_distribution["barrier_outcome"].astype(str).eq("ALL")]
                risk_label_rate = float(all_label.iloc[0]["risk_label_rate"]) if not all_label.empty else np.nan
                label_count = int(all_label.iloc[0]["rows"]) if not all_label.empty else int(result.label_distribution["rows"].sum()) if "rows" in result.label_distribution else 0
                barrier_rows["config_id"] = config_id
                barrier_rows["horizon_days"] = int(horizon_days)
                barrier_rows["pt_mult"] = float(pt_mult)
                barrier_rows["sl_mult"] = float(sl_mult)
                barrier_rows["min_ret"] = float(min_ret)
                barrier_rows["risk_label_rate"] = risk_label_rate
                barrier_rows["label_rows"] = label_count
                barrier_rows["error"] = ""
                summary_rows.append(barrier_rows)
                labels = result.label_distribution.copy()
                labels["config_id"] = config_id
                labels["horizon_days"] = int(horizon_days)
                labels["pt_mult"] = float(pt_mult)
                labels["sl_mult"] = float(sl_mult)
                labels["min_ret"] = float(min_ret)
                label_rows.append(labels)
                logging.info("Barrier-risk grid config %s/%s complete: %s", config_index, total_configs, config_id)

    summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    label_distribution = pd.concat(label_rows, ignore_index=True) if label_rows else pd.DataFrame()
    if not summary.empty:
        sort_columns = [
            column
            for column in ("phase_pass", "filter_pass_rate", "avg_future_max_drawdown_5d_delta", "avg_future_return_5d_delta")
            if column in summary.columns
        ]
        if sort_columns:
            summary = summary.sort_values(sort_columns, ascending=[False] * len(sort_columns)).reset_index(drop=True)
    summary_path = report_dir / "barrier_risk_grid_summary.csv"
    label_distribution_path = report_dir / "barrier_risk_grid_label_distribution.csv"
    config_path = report_dir / "barrier_risk_grid_config.json"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    label_distribution.to_csv(label_distribution_path, index=False, encoding="utf-8-sig")
    config_path.write_text(
        json.dumps(
            {
                "start_date": start_date.isoformat() if start_date else "",
                "end_date": end_date.isoformat() if end_date else "",
                "limit": limit,
                "train_days": int(train_days),
                "valid_days": int(valid_days),
                "step_days": int(step_days),
                "max_windows": max_windows,
                "horizon_days_grid": [int(value) for value in horizon_days_grid],
                "pt_sl_grid": [[float(pt), float(sl)] for pt, sl in pt_sl_grid],
                "min_ret_grid": [float(value) for value in min_ret_grid],
                "volatility_lookback": int(volatility_lookback),
                "model_names": list(model_names),
                "filter_rates": [float(value) for value in filter_rates],
                "return_tolerance": float(return_tolerance),
                "allow_short_sample": bool(allow_short_sample),
                "min_training_rows": int(min_training_rows),
                "configs": configs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return BarrierRiskGridResult(
        summary=summary,
        label_distribution=label_distribution,
        summary_path=summary_path,
        label_distribution_path=label_distribution_path,
        config_path=config_path,
    )


def build_tail_risk_walkforward_windows(
    dataset: pd.DataFrame,
    *,
    train_days: int,
    valid_days: int,
    step_days: int,
    embargo_days: int = 1,
    max_windows: int | None = None,
) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    trade_dates = pd.Series(pd.to_datetime(dataset["trade_date"], errors="coerce").dropna().unique()).sort_values().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    start_index = 0
    while True:
        train_start_index = start_index
        train_end_index = train_start_index + train_days - 1
        valid_start_index = train_end_index + 1 + max(int(embargo_days), 0)
        valid_end_index = valid_start_index + valid_days - 1
        if valid_end_index >= len(trade_dates):
            break
        rows.append(
            {
                "window_id": f"wf_{len(rows) + 1:02d}",
                "train_start": pd.Timestamp(trade_dates.iloc[train_start_index]).date().isoformat(),
                "train_end": pd.Timestamp(trade_dates.iloc[train_end_index]).date().isoformat(),
                "valid_start": pd.Timestamp(trade_dates.iloc[valid_start_index]).date().isoformat(),
                "valid_end": pd.Timestamp(trade_dates.iloc[valid_end_index]).date().isoformat(),
                "train_days": int(train_days),
                "valid_days": int(valid_days),
                "embargo_days": int(embargo_days),
            }
        )
        if max_windows is not None and len(rows) >= max_windows:
            break
        start_index += step_days
    return pd.DataFrame(rows)


def summarize_tail_risk_walkforward(metrics: pd.DataFrame, deciles: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for model_name, group in metrics.groupby("model_name", sort=False):
        successful = group[group["error"].fillna("").eq("")].copy()
        decile_checks = _walkforward_decile_checks(deciles, model_name=model_name)
        rows.append(
            {
                "model_name": model_name,
                "windows": int(group["window_id"].nunique()) if "window_id" in group.columns else int(len(group)),
                "successful_windows": int(len(successful)),
                "avg_pr_auc": _safe_mean(successful.get("pr_auc", pd.Series(dtype=float))),
                "avg_pr_auc_baseline": _safe_mean(successful.get("pr_auc_baseline", pd.Series(dtype=float))),
                "avg_roc_auc": _safe_mean(successful.get("roc_auc", pd.Series(dtype=float))),
                "pr_auc_beat_baseline_rate": _mean_bool(successful["pr_auc"].gt(successful["pr_auc_baseline"])) if not successful.empty else 0.0,
                "top_decile_higher_risk_rate": _safe_mean(decile_checks.get("top_decile_higher_risk", pd.Series(dtype=float))),
                "top_decile_worse_drawdown_rate": _safe_mean(decile_checks.get("top_decile_worse_drawdown", pd.Series(dtype=float))),
                "phase1_pass": _phase1_pass(successful, decile_checks),
            }
        )
    return pd.DataFrame(rows)


def build_barrier_vs_tail_comparison(
    project_root: Path,
    *,
    barrier_summary: pd.DataFrame,
    barrier_filter_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    rows.extend(_comparison_rows("barrier", barrier_summary, barrier_filter_summary))
    report_dir = full_market_report_dir(project_root)
    tail_summary_path = report_dir / "tail_risk_walkforward_summary.csv"
    tail_filter_path = report_dir / "tail_risk_filter_impact_summary.csv"
    if tail_summary_path.exists():
        tail_summary = pd.read_csv(tail_summary_path)
        tail_filter = pd.read_csv(tail_filter_path) if tail_filter_path.exists() else pd.DataFrame()
        rows.extend(_comparison_rows("tail", tail_summary, tail_filter))
    return pd.DataFrame(rows)


def _comparison_rows(risk_target: str, summary: pd.DataFrame, filter_summary: pd.DataFrame) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    filters = filter_summary.copy()
    if not filters.empty and "filter_rate" in filters.columns:
        filters["filter_rate"] = pd.to_numeric(filters["filter_rate"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        model_name = str(row["model_name"])
        filter_row = pd.Series(dtype=object)
        if not filters.empty:
            model_filters = filters[filters["model_name"].astype(str).eq(model_name)].copy()
            if not model_filters.empty:
                model_filters["_filter_distance"] = (pd.to_numeric(model_filters["filter_rate"], errors="coerce") - 0.2).abs()
                filter_row = model_filters.sort_values("_filter_distance").iloc[0]
        rows.append(
            {
                "risk_target": risk_target,
                "model_name": model_name,
                "windows": int(row.get("windows", 0)),
                "successful_windows": int(row.get("successful_windows", 0)),
                "avg_pr_auc": float(row.get("avg_pr_auc", np.nan)),
                "avg_pr_auc_baseline": float(row.get("avg_pr_auc_baseline", np.nan)),
                "avg_roc_auc": float(row.get("avg_roc_auc", np.nan)),
                "pr_auc_beat_baseline_rate": float(row.get("pr_auc_beat_baseline_rate", np.nan)),
                "top_decile_higher_risk_rate": float(row.get("top_decile_higher_risk_rate", np.nan)),
                "top_decile_worse_drawdown_rate": float(row.get("top_decile_worse_drawdown_rate", np.nan)),
                "filter_pass_rate": float(filter_row.get("filter_pass_rate", np.nan)) if not filter_row.empty else np.nan,
                "avg_future_return_5d_delta": float(filter_row.get("avg_future_return_5d_delta", np.nan)) if not filter_row.empty else np.nan,
                "avg_future_max_drawdown_5d_delta": float(filter_row.get("avg_future_max_drawdown_5d_delta", np.nan)) if not filter_row.empty else np.nan,
                "phase_pass": bool(row.get("phase1_pass", False)) and (bool(filter_row.get("phase1_filter_pass", False)) if not filter_row.empty else True),
            }
        )
    return rows


def build_tail_risk_index_dataset(
    *,
    storage: Storage,
    project_root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    lookback_days: int = 100,
    quantile: float = 0.05,
    horizon_days: int = 1,
    source_column: str = "synthetic_equal_weight_index",
    min_stock_count: int = 500,
) -> pd.DataFrame:
    market = _load_or_build_synthetic_market(
        storage=storage,
        project_root=project_root,
        start_date=start_date,
        end_date=end_date,
        min_stock_count=min_stock_count,
    )
    if source_column not in market.columns:
        raise ValueError(f"Synthetic market file does not contain source column: {source_column}")
    frame = market.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    index_value = pd.to_numeric(frame[source_column], errors="coerce")
    log_return = np.log(index_value / index_value.shift(1)).replace([np.inf, -np.inf], np.nan)
    tail_threshold = log_return.shift(1).rolling(lookback_days, min_periods=lookback_days).quantile(quantile)
    tail_event = log_return.lt(tail_threshold)
    result = pd.DataFrame(
        {
            "trade_date": frame["trade_date"],
            "symbol": source_column,
            "name": source_column,
            "index_source_column": source_column,
            "index_value": index_value,
            "log_return_1d": log_return,
            "tail_threshold_past": tail_threshold,
            "tail_event_today": tail_event.astype("float"),
            # Noh 2026 is an index-level risk-state classifier. The exact reproduction
            # classifies whether the observed index return is below the rolling tail threshold.
            "risk_label": tail_event.astype("float"),
            "forward_log_return": log_return.shift(-horizon_days),
            "future_return_5d": index_value.shift(-5).div(index_value).sub(1.0),
            "future_max_drawdown_5d": _future_min_return(index_value, horizon=5),
        }
    )
    if "stock_count" in frame.columns:
        result["stock_count"] = pd.to_numeric(frame["stock_count"], errors="coerce")
    if start_date is not None:
        result = result[result["trade_date"].dt.date >= start_date]
    if end_date is not None:
        result = result[result["trade_date"].dt.date <= end_date]
    result = result.dropna(subset=["risk_label", *NOH_INDEX_FEATURE_COLUMNS]).copy()
    return result.reset_index(drop=True)


def reproduce_tail_risk_index_models(
    dataset: pd.DataFrame,
    *,
    train_end: date,
    valid_end: date,
    min_training_rows: int = 200,
) -> pd.DataFrame:
    train = dataset[dataset["trade_date"].dt.date <= train_end].copy()
    valid = dataset[(dataset["trade_date"].dt.date > train_end) & (dataset["trade_date"].dt.date <= valid_end)].copy()
    test = dataset[dataset["trade_date"].dt.date > valid_end].copy()
    if len(train) < min_training_rows or valid.empty:
        raise RuntimeError(f"Insufficient tail-risk index split rows: train={len(train)} valid={len(valid)}")
    splits = {"valid": valid}
    if not test.empty:
        splits["test"] = test
    result = _fit_score_risk_models(
        train=train,
        splits=splits,
        feature_columns=NOH_INDEX_FEATURE_COLUMNS,
        scope="index",
        min_training_rows=min_training_rows,
        skip_large_knn=False,
    )
    return result["metrics"]


def build_risk_decile_report(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = [column for column in ("window_id", "scope", "split", "model_name") if column in scored.columns]
    for key, group in scored.groupby(group_columns, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        key_map = dict(zip(group_columns, key_values, strict=False))
        frame = group.copy()
        frame["risk_decile"] = pd.qcut(frame["risk_score"].rank(method="first"), 10, labels=False, duplicates="drop")
        for decile, decile_frame in frame.groupby("risk_decile", sort=True):
            rows.append(
                {
                    **key_map,
                    "risk_decile": int(decile),
                    "rows": int(len(decile_frame)),
                    "risk_label_rate": float(decile_frame["risk_label"].mean()),
                    "avg_forward_log_return": float(pd.to_numeric(decile_frame["forward_log_return"], errors="coerce").mean()),
                    "avg_future_return_5d": float(pd.to_numeric(decile_frame["future_return_5d"], errors="coerce").mean()),
                    "avg_future_max_drawdown_5d": float(pd.to_numeric(decile_frame["future_max_drawdown_5d"], errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows)


def build_risk_filter_impact(
    scored: pd.DataFrame,
    *,
    filter_rates: tuple[float, ...] = (0.2,),
    return_tolerance: float = 0.001,
) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = [column for column in ("window_id", "scope", "split", "model_name") if column in scored.columns]
    groups = scored.groupby(group_columns, sort=False) if group_columns else [((), scored)]
    for key, group in groups:
        key_values = key if isinstance(key, tuple) else (key,)
        key_map = dict(zip(group_columns, key_values, strict=False))
        frame = group.copy()
        frame["risk_score"] = pd.to_numeric(frame["risk_score"], errors="coerce")
        frame = frame.dropna(subset=["risk_score", "future_return_5d", "future_max_drawdown_5d"]).copy()
        if frame.empty:
            continue
        ordered = frame.sort_values(["risk_score", "symbol"], ascending=[False, True])
        for filter_rate in filter_rates:
            rate = float(filter_rate)
            if rate <= 0 or rate >= 1:
                raise ValueError(f"filter_rates must be between 0 and 1, got {filter_rate}")
            removed_rows = max(1, int(np.ceil(len(ordered) * rate)))
            kept = ordered.iloc[removed_rows:].copy()
            removed = ordered.iloc[:removed_rows].copy()
            baseline_return = _safe_mean(ordered["future_return_5d"])
            kept_return = _safe_mean(kept["future_return_5d"])
            baseline_drawdown = _safe_mean(ordered["future_max_drawdown_5d"])
            kept_drawdown = _safe_mean(kept["future_max_drawdown_5d"])
            baseline_risk_rate = _safe_mean(ordered["risk_label"]) if "risk_label" in ordered.columns else float("nan")
            kept_risk_rate = _safe_mean(kept["risk_label"]) if "risk_label" in kept.columns else float("nan")
            drawdown_delta = kept_drawdown - baseline_drawdown
            return_delta = kept_return - baseline_return
            rows.append(
                {
                    **key_map,
                    "filter_rate": rate,
                    "rows": int(len(ordered)),
                    "removed_rows": int(len(removed)),
                    "kept_rows": int(len(kept)),
                    "kept_rate": float(len(kept) / len(ordered)),
                    "baseline_risk_label_rate": baseline_risk_rate,
                    "kept_risk_label_rate": kept_risk_rate,
                    "removed_risk_label_rate": _safe_mean(removed["risk_label"]) if "risk_label" in removed.columns else float("nan"),
                    "risk_label_rate_delta": kept_risk_rate - baseline_risk_rate,
                    "baseline_avg_future_return_5d": baseline_return,
                    "kept_avg_future_return_5d": kept_return,
                    "removed_avg_future_return_5d": _safe_mean(removed["future_return_5d"]),
                    "future_return_5d_delta": return_delta,
                    "baseline_avg_future_max_drawdown_5d": baseline_drawdown,
                    "kept_avg_future_max_drawdown_5d": kept_drawdown,
                    "removed_avg_future_max_drawdown_5d": _safe_mean(removed["future_max_drawdown_5d"]),
                    "future_max_drawdown_5d_delta": drawdown_delta,
                    "drawdown_improved": bool(drawdown_delta > 0),
                    "return_not_materially_worse": bool(return_delta >= -float(return_tolerance)),
                    "filter_pass": bool(drawdown_delta > 0 and return_delta >= -float(return_tolerance)),
                }
            )
    return pd.DataFrame(rows)


def summarize_risk_filter_impact(filter_impact: pd.DataFrame) -> pd.DataFrame:
    if filter_impact.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = [column for column in ("model_name", "filter_rate") if column in filter_impact.columns]
    for key, group in filter_impact.groupby(group_columns, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        key_map = dict(zip(group_columns, key_values, strict=False))
        rows.append(
            {
                **key_map,
                "windows": int(group["window_id"].nunique()) if "window_id" in group.columns else int(len(group)),
                "avg_kept_rate": _safe_mean(group["kept_rate"]),
                "avg_risk_label_rate_delta": _safe_mean(group["risk_label_rate_delta"]),
                "avg_future_return_5d_delta": _safe_mean(group["future_return_5d_delta"]),
                "avg_future_max_drawdown_5d_delta": _safe_mean(group["future_max_drawdown_5d_delta"]),
                "drawdown_improved_rate": _mean_bool(group["drawdown_improved"]),
                "return_not_materially_worse_rate": _mean_bool(group["return_not_materially_worse"]),
                "filter_pass_rate": _mean_bool(group["filter_pass"]),
                "phase1_filter_pass": bool(_mean_bool(group["filter_pass"]) >= 0.70),
            }
        )
    return pd.DataFrame(rows)


def _log_prediction_progress(current: int, total: int) -> None:
    if total <= 0:
        return
    if current == 1 or current % 500 == 0 or current == total:
        logging.info("Tail-risk prediction progress: %s/%s", current, total)


def _window_slice(dataset: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    trade_date = pd.to_datetime(dataset["trade_date"], errors="coerce")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return dataset[trade_date.ge(start_ts) & trade_date.le(end_ts)].copy()


def _attach_window_columns(frame: pd.DataFrame, window: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    for column in ("window_id", "train_start", "train_end", "valid_start", "valid_end"):
        result[column] = window[column]
    return result


def _empty_window_metrics(
    *,
    window: dict[str, Any],
    model_names: tuple[str, ...],
    train_rows: int,
    valid_rows: int,
    error: str,
) -> pd.DataFrame:
    rows = []
    y_true = pd.Series([0] * valid_rows, dtype=int)
    for model_name in model_names:
        row = _error_metric_row(
            y_true,
            model_name=model_name,
            split="valid",
            scope="panel_walkforward",
            error=error,
        )
        row["train_rows"] = int(train_rows)
        for column in ("window_id", "train_start", "train_end", "valid_start", "valid_end"):
            row[column] = window[column]
        rows.append(row)
    return pd.DataFrame(rows)


def _walkforward_decile_checks(deciles: pd.DataFrame, *, model_name: str) -> pd.DataFrame:
    if deciles.empty or "window_id" not in deciles.columns:
        return pd.DataFrame()
    rows = []
    model_deciles = deciles[deciles["model_name"].eq(model_name)].copy()
    for window_id, group in model_deciles.groupby("window_id", sort=False):
        ordered = group.sort_values("risk_decile")
        if ordered.empty:
            continue
        low = ordered.iloc[0]
        high = ordered.iloc[-1]
        rows.append(
            {
                "model_name": model_name,
                "window_id": window_id,
                "top_decile_higher_risk": float(high["risk_label_rate"] > low["risk_label_rate"]),
                "top_decile_worse_drawdown": float(high["avg_future_max_drawdown_5d"] < low["avg_future_max_drawdown_5d"]),
            }
        )
    return pd.DataFrame(rows)


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().empty:
        return float("nan")
    return float(numeric.mean())


def _mean_bool(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(values.fillna(False).astype(bool).mean())


def _phase1_pass(metrics: pd.DataFrame, decile_checks: pd.DataFrame) -> bool:
    if metrics.empty or decile_checks.empty:
        return False
    pr_rate = _mean_bool(metrics["pr_auc"].gt(metrics["pr_auc_baseline"]))
    risk_rate = _safe_mean(decile_checks["top_decile_higher_risk"])
    drawdown_rate = _safe_mean(decile_checks["top_decile_worse_drawdown"])
    return bool(pr_rate >= 0.70 and risk_rate >= 0.70 and drawdown_rate >= 0.70)


def _fit_score_risk_models(
    *,
    train: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    feature_columns: tuple[str, ...],
    scope: str,
    min_training_rows: int,
    skip_large_knn: bool,
    model_names: tuple[str, ...] = ALL_TAIL_RISK_MODEL_NAMES,
) -> dict[str, pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    scored_parts: list[pd.DataFrame] = []
    if len(train) < min_training_rows:
        raise RuntimeError(f"Insufficient {scope} training rows: {len(train)}")
    X_train = train.loc[:, feature_columns]
    y_train = train["risk_label"].astype(int)
    models = _tail_risk_models(model_names)
    total_models = len(models)
    for model_index, (model_name, model) in enumerate(models.items(), start=1):
        logging.info(
            "Tail-risk %s model %s/%s started: %s train_rows=%s",
            scope,
            model_index,
            total_models,
            model_name,
            len(train),
        )
        if skip_large_knn and model_name == "knn" and _is_large_knn_problem(train, splits):
            for split_name, split in splits.items():
                metrics_rows.append(
                    _error_metric_row(
                        split["risk_label"].astype(int),
                        model_name=model_name,
                        split=split_name,
                        scope=scope,
                        error=f"skipped: KNN over panel rows exceeds {PANEL_KNN_MAX_ROWS}",
                    )
                )
            logging.info("Tail-risk %s model skipped: %s", scope, model_name)
            continue
        try:
            fitted = model.fit(X_train, y_train)
        except Exception as exc:
            for split_name, split in splits.items():
                metrics_rows.append(
                    _error_metric_row(
                        split["risk_label"].astype(int),
                        model_name=model_name,
                        split=split_name,
                        scope=scope,
                        error=f"fit_failed: {type(exc).__name__}: {exc}",
                    )
                )
            logging.warning("Tail-risk %s model fit failed: %s", scope, model_name)
            continue
        logging.info("Tail-risk %s model fitted: %s", scope, model_name)
        for split_name, split in splits.items():
            logging.info(
                "Tail-risk %s model predicting split=%s model=%s rows=%s",
                scope,
                split_name,
                model_name,
                len(split),
            )
            try:
                proba = _predict_risk_proba(fitted, split.loc[:, feature_columns])
            except Exception as exc:
                metrics_rows.append(
                    _error_metric_row(
                        split["risk_label"].astype(int),
                        model_name=model_name,
                        split=split_name,
                        scope=scope,
                        error=f"predict_failed: {type(exc).__name__}: {exc}",
                    )
                )
                continue
            metrics_rows.append(
                _classification_metric_row(
                    split["risk_label"].astype(int),
                    proba,
                    model_name=model_name,
                    split=split_name,
                    scope=scope,
                    feature_count=len(feature_columns),
                    train_rows=len(train),
                )
            )
            scored = _scored_columns(split).copy()
            scored["scope"] = scope
            scored["split"] = split_name
            scored["model_name"] = model_name
            scored["risk_score"] = proba
            scored_parts.append(scored)
        logging.info("Tail-risk %s model %s/%s complete: %s", scope, model_index, total_models, model_name)
    return {
        "metrics": pd.DataFrame(metrics_rows),
        "scored": pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame(),
    }


def _tail_risk_models(model_names: tuple[str, ...] = ALL_TAIL_RISK_MODEL_NAMES) -> dict[str, Any]:
    available = {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=1),
        "linear_discriminant_analysis": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "naive_bayes": GaussianNB(),
        "quadratic_discriminant_analysis": make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.01)),
        "adaboost": AdaBoostClassifier(n_estimators=80, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    if LGBMClassifier is not None:
        available["lightgbm_classifier"] = LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )
    unknown = [model_name for model_name in model_names if model_name not in available]
    if unknown:
        raise ValueError(f"Unknown tail-risk model names: {', '.join(unknown)}")
    return {model_name: available[model_name] for model_name in model_names}


def _is_large_knn_problem(train: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> bool:
    if len(train) > PANEL_KNN_MAX_ROWS:
        return True
    return any(len(split) > PANEL_KNN_MAX_ROWS for split in splits.values())


def _scored_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trade_date",
        "symbol",
        "name",
        "risk_label",
        "forward_log_return",
        "future_return_5d",
        "future_max_drawdown_5d",
    ]
    available = [column for column in columns if column in frame.columns]
    return frame.loc[:, available].copy()


def _load_or_build_synthetic_market(
    *,
    storage: Storage,
    project_root: Path,
    start_date: date | None,
    end_date: date | None,
    min_stock_count: int,
) -> pd.DataFrame:
    path = synthetic_market_path(project_root)
    if path.exists():
        cached = pd.read_csv(path)
        if _synthetic_market_covers(cached, start_date=start_date, end_date=end_date):
            return cached
    result = build_synthetic_market_index(
        storage=storage,
        project_root=project_root,
        start_date=start_date.isoformat() if start_date else None,
        end_date=end_date.isoformat() if end_date else None,
        min_stock_count=min_stock_count,
    )
    return result.frame


def _synthetic_market_covers(market: pd.DataFrame, *, start_date: date | None, end_date: date | None) -> bool:
    if market.empty or "trade_date" not in market.columns:
        return False
    trade_dates = pd.to_datetime(market["trade_date"], errors="coerce").dropna()
    if trade_dates.empty:
        return False
    if start_date is not None and trade_dates.min().date() > start_date:
        return False
    if end_date is not None and trade_dates.max().date() < end_date:
        return False
    return True


def _assert_enough_index_history(dataset: pd.DataFrame, *, lookback_days: int, allow_short_sample: bool) -> None:
    if dataset.empty:
        raise RuntimeError("Tail-risk index reproduction has no labeled rows. Build synthetic_market.csv first.")
    if not allow_short_sample and dataset["trade_date"].dt.date.nunique() < lookback_days + 252:
        raise RuntimeError(
            "Tail-risk index reproduction needs more market history. "
            "Build synthetic market data from 20150101 or pass allow_short_sample only for smoke tests."
        )


def _future_min_return(index_value: pd.Series, *, horizon: int) -> pd.Series:
    values = []
    for offset in range(1, horizon + 1):
        values.append(index_value.shift(-offset).div(index_value).sub(1.0))
    if not values:
        return pd.Series(np.nan, index=index_value.index)
    return pd.concat(values, axis=1).min(axis=1)


def _predict_risk_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "steps"):
            classes = getattr(model.steps[-1][1], "classes_", None)
        if classes is not None and int(classes[0]) == 1:
            return np.ones(len(X), dtype=float)
        return np.zeros(len(X), dtype=float)
    return proba[:, 1]


def _classification_metric_row(
    y_true: pd.Series,
    score: np.ndarray,
    *,
    model_name: str,
    split: str,
    scope: str,
    feature_count: int,
    train_rows: int,
) -> dict[str, Any]:
    pred = score >= 0.5
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, pred.astype(int), labels=labels).ravel()
    return {
        "scope": scope,
        "split": split,
        "model_name": model_name,
        "feature_count": int(feature_count),
        "train_rows": int(train_rows),
        "rows": int(len(y_true)),
        "risk_rate": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, pred)),
        "non_risk_precision": float(precision_score(y_true, pred, pos_label=0, zero_division=0)),
        "non_risk_recall": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        "non_risk_f1": float(f1_score(y_true, pred, pos_label=0, zero_division=0)),
        "risk_precision": float(precision_score(y_true, pred, zero_division=0)),
        "risk_recall": float(recall_score(y_true, pred, zero_division=0)),
        "risk_f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": _safe_auc(y_true, score),
        "pr_auc": _safe_average_precision(y_true, score),
        "pr_auc_baseline": float(y_true.mean()),
        "brier": float(brier_score_loss(y_true, score)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "error": "",
    }


def _error_metric_row(y_true: pd.Series, *, model_name: str, split: str, scope: str, error: str) -> dict[str, Any]:
    return {
        "scope": scope,
        "split": split,
        "model_name": model_name,
        "feature_count": 0,
        "train_rows": 0,
        "rows": int(len(y_true)),
        "risk_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "accuracy": float("nan"),
        "non_risk_precision": float("nan"),
        "non_risk_recall": float("nan"),
        "non_risk_f1": float("nan"),
        "risk_precision": float("nan"),
        "risk_recall": float("nan"),
        "risk_f1": float("nan"),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "pr_auc_baseline": float(y_true.mean()) if len(y_true) else float("nan"),
        "brier": float("nan"),
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 0,
        "error": error,
    }


def _safe_auc(y_true: pd.Series, score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, score)) if y_true.nunique() > 1 else float("nan")


def _safe_average_precision(y_true: pd.Series, score: np.ndarray) -> float:
    return float(average_precision_score(y_true, score)) if y_true.nunique() > 1 else float("nan")
