from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
import json
import logging

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

from .full_market_labels import TAIL_RISK_FEATURE_COLUMNS, build_tail_risk_panel
from .full_market_panel import full_market_report_dir
from .storage import Storage
from .synthetic_market import build_synthetic_market_index, synthetic_market_path


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
    group_columns = [column for column in ("scope", "split", "model_name") if column in scored.columns]
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
