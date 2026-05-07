from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .full_market_labels import TAIL_RISK_FEATURE_COLUMNS, build_tail_risk_panel
from .full_market_panel import full_market_report_dir
from .storage import Storage


@dataclass(slots=True)
class TailRiskReproductionResult:
    dataset: pd.DataFrame
    skipped: pd.DataFrame
    metrics: pd.DataFrame
    deciles: pd.DataFrame
    report_dir: Path
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
) -> TailRiskReproductionResult:
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
            "Tail-risk reproduction needs more daily history. "
            "Run update from 20150101 or pass allow_short_sample only for smoke tests."
        )

    train = dataset[dataset["trade_date"].dt.date <= train_end].copy()
    valid = dataset[(dataset["trade_date"].dt.date > train_end) & (dataset["trade_date"].dt.date <= valid_end)].copy()
    if len(train) < min_training_rows or valid.empty:
        raise RuntimeError(f"Insufficient tail-risk split rows: train={len(train)} valid={len(valid)}")

    metrics_rows: list[dict[str, Any]] = []
    scored_parts: list[pd.DataFrame] = []
    for model_name, model in _tail_risk_models().items():
        fitted = model.fit(train.loc[:, TAIL_RISK_FEATURE_COLUMNS], train["risk_label"].astype(int))
        proba = _predict_risk_proba(fitted, valid.loc[:, TAIL_RISK_FEATURE_COLUMNS])
        metrics_rows.append(_classification_metric_row(valid["risk_label"].astype(int), proba, model_name=model_name))
        scored = valid.loc[:, ["trade_date", "symbol", "name", "risk_label", "forward_log_return", "future_return_5d", "future_max_drawdown_5d"]].copy()
        scored["model_name"] = model_name
        scored["risk_score"] = proba
        scored_parts.append(scored)

    metrics = pd.DataFrame(metrics_rows)
    scored_valid = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    deciles = build_risk_decile_report(scored_valid)
    report_dir = full_market_report_dir(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = report_dir / "tail_risk_dataset.csv"
    skipped_path = report_dir / "tail_risk_skipped.csv"
    metrics_path = report_dir / "tail_risk_panel_metrics.csv"
    deciles_path = report_dir / "tail_risk_decile_report.csv"
    summary_path = report_dir / "tail_risk_summary.json"
    dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    skipped.to_csv(skipped_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    deciles.to_csv(deciles_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(
            {
                "rows": int(len(dataset)),
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "start_date": start_date.isoformat() if start_date else "",
                "end_date": end_date.isoformat() if end_date else "",
                "train_end": train_end.isoformat(),
                "valid_end": valid_end.isoformat(),
                "lookback_days": int(lookback_days),
                "quantile": float(quantile),
                "horizon_days": int(horizon_days),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return TailRiskReproductionResult(
        dataset=dataset,
        skipped=skipped,
        metrics=metrics,
        deciles=deciles,
        report_dir=report_dir,
        dataset_path=dataset_path,
        skipped_path=skipped_path,
        metrics_path=metrics_path,
        deciles_path=deciles_path,
        summary_path=summary_path,
    )


def build_risk_decile_report(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for model_name, group in scored.groupby("model_name", sort=False):
        frame = group.copy()
        frame["risk_decile"] = pd.qcut(frame["risk_score"].rank(method="first"), 10, labels=False, duplicates="drop")
        for decile, decile_frame in frame.groupby("risk_decile", sort=True):
            rows.append(
                {
                    "model_name": model_name,
                    "risk_decile": int(decile),
                    "rows": int(len(decile_frame)),
                    "risk_label_rate": float(decile_frame["risk_label"].mean()),
                    "avg_forward_log_return": float(pd.to_numeric(decile_frame["forward_log_return"], errors="coerce").mean()),
                    "avg_future_return_5d": float(pd.to_numeric(decile_frame["future_return_5d"], errors="coerce").mean()),
                    "avg_future_max_drawdown_5d": float(pd.to_numeric(decile_frame["future_max_drawdown_5d"], errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows)


def _tail_risk_models() -> dict[str, Any]:
    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=80, min_samples_leaf=20, random_state=42, n_jobs=1, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }


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


def _classification_metric_row(y_true: pd.Series, score: np.ndarray, *, model_name: str) -> dict[str, Any]:
    pred = score >= 0.5
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, pred.astype(int), labels=labels).ravel()
    return {
        "model_name": model_name,
        "rows": int(len(y_true)),
        "risk_rate": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, pred)),
        "risk_precision": float(precision_score(y_true, pred, zero_division=0)),
        "risk_recall": float(recall_score(y_true, pred, zero_division=0)),
        "risk_f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": _safe_auc(y_true, score),
        "pr_auc": _safe_average_precision(y_true, score),
        "brier": float(brier_score_loss(y_true, score)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def _safe_auc(y_true: pd.Series, score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, score)) if y_true.nunique() > 1 else float("nan")


def _safe_average_precision(y_true: pd.Series, score: np.ndarray) -> float:
    return float(average_precision_score(y_true, score)) if y_true.nunique() > 1 else float("nan")
