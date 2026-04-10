from __future__ import annotations

import json
import logging
import pickle
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, precision_recall_curve, roc_auc_score, auc
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .ml_dataset import DatasetSplit


AVAILABLE_MODELS = ("xgboost",)


@dataclass(slots=True)
class TrainedModelArtifact:
    model_name: str
    backend: str
    estimator: object
    feature_columns: list[str]
    metrics: dict[str, float]
    model_path: Path
    metadata_path: Path


def normalize_model_names(model_name: str) -> list[str]:
    aliases = {
        "all": "xgboost",
        "xgb": "xgboost",
    }
    normalized = aliases.get(model_name, model_name)
    if normalized not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    return [normalized]


def train_and_save_models(split: DatasetSplit, model_names: list[str], output_dir: Path) -> list[TrainedModelArtifact]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[TrainedModelArtifact] = []
    for model_name in model_names:
        runtime = _resolve_runtime_config(model_name)
        logging.info("Training probability model: %s (device=%s)", model_name, runtime["backend"])
        started_at = perf_counter()
        X_train = split.train.loc[:, split.feature_columns]
        y_train = split.train.loc[:, split.label_column].astype(int)
        X_valid = split.valid.loc[:, split.feature_columns]
        y_valid = split.valid.loc[:, split.label_column].astype(int)
        X_test = split.test.loc[:, split.feature_columns]
        y_test = split.test.loc[:, split.label_column].astype(int)

        estimator = _fit_model_with_progress(
            model_name=model_name,
            runtime=runtime,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
        )
        valid_prob = estimator.predict_proba(X_valid)[:, 1]
        test_prob = estimator.predict_proba(X_test)[:, 1]

        metrics = _evaluate_probabilities(
            valid_true=y_valid,
            valid_prob=valid_prob,
            test_true=y_test,
            test_prob=test_prob,
        )

        model_path = output_dir / f"{model_name}.pkl"
        metadata_path = output_dir / f"{model_name}.json"
        _save_artifact(
            model_path=model_path,
            metadata_path=metadata_path,
            model_name=model_name,
            backend=runtime["backend"],
            estimator=estimator,
            feature_columns=split.feature_columns,
            metrics=metrics,
        )
        artifacts.append(
            TrainedModelArtifact(
                model_name=model_name,
                backend=runtime["backend"],
                estimator=estimator,
                feature_columns=split.feature_columns,
                metrics=metrics,
                model_path=model_path,
                metadata_path=metadata_path,
            )
        )
        elapsed = perf_counter() - started_at
        logging.info(
            "Finished probability model: %s in %.2fs (device=%s, valid_roc_auc=%.4f, test_roc_auc=%.4f)",
            model_name,
            elapsed,
            runtime["backend"],
            metrics["valid_roc_auc"],
            metrics["test_roc_auc"],
        )
    return artifacts


def load_model_artifact(model_path: Path) -> dict[str, object]:
    with model_path.open("rb") as file:
        return pickle.load(file)


def predict_with_model(
    artifact: dict[str, object],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    estimator = artifact["estimator"]
    feature_columns = artifact["feature_columns"]
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns for prediction: {missing[:10]}")

    result = frame.copy()
    result["success_prob"] = estimator.predict_proba(result.loc[:, feature_columns])[:, 1]
    result = result.sort_values("success_prob", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    result["model_name"] = str(artifact["model_name"])
    return result


def _build_estimator(model_name: str, runtime: dict[str, object]) -> object:
    if model_name == "xgboost":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        eval_metric="logloss",
                        **runtime["params"],
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported model: {model_name}")


def _fit_model_with_progress(
    model_name: str,
    runtime: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> object:
    if model_name == "xgboost":
        return _fit_xgboost_with_progress(runtime, X_train, y_train, X_valid, y_valid)
    raise ValueError(f"Unsupported model: {model_name}")


def _fit_xgboost_with_progress(
    runtime: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> Pipeline:
    pipeline = _build_estimator("xgboost", runtime)
    imputer = pipeline.named_steps["imputer"]
    model = pipeline.named_steps["model"]
    X_train_transformed = imputer.fit_transform(X_train)
    X_valid_transformed = imputer.transform(X_valid)

    total_rounds = int(model.get_params().get("n_estimators", 300))
    model.set_params(callbacks=[_XGBoostProgressCallback(total_rounds=total_rounds, log_every=25)])
    model.fit(
        X_train_transformed,
        y_train,
        eval_set=[(X_valid_transformed, y_valid)],
        verbose=False,
    )
    return pipeline


class _XGBoostProgressCallback(xgb.callback.TrainingCallback):
    def __init__(self, total_rounds: int, log_every: int) -> None:
        self.total_rounds = total_rounds
        self.log_every = log_every
        self.started_at = perf_counter()

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        current = epoch + 1
        if current == 1 or current == self.total_rounds or current % self.log_every == 0:
            elapsed = perf_counter() - self.started_at
            eta = max((elapsed / current) * (self.total_rounds - current), 0.0)
            logging.info("[xgboost] round %s/%s elapsed=%.2fs eta=%.2fs", current, self.total_rounds, elapsed, eta)
        return False


def _evaluate_probabilities(
    valid_true: pd.Series,
    valid_prob: pd.Series,
    test_true: pd.Series,
    test_prob: pd.Series,
) -> dict[str, float]:
    return {
        "valid_roc_auc": float(roc_auc_score(valid_true, valid_prob)),
        "valid_pr_auc": float(auc(*precision_recall_curve(valid_true, valid_prob)[1::-1])),
        "valid_log_loss": float(log_loss(valid_true, valid_prob, labels=[0, 1])),
        "valid_brier_score": float(brier_score_loss(valid_true, valid_prob)),
        "test_roc_auc": float(roc_auc_score(test_true, test_prob)),
        "test_pr_auc": float(auc(*precision_recall_curve(test_true, test_prob)[1::-1])),
        "test_log_loss": float(log_loss(test_true, test_prob, labels=[0, 1])),
        "test_brier_score": float(brier_score_loss(test_true, test_prob)),
    }


def _save_artifact(
    model_path: Path,
    metadata_path: Path,
    model_name: str,
    backend: str,
    estimator: object,
    feature_columns: list[str],
    metrics: dict[str, float],
) -> None:
    payload = {
        "model_name": model_name,
        "backend": backend,
        "estimator": estimator,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with model_path.open("wb") as file:
        pickle.dump(payload, file)

    metadata = {
        "model_name": model_name,
        "backend": backend,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "trained_at": payload["trained_at"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_runtime_config(model_name: str) -> dict[str, object]:
    if model_name == "xgboost":
        if _cuda_runtime_available() and _xgboost_cuda_supported():
            return {"backend": "cuda", "params": {"device": "cuda", "tree_method": "hist"}}
        return {"backend": "cpu", "params": {"device": "cpu", "tree_method": "hist"}}

    raise ValueError(f"Unsupported model: {model_name}")


def _cuda_runtime_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:
        return False
    return bool(result.stdout.strip())


def _xgboost_cuda_supported() -> bool:
    try:
        build_info = xgb.build_info()
    except Exception:
        return False
    return bool(build_info.get("USE_CUDA"))
