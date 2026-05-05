from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


PREDICT_MODEL_VERSION = "v42_gate_v4_rank"


def predict_model_report_dir(project_root: Path) -> Path:
    return project_root / "reports" / "predict_model"


def predict_model_predictions_path(project_root: Path, trade_date: date) -> Path:
    return predict_model_report_dir(project_root) / f"predictions_{trade_date.isoformat()}.csv"


def save_predict_model_predictions(
    predictions: pd.DataFrame,
    *,
    project_root: Path,
    trade_date: date,
    output: str | None = None,
    model_version: str = PREDICT_MODEL_VERSION,
) -> Path:
    target = Path(output) if output else predict_model_predictions_path(project_root, trade_date)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = predictions.copy()
    if "model_version" not in frame.columns:
        insert_at = 1 if "trade_date" in frame.columns else 0
        frame.insert(insert_at, "model_version", model_version)
    frame.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def load_predict_model_predictions(*, project_root: Path, trade_date: date) -> pd.DataFrame:
    path = predict_model_predictions_path(project_root, trade_date)
    if not path.exists():
        raise FileNotFoundError(f"Predict model predictions not found: {path}")
    return pd.read_csv(path)
