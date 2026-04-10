from pathlib import Path
from uuid import uuid4

import logging
import numpy as np
import pandas as pd

from stocks_analyzer.ml_dataset import DatasetSplit
from stocks_analyzer.ml_models import (
    _resolve_runtime_config,
    load_model_artifact,
    normalize_model_names,
    predict_with_model,
    train_and_save_models,
)


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_split() -> DatasetSplit:
    rng = np.random.default_rng(42)
    size = 180
    feature_a = rng.normal(size=size)
    feature_b = rng.normal(size=size)
    feature_c = rng.normal(size=size)
    raw_score = 1.3 * feature_a - 0.8 * feature_b + 0.5 * feature_c
    label = (raw_score > 0).astype(int)

    dataframe = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=size, freq="D"),
            "symbol": [f"{600000 + idx % 3:06d}" for idx in range(size)],
            "feature_a": feature_a,
            "feature_b": feature_b,
            "feature_c": feature_c,
            "label_stable_up": label,
        }
    )
    feature_columns = ["feature_a", "feature_b", "feature_c"]
    return DatasetSplit(
        train=dataframe.iloc[:100].reset_index(drop=True),
        valid=dataframe.iloc[100:140].reset_index(drop=True),
        test=dataframe.iloc[140:].reset_index(drop=True),
        feature_columns=feature_columns,
        label_column="label_stable_up",
    )


def test_normalize_model_names_accepts_aliases() -> None:
    assert normalize_model_names("all") == ["xgboost"]
    assert normalize_model_names("xgb") == ["xgboost"]


def test_train_and_predict_models_round_trip() -> None:
    split = _build_split()
    tmp_path = _make_workspace_tmp_dir("ml_models_round_trip")

    artifacts = train_and_save_models(split, ["xgboost"], tmp_path)

    assert {artifact.model_name for artifact in artifacts} == {"xgboost"}
    for artifact in artifacts:
        assert artifact.model_path.exists()
        assert artifact.metadata_path.exists()
        assert artifact.backend in {"cpu", "cuda", "gpu"}
        payload = load_model_artifact(artifact.model_path)
        assert payload["backend"] in {"cpu", "cuda", "gpu"}
        predictions = predict_with_model(payload, split.test.copy())
        assert "success_prob" in predictions.columns
        assert predictions["success_prob"].between(0, 1).all()


def test_train_and_save_models_logs_model_progress(caplog) -> None:
    split = _build_split()
    tmp_path = _make_workspace_tmp_dir("ml_models_logs")

    with caplog.at_level(logging.INFO):
        train_and_save_models(split, ["xgboost"], tmp_path)

    messages = [record.getMessage() for record in caplog.records]
    assert any("Training probability model: xgboost" in message for message in messages)
    assert any("Finished probability model: xgboost" in message for message in messages)


def test_resolve_runtime_config_prefers_cuda_for_xgboost(monkeypatch) -> None:
    monkeypatch.setattr("stocks_analyzer.ml_models._cuda_runtime_available", lambda: True)
    monkeypatch.setattr("stocks_analyzer.ml_models._xgboost_cuda_supported", lambda: True)

    runtime = _resolve_runtime_config("xgboost")

    assert runtime["backend"] == "cuda"
    assert runtime["params"]["device"] == "cuda"
