from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from stocks_analyzer.ml_dataset import DatasetSplit
from stocks_analyzer.ml_evaluation import evaluate_trained_artifact
from stocks_analyzer.ml_models import train_and_save_models


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_split() -> DatasetSplit:
    rng = np.random.default_rng(7)
    size = 200
    feature_a = rng.normal(size=size)
    feature_b = rng.normal(size=size)
    feature_c = rng.normal(size=size)
    raw_score = 1.2 * feature_a - 0.7 * feature_b + 0.4 * feature_c
    label = (raw_score > 0.1).astype(int)
    future_return = raw_score / 20
    future_drawdown = np.clip(0.08 - (raw_score / 40), 0.0, 0.2)

    dataframe = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=size, freq="B"),
            "symbol": [f"{600000 + idx % 5:06d}" for idx in range(size)],
            "name": ["测试"] * size,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "feature_c": feature_c,
            "future_20d_return": future_return,
            "future_20d_max_drawdown": future_drawdown,
            "label_stable_up": label,
        }
    )
    return DatasetSplit(
        train=dataframe.iloc[:120].reset_index(drop=True),
        valid=dataframe.iloc[120:160].reset_index(drop=True),
        test=dataframe.iloc[160:].reset_index(drop=True),
        feature_columns=["feature_a", "feature_b", "feature_c"],
        label_column="label_stable_up",
    )


def test_evaluate_trained_artifact_generates_topn_and_quantile_reports() -> None:
    split = _build_split()
    tmp_path = _make_workspace_tmp_dir("ml_eval")
    artifact = train_and_save_models(split, ["xgboost"], tmp_path)[0]

    report = evaluate_trained_artifact(artifact, split, top_n_list=(5, 10))

    assert not report.valid_topn.empty
    assert not report.test_topn.empty
    assert not report.valid_quantiles.empty
    assert not report.test_quantiles.empty
    assert report.test_topn["top_n"].tolist() == [5, 10]
