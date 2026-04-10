from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .ml_dataset import DatasetSplit
from .ml_models import TrainedModelArtifact


@dataclass(slots=True)
class ModelEvaluationReport:
    model_name: str
    valid_quantiles: pd.DataFrame
    test_quantiles: pd.DataFrame
    valid_topn: pd.DataFrame
    test_topn: pd.DataFrame


def evaluate_trained_artifact(
    artifact: TrainedModelArtifact,
    split: DatasetSplit,
    top_n_list: tuple[int, ...],
    quantiles: int = 10,
) -> ModelEvaluationReport:
    valid_scored = _score_frame(artifact, split.valid, split.feature_columns, split.label_column)
    test_scored = _score_frame(artifact, split.test, split.feature_columns, split.label_column)
    return ModelEvaluationReport(
        model_name=artifact.model_name,
        valid_quantiles=_build_quantile_report(valid_scored, split.label_column, quantiles),
        test_quantiles=_build_quantile_report(test_scored, split.label_column, quantiles),
        valid_topn=_build_topn_report(valid_scored, split.label_column, top_n_list),
        test_topn=_build_topn_report(test_scored, split.label_column, top_n_list),
    )


def _score_frame(
    artifact: TrainedModelArtifact,
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> pd.DataFrame:
    scored = frame.copy()
    scored["success_prob"] = artifact.estimator.predict_proba(scored.loc[:, feature_columns])[:, 1]
    return scored.loc[
        :,
        [
            column
            for column in [
                "trade_date",
                "symbol",
                "name",
                label_column,
                "future_20d_return",
                "future_20d_max_drawdown",
                "success_prob",
            ]
            if column in scored.columns
        ],
    ].copy()


def _build_quantile_report(scored: pd.DataFrame, label_column: str, quantiles: int) -> pd.DataFrame:
    frame = scored.copy()
    rank = frame["success_prob"].rank(method="first", pct=True)
    # bucket 1 is the highest-probability group.
    frame["prob_bucket"] = ((1 - rank) * quantiles).astype(int) + 1
    frame["prob_bucket"] = frame["prob_bucket"].clip(upper=quantiles)

    grouped = (
        frame.groupby("prob_bucket", sort=True)
        .agg(
            count=("success_prob", "size"),
            avg_prob=("success_prob", "mean"),
            hit_rate=(label_column, "mean"),
            avg_future_return=("future_20d_return", "mean"),
            avg_future_drawdown=("future_20d_max_drawdown", "mean"),
        )
        .reset_index()
        .sort_values("prob_bucket")
        .reset_index(drop=True)
    )
    return grouped


def _build_topn_report(scored: pd.DataFrame, label_column: str, top_n_list: tuple[int, ...]) -> pd.DataFrame:
    grouped_days = list(scored.groupby("trade_date", sort=True))
    rows: list[dict[str, float | int]] = []

    for top_n in top_n_list:
        daily_rows: list[dict[str, float]] = []
        for _, day_frame in grouped_days:
            subset = day_frame.sort_values("success_prob", ascending=False).head(top_n)
            if subset.empty:
                continue
            daily_rows.append(
                {
                    "hit_rate": float(subset[label_column].mean()),
                    "avg_future_return": float(subset["future_20d_return"].mean()),
                    "avg_future_drawdown": float(subset["future_20d_max_drawdown"].mean()),
                }
            )

        if not daily_rows:
            continue

        daily = pd.DataFrame(daily_rows)
        rows.append(
            {
                "top_n": top_n,
                "days": len(daily_rows),
                "hit_rate": float(daily["hit_rate"].mean()),
                "avg_future_return": float(daily["avg_future_return"].mean()),
                "avg_future_drawdown": float(daily["avg_future_drawdown"].mean()),
            }
        )

    return pd.DataFrame(rows)
