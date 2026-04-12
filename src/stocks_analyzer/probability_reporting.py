from __future__ import annotations

from pathlib import Path

import pandas as pd

from .ml_evaluation import ModelEvaluationReport
from .ml_models import TrainedModelArtifact


def format_training_summary(artifacts: list[TrainedModelArtifact]) -> str:
    if not artifacts:
        return "No models were trained."

    rows = []
    for artifact in artifacts:
        rows.append(
            {
                "model": artifact.model_name,
                "device": artifact.backend,
                "valid_roc_auc": round(artifact.metrics["valid_roc_auc"], 4),
                "test_roc_auc": round(artifact.metrics["test_roc_auc"], 4),
                "valid_pr_auc": round(artifact.metrics["valid_pr_auc"], 4),
                "test_pr_auc": round(artifact.metrics["test_pr_auc"], 4),
                "model_path": str(artifact.model_path),
            }
        )
    return pd.DataFrame(rows).to_string(index=False)


def format_evaluation_summary(reports: list[ModelEvaluationReport]) -> str:
    if not reports:
        return "No evaluation reports generated."

    parts: list[str] = []
    for report in reports:
        parts.append(f"[{report.model_name}] Valid TopN")
        parts.append(_format_percent_table(report.valid_topn))
        parts.append("")
        parts.append(f"[{report.model_name}] Test TopN")
        parts.append(_format_percent_table(report.test_topn))
        parts.append("")
        parts.append(f"[{report.model_name}] Test Quantiles")
        parts.append(_format_percent_table(report.test_quantiles))
        parts.append("")
    return "\n".join(parts).strip()


def format_prediction_summary(predictions: pd.DataFrame, limit: int) -> str:
    if predictions.empty:
        return "No probability predictions generated."
    columns = [column for column in ("trade_date", "symbol", "name", "success_prob", "rank", "model_name", "all_rating") if column in predictions.columns]
    display = predictions.loc[:, columns].head(limit).copy()
    if "success_prob" in display.columns:
        display["success_prob"] = display["success_prob"].map(lambda value: f"{value:.4f}")
    if "all_rating" in display.columns:
        display["all_rating"] = display["all_rating"].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    return display.to_string(index=False)


def format_tradingview_summary(scores: pd.DataFrame, limit: int) -> str:
    if scores.empty:
        return "No TradingView ratings generated."
    rating_date_columns = sorted(column for column in scores.columns if column.startswith("all_rating_20"))
    columns = [
        column
        for column in (
            "symbol",
            "name",
            *rating_date_columns,
            "avg_all_rating_5d",
            "ma_rating",
            "osc_rating",
            "all_rating",
            "all_rating_label",
        )
        if column in scores.columns
    ]
    display = scores.loc[:, columns].head(limit).copy()
    for column in ("ma_rating", "osc_rating", "all_rating", "avg_all_rating_5d", *rating_date_columns):
        if column in display.columns:
            display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    return display.to_string(index=False)


def save_predictions_report(predictions: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def save_evaluation_reports(reports: list[ModelEvaluationReport], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for report in reports:
        pairs = {
            f"{report.model_name}_valid_topn.csv": report.valid_topn,
            f"{report.model_name}_test_topn.csv": report.test_topn,
            f"{report.model_name}_valid_quantiles.csv": report.valid_quantiles,
            f"{report.model_name}_test_quantiles.csv": report.test_quantiles,
        }
        for filename, frame in pairs.items():
            path = output_dir / filename
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            saved.append(path)
    return saved


def _format_percent_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    display = frame.copy()
    for column in ("avg_prob", "hit_rate", "avg_future_return", "avg_future_drawdown"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: f"{value:.2%}")
    return display.to_string(index=False)
