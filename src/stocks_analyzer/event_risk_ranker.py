from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json
import math
import pickle
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .event_features import FEATURE_COLUMNS, add_rule_baseline_scores, build_event_features
from .event_labels import (
    DEFAULT_HOLDING_DAYS_GRID,
    DEFAULT_STOP_ATR_GRID,
    DEFAULT_TAKE_ATR_GRID,
    EventLabelConfig,
    add_rank_labels,
    build_event_labels,
    build_prediction_events,
)
from .event_watchlist import build_watchlist_event_payload, watchlist_event_path, write_watchlist_event
from .models import AppConfig
from .pattern_backtest import scan_pattern_backtest_signals
from .storage import DailyBarsReadError, Storage
from .strategies import STRATEGY_NAMES


EVENT_RISK_RANKER_VERSION = "event_risk_ranker_v1"


@dataclass(slots=True)
class EventRiskRankerResult:
    labels: pd.DataFrame
    features: pd.DataFrame
    skipped_events: pd.DataFrame
    predictions: pd.DataFrame
    topn_metrics: pd.DataFrame
    summary: pd.DataFrame
    model_path: Path
    metadata_path: Path
    report_dir: Path


class ConstantRiskModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


class ConstantRankModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value, dtype=float)


def event_risk_ranker_report_dir(project_root: Path) -> Path:
    return project_root / "reports" / "event_risk_ranker"


def event_risk_ranker_model_dir(project_root: Path) -> Path:
    return project_root / "data" / "ml" / "event_risk_ranker"


def event_risk_ranker_predictions_path(project_root: Path, trade_date: date) -> Path:
    return event_risk_ranker_report_dir(project_root) / f"predictions_{trade_date.isoformat()}.csv"


def build_event_risk_ranker_dataset(
    *,
    storage: Storage,
    config: AppConfig,
    start_date: date,
    end_date: date,
    limit: int | None = None,
    label_config: EventLabelConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_storage = _LimitedStorage(storage, limit)
    effective_config = label_config or EventLabelConfig()
    signals = scan_pattern_backtest_signals(
        selected_storage,
        config,
        start_date=start_date,
        end_date=end_date,
        selected_strategies=STRATEGY_NAMES,
        cooldown_trading_days=1,
        progress_callback=progress_callback,
    )
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    histories = _load_history_for_symbols(storage, signals["symbol"].astype(str).str.zfill(6).unique())
    labels, skipped = build_event_labels(signals, histories, config=effective_config)
    labels = add_rank_labels(labels)
    if labels.empty:
        return labels, pd.DataFrame(), skipped
    features = build_event_features(labels, histories)
    features = add_rule_baseline_scores(features, min_avg_amount_20d=float(config.universe.min_avg_amount_20d))
    return labels, features, skipped


def train_event_risk_ranker_model(
    *,
    storage: Storage,
    config: AppConfig,
    project_root: Path,
    start_date: date,
    end_date: date,
    limit: int | None = None,
    top_n_list: tuple[int, ...] = (10, 20),
    stop_atr_grid: tuple[float, ...] = DEFAULT_STOP_ATR_GRID,
    take_atr_grid: tuple[float, ...] = DEFAULT_TAKE_ATR_GRID,
    holding_days_grid: tuple[int, ...] = DEFAULT_HOLDING_DAYS_GRID,
    max_iter: int = 80,
    prediction_date: date | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EventRiskRankerResult:
    report_dir = event_risk_ranker_report_dir(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    label_config = _select_label_config(stop_atr_grid, take_atr_grid, holding_days_grid)
    labels, features, skipped = build_event_risk_ranker_dataset(
        storage=storage,
        config=config,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        label_config=label_config,
        progress_callback=progress_callback,
    )
    if features.empty:
        raise RuntimeError("No event-risk-ranker training dataset could be built from local pattern events.")

    train, valid = _last_fraction_split(features, valid_fraction=0.25)
    artifact = _fit_artifact(
        train,
        valid,
        top_n_list=top_n_list,
        max_iter=max_iter,
        label_config=label_config,
        stop_atr_grid=stop_atr_grid,
        take_atr_grid=take_atr_grid,
        holding_days_grid=holding_days_grid,
    )
    scored = score_event_risk_ranker_frame(features, artifact=artifact)
    metrics = evaluate_event_ranker_topn(scored, top_n_list=top_n_list, model_name="event_risk_ranker", dataset_split="train_full")
    summary = summarize_event_ranker_metrics(metrics)
    model_path, metadata_path = save_event_risk_ranker_artifact(project_root, artifact)
    _save_training_reports(
        report_dir=report_dir,
        start_date=start_date,
        end_date=end_date,
        labels=labels,
        features=features,
        skipped=skipped,
        metrics=metrics,
        summary=summary,
    )

    latest_predictions = pd.DataFrame()
    if prediction_date is not None:
        latest_predictions = predict_event_risk_ranker(
            storage=storage,
            config=config,
            project_root=project_root,
            trade_date=prediction_date,
        )
        latest_predictions.to_csv(event_risk_ranker_predictions_path(project_root, prediction_date), index=False, encoding="utf-8-sig")
        payload = build_watchlist_event_payload(latest_predictions, trade_date=prediction_date, limit=max(top_n_list))
        write_watchlist_event(payload, watchlist_event_path(project_root, prediction_date))

    return EventRiskRankerResult(
        labels=labels,
        features=features,
        skipped_events=skipped,
        predictions=latest_predictions,
        topn_metrics=metrics,
        summary=summary,
        model_path=model_path,
        metadata_path=metadata_path,
        report_dir=report_dir,
    )


def validate_event_risk_ranker_walkforward(
    *,
    storage: Storage,
    config: AppConfig,
    project_root: Path,
    start_date: date,
    end_date: date,
    limit: int | None = None,
    windows: int = 8,
    train_days: int = 280,
    valid_days: int = 60,
    test_days: int = 60,
    top_n_list: tuple[int, ...] = (10, 20),
    stop_atr_grid: tuple[float, ...] = DEFAULT_STOP_ATR_GRID,
    take_atr_grid: tuple[float, ...] = DEFAULT_TAKE_ATR_GRID,
    holding_days_grid: tuple[int, ...] = DEFAULT_HOLDING_DAYS_GRID,
    max_iter: int = 40,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    report_dir = event_risk_ranker_report_dir(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    label_config = _select_label_config(stop_atr_grid, take_atr_grid, holding_days_grid)
    labels, features, skipped = build_event_risk_ranker_dataset(
        storage=storage,
        config=config,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        label_config=label_config,
        progress_callback=progress_callback,
    )
    if features.empty:
        raise RuntimeError("No event-risk-ranker validation dataset could be built from local pattern events.")

    dates = sorted(pd.to_datetime(features["signal_date"]).dt.date.unique())
    windows_frame = _build_walkforward_windows(
        dates,
        windows=windows,
        train_days=train_days,
        valid_days=valid_days,
        test_days=test_days,
        embargo_days=max(holding_days_grid),
    )
    metric_frames: list[pd.DataFrame] = []
    for window in windows_frame.to_dict("records"):
        train = _date_slice(features, window["train_start"], window["train_end"])
        valid = _date_slice(features, window["valid_start"], window["valid_end"])
        test = _date_slice(features, window["test_start"], window["test_end"])
        if train.empty or valid.empty or test.empty:
            continue
        artifact = _fit_artifact(
            train,
            valid,
            top_n_list=top_n_list,
            max_iter=max_iter,
            label_config=label_config,
            stop_atr_grid=stop_atr_grid,
            take_atr_grid=take_atr_grid,
            holding_days_grid=holding_days_grid,
        )
        scored_test = score_event_risk_ranker_frame(test, artifact=artifact)
        metrics = evaluate_event_ranker_topn(
            scored_test,
            top_n_list=top_n_list,
            model_name="event_risk_ranker",
            dataset_split="test",
        )
        metrics["window_id"] = int(window["window_id"])
        metric_frames.append(metrics)

    metrics_frame = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    summary = summarize_event_ranker_metrics(metrics_frame)
    labels.to_csv(report_dir / f"event_labels_{start_date.isoformat()}_{end_date.isoformat()}.csv", index=False, encoding="utf-8-sig")
    features.to_csv(report_dir / f"event_features_{start_date.isoformat()}_{end_date.isoformat()}.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(report_dir / "skipped_events.csv", index=False, encoding="utf-8-sig")
    windows_frame.to_csv(report_dir / "walkforward_windows.csv", index=False, encoding="utf-8-sig")
    metrics_frame.to_csv(report_dir / "walkforward_topn_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(report_dir / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    return windows_frame, metrics_frame, summary


def predict_event_risk_ranker(
    *,
    storage: Storage,
    config: AppConfig,
    project_root: Path,
    trade_date: date,
    output: str | None = None,
) -> pd.DataFrame:
    artifact = load_event_risk_ranker_artifact(project_root)
    signals = scan_pattern_backtest_signals(
        storage,
        config,
        start_date=trade_date,
        end_date=trade_date,
        selected_strategies=STRATEGY_NAMES,
        cooldown_trading_days=1,
    )
    if signals.empty:
        predictions = pd.DataFrame()
    else:
        histories = _load_history_for_symbols(storage, signals["symbol"].astype(str).str.zfill(6).unique())
        events, skipped = build_prediction_events(signals, histories, config=artifact["label_config"])
        if events.empty:
            predictions = _skipped_prediction_rows(skipped, trade_date=trade_date)
        else:
            features = build_event_features(events, histories)
            features = add_rule_baseline_scores(features, min_avg_amount_20d=float(config.universe.min_avg_amount_20d))
            predictions = score_event_risk_ranker_frame(features, artifact=artifact)
            if not skipped.empty:
                predictions = pd.concat([predictions, _skipped_prediction_rows(skipped, trade_date=trade_date)], ignore_index=True)

    output_path = Path(output) if output else event_risk_ranker_predictions_path(project_root, trade_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
    if not predictions.empty:
        payload = build_watchlist_event_payload(predictions, trade_date=trade_date)
        write_watchlist_event(payload, watchlist_event_path(project_root, trade_date))
    return predictions


def score_event_risk_ranker_frame(frame: pd.DataFrame, *, artifact: dict[str, Any]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    X = _model_matrix(result, artifact["feature_columns"])
    result["p_stop_first"] = artifact["risk_model"].predict_proba(X)[:, 1]
    risk_threshold = float(artifact["risk_threshold"])
    result["risk_pass"] = result["p_stop_first"] <= risk_threshold
    result["risk_tier"] = np.select(
        [result["p_stop_first"] <= risk_threshold, result["p_stop_first"] <= risk_threshold + 0.15],
        ["low", "medium"],
        default="high",
    )
    result["risk_reason"] = np.where(result["risk_pass"], "passed", "p_stop_first_threshold")
    result["rank_score"] = artifact["rank_model"].predict(X)
    result["expected_R_score"] = result["rank_score"]
    result["rank_pct"] = _daily_pct(result, "rank_score")
    result["expected_R_score_pct"] = _daily_pct(result, "expected_R_score")
    result["rule_score_pct"] = _daily_pct(result, "rule_score")
    result["final_score"] = (
        0.65 * result["rank_pct"].fillna(0.0)
        + 0.25 * result["expected_R_score_pct"].fillna(0.0)
        + 0.10 * result["rule_score_pct"].fillna(0.0)
    )
    result["trade_permission"] = "no_trade"
    for signal_date, day in result.groupby("signal_date", sort=False):
        risk_passed = day[day["risk_pass"].fillna(False).astype(bool)]
        allow = not risk_passed.empty and float(risk_passed["final_score"].median()) >= float(artifact.get("opportunity_threshold", 0.0))
        result.loc[day.index, "trade_permission"] = "allow" if allow else "no_trade"
    result["suggested_action"] = np.select(
        [result["risk_pass"] & result["trade_permission"].eq("allow"), result["risk_pass"]],
        ["candidate", "observe"],
        default="avoid",
    )
    result["pattern_ids"] = result["pattern_id"].astype(str)
    result["entry_price_ref"] = pd.to_numeric(result.get("entry_price"), errors="coerce")
    result["stop_loss_price_ref"] = pd.to_numeric(result.get("stop_loss_price"), errors="coerce")
    result["take_profit_price_ref"] = pd.to_numeric(result.get("take_profit_price"), errors="coerce")
    result["model_version"] = EVENT_RISK_RANKER_VERSION
    result["_trade_permission_rank"] = result["trade_permission"].map({"allow": 0, "no_trade": 1}).fillna(2)
    result["_action_rank"] = result["suggested_action"].map({"candidate": 0, "observe": 1, "avoid": 2}).fillna(3)
    return (
        result.sort_values(["_trade_permission_rank", "_action_rank", "final_score"], ascending=[True, True, False])
        .drop(columns=["_trade_permission_rank", "_action_rank"])
        .reset_index(drop=True)
    )


def evaluate_event_ranker_topn(
    frame: pd.DataFrame,
    *,
    top_n_list: tuple[int, ...],
    model_name: str,
    dataset_split: str,
) -> pd.DataFrame:
    if frame.empty or "realized_R" not in frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for scope in ("event", "symbol_dedup"):
        scoped = frame.copy()
        if scope == "symbol_dedup":
            scoped = scoped.sort_values(["signal_date", "symbol", "final_score"], ascending=[True, True, False]).drop_duplicates(
                ["signal_date", "symbol"], keep="first"
            )
        tradable = scoped[scoped["suggested_action"].isin(["candidate", "observe"])].copy()
        for top_n in top_n_list:
            selected_parts = []
            for _, day in tradable.groupby("signal_date", sort=False):
                allowed = day[day["trade_permission"].eq("allow")]
                selected_parts.append(allowed.sort_values("final_score", ascending=False).head(top_n))
            selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
            rows.append(_metric_row(selected, frame=scoped, top_n=top_n, model_name=model_name, dataset_split=dataset_split, scope=scope))
    return pd.DataFrame(rows)


def summarize_event_ranker_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    for (model_name, scope, top_n), group in metrics.groupby(["model_name", "scope", "top_n"], dropna=False):
        row = {"model_name": model_name, "scope": scope, "top_n": top_n, "windows": int(len(group))}
        for column in ("avg_realized_R", "median_realized_R", "profit_factor_R", "stop_first_rate", "coverage_days_rate"):
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean()) if values.notna().any() else math.nan
            row[f"{column}_min"] = float(values.min()) if values.notna().any() else math.nan
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def save_event_risk_ranker_artifact(project_root: Path, artifact: dict[str, Any]) -> tuple[Path, Path]:
    model_dir = event_risk_ranker_model_dir(project_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "event_risk_ranker.pkl"
    metadata_path = model_dir / "event_risk_ranker_metadata.json"
    with model_path.open("wb") as file:
        pickle.dump(artifact, file)
    metadata = {
        "model_version": EVENT_RISK_RANKER_VERSION,
        "feature_columns": artifact["feature_columns"],
        "risk_threshold": artifact["risk_threshold"],
        "opportunity_threshold": artifact.get("opportunity_threshold", 0.0),
        "label_config": {
            "stop_atr_mult": artifact["label_config"].stop_atr_mult,
            "take_atr_mult": artifact["label_config"].take_atr_mult,
            "max_holding_days": artifact["label_config"].max_holding_days,
            "min_history_days": artifact["label_config"].min_history_days,
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_path, metadata_path


def load_event_risk_ranker_artifact(project_root: Path) -> dict[str, Any]:
    model_path = event_risk_ranker_model_dir(project_root) / "event_risk_ranker.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Event risk ranker model not found: {model_path}. Run train-event-risk-ranker first.")
    with model_path.open("rb") as file:
        return pickle.load(file)


def format_event_ranker_prediction_table(frame: pd.DataFrame, *, limit: int = 20) -> str:
    if frame.empty:
        return "No event risk ranker predictions."
    columns = [
        "symbol",
        "name",
        "pattern_id",
        "suggested_action",
        "trade_permission",
        "p_stop_first",
        "final_score",
        "entry_price_ref",
        "stop_loss_price_ref",
        "take_profit_price_ref",
    ]
    display = frame[[column for column in columns if column in frame.columns]].head(limit)
    return display.to_string(index=False)


def _fit_artifact(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    top_n_list: tuple[int, ...],
    max_iter: int,
    label_config: EventLabelConfig,
    stop_atr_grid: tuple[float, ...],
    take_atr_grid: tuple[float, ...],
    holding_days_grid: tuple[int, ...],
) -> dict[str, Any]:
    feature_columns = [column for column in FEATURE_COLUMNS if column in train.columns]
    risk_model = _fit_risk_model(train, feature_columns, max_iter=max_iter)
    risk_threshold = _select_risk_threshold(risk_model, valid, feature_columns)
    train_scored = train.copy()
    train_scored["p_stop_first"] = risk_model.predict_proba(_model_matrix(train_scored, feature_columns))[:, 1]
    rank_train = train_scored[train_scored["p_stop_first"].le(risk_threshold) & train_scored["rank_train_eligible"].fillna(False).astype(bool)]
    rank_model = _fit_rank_model(rank_train, feature_columns, max_iter=max_iter)
    valid_scored = score_event_risk_ranker_frame(
        valid,
        artifact={
            "feature_columns": feature_columns,
            "risk_model": risk_model,
            "rank_model": rank_model,
            "risk_threshold": risk_threshold,
            "opportunity_threshold": 0.0,
            "label_config": label_config,
        },
    )
    opportunity_threshold = _select_opportunity_threshold(valid_scored, top_n_list=top_n_list)
    return {
        "model_version": EVENT_RISK_RANKER_VERSION,
        "feature_columns": feature_columns,
        "risk_model": risk_model,
        "rank_model": rank_model,
        "risk_threshold": float(risk_threshold),
        "opportunity_threshold": float(opportunity_threshold),
        "label_config": label_config,
        "stop_atr_grid": tuple(stop_atr_grid),
        "take_atr_grid": tuple(take_atr_grid),
        "holding_days_grid": tuple(holding_days_grid),
    }


def _fit_risk_model(train: pd.DataFrame, feature_columns: list[str], *, max_iter: int) -> Any:
    y = pd.to_numeric(train.get("risk_label", pd.Series(dtype=float)), errors="coerce")
    valid = y.notna()
    if int(valid.sum()) < 20 or y.loc[valid].nunique() < 2:
        return ConstantRiskModel(float(y.loc[valid].mean()) if valid.any() else 0.5)
    model = HistGradientBoostingClassifier(max_iter=max_iter, random_state=42)
    model.fit(_model_matrix(train.loc[valid], feature_columns), y.loc[valid].astype(int), sample_weight=_sample_weight(train.loc[valid]))
    return model


def _fit_rank_model(train: pd.DataFrame, feature_columns: list[str], *, max_iter: int) -> Any:
    y = pd.to_numeric(train.get("rank_target", pd.Series(dtype=float)), errors="coerce")
    valid = y.notna()
    if int(valid.sum()) < 20 or y.loc[valid].nunique() < 2:
        return ConstantRankModel(float(y.loc[valid].mean()) if valid.any() else 0.5)
    model = HistGradientBoostingRegressor(max_iter=max_iter, random_state=42)
    model.fit(_model_matrix(train.loc[valid], feature_columns), y.loc[valid], sample_weight=_sample_weight(train.loc[valid]))
    return model


def _select_risk_threshold(model: Any, valid: pd.DataFrame, feature_columns: list[str]) -> float:
    if valid.empty:
        return 0.5
    scores = pd.Series(model.predict_proba(_model_matrix(valid, feature_columns))[:, 1], index=valid.index)
    thresholds = sorted(set(float(item) for item in scores.quantile([0.15, 0.25, 0.35, 0.50, 0.65, 0.80]).dropna().tolist()))
    if not thresholds:
        return float(scores.median()) if scores.notna().any() else 0.5
    best_threshold = thresholds[0]
    best_score = -math.inf
    for threshold in thresholds:
        selected = valid.loc[scores.le(threshold)]
        if selected.empty:
            continue
        avg_r = float(pd.to_numeric(selected["realized_R"], errors="coerce").mean())
        stop_rate = float(selected["barrier_outcome"].eq("stop_loss_first").mean())
        coverage = float(selected["signal_date"].nunique() / max(valid["signal_date"].nunique(), 1))
        objective = avg_r - 0.5 * stop_rate - max(0.0, 0.25 - coverage)
        if objective > best_score:
            best_score = objective
            best_threshold = threshold
    return float(best_threshold)


def _select_opportunity_threshold(scored_valid: pd.DataFrame, *, top_n_list: tuple[int, ...]) -> float:
    if scored_valid.empty or "final_score" not in scored_valid.columns:
        return 0.0
    risk_passed = scored_valid[scored_valid["risk_pass"].fillna(False).astype(bool)]
    if risk_passed.empty:
        return 0.0
    daily_median = pd.to_numeric(risk_passed["final_score"], errors="coerce").groupby(risk_passed["signal_date"]).median().dropna()
    if daily_median.empty:
        return 0.0
    thresholds = sorted(set(float(item) for item in daily_median.quantile([0.0, 0.25, 0.50, 0.65, 0.75, 0.85]).dropna()))
    top_n = min(top_n_list) if top_n_list else 10
    total_days = max(int(scored_valid["signal_date"].nunique()), 1)
    best_threshold = thresholds[0]
    best_score = -math.inf
    for threshold in thresholds:
        allowed_days = set(daily_median[daily_median.ge(threshold)].index)
        allowed = risk_passed[risk_passed["signal_date"].isin(allowed_days)]
        if allowed.empty:
            continue
        selected = allowed.sort_values(["signal_date", "final_score"], ascending=[True, False]).groupby("signal_date", group_keys=False).head(top_n)
        realized = pd.to_numeric(selected["realized_R"], errors="coerce")
        avg_r = float(realized.mean()) if realized.notna().any() else -math.inf
        stop_rate = float(selected["barrier_outcome"].eq("stop_loss_first").mean())
        coverage = float(selected["signal_date"].nunique() / total_days)
        objective = avg_r - 0.5 * stop_rate - max(0.0, 0.25 - coverage)
        if objective > best_score:
            best_score = objective
            best_threshold = threshold
    return float(best_threshold)


def _metric_row(selected: pd.DataFrame, *, frame: pd.DataFrame, top_n: int, model_name: str, dataset_split: str, scope: str) -> dict[str, Any]:
    realized = pd.to_numeric(selected.get("realized_R", pd.Series(dtype=float)), errors="coerce")
    positive_sum = float(realized[realized > 0].sum()) if not realized.empty else 0.0
    negative_sum = float(realized[realized < 0].sum()) if not realized.empty else 0.0
    profit_factor = math.inf if negative_sum == 0 and positive_sum > 0 else (positive_sum / abs(negative_sum) if negative_sum else math.nan)
    total_days = max(int(frame["signal_date"].nunique()) if "signal_date" in frame.columns else 0, 1)
    selected_days = int(selected["signal_date"].nunique()) if not selected.empty and "signal_date" in selected.columns else 0
    return {
        "dataset_split": dataset_split,
        "model_name": model_name,
        "scope": scope,
        "top_n": int(top_n),
        "coverage_days": selected_days,
        "coverage_days_rate": float(selected_days / total_days),
        "selected_events": int(len(selected)),
        "selected_symbols": int(selected["symbol"].nunique()) if "symbol" in selected.columns else 0,
        "avg_realized_R": float(realized.mean()) if realized.notna().any() else math.nan,
        "median_realized_R": float(realized.median()) if realized.notna().any() else math.nan,
        "win_rate_R_positive": float(realized.gt(0).mean()) if realized.notna().any() else math.nan,
        "stop_first_rate": float(selected.get("barrier_outcome", pd.Series(dtype=str)).eq("stop_loss_first").mean()) if len(selected) else math.nan,
        "take_profit_first_rate": float(selected.get("barrier_outcome", pd.Series(dtype=str)).eq("take_profit_first").mean()) if len(selected) else math.nan,
        "timeout_rate": float(selected.get("barrier_outcome", pd.Series(dtype=str)).eq("timeout").mean()) if len(selected) else math.nan,
        "avg_max_drawdown_R": float(pd.to_numeric(selected.get("max_drawdown_R", pd.Series(dtype=float)), errors="coerce").mean())
        if len(selected)
        else math.nan,
        "profit_factor_R": profit_factor,
        "avg_holding_days": float(pd.to_numeric(selected.get("holding_days", pd.Series(dtype=float)), errors="coerce").mean())
        if len(selected)
        else math.nan,
    }


def _build_walkforward_windows(
    dates: list[date],
    *,
    windows: int,
    train_days: int,
    valid_days: int,
    test_days: int,
    embargo_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(dates)
    span = train_days + embargo_days + valid_days + embargo_days + test_days
    if total < span:
        return pd.DataFrame(rows)
    max_start = total - span
    starts = [0] if windows <= 1 else sorted(set(round(index * max_start / max(windows - 1, 1)) for index in range(windows)))
    for window_id, start in enumerate(starts, start=1):
        train_start = start
        train_end = train_start + train_days - 1
        valid_start = train_end + 1 + embargo_days
        valid_end = valid_start + valid_days - 1
        test_start = valid_end + 1 + embargo_days
        test_end = test_start + test_days - 1
        if test_end >= total:
            continue
        rows.append(
            {
                "window_id": window_id,
                "train_start": dates[train_start],
                "train_end": dates[train_end],
                "valid_start": dates[valid_start],
                "valid_end": dates[valid_end],
                "test_start": dates[test_start],
                "test_end": dates[test_end],
                "train_days": train_days,
                "valid_days": valid_days,
                "test_days": test_days,
                "embargo_days": embargo_days,
            }
        )
    return pd.DataFrame(rows)


def _save_training_reports(
    *,
    report_dir: Path,
    start_date: date,
    end_date: date,
    labels: pd.DataFrame,
    features: pd.DataFrame,
    skipped: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    labels.to_csv(report_dir / f"event_labels_{start_date.isoformat()}_{end_date.isoformat()}.csv", index=False, encoding="utf-8-sig")
    features.to_csv(report_dir / f"event_features_{start_date.isoformat()}_{end_date.isoformat()}.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(report_dir / "skipped_events.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(report_dir / "train_topn_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(report_dir / "train_summary.csv", index=False, encoding="utf-8-sig")


def _last_fraction_split(frame: pd.DataFrame, *, valid_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(frame["signal_date"]).dt.date.unique())
    if len(dates) < 4:
        return frame.copy(), frame.copy()
    split_index = max(1, int(len(dates) * (1 - valid_fraction)))
    train_dates = set(dates[:split_index])
    train = frame[pd.to_datetime(frame["signal_date"]).dt.date.isin(train_dates)].copy()
    valid = frame[~pd.to_datetime(frame["signal_date"]).dt.date.isin(train_dates)].copy()
    return train, valid if not valid.empty else train.copy()


def _date_slice(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    days = pd.to_datetime(frame["signal_date"]).dt.date
    return frame[(days >= start) & (days <= end)].copy()


def _select_label_config(
    stop_atr_grid: tuple[float, ...],
    take_atr_grid: tuple[float, ...],
    holding_days_grid: tuple[int, ...],
) -> EventLabelConfig:
    stop_values = sorted(float(item) for item in stop_atr_grid)
    take_values = sorted(float(item) for item in take_atr_grid)
    holding_values = sorted(int(item) for item in holding_days_grid)
    return EventLabelConfig(
        stop_atr_mult=stop_values[len(stop_values) // 2],
        take_atr_mult=take_values[len(take_values) // 2],
        max_holding_days=holding_values[len(holding_values) // 2],
    )


def _model_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if not feature_columns:
        return pd.DataFrame({"constant": [0.0] * len(frame)}, index=frame.index)
    return frame.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _sample_weight(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame.get("sample_weight", pd.Series(1.0, index=frame.index)), errors="coerce").fillna(1.0)


def _daily_pct(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("signal_date")[column].rank(method="average", pct=True).fillna(0.5)


def _load_history_for_symbols(storage: Storage, symbols: Any) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for symbol in sorted({str(item).zfill(6) for item in symbols}):
        try:
            histories[symbol] = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError):
            continue
    return histories


def _skipped_prediction_rows(skipped: pd.DataFrame, *, trade_date: date) -> pd.DataFrame:
    if skipped.empty:
        return pd.DataFrame()
    result = skipped.copy()
    result["trade_date"] = pd.Timestamp(trade_date)
    result["pattern_ids"] = result["pattern_id"].astype(str)
    result["risk_pass"] = False
    result["risk_tier"] = "high"
    result["p_stop_first"] = 1.0
    result["expected_R_score"] = 0.0
    result["rank_score"] = 0.0
    result["final_score"] = 0.0
    result["trade_permission"] = "no_trade"
    result["suggested_action"] = "avoid"
    result["risk_reason"] = result.get("skip_reason", "skipped")
    result["entry_price_ref"] = pd.NA
    result["stop_loss_price_ref"] = pd.NA
    result["take_profit_price_ref"] = pd.NA
    result["max_holding_days"] = pd.NA
    result["model_version"] = EVENT_RISK_RANKER_VERSION
    return result


class _LimitedStorage:
    def __init__(self, storage: Storage, limit: int | None) -> None:
        self._storage = storage
        self._limit = limit
        self.paths = storage.paths

    def load_universe(self) -> pd.DataFrame:
        universe = self._storage.load_universe()
        return universe.head(self._limit).copy() if self._limit is not None else universe

    def load_daily_bars(self, symbol: str) -> pd.DataFrame:
        return self._storage.load_daily_bars(symbol)
