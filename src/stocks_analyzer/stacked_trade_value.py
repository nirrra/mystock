from __future__ import annotations

import json
import math
import pickle
import traceback
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, ndcg_score, roc_auc_score

from .models import AppConfig
from .storage import DailyBarsReadError, Storage

try:  # LightGBM is optional in tests, but preferred for the V4 cross-sectional ranker.
    from lightgbm import LGBMRanker
except Exception:  # pragma: no cover - exercised only when LightGBM is absent locally.
    LGBMRanker = None


OUTCOME_TO_CLASS = {"up": 0, "down": 1, "neutral": 2}
CLASS_TO_OUTCOME = {value: key for key, value in OUTCOME_TO_CLASS.items()}
STAGE1_CLASSES = (0, 1, 2)


@dataclass(frozen=True, slots=True)
class HorizonSpec:
    horizon_days: int
    upside_target: float
    downside_threshold: float
    trade_value_weight: float
    name: str


HORIZONS: tuple[HorizonSpec, ...] = (
    HorizonSpec(5, 0.08, 0.05, 0.10, "5d"),
    HorizonSpec(10, 0.12, 0.06, 0.20, "10d"),
    HorizonSpec(20, 0.15, 0.08, 0.40, "20d"),
    HorizonSpec(60, 0.30, 0.15, 0.30, "60d"),
)
LONG_UPSIDE_HORIZON_NAMES = ("20d", "60d")
LONG_UPSIDE_WEIGHTS = {"20d": 0.40, "60d": 0.60}
LONG_UPSIDE_HORIZONS: tuple[HorizonSpec, ...] = tuple(
    spec for spec in HORIZONS if spec.name in LONG_UPSIDE_HORIZON_NAMES
)


@dataclass(slots=True)
class OpportunityRankerTrainResult:
    model_path: Path
    metadata_path: Path
    split_metrics: pd.DataFrame
    topn_metrics: pd.DataFrame
    risk_filter_metrics: pd.DataFrame
    opportunity_metrics: pd.DataFrame
    ranker_metrics: pd.DataFrame
    threshold_grid: pd.DataFrame
    comparison_metrics: pd.DataFrame
    risk_gate_grid: pd.DataFrame
    selected_risk_params: dict[str, object]
    selected_opportunity_params: dict[str, object]
    selected_hybrid_opportunity_params: dict[str, object]
    prediction_path: Path | None
    latest_predictions: pd.DataFrame
    skipped_symbols: pd.DataFrame


@dataclass(slots=True)
class VolumePriceFusionTrainResult:
    model_path: Path
    metadata_path: Path
    topn_metrics: pd.DataFrame
    volume_price_risk_metrics: pd.DataFrame
    volume_price_quality_metrics: pd.DataFrame
    comparison_metrics: pd.DataFrame
    prediction_path: Path | None
    latest_predictions: pd.DataFrame
    skipped_symbols: pd.DataFrame


@dataclass(slots=True)
class CandidateRankerTrainResult:
    model_path: Path
    metadata_path: Path
    topn_metrics: pd.DataFrame
    ranker_metrics: pd.DataFrame
    comparison_metrics: pd.DataFrame
    blend_grid: pd.DataFrame
    selected_blend_params: dict[str, object]
    prediction_path: Path | None
    latest_predictions: pd.DataFrame
    skipped_symbols: pd.DataFrame


@dataclass(slots=True)
class WalkForwardValidationResult:
    report_dir: Path
    config_path: Path
    windows_path: Path
    topn_metrics_path: Path
    summary_path: Path
    windows: pd.DataFrame
    topn_metrics: pd.DataFrame
    summary: pd.DataFrame
    skipped_symbols: pd.DataFrame


class ConstantClassifier:
    def __init__(self, class_label: int) -> None:
        self.classes_ = np.array([class_label], dtype=int)
        self.class_label = int(class_label)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones((len(X), 1), dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.class_label, dtype=int)


class ConstantRegressor:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value, dtype=float)


def opportunity_ranker_model_dir(project_root: Path) -> Path:
    return project_root / "data" / "ml" / "v42_opportunity_ranker"


def opportunity_ranker_report_dir(project_root: Path) -> Path:
    return project_root / "reports" / "v42_opportunity_ranker"


def volume_price_fusion_model_dir(project_root: Path) -> Path:
    return project_root / "data" / "ml" / "v5_volume_price_fusion"


def volume_price_fusion_report_dir(project_root: Path) -> Path:
    return project_root / "reports" / "v5_volume_price_fusion"


def candidate_ranker_model_dir(project_root: Path) -> Path:
    return project_root / "data" / "ml" / "v51_candidate_ranker"


def candidate_ranker_report_dir(project_root: Path) -> Path:
    return project_root / "reports" / "v51_candidate_ranker"


def model_walkforward_report_dir(project_root: Path) -> Path:
    return project_root / "reports" / "model_walkforward"


def build_stacked_dataset(
    *,
    storage: Storage,
    config: AppConfig,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = storage.load_universe().copy()
    if limit is not None:
        universe = universe.head(limit).copy()

    rows: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []
    min_history_days = max(config.universe.min_history_days, 260)

    for instrument in universe.to_dict("records"):
        symbol = str(instrument.get("symbol", "")).zfill(6)
        name = str(instrument.get("name", ""))
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError) as exc:
            skipped.append({"symbol": symbol, "name": name, "reason": type(exc).__name__})
            continue
        if len(bars) < min_history_days:
            skipped.append({"symbol": symbol, "name": name, "reason": "insufficient_history"})
            continue

        frame = build_stacked_feature_frame(bars)
        frame = add_all_horizon_labels(frame)
        frame["symbol"] = symbol
        frame["name"] = name
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
        frame = frame.dropna(subset=["trade_date"]).copy()
        if start_date is not None:
            frame = frame[frame["trade_date"].dt.date >= start_date]
        if end_date is not None:
            frame = frame[frame["trade_date"].dt.date <= end_date]
        if "amount_ma_20" in frame.columns:
            amount_ma_20 = pd.to_numeric(frame["amount_ma_20"], errors="coerce")
            frame = frame[amount_ma_20 >= config.universe.min_avg_amount_20d]
        if frame.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "no_training_rows"})
            continue
        rows.append(frame)

    dataset = pd.concat(rows, ignore_index=True, copy=False) if rows else pd.DataFrame()
    if not dataset.empty:
        dataset = dataset.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
        dataset = _downcast_frame(dataset)
    return dataset, pd.DataFrame(skipped)


def build_prediction_frame(*, storage: Storage, config: AppConfig, trade_date: date) -> pd.DataFrame:
    universe = storage.load_universe().copy()
    rows: list[pd.DataFrame] = []
    for instrument in universe.to_dict("records"):
        symbol = str(instrument.get("symbol", "")).zfill(6)
        name = str(instrument.get("name", ""))
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError):
            continue
        frame = build_stacked_feature_frame(bars)
        frame["symbol"] = symbol
        frame["name"] = name
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
        current = frame[frame["trade_date"].dt.date == trade_date].copy()
        if current.empty:
            continue
        if "amount_ma_20" in current.columns:
            amount_ma_20 = pd.to_numeric(current["amount_ma_20"], errors="coerce")
            current = current[amount_ma_20 >= config.universe.min_avg_amount_20d]
        if not current.empty:
            rows.append(current)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True, copy=False)
    return _downcast_frame(result.sort_values(["trade_date", "symbol"]).reset_index(drop=True))


def train_opportunity_ranker_model(
    *,
    storage: Storage,
    config: AppConfig,
    project_root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    train_end: date | None = None,
    valid_end: date | None = None,
    test_end: date | None = None,
    limit: int | None = None,
    max_iter: int = 80,
    top_n_list: tuple[int, ...] = (20, 50),
    prediction_date: date | None = None,
) -> OpportunityRankerTrainResult:
    started_at = perf_counter()
    model_dir = opportunity_ranker_model_dir(project_root)
    report_dir = opportunity_ranker_report_dir(project_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset, skipped_symbols = build_stacked_dataset(
        storage=storage,
        config=config,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    if dataset.empty:
        raise RuntimeError("No V4.2 opportunity-ranker dataset could be built from local daily bars.")
    dataset = add_v4_risk_upside_labels(dataset)

    train_end, valid_end, test_end = resolve_split_dates(dataset, train_end=train_end, valid_end=valid_end, test_end=test_end)
    split_frames = split_dataset(dataset, train_end=train_end, valid_end=valid_end, test_end=test_end)

    risk_feature_columns_by_horizon = {spec.name: stage1_feature_columns(dataset, spec) for spec in HORIZONS}
    long_upside_feature_columns_by_horizon = {
        spec.name: stage1_feature_columns(dataset, spec) for spec in LONG_UPSIDE_HORIZONS
    }
    risk_stage2_train = build_stage1_oof_predictions(
        split_frames["train"],
        feature_columns_by_horizon=risk_feature_columns_by_horizon,
        max_iter=max_iter,
    )
    long_stage2_train = build_stage1_oof_predictions_for_horizons(
        split_frames["train"],
        feature_columns_by_horizon=long_upside_feature_columns_by_horizon,
        horizons=LONG_UPSIDE_HORIZONS,
        max_iter=max_iter,
        column_prefix="long_",
    )
    if risk_stage2_train.empty or long_stage2_train.empty:
        raise RuntimeError("Stage 1 walk-forward out-of-fold predictions are empty; cannot train V4.2 model.")

    long_columns = ["trade_date", "symbol"] + [column for column in long_stage2_train.columns if column.startswith("long_")]
    stage2_train = risk_stage2_train.merge(
        long_stage2_train.loc[:, long_columns],
        on=["trade_date", "symbol"],
        how="inner",
    )
    stage2_train = add_v4_risk_upside_labels(stage2_train)
    stage2_train = add_long_upside_stage2_features(stage2_train)
    if stage2_train.empty:
        raise RuntimeError("V4.2 stage2 training frame is empty after aligning risk and long-upside OOF rows.")

    risk_features = stage2_feature_columns(stage2_train)
    long_upside_features = long_upside_feature_columns(stage2_train)
    risk_model = fit_risk_filter_model(stage2_train, feature_columns=risk_features, max_iter=max_iter)
    long_upside_model = fit_long_upside_model(stage2_train, feature_columns=long_upside_features, max_iter=max_iter)

    eval_risk_stage1_models = train_stage1_models(
        split_frames["train"],
        feature_columns_by_horizon=risk_feature_columns_by_horizon,
        max_iter=max_iter,
    )
    eval_long_stage1_models = train_stage1_models_for_horizons(
        split_frames["train"],
        feature_columns_by_horizon=long_upside_feature_columns_by_horizon,
        horizons=LONG_UPSIDE_HORIZONS,
        max_iter=max_iter,
    )
    raw_train = score_v4_risk_upside_raw_frame(
        stage2_train,
        risk_model=risk_model,
        long_upside_model=long_upside_model,
        risk_features=risk_features,
        long_upside_features=long_upside_features,
    )
    raw_valid = score_v4_risk_upside_full_frame(
        split_frames["valid"],
        risk_stage1_models=eval_risk_stage1_models,
        long_upside_stage1_models=eval_long_stage1_models,
        risk_model=risk_model,
        long_upside_model=long_upside_model,
        risk_feature_columns_by_horizon=risk_feature_columns_by_horizon,
        long_upside_feature_columns_by_horizon=long_upside_feature_columns_by_horizon,
        risk_features=risk_features,
        long_upside_features=long_upside_features,
    )
    raw_test = score_v4_risk_upside_full_frame(
        split_frames["test"],
        risk_stage1_models=eval_risk_stage1_models,
        long_upside_stage1_models=eval_long_stage1_models,
        risk_model=risk_model,
        long_upside_model=long_upside_model,
        risk_feature_columns_by_horizon=risk_feature_columns_by_horizon,
        long_upside_feature_columns_by_horizon=long_upside_feature_columns_by_horizon,
        risk_features=risk_features,
        long_upside_features=long_upside_features,
    )

    selected_risk_params, risk_gate_grid = select_v41_risk_gate_params(raw_valid, top_n_list=top_n_list)
    gated_train = apply_v41_risk_gate_decision(raw_train, selected_risk_params).assign(dataset_split="train_oof")
    gated_valid = apply_v41_risk_gate_decision(raw_valid, selected_risk_params).assign(dataset_split="valid")
    gated_test = apply_v41_risk_gate_decision(raw_test, selected_risk_params).assign(dataset_split="test")

    labeled_train = add_v41_long_quality_labels(gated_train)
    labeled_valid = add_v41_long_quality_labels(gated_valid)
    labeled_test = add_v41_long_quality_labels(gated_test)

    opportunity_train = build_v42_opportunity_frame(labeled_train)
    opportunity_valid = build_v42_opportunity_frame(labeled_valid)
    opportunity_test = build_v42_opportunity_frame(labeled_test)
    opportunity_features = opportunity_gate_feature_columns(opportunity_train)
    opportunity_model = fit_v42_opportunity_gate_model(
        opportunity_train,
        feature_columns=opportunity_features,
        max_iter=max_iter,
    )
    opportunity_train_scored = score_v42_opportunity_gate_frame(
        opportunity_train,
        opportunity_model=opportunity_model,
        opportunity_features=opportunity_features,
    )
    opportunity_valid_scored = score_v42_opportunity_gate_frame(
        opportunity_valid,
        opportunity_model=opportunity_model,
        opportunity_features=opportunity_features,
    )
    opportunity_test_scored = score_v42_opportunity_gate_frame(
        opportunity_test,
        opportunity_model=opportunity_model,
        opportunity_features=opportunity_features,
    )

    ranker_features = long_quality_ranker_feature_columns(add_v41_cross_sectional_rank_features(labeled_train))
    ranker_model, ranker_engine, ranker_train_scope = fit_v42_conditional_ranker_model(
        labeled_train,
        opportunity_train,
        feature_columns=ranker_features,
        max_iter=max_iter,
    )
    rank_scored_train = score_v42_ranker_frame(
        labeled_train,
        ranker_model=ranker_model,
        ranker_features=ranker_features,
    )
    rank_scored_valid = score_v42_ranker_frame(
        labeled_valid,
        ranker_model=ranker_model,
        ranker_features=ranker_features,
    )
    rank_scored_test = score_v42_ranker_frame(
        labeled_test,
        ranker_model=ranker_model,
        ranker_features=ranker_features,
    )
    hybrid_rank_train = score_v42_v4_rank_frame(labeled_train)
    hybrid_rank_valid = score_v42_v4_rank_frame(labeled_valid)
    hybrid_rank_test = score_v42_v4_rank_frame(labeled_test)

    selected_opportunity_params, threshold_grid = select_v42_opportunity_threshold(
        rank_scored_valid,
        opportunity_valid_scored,
        top_n_list=top_n_list,
    )
    selected_hybrid_opportunity_params, hybrid_threshold_grid = select_v42_opportunity_threshold(
        hybrid_rank_valid,
        opportunity_valid_scored,
        top_n_list=top_n_list,
    )
    scored_train = apply_v42_opportunity_decision(rank_scored_train, opportunity_train_scored, selected_opportunity_params)
    scored_valid = apply_v42_opportunity_decision(rank_scored_valid, opportunity_valid_scored, selected_opportunity_params)
    scored_test = apply_v42_opportunity_decision(rank_scored_test, opportunity_test_scored, selected_opportunity_params)
    hybrid_scored_train = apply_v42_opportunity_decision(
        hybrid_rank_train,
        opportunity_train_scored,
        selected_hybrid_opportunity_params,
    )
    hybrid_scored_valid = apply_v42_opportunity_decision(
        hybrid_rank_valid,
        opportunity_valid_scored,
        selected_hybrid_opportunity_params,
    )
    hybrid_scored_test = apply_v42_opportunity_decision(
        hybrid_rank_test,
        opportunity_test_scored,
        selected_hybrid_opportunity_params,
    )
    opportunity_train_scored = apply_v42_opportunity_threshold_to_daily(opportunity_train_scored, selected_opportunity_params)
    opportunity_valid_scored = apply_v42_opportunity_threshold_to_daily(opportunity_valid_scored, selected_opportunity_params)
    opportunity_test_scored = apply_v42_opportunity_threshold_to_daily(opportunity_test_scored, selected_opportunity_params)

    evaluated = pd.concat([scored_train, scored_valid, scored_test], ignore_index=True, copy=False)
    hybrid_evaluated = pd.concat(
        [hybrid_scored_train, hybrid_scored_valid, hybrid_scored_test],
        ignore_index=True,
        copy=False,
    )
    opportunity_evaluated = pd.concat(
        [opportunity_train_scored, opportunity_valid_scored, opportunity_test_scored],
        ignore_index=True,
        copy=False,
    )
    baseline_evaluated = pd.concat([labeled_train, labeled_valid, labeled_test], ignore_index=True, copy=False)

    split_metrics = evaluate_v42_split_metrics(evaluated)
    topn_metrics = evaluate_v42_topn_metrics(evaluated, top_n_list=top_n_list)
    risk_filter_metrics = evaluate_risk_filter_metrics(evaluated)
    opportunity_metrics = evaluate_v42_opportunity_metrics(opportunity_evaluated)
    ranker_metrics = evaluate_v42_ranker_metrics(evaluated, top_n_list=top_n_list)
    comparison_metrics = compare_v42_topn_metrics(
        baseline_scored=baseline_evaluated,
        v42_scored=evaluated,
        hybrid_v4_scored=hybrid_evaluated,
        top_n_list=top_n_list,
    )

    deployment_risk_stage1_models = train_stage1_models(
        dataset,
        feature_columns_by_horizon=risk_feature_columns_by_horizon,
        max_iter=max_iter,
    )
    deployment_long_stage1_models = train_stage1_models_for_horizons(
        dataset,
        feature_columns_by_horizon=long_upside_feature_columns_by_horizon,
        horizons=LONG_UPSIDE_HORIZONS,
        max_iter=max_iter,
    )
    deployment_raw = attach_stage1_predictions(
        dataset,
        models=deployment_risk_stage1_models,
        feature_columns_by_horizon=risk_feature_columns_by_horizon,
    )
    deployment_raw = attach_stage1_predictions_for_horizons(
        deployment_raw,
        models=deployment_long_stage1_models,
        feature_columns_by_horizon=long_upside_feature_columns_by_horizon,
        horizons=LONG_UPSIDE_HORIZONS,
        column_prefix="long_",
    )
    deployment_raw = add_v4_risk_upside_labels(deployment_raw)
    deployment_raw = add_long_upside_stage2_features(deployment_raw)
    deployment_risk_model = fit_risk_filter_model(deployment_raw, feature_columns=risk_features, max_iter=max_iter)
    deployment_long_upside_model = fit_long_upside_model(
        deployment_raw,
        feature_columns=long_upside_features,
        max_iter=max_iter,
    )
    deployment_scored_raw = score_v4_risk_upside_raw_frame(
        deployment_raw,
        risk_model=deployment_risk_model,
        long_upside_model=deployment_long_upside_model,
        risk_features=risk_features,
        long_upside_features=long_upside_features,
    )
    deployment_gated = apply_v41_risk_gate_decision(deployment_scored_raw, selected_risk_params)
    deployment_labeled = add_v41_long_quality_labels(deployment_gated)
    deployment_opportunity = build_v42_opportunity_frame(deployment_labeled)
    deployment_opportunity_model = fit_v42_opportunity_gate_model(
        deployment_opportunity,
        feature_columns=opportunity_features,
        max_iter=max_iter,
    )
    deployment_ranker_model, deployment_ranker_engine, deployment_ranker_scope = fit_v42_conditional_ranker_model(
        deployment_labeled,
        deployment_opportunity,
        feature_columns=ranker_features,
        max_iter=max_iter,
    )

    artifact = {
        "kind": "v42_opportunity_ranker",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "horizons": [
            {
                "horizon_days": spec.horizon_days,
                "upside_target": spec.upside_target,
                "downside_threshold": spec.downside_threshold,
                "trade_value_weight": spec.trade_value_weight,
                "name": spec.name,
            }
            for spec in HORIZONS
        ],
        "risk_feature_columns_by_horizon": risk_feature_columns_by_horizon,
        "long_upside_feature_columns_by_horizon": long_upside_feature_columns_by_horizon,
        "risk_features": risk_features,
        "long_upside_features": long_upside_features,
        "opportunity_features": opportunity_features,
        "ranker_features": ranker_features,
        "risk_stage1_models": deployment_risk_stage1_models,
        "long_upside_stage1_models": deployment_long_stage1_models,
        "risk_model": deployment_risk_model,
        "long_upside_model": deployment_long_upside_model,
        "opportunity_model": deployment_opportunity_model,
        "ranker_model": deployment_ranker_model,
        "ranker_engine": deployment_ranker_engine,
        "validation_ranker_engine": ranker_engine,
        "ranker_train_scope": deployment_ranker_scope,
        "validation_ranker_train_scope": ranker_train_scope,
        "selected_risk_params": selected_risk_params,
        "selected_opportunity_params": selected_opportunity_params,
        "selected_hybrid_opportunity_params": selected_hybrid_opportunity_params,
        "train_end": train_end.isoformat(),
        "valid_end": valid_end.isoformat(),
        "test_end": test_end.isoformat(),
        "dataset_rows": int(len(dataset)),
        "elapsed_seconds": round(perf_counter() - started_at, 3),
    }
    model_path = model_dir / "v42_opportunity_ranker.pkl"
    with model_path.open("wb") as file:
        pickle.dump(artifact, file)

    metadata = {
        key: value
        for key, value in artifact.items()
        if key
        not in {
            "risk_stage1_models",
            "long_upside_stage1_models",
            "risk_model",
            "long_upside_model",
            "opportunity_model",
            "ranker_model",
        }
    }
    metadata_path = model_dir / "v42_opportunity_ranker_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    split_metrics.to_csv(report_dir / "v42_split_metrics.csv", index=False, encoding="utf-8-sig")
    topn_metrics.to_csv(report_dir / "v42_topn_metrics.csv", index=False, encoding="utf-8-sig")
    risk_filter_metrics.to_csv(report_dir / "v42_risk_filter_metrics.csv", index=False, encoding="utf-8-sig")
    opportunity_metrics.to_csv(report_dir / "v42_opportunity_metrics.csv", index=False, encoding="utf-8-sig")
    ranker_metrics.to_csv(report_dir / "v42_ranker_metrics.csv", index=False, encoding="utf-8-sig")
    threshold_grid.to_csv(report_dir / "v42_threshold_grid.csv", index=False, encoding="utf-8-sig")
    hybrid_threshold_grid.to_csv(report_dir / "v42_gate_v4_rank_threshold_grid.csv", index=False, encoding="utf-8-sig")
    comparison_metrics.to_csv(report_dir / "v42_comparison.csv", index=False, encoding="utf-8-sig")
    risk_gate_grid.to_csv(report_dir / "v42_risk_gate_grid.csv", index=False, encoding="utf-8-sig")
    skipped_symbols.to_csv(report_dir / "v42_skipped_symbols.csv", index=False, encoding="utf-8-sig")

    prediction_path: Path | None = None
    latest_predictions = pd.DataFrame()
    if prediction_date is not None:
        latest_predictions = predict_opportunity_ranker(
            storage=storage,
            config=config,
            project_root=project_root,
            trade_date=prediction_date,
        )
        prediction_path = report_dir / f"predictions_{prediction_date.isoformat()}.csv"
        latest_predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    return OpportunityRankerTrainResult(
        model_path=model_path,
        metadata_path=metadata_path,
        split_metrics=split_metrics,
        topn_metrics=topn_metrics,
        risk_filter_metrics=risk_filter_metrics,
        opportunity_metrics=opportunity_metrics,
        ranker_metrics=ranker_metrics,
        threshold_grid=threshold_grid,
        comparison_metrics=comparison_metrics,
        risk_gate_grid=risk_gate_grid,
        selected_risk_params=selected_risk_params,
        selected_opportunity_params=selected_opportunity_params,
        selected_hybrid_opportunity_params=selected_hybrid_opportunity_params,
        prediction_path=prediction_path,
        latest_predictions=latest_predictions,
        skipped_symbols=skipped_symbols,
    )


def predict_opportunity_ranker(
    *,
    storage: Storage,
    config: AppConfig,
    project_root: Path,
    trade_date: date,
    rank_source: str = "v42",
) -> pd.DataFrame:
    if rank_source not in {"v42", "v4"}:
        raise ValueError("rank_source must be 'v42' or 'v4'.")
    artifact = load_opportunity_ranker_artifact(project_root)
    frame = build_prediction_frame(storage=storage, config=config, trade_date=trade_date)
    if frame.empty:
        return pd.DataFrame()
    frame = frame[pd.to_datetime(frame["trade_date"]).dt.date == trade_date].copy()
    if frame.empty:
        return pd.DataFrame()
    raw = score_v4_risk_upside_full_frame(
        frame,
        risk_stage1_models=artifact["risk_stage1_models"],
        long_upside_stage1_models=artifact["long_upside_stage1_models"],
        risk_model=artifact["risk_model"],
        long_upside_model=artifact["long_upside_model"],
        risk_feature_columns_by_horizon=artifact["risk_feature_columns_by_horizon"],
        long_upside_feature_columns_by_horizon=artifact["long_upside_feature_columns_by_horizon"],
        risk_features=artifact["risk_features"],
        long_upside_features=artifact["long_upside_features"],
    )
    gated = apply_v41_risk_gate_decision(raw, artifact["selected_risk_params"])
    opportunity = build_v42_opportunity_frame(gated)
    opportunity = score_v42_opportunity_gate_frame(
        opportunity,
        opportunity_model=artifact["opportunity_model"],
        opportunity_features=artifact["opportunity_features"],
    )
    if rank_source == "v4":
        ranked = score_v42_v4_rank_frame(gated)
        opportunity_params = artifact.get("selected_hybrid_opportunity_params", artifact["selected_opportunity_params"])
    else:
        ranked = score_v42_ranker_frame(
            gated,
            ranker_model=artifact["ranker_model"],
            ranker_features=artifact["ranker_features"],
        )
        ranked["rank_source_v42"] = "v42_conditional_ranker"
        opportunity_params = artifact["selected_opportunity_params"]
    scored = apply_v42_opportunity_decision(ranked, opportunity, opportunity_params)
    scored = scored.sort_values(["action_rank_v42", "final_score_v42"], ascending=[True, False]).reset_index(drop=True)
    scored["rank"] = range(1, len(scored) + 1)
    return prepare_opportunity_ranker_prediction_report(scored)


def _score_v42_hybrid_from_artifact(frame: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    raw = score_v4_risk_upside_full_frame(
        frame,
        risk_stage1_models=artifact["risk_stage1_models"],
        long_upside_stage1_models=artifact["long_upside_stage1_models"],
        risk_model=artifact["risk_model"],
        long_upside_model=artifact["long_upside_model"],
        risk_feature_columns_by_horizon=artifact["risk_feature_columns_by_horizon"],
        long_upside_feature_columns_by_horizon=artifact["long_upside_feature_columns_by_horizon"],
        risk_features=artifact["risk_features"],
        long_upside_features=artifact["long_upside_features"],
    )
    gated = apply_v41_risk_gate_decision(raw, artifact["selected_risk_params"])
    opportunity = build_v42_opportunity_frame(gated)
    opportunity = score_v42_opportunity_gate_frame(
        opportunity,
        opportunity_model=artifact["opportunity_model"],
        opportunity_features=artifact["opportunity_features"],
    )
    ranked = score_v42_v4_rank_frame(gated)
    opportunity_params = artifact.get("selected_hybrid_opportunity_params", artifact["selected_opportunity_params"])
    return apply_v42_opportunity_decision(ranked, opportunity, opportunity_params)


def _train_v42_hybrid_base_for_v5(
    dataset: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    *,
    max_iter: int,
    top_n_list: tuple[int, ...],
    train_end: date,
    valid_end: date,
    test_end: date,
    include_deployment_artifact: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    risk_feature_columns_by_horizon = {spec.name: stage1_feature_columns(dataset, spec) for spec in HORIZONS}
    long_upside_feature_columns_by_horizon = {
        spec.name: stage1_feature_columns(dataset, spec) for spec in LONG_UPSIDE_HORIZONS
    }

    risk_stage2_train = build_stage1_oof_predictions(
        split_frames["train"],
        feature_columns_by_horizon=risk_feature_columns_by_horizon,
        max_iter=max_iter,
    )
    long_stage2_train = build_stage1_oof_predictions_for_horizons(
        split_frames["train"],
        feature_columns_by_horizon=long_upside_feature_columns_by_horizon,
        horizons=LONG_UPSIDE_HORIZONS,
        max_iter=max_iter,
        column_prefix="long_",
    )
    if risk_stage2_train.empty or long_stage2_train.empty:
        raise RuntimeError("V5 base V4.2 OOF stage1 predictions are empty; cannot train volume-price fusion.")

    long_columns = ["trade_date", "symbol"] + [column for column in long_stage2_train.columns if column.startswith("long_")]
    stage2_train = risk_stage2_train.merge(
        long_stage2_train.loc[:, long_columns],
        on=["trade_date", "symbol"],
        how="inner",
    )
    stage2_train = add_v4_risk_upside_labels(stage2_train)
    stage2_train = add_long_upside_stage2_features(stage2_train)
    if stage2_train.empty:
        raise RuntimeError("V5 base V4.2 stage2 training frame is empty after OOF alignment.")

    risk_features = stage2_feature_columns(stage2_train)
    long_upside_features = long_upside_feature_columns(stage2_train)
    risk_model = fit_risk_filter_model(stage2_train, feature_columns=risk_features, max_iter=max_iter)
    long_upside_model = fit_long_upside_model(stage2_train, feature_columns=long_upside_features, max_iter=max_iter)

    eval_risk_stage1_models = train_stage1_models(
        split_frames["train"],
        feature_columns_by_horizon=risk_feature_columns_by_horizon,
        max_iter=max_iter,
    )
    eval_long_stage1_models = train_stage1_models_for_horizons(
        split_frames["train"],
        feature_columns_by_horizon=long_upside_feature_columns_by_horizon,
        horizons=LONG_UPSIDE_HORIZONS,
        max_iter=max_iter,
    )
    raw_train = score_v4_risk_upside_raw_frame(
        stage2_train,
        risk_model=risk_model,
        long_upside_model=long_upside_model,
        risk_features=risk_features,
        long_upside_features=long_upside_features,
    )
    raw_valid = score_v4_risk_upside_full_frame(
        split_frames["valid"],
        risk_stage1_models=eval_risk_stage1_models,
        long_upside_stage1_models=eval_long_stage1_models,
        risk_model=risk_model,
        long_upside_model=long_upside_model,
        risk_feature_columns_by_horizon=risk_feature_columns_by_horizon,
        long_upside_feature_columns_by_horizon=long_upside_feature_columns_by_horizon,
        risk_features=risk_features,
        long_upside_features=long_upside_features,
    )
    raw_test = score_v4_risk_upside_full_frame(
        split_frames["test"],
        risk_stage1_models=eval_risk_stage1_models,
        long_upside_stage1_models=eval_long_stage1_models,
        risk_model=risk_model,
        long_upside_model=long_upside_model,
        risk_feature_columns_by_horizon=risk_feature_columns_by_horizon,
        long_upside_feature_columns_by_horizon=long_upside_feature_columns_by_horizon,
        risk_features=risk_features,
        long_upside_features=long_upside_features,
    )

    selected_risk_params, _ = select_v41_risk_gate_params(raw_valid, top_n_list=top_n_list)
    gated_train = apply_v41_risk_gate_decision(raw_train, selected_risk_params).assign(dataset_split="train_oof")
    gated_valid = apply_v41_risk_gate_decision(raw_valid, selected_risk_params).assign(dataset_split="valid")
    gated_test = apply_v41_risk_gate_decision(raw_test, selected_risk_params).assign(dataset_split="test")

    labeled_train = add_v41_long_quality_labels(gated_train)
    labeled_valid = add_v41_long_quality_labels(gated_valid)
    labeled_test = add_v41_long_quality_labels(gated_test)

    opportunity_train = build_v42_opportunity_frame(labeled_train)
    opportunity_valid = build_v42_opportunity_frame(labeled_valid)
    opportunity_test = build_v42_opportunity_frame(labeled_test)
    opportunity_features = opportunity_gate_feature_columns(opportunity_train)
    opportunity_model = fit_v42_opportunity_gate_model(
        opportunity_train,
        feature_columns=opportunity_features,
        max_iter=max_iter,
    )
    opportunity_train_scored = score_v42_opportunity_gate_frame(
        opportunity_train,
        opportunity_model=opportunity_model,
        opportunity_features=opportunity_features,
    )
    opportunity_valid_scored = score_v42_opportunity_gate_frame(
        opportunity_valid,
        opportunity_model=opportunity_model,
        opportunity_features=opportunity_features,
    )
    opportunity_test_scored = score_v42_opportunity_gate_frame(
        opportunity_test,
        opportunity_model=opportunity_model,
        opportunity_features=opportunity_features,
    )

    hybrid_rank_train = score_v42_v4_rank_frame(labeled_train)
    hybrid_rank_valid = score_v42_v4_rank_frame(labeled_valid)
    hybrid_rank_test = score_v42_v4_rank_frame(labeled_test)
    selected_hybrid_opportunity_params, _ = select_v42_opportunity_threshold(
        hybrid_rank_valid,
        opportunity_valid_scored,
        top_n_list=top_n_list,
    )
    base_train = apply_v42_opportunity_decision(
        hybrid_rank_train,
        opportunity_train_scored,
        selected_hybrid_opportunity_params,
    )
    base_valid = apply_v42_opportunity_decision(
        hybrid_rank_valid,
        opportunity_valid_scored,
        selected_hybrid_opportunity_params,
    )
    base_test = apply_v42_opportunity_decision(
        hybrid_rank_test,
        opportunity_test_scored,
        selected_hybrid_opportunity_params,
    )

    if not include_deployment_artifact:
        base_artifact = {
            "kind": "v42_hybrid_base_evaluation",
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "risk_feature_columns_by_horizon": risk_feature_columns_by_horizon,
            "long_upside_feature_columns_by_horizon": long_upside_feature_columns_by_horizon,
            "risk_features": risk_features,
            "long_upside_features": long_upside_features,
            "opportunity_features": opportunity_features,
            "selected_risk_params": selected_risk_params,
            "selected_opportunity_params": selected_hybrid_opportunity_params,
            "selected_hybrid_opportunity_params": selected_hybrid_opportunity_params,
            "train_end": train_end.isoformat(),
            "valid_end": valid_end.isoformat(),
            "test_end": test_end.isoformat(),
        }
        return base_train, base_valid, base_test, base_artifact

    deployment_risk_stage1_models = train_stage1_models(
        dataset,
        feature_columns_by_horizon=risk_feature_columns_by_horizon,
        max_iter=max_iter,
    )
    deployment_long_stage1_models = train_stage1_models_for_horizons(
        dataset,
        feature_columns_by_horizon=long_upside_feature_columns_by_horizon,
        horizons=LONG_UPSIDE_HORIZONS,
        max_iter=max_iter,
    )
    deployment_raw = attach_stage1_predictions(
        dataset,
        models=deployment_risk_stage1_models,
        feature_columns_by_horizon=risk_feature_columns_by_horizon,
    )
    deployment_raw = attach_stage1_predictions_for_horizons(
        deployment_raw,
        models=deployment_long_stage1_models,
        feature_columns_by_horizon=long_upside_feature_columns_by_horizon,
        horizons=LONG_UPSIDE_HORIZONS,
        column_prefix="long_",
    )
    deployment_raw = add_v4_risk_upside_labels(deployment_raw)
    deployment_raw = add_long_upside_stage2_features(deployment_raw)
    deployment_risk_model = fit_risk_filter_model(deployment_raw, feature_columns=risk_features, max_iter=max_iter)
    deployment_long_upside_model = fit_long_upside_model(
        deployment_raw,
        feature_columns=long_upside_features,
        max_iter=max_iter,
    )
    deployment_scored_raw = score_v4_risk_upside_raw_frame(
        deployment_raw,
        risk_model=deployment_risk_model,
        long_upside_model=deployment_long_upside_model,
        risk_features=risk_features,
        long_upside_features=long_upside_features,
    )
    deployment_gated = apply_v41_risk_gate_decision(deployment_scored_raw, selected_risk_params)
    deployment_labeled = add_v41_long_quality_labels(deployment_gated)
    deployment_opportunity = build_v42_opportunity_frame(deployment_labeled)
    deployment_opportunity_model = fit_v42_opportunity_gate_model(
        deployment_opportunity,
        feature_columns=opportunity_features,
        max_iter=max_iter,
    )

    base_artifact = {
        "kind": "v42_hybrid_base_for_v5",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "risk_feature_columns_by_horizon": risk_feature_columns_by_horizon,
        "long_upside_feature_columns_by_horizon": long_upside_feature_columns_by_horizon,
        "risk_features": risk_features,
        "long_upside_features": long_upside_features,
        "opportunity_features": opportunity_features,
        "risk_stage1_models": deployment_risk_stage1_models,
        "long_upside_stage1_models": deployment_long_stage1_models,
        "risk_model": deployment_risk_model,
        "long_upside_model": deployment_long_upside_model,
        "opportunity_model": deployment_opportunity_model,
        "selected_risk_params": selected_risk_params,
        "selected_opportunity_params": selected_hybrid_opportunity_params,
        "selected_hybrid_opportunity_params": selected_hybrid_opportunity_params,
        "train_end": train_end.isoformat(),
        "valid_end": valid_end.isoformat(),
        "test_end": test_end.isoformat(),
    }
    return base_train, base_valid, base_test, base_artifact


def build_volume_price_oof_predictions(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    max_iter: int,
    folds: int = 3,
) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(frame["trade_date"]).dt.date.unique())
    if len(dates) < 20:
        return pd.DataFrame()
    cut_indices = np.linspace(0.35, 0.90, folds)
    rows: list[pd.DataFrame] = []
    for ratio in cut_indices:
        train_cut_index = min(max(int(len(dates) * ratio), 1), len(dates) - 2)
        train_cut = dates[train_cut_index]
        next_index = min(max(int(len(dates) * (ratio + 0.15)), train_cut_index + 1), len(dates) - 1)
        predict_end = dates[next_index]
        train_frame = frame[pd.to_datetime(frame["trade_date"]).dt.date <= train_cut].copy()
        predict_frame = frame[
            (pd.to_datetime(frame["trade_date"]).dt.date > train_cut)
            & (pd.to_datetime(frame["trade_date"]).dt.date <= predict_end)
        ].copy()
        if train_frame.empty or predict_frame.empty:
            continue
        train_frame = add_v5_volume_price_labels(train_frame)
        risk_model = fit_volume_price_risk_model(train_frame, feature_columns=feature_columns, max_iter=max_iter)
        quality_model = fit_volume_price_quality_model(train_frame, feature_columns=feature_columns, max_iter=max_iter)
        rows.append(
            score_volume_price_submodels(
                predict_frame,
                risk_model=risk_model,
                quality_model=quality_model,
                feature_columns=feature_columns,
            )
        )
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True, copy=False)
    return result.drop_duplicates(subset=["trade_date", "symbol"], keep="last").reset_index(drop=True)


def train_volume_price_fusion_model(
    *,
    storage: Storage,
    config: AppConfig,
    project_root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    train_end: date | None = None,
    valid_end: date | None = None,
    test_end: date | None = None,
    limit: int | None = None,
    max_iter: int = 80,
    top_n_list: tuple[int, ...] = (20, 50),
    prediction_date: date | None = None,
    reuse_base_artifact: bool = False,
) -> VolumePriceFusionTrainResult:
    model_dir = volume_price_fusion_model_dir(project_root)
    report_dir = volume_price_fusion_report_dir(project_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset, skipped_symbols = build_stacked_dataset(
        storage=storage,
        config=config,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    if dataset.empty:
        raise RuntimeError("No V5 volume-price fusion dataset could be built from local daily bars.")
    dataset = add_v5_volume_price_labels(add_v4_risk_upside_labels(dataset))

    train_end, valid_end, test_end = resolve_split_dates(dataset, train_end=train_end, valid_end=valid_end, test_end=test_end)
    split_frames = split_dataset(dataset, train_end=train_end, valid_end=valid_end, test_end=test_end)
    if reuse_base_artifact:
        base_artifact = load_opportunity_ranker_artifact(project_root)
        base_train = _score_v42_hybrid_from_artifact(split_frames["train"], base_artifact).assign(dataset_split="train")
        base_valid = _score_v42_hybrid_from_artifact(split_frames["valid"], base_artifact).assign(dataset_split="valid")
        base_test = _score_v42_hybrid_from_artifact(split_frames["test"], base_artifact).assign(dataset_split="test")
        base_training_mode = "reuse_existing_v42_artifact"
    else:
        base_train, base_valid, base_test, base_artifact = _train_v42_hybrid_base_for_v5(
            dataset,
            split_frames,
            max_iter=max_iter,
            top_n_list=top_n_list,
            train_end=train_end,
            valid_end=valid_end,
            test_end=test_end,
        )
        base_training_mode = "retrain_v42_hybrid_oof"

    vp_features = volume_price_feature_columns(base_train)
    vp_risk_model = fit_volume_price_risk_model(base_train, feature_columns=vp_features, max_iter=max_iter)
    vp_quality_model = fit_volume_price_quality_model(base_train, feature_columns=vp_features, max_iter=max_iter)

    scored_train = build_volume_price_oof_predictions(base_train, feature_columns=vp_features, max_iter=max_iter)
    if scored_train.empty:
        scored_train = score_volume_price_submodels(
            base_train,
            risk_model=vp_risk_model,
            quality_model=vp_quality_model,
            feature_columns=vp_features,
        )
    scored_valid = score_volume_price_submodels(
        base_valid,
        risk_model=vp_risk_model,
        quality_model=vp_quality_model,
        feature_columns=vp_features,
    )
    scored_test = score_volume_price_submodels(
        base_test,
        risk_model=vp_risk_model,
        quality_model=vp_quality_model,
        feature_columns=vp_features,
    )
    scored_train = add_v5_fusion_target(scored_train)
    scored_valid = add_v5_fusion_target(scored_valid)
    scored_test = add_v5_fusion_target(scored_test)
    fusion_features = v5_fusion_feature_columns(scored_train)
    fusion_model = fit_v5_fusion_model(scored_train, feature_columns=fusion_features, max_iter=max_iter)

    v5_train = apply_v5_decision(score_v5_fusion_frame(scored_train, fusion_model=fusion_model, fusion_features=fusion_features))
    v5_valid = apply_v5_decision(score_v5_fusion_frame(scored_valid, fusion_model=fusion_model, fusion_features=fusion_features))
    v5_test = apply_v5_decision(score_v5_fusion_frame(scored_test, fusion_model=fusion_model, fusion_features=fusion_features))
    evaluated = pd.concat([v5_train, v5_valid, v5_test], ignore_index=True, copy=False)
    baseline = pd.concat([base_train, base_valid, base_test], ignore_index=True, copy=False)

    topn_metrics = evaluate_v5_topn_metrics(evaluated, top_n_list=top_n_list)
    risk_metrics = evaluate_volume_price_risk_metrics(evaluated)
    quality_metrics = evaluate_volume_price_quality_metrics(evaluated)
    comparison_metrics = compare_v5_topn_metrics(baseline, evaluated, top_n_list=top_n_list)

    topn_metrics.to_csv(report_dir / "v5_topn_metrics.csv", index=False, encoding="utf-8-sig")
    risk_metrics.to_csv(report_dir / "v5_volume_price_risk_metrics.csv", index=False, encoding="utf-8-sig")
    quality_metrics.to_csv(report_dir / "v5_volume_price_quality_metrics.csv", index=False, encoding="utf-8-sig")
    comparison_metrics.to_csv(report_dir / "v5_comparison.csv", index=False, encoding="utf-8-sig")

    artifact = {
        "model_version": "v5_volume_price_fusion",
        "created_at": datetime.now(UTC).isoformat(),
        "base_training_mode": base_training_mode,
        "base_artifact": base_artifact,
        "volume_price_feature_columns": vp_features,
        "volume_price_risk_model": vp_risk_model,
        "volume_price_quality_model": vp_quality_model,
        "fusion_features": fusion_features,
        "fusion_model": fusion_model,
        "top_n_list": top_n_list,
        "split_dates": {
            "train_end": train_end.isoformat(),
            "valid_end": valid_end.isoformat(),
            "test_end": test_end.isoformat(),
        },
    }
    model_path = model_dir / "v5_volume_price_fusion.pkl"
    with model_path.open("wb") as file:
        pickle.dump(artifact, file)
    metadata_path = model_dir / "v5_volume_price_fusion_metadata.json"
    metadata = {
        "model_version": "v5_volume_price_fusion",
        "created_at": artifact["created_at"],
        "base_training_mode": base_training_mode,
        "volume_price_feature_count": len(vp_features),
        "fusion_feature_count": len(fusion_features),
        "top_n_list": list(top_n_list),
        "split_dates": artifact["split_dates"],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    prediction_path: Path | None = None
    latest_predictions = pd.DataFrame()
    if prediction_date is not None:
        latest_predictions = predict_volume_price_fusion(
            storage=storage,
            config=config,
            project_root=project_root,
            trade_date=prediction_date,
        )
        if not latest_predictions.empty:
            prediction_path = report_dir / f"predictions_{prediction_date.isoformat()}.csv"
            latest_predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    return VolumePriceFusionTrainResult(
        model_path=model_path,
        metadata_path=metadata_path,
        topn_metrics=topn_metrics,
        volume_price_risk_metrics=risk_metrics,
        volume_price_quality_metrics=quality_metrics,
        comparison_metrics=comparison_metrics,
        prediction_path=prediction_path,
        latest_predictions=latest_predictions,
        skipped_symbols=skipped_symbols,
    )


def predict_volume_price_fusion(
    *,
    storage: Storage,
    config: AppConfig,
    project_root: Path,
    trade_date: date,
) -> pd.DataFrame:
    artifact = load_volume_price_fusion_artifact(project_root)
    frame = build_prediction_frame(storage=storage, config=config, trade_date=trade_date)
    if frame.empty:
        return pd.DataFrame()
    base = _score_v42_hybrid_from_artifact(frame, artifact["base_artifact"])
    scored = score_volume_price_submodels(
        base,
        risk_model=artifact["volume_price_risk_model"],
        quality_model=artifact["volume_price_quality_model"],
        feature_columns=artifact["volume_price_feature_columns"],
    )
    fused = score_v5_fusion_frame(scored, fusion_model=artifact["fusion_model"], fusion_features=artifact["fusion_features"])
    fused = apply_v5_decision(fused)
    fused = fused.sort_values(["action_rank_v5", "final_score_v5"], ascending=[True, False]).reset_index(drop=True)
    fused["rank"] = range(1, len(fused) + 1)
    return prepare_v5_prediction_report(fused)


def load_volume_price_fusion_artifact(project_root: Path) -> dict[str, Any]:
    model_path = volume_price_fusion_model_dir(project_root) / "v5_volume_price_fusion.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"V5 volume-price fusion model not found: {model_path}")
    with model_path.open("rb") as file:
        return pickle.load(file)


def _score_v5_from_artifact_frame(frame: pd.DataFrame, artifact: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = _score_v42_hybrid_from_artifact(frame, artifact["base_artifact"])
    scored = score_volume_price_submodels(
        base,
        risk_model=artifact["volume_price_risk_model"],
        quality_model=artifact["volume_price_quality_model"],
        feature_columns=artifact["volume_price_feature_columns"],
    )
    fused = score_v5_fusion_frame(scored, fusion_model=artifact["fusion_model"], fusion_features=artifact["fusion_features"])
    v5 = apply_v5_decision(fused)
    return base, v5


def train_candidate_ranker_model(
    *,
    storage: Storage,
    config: AppConfig,
    project_root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    train_end: date | None = None,
    valid_end: date | None = None,
    test_end: date | None = None,
    limit: int | None = None,
    max_iter: int = 80,
    top_n_list: tuple[int, ...] = (20, 50),
    prediction_date: date | None = None,
) -> CandidateRankerTrainResult:
    model_dir = candidate_ranker_model_dir(project_root)
    report_dir = candidate_ranker_report_dir(project_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    v5_artifact = load_volume_price_fusion_artifact(project_root)
    split_dates = v5_artifact.get("split_dates", {})
    if train_end is None and isinstance(split_dates, dict) and split_dates.get("train_end"):
        train_end = date.fromisoformat(str(split_dates["train_end"]))
    if valid_end is None and isinstance(split_dates, dict) and split_dates.get("valid_end"):
        valid_end = date.fromisoformat(str(split_dates["valid_end"]))
    if test_end is None and isinstance(split_dates, dict) and split_dates.get("test_end"):
        test_end = date.fromisoformat(str(split_dates["test_end"]))

    dataset, skipped_symbols = build_stacked_dataset(
        storage=storage,
        config=config,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    if dataset.empty:
        raise RuntimeError("No V5.1 candidate-ranker dataset could be built from local daily bars.")
    dataset = add_v5_volume_price_labels(add_v4_risk_upside_labels(dataset))

    train_end, valid_end, test_end = resolve_split_dates(dataset, train_end=train_end, valid_end=valid_end, test_end=test_end)
    split_frames = split_dataset(dataset, train_end=train_end, valid_end=valid_end, test_end=test_end)

    base_train, v5_train = _score_v5_from_artifact_frame(split_frames["train"], v5_artifact)
    base_valid, v5_valid = _score_v5_from_artifact_frame(split_frames["valid"], v5_artifact)
    base_test, v5_test = _score_v5_from_artifact_frame(split_frames["test"], v5_artifact)
    base_train = base_train.assign(dataset_split="train")
    base_valid = base_valid.assign(dataset_split="valid")
    base_test = base_test.assign(dataset_split="test")
    v5_train = v5_train.assign(dataset_split="train")
    v5_valid = v5_valid.assign(dataset_split="valid")
    v5_test = v5_test.assign(dataset_split="test")

    train_labeled = add_v51_cross_sectional_rank_features(add_v51_candidate_rank_labels(v5_train))
    ranker_features = v51_candidate_ranker_feature_columns(train_labeled)
    ranker_model, ranker_engine = fit_v51_candidate_ranker_model(
        train_labeled,
        feature_columns=ranker_features,
        max_iter=max_iter,
    )

    scored_train = score_v51_candidate_ranker_frame(v5_train, ranker_model=ranker_model, feature_columns=ranker_features)
    scored_valid = score_v51_candidate_ranker_frame(v5_valid, ranker_model=ranker_model, feature_columns=ranker_features)
    scored_test = score_v51_candidate_ranker_frame(v5_test, ranker_model=ranker_model, feature_columns=ranker_features)
    selected_blend_params, blend_grid = select_v51_blend_params(scored_valid, top_n_list=top_n_list)
    v51_train = apply_v51_blend_score(scored_train, selected_blend_params)
    v51_valid = apply_v51_blend_score(scored_valid, selected_blend_params)
    v51_test = apply_v51_blend_score(scored_test, selected_blend_params)

    v51_evaluated = pd.concat([v51_train, v51_valid, v51_test], ignore_index=True, copy=False)
    v5_evaluated = pd.concat([v5_train, v5_valid, v5_test], ignore_index=True, copy=False)
    baseline = pd.concat([base_train, base_valid, base_test], ignore_index=True, copy=False)

    topn_metrics = evaluate_v51_topn_metrics(v51_evaluated, top_n_list=top_n_list)
    ranker_metrics = evaluate_v51_ranker_metrics(v51_evaluated, top_n_list=top_n_list)
    comparison_metrics = compare_v51_topn_metrics(
        baseline_scored=baseline,
        v5_scored=v5_evaluated,
        v51_scored=v51_evaluated,
        top_n_list=top_n_list,
    )

    topn_metrics.to_csv(report_dir / "v51_topn_metrics.csv", index=False, encoding="utf-8-sig")
    ranker_metrics.to_csv(report_dir / "v51_ranker_metrics.csv", index=False, encoding="utf-8-sig")
    comparison_metrics.to_csv(report_dir / "v51_comparison.csv", index=False, encoding="utf-8-sig")
    blend_grid.to_csv(report_dir / "v51_blend_grid.csv", index=False, encoding="utf-8-sig")
    skipped_symbols.to_csv(report_dir / "v51_skipped_symbols.csv", index=False, encoding="utf-8-sig")

    artifact = {
        "model_version": "v51_candidate_ranker",
        "created_at": datetime.now(UTC).isoformat(),
        "v5_artifact": v5_artifact,
        "ranker_features": ranker_features,
        "ranker_model": ranker_model,
        "ranker_engine": ranker_engine,
        "selected_blend_params": selected_blend_params,
        "top_n_list": top_n_list,
        "split_dates": {
            "train_end": train_end.isoformat(),
            "valid_end": valid_end.isoformat(),
            "test_end": test_end.isoformat(),
        },
    }
    model_path = model_dir / "v51_candidate_ranker.pkl"
    with model_path.open("wb") as file:
        pickle.dump(artifact, file)
    metadata_path = model_dir / "v51_candidate_ranker_metadata.json"
    metadata = {
        "model_version": artifact["model_version"],
        "created_at": artifact["created_at"],
        "ranker_engine": ranker_engine,
        "ranker_feature_count": len(ranker_features),
        "selected_blend_params": selected_blend_params,
        "top_n_list": list(top_n_list),
        "split_dates": artifact["split_dates"],
        "v5_base_training_mode": v5_artifact.get("base_training_mode"),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    prediction_path: Path | None = None
    latest_predictions = pd.DataFrame()
    if prediction_date is not None:
        latest_predictions = predict_candidate_ranker(
            storage=storage,
            config=config,
            project_root=project_root,
            trade_date=prediction_date,
        )
        if not latest_predictions.empty:
            prediction_path = report_dir / f"predictions_{prediction_date.isoformat()}.csv"
            latest_predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    return CandidateRankerTrainResult(
        model_path=model_path,
        metadata_path=metadata_path,
        topn_metrics=topn_metrics,
        ranker_metrics=ranker_metrics,
        comparison_metrics=comparison_metrics,
        blend_grid=blend_grid,
        selected_blend_params=selected_blend_params,
        prediction_path=prediction_path,
        latest_predictions=latest_predictions,
        skipped_symbols=skipped_symbols,
    )


def predict_candidate_ranker(
    *,
    storage: Storage,
    config: AppConfig,
    project_root: Path,
    trade_date: date,
) -> pd.DataFrame:
    artifact = load_candidate_ranker_artifact(project_root)
    frame = build_prediction_frame(storage=storage, config=config, trade_date=trade_date)
    if frame.empty:
        return pd.DataFrame()
    _, v5 = _score_v5_from_artifact_frame(frame, artifact["v5_artifact"])
    scored = score_v51_candidate_ranker_frame(
        v5,
        ranker_model=artifact["ranker_model"],
        feature_columns=artifact["ranker_features"],
    )
    blended = apply_v51_blend_score(scored, artifact["selected_blend_params"])
    blended = blended.sort_values(["action_rank_v51", "final_score_v51"], ascending=[True, False]).reset_index(drop=True)
    blended["rank"] = range(1, len(blended) + 1)
    return prepare_v51_prediction_report(blended)


def load_candidate_ranker_artifact(project_root: Path) -> dict[str, Any]:
    model_path = candidate_ranker_model_dir(project_root) / "v51_candidate_ranker.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"V5.1 candidate ranker model not found: {model_path}")
    with model_path.open("rb") as file:
        return pickle.load(file)


def validate_model_walkforward(
    *,
    storage: Storage,
    config: AppConfig,
    project_root: Path,
    model: str = "v42",
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    windows: int = 8,
    train_days: int = 280,
    valid_days: int = 60,
    test_days: int = 60,
    min_train_days: int = 220,
    max_iter: int = 40,
    top_n_list: tuple[int, ...] = (20, 50),
) -> WalkForwardValidationResult:
    if model != "v42":
        raise ValueError("Only model='v42' is supported in the first walk-forward validator.")
    report_dir = model_walkforward_report_dir(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset, skipped_symbols = build_stacked_dataset(
        storage=storage,
        config=config,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    if dataset.empty:
        raise RuntimeError("No walk-forward dataset could be built from local daily bars.")
    dataset = add_v4_risk_upside_labels(dataset)
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"], errors="coerce")
    dataset = dataset[dataset["trade_date"].notna()].copy()
    trade_dates = sorted(dataset["trade_date"].dt.date.unique())
    windows_frame = generate_walkforward_windows(
        trade_dates,
        windows=windows,
        train_days=train_days,
        valid_days=valid_days,
        test_days=test_days,
        min_train_days=min_train_days,
    )
    if windows_frame.empty:
        raise RuntimeError(
            "Not enough distinct trade dates for walk-forward validation. "
            "Reduce --train-days, --valid-days, --test-days, or --min-train-days."
        )

    window_records: list[dict[str, object]] = []
    metric_frames: list[pd.DataFrame] = []
    date_values = dataset["trade_date"].dt.date
    for window in windows_frame.to_dict("records"):
        window_started_at = perf_counter()
        window_record = dict(window)
        window_dataset = dataset[
            (date_values >= window["train_start"]) & (date_values <= window["test_end"])
        ].reset_index(drop=True)
        try:
            split_frames = split_dataset(
                window_dataset,
                train_end=window["train_end"],
                valid_end=window["valid_end"],
                test_end=window["test_end"],
            )
            base_train, base_valid, base_test, _ = _train_v42_hybrid_base_for_v5(
                window_dataset,
                split_frames,
                max_iter=max_iter,
                top_n_list=top_n_list,
                train_end=window["train_end"],
                valid_end=window["valid_end"],
                test_end=window["test_end"],
                include_deployment_artifact=False,
            )
            base_train = base_train.assign(dataset_split="train")
            base_valid = base_valid.assign(dataset_split="valid")
            base_test = base_test.assign(dataset_split="test")
            evaluated = pd.concat([base_train, base_valid, base_test], ignore_index=True, copy=False)
            metrics = evaluate_v42_topn_metrics(evaluated, top_n_list=top_n_list)
            if not metrics.empty and "dataset_split" in metrics.columns:
                metrics = metrics[metrics["dataset_split"].eq("test")].copy()
            if metrics.empty or "dataset_split" not in metrics.columns:
                metrics = _empty_walkforward_topn_metrics(
                    window=window,
                    top_n_list=top_n_list,
                    model_version="v42_gate_v4_rank",
                )
                metric_frames.append(metrics)
                window_record["status"] = "no_candidates"
                window_record["top20_coverage"] = 0.0
            else:
                metrics["window_id"] = int(window["window_id"])
                metrics["model_version"] = "v42_gate_v4_rank"
                metrics["train_start"] = window["train_start"]
                metrics["train_end"] = window["train_end"]
                metrics["valid_start"] = window["valid_start"]
                metrics["valid_end"] = window["valid_end"]
                metrics["test_start"] = window["test_start"]
                metrics["test_end"] = window["test_end"]
                metrics["train_days"] = int(window["train_days"])
                metrics["valid_days"] = int(window["valid_days"])
                metrics["test_days"] = int(window["test_days"])
                metrics["allowed_days"] = pd.to_numeric(metrics["days"], errors="coerce").fillna(0).astype(int)
                metrics["coverage"] = metrics["allowed_days"] / max(int(window["test_days"]), 1)
                metric_frames.append(metrics)
                window_record["status"] = "ok"
                top20 = metrics[pd.to_numeric(metrics["top_n"], errors="coerce").eq(20)]
                if not top20.empty:
                    window_record["top20_win_rate"] = float(top20.iloc[0].get("win_rate", float("nan")))
                    window_record["top20_avg_return_20d"] = float(top20.iloc[0].get("avg_return_20d", float("nan")))
                    window_record["top20_stop_loss_rate_20d"] = float(
                        top20.iloc[0].get("stop_loss_rate_20d", float("nan"))
                    )
                    window_record["top20_bad_risk_rate"] = float(top20.iloc[0].get("bad_risk_rate", float("nan")))
                    window_record["top20_coverage"] = float(top20.iloc[0].get("coverage", float("nan")))
        except Exception as exc:  # pragma: no cover - defensive for long-running local experiments.
            window_record["status"] = "failed"
            window_record["error"] = str(exc)
            window_record["error_trace"] = traceback.format_exc(limit=6)
        window_record["elapsed_seconds"] = round(perf_counter() - window_started_at, 3)
        window_records.append(window_record)

    windows_result = pd.DataFrame(window_records)
    topn_metrics = pd.concat(metric_frames, ignore_index=True, copy=False) if metric_frames else pd.DataFrame()
    if topn_metrics.empty:
        windows_path = report_dir / "v42_walkforward_windows.csv"
        windows_result.to_csv(windows_path, index=False, encoding="utf-8-sig")
        if "error" in windows_result.columns:
            error_columns = ["error"]
            if "error_trace" in windows_result.columns:
                error_columns.append("error_trace")
            errors = (
                windows_result.loc[windows_result["error"].notna(), error_columns]
                .astype(str)
                .head(3)
                .agg(" | ".join, axis=1)
                .tolist()
            )
        else:
            errors = []
        detail = f" First errors: {' | '.join(errors)}" if errors else ""
        raise RuntimeError(f"Walk-forward validation did not produce any TopN metrics.{detail}")
    summary = summarize_walkforward_topn_metrics(topn_metrics)

    config_payload = {
        "model": model,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "windows_requested": windows,
        "windows_completed": (
            int(windows_result["status"].isin(["ok", "no_candidates"]).sum())
            if "status" in windows_result.columns
            else 0
        ),
        "train_days": train_days,
        "valid_days": valid_days,
        "test_days": test_days,
        "min_train_days": min_train_days,
        "effective_windows": [
            {
                key: (value.isoformat() if isinstance(value, date) else value)
                for key, value in row.items()
            }
            for row in windows_frame.to_dict("records")
        ],
        "max_iter": max_iter,
        "top_n_list": list(top_n_list),
        "thresholds_top20": WALKFORWARD_TOP20_THRESHOLDS,
    }

    windows_path = report_dir / "v42_walkforward_windows.csv"
    topn_metrics_path = report_dir / "v42_walkforward_topn_metrics.csv"
    summary_path = report_dir / "v42_walkforward_summary.csv"
    config_path = report_dir / "v42_walkforward_config.json"
    windows_result.to_csv(windows_path, index=False, encoding="utf-8-sig")
    topn_metrics.to_csv(topn_metrics_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return WalkForwardValidationResult(
        report_dir=report_dir,
        config_path=config_path,
        windows_path=windows_path,
        topn_metrics_path=topn_metrics_path,
        summary_path=summary_path,
        windows=windows_result,
        topn_metrics=topn_metrics,
        summary=summary,
        skipped_symbols=skipped_symbols,
    )


def load_opportunity_ranker_artifact(project_root: Path) -> dict[str, Any]:
    model_path = opportunity_ranker_model_dir(project_root) / "v42_opportunity_ranker.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"V4.2 opportunity-ranker model not found: {model_path}")
    with model_path.open("rb") as file:
        return pickle.load(file)


def build_stacked_feature_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy().sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for column in ("open", "high", "low", "close", "volume", "amount"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "turnover" in df.columns:
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")

    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    amount = df["amount"]

    for window in (3, 5, 10, 20, 40, 60, 120, 180):
        df[f"return_{window}d"] = close.pct_change(window, fill_method=None)
        df[f"ma_{window}"] = close.rolling(window).mean()
        df[f"distance_to_ma{window}"] = close.div(df[f"ma_{window}"].replace(0.0, np.nan)) - 1
        df[f"ma{window}_slope_1d"] = df[f"ma_{window}"].pct_change(fill_method=None)
        df[f"ma{window}_slope_5d"] = df[f"ma_{window}"].pct_change(5, fill_method=None)
        df[f"ma{window}_slope_10d"] = df[f"ma_{window}"].pct_change(10, fill_method=None)

    for window in (2, 3, 5, 10, 20, 60):
        df[f"volume_ma_{window}"] = volume.rolling(window).mean()
        df[f"amount_ma_{window}"] = amount.rolling(window).mean()
        df[f"volume_ratio_{window}"] = volume.div(df[f"volume_ma_{window}"].replace(0.0, np.nan))
        df[f"amount_ratio_{window}"] = amount.div(df[f"amount_ma_{window}"].replace(0.0, np.nan))
        df[f"volume_change_{window}d"] = volume.div(volume.shift(window).replace(0.0, np.nan)) - 1
        df[f"amount_change_{window}d"] = amount.div(amount.shift(window).replace(0.0, np.nan)) - 1

    df = df.copy()
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    amount = df["amount"]

    df["ma5_accel"] = df["ma5_slope_1d"].diff()
    df["ma5_to_ma10"] = df["ma_5"].div(df["ma_10"].replace(0.0, np.nan)) - 1
    df["ma20_to_ma60"] = df["ma_20"].div(df["ma_60"].replace(0.0, np.nan)) - 1
    df["ma5_to_ma10_change_5d"] = df["ma5_to_ma10"].diff(5)
    df["ma20_to_ma60_change_20d"] = df["ma20_to_ma60"].diff(20)

    candle_range = (high - low).replace(0.0, np.nan)
    df["intraday_range_pct"] = candle_range.div(close.replace(0.0, np.nan))
    df["body_pct"] = (close - open_).abs().div(candle_range)
    df["upper_shadow_pct"] = (high - pd.concat([open_, close], axis=1).max(axis=1)).div(candle_range)
    df["lower_shadow_pct"] = (pd.concat([open_, close], axis=1).min(axis=1) - low).div(candle_range)
    df["gap_pct"] = open_.div(close.shift(1).replace(0.0, np.nan)) - 1

    df["return_1d"] = close.pct_change(fill_method=None)
    df["return_2d"] = close.pct_change(2, fill_method=None)
    up = close > close.shift(1)
    down = close < close.shift(1)
    for window in (3, 5, 10):
        df[f"up_days_{window}"] = up.rolling(window).sum()
        df[f"down_days_{window}"] = down.rolling(window).sum()

    for window in (5, 10, 20, 60, 120):
        rolling_high = high.rolling(window).max()
        rolling_low = low.rolling(window).min()
        df[f"volatility_{window}d"] = df["return_1d"].rolling(window).std() * math.sqrt(window)
        df[f"drawdown_{window}d"] = 1 - low.rolling(window).min().div(rolling_high.replace(0.0, np.nan))
        df[f"position_in_range_{window}d"] = (close - rolling_low).div((rolling_high - rolling_low).replace(0.0, np.nan))
        df[f"distance_to_{window}d_high"] = close.div(rolling_high.replace(0.0, np.nan)) - 1
        df[f"distance_to_{window}d_low"] = close.div(rolling_low.replace(0.0, np.nan)) - 1

    df = _add_low_level_technical_indicators(df)
    df = _add_block_features(df, window=20, blocks=3)
    df = _add_block_features(df, window=60, blocks=3)
    df = add_v5_volume_price_features(df)

    return _clean_numeric_frame(df)


def add_v5_volume_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add daily-data volume-price features for V5.

    The features intentionally use only current and historical rows. They are
    grouped by meaning rather than by model layer: 1d trigger/risk hints, 5d
    buy-point confirmation, 20d structure, and 5d-vs-20d acceleration.
    """
    result = frame.copy()
    close = _numeric_feature(result, "close")
    open_ = _numeric_feature(result, "open")
    high = _numeric_feature(result, "high")
    low = _numeric_feature(result, "low")
    volume = _numeric_feature(result, "volume")
    amount = _numeric_feature(result, "amount")

    prev_close = close.shift(1)
    candle_range = (high - low).replace(0.0, np.nan)
    result["vp_close_position_1d"] = (close - low).div(candle_range)
    result["vp_signed_body_1d"] = (close - open_).div(candle_range)
    result["vp_body_abs_1d"] = (close - open_).abs().div(candle_range)
    result["vp_upper_shadow_1d"] = (high - pd.concat([open_, close], axis=1).max(axis=1)).div(candle_range)
    result["vp_lower_shadow_1d"] = (pd.concat([open_, close], axis=1).min(axis=1) - low).div(candle_range)
    result["vp_return_1d"] = close.div(prev_close.replace(0.0, np.nan)) - 1
    result["vp_gap_1d"] = open_.div(prev_close.replace(0.0, np.nan)) - 1

    volume_ma_5 = volume.rolling(5).mean()
    volume_ma_20 = volume.rolling(20).mean()
    amount_ma_5 = amount.rolling(5).mean()
    amount_ma_20 = amount.rolling(20).mean()
    result["vp_volume_ratio_1d_to_5d"] = volume.div(volume_ma_5.replace(0.0, np.nan))
    result["vp_volume_ratio_1d_to_20d"] = volume.div(volume_ma_20.replace(0.0, np.nan))
    result["vp_amount_ratio_1d_to_5d"] = amount.div(amount_ma_5.replace(0.0, np.nan))
    result["vp_amount_ratio_1d_to_20d"] = amount.div(amount_ma_20.replace(0.0, np.nan))
    if "turnover" in result.columns:
        result["vp_turnover_1d"] = _numeric_feature(result, "turnover")

    high_volume_1d = result["vp_volume_ratio_1d_to_20d"] >= 1.6
    weak_close_1d = result["vp_close_position_1d"] <= 0.45
    bearish_body_1d = result["vp_signed_body_1d"] <= -0.25
    result["vp_high_volume_upper_shadow_flag"] = (high_volume_1d & weak_close_1d & (result["vp_upper_shadow_1d"] >= 0.35)).astype(float)
    result["vp_high_volume_bearish_flag"] = (high_volume_1d & bearish_body_1d).astype(float)
    result["vp_low_volume_stabilization_flag"] = (
        (result["vp_volume_ratio_1d_to_20d"] <= 0.85)
        & (result["vp_close_position_1d"] >= 0.50)
        & (result["vp_return_1d"] >= -0.015)
    ).astype(float)
    result["vp_failed_breakout_1d_flag"] = (
        high_volume_1d
        & (result.get("distance_to_20d_high", pd.Series(np.nan, index=result.index)) >= -0.02)
        & weak_close_1d
    ).astype(float)

    return_5d = close.pct_change(5, fill_method=None)
    return_20d = close.pct_change(20, fill_method=None)
    result["vp_return_5d"] = return_5d
    result["vp_return_20d"] = return_20d
    result["vp_volume_change_5d"] = volume.div(volume.shift(5).replace(0.0, np.nan)) - 1
    result["vp_amount_change_5d"] = amount.div(amount.shift(5).replace(0.0, np.nan)) - 1
    result["vp_volume_change_20d"] = volume.div(volume.shift(20).replace(0.0, np.nan)) - 1
    result["vp_amount_change_20d"] = amount.div(amount.shift(20).replace(0.0, np.nan)) - 1

    up_day = close > prev_close
    down_day = close < prev_close
    high_volume_day = result["vp_volume_ratio_1d_to_20d"] >= 1.4
    weak_high_volume_day = high_volume_day & weak_close_1d
    strong_high_volume_day = high_volume_day & (result["vp_close_position_1d"] >= 0.60) & up_day
    up_volume_5d = volume.where(up_day, 0.0).rolling(5).sum()
    down_volume_5d = volume.where(down_day, 0.0).rolling(5).sum()
    total_volume_5d = volume.rolling(5).sum()
    result["vp_5d_up_volume_share"] = up_volume_5d.div(total_volume_5d.replace(0.0, np.nan))
    result["vp_5d_down_volume_share"] = down_volume_5d.div(total_volume_5d.replace(0.0, np.nan))
    result["vp_5d_up_down_volume_ratio"] = up_volume_5d.div(down_volume_5d.replace(0.0, np.nan))
    result["vp_5d_high_volume_weak_days"] = weak_high_volume_day.astype(float).rolling(5).sum()
    result["vp_5d_high_volume_strong_days"] = strong_high_volume_day.astype(float).rolling(5).sum()
    result["vp_5d_upper_shadow_pressure"] = result["vp_upper_shadow_1d"].rolling(5).mean()
    result["vp_5d_lower_shadow_support"] = result["vp_lower_shadow_1d"].rolling(5).mean()
    result["vp_5d_volume_concentration"] = volume.rolling(5).max().div(total_volume_5d.replace(0.0, np.nan))
    result["vp_5d_price_volume_confirm"] = return_5d * result["vp_volume_change_5d"]
    result["vp_5d_volume_without_price"] = result["vp_volume_change_5d"].clip(lower=0.0) * (-return_5d).clip(lower=0.0)
    result["vp_5d_shrink_pullback_score"] = (-return_5d).clip(lower=0.0) * (1 - result["vp_volume_ratio_1d_to_20d"]).clip(lower=0.0)

    rolling_high_20 = high.rolling(20).max()
    rolling_low_20 = low.rolling(20).min()
    result["vp_20d_range_position"] = (close - rolling_low_20).div((rolling_high_20 - rolling_low_20).replace(0.0, np.nan))
    up_volume_20d = volume.where(up_day, 0.0).rolling(20).sum()
    down_volume_20d = volume.where(down_day, 0.0).rolling(20).sum()
    total_volume_20d = volume.rolling(20).sum()
    result["vp_20d_up_volume_share"] = up_volume_20d.div(total_volume_20d.replace(0.0, np.nan))
    result["vp_20d_down_volume_share"] = down_volume_20d.div(total_volume_20d.replace(0.0, np.nan))
    result["vp_20d_up_down_volume_ratio"] = up_volume_20d.div(down_volume_20d.replace(0.0, np.nan))
    result["vp_20d_high_volume_weak_days"] = weak_high_volume_day.astype(float).rolling(20).sum()
    result["vp_20d_high_volume_strong_days"] = strong_high_volume_day.astype(float).rolling(20).sum()
    result["vp_20d_volume_trend"] = volume_ma_5.div(volume_ma_20.replace(0.0, np.nan)) - 1
    result["vp_20d_amount_trend"] = amount_ma_5.div(amount_ma_20.replace(0.0, np.nan)) - 1
    result["vp_20d_price_progress_per_volume"] = return_20d.div(result["vp_volume_change_20d"].abs().add(0.20))
    result["vp_20d_amount_progress_per_return"] = result["vp_amount_change_20d"].div(return_20d.abs().add(0.05))
    result["vp_20d_accumulation_score"] = (
        0.35 * result["vp_20d_range_position"].clip(0.0, 1.0)
        + 0.30 * result["vp_20d_up_volume_share"].clip(0.0, 1.0)
        + 0.20 * result["vp_5d_lower_shadow_support"].clip(0.0, 1.0)
        + 0.15 * (1 - result["vp_20d_high_volume_weak_days"].div(20).clip(0.0, 1.0))
    )
    high_position_20d = result["vp_20d_range_position"] >= 0.70
    result["vp_20d_distribution_score"] = (
        0.35 * high_position_20d.astype(float)
        + 0.30 * result["vp_20d_high_volume_weak_days"].div(20).clip(0.0, 1.0)
        + 0.20 * result["vp_5d_upper_shadow_pressure"].clip(0.0, 1.0)
        + 0.15 * result["vp_5d_volume_without_price"].clip(0.0, 1.0)
    )

    result["vp_5d_vs_20d_return_accel"] = return_5d - (return_20d / 4)
    result["vp_5d_vs_20d_volume_accel"] = result["vp_volume_change_5d"] - (result["vp_volume_change_20d"] / 4)
    result["vp_5d_vs_20d_amount_accel"] = result["vp_amount_change_5d"] - (result["vp_amount_change_20d"] / 4)
    result["vp_volume_accel_without_price"] = result["vp_5d_vs_20d_volume_accel"].clip(lower=0.0) * (
        -result["vp_5d_vs_20d_return_accel"]
    ).clip(lower=0.0)
    result["vp_short_shrink_after_strength"] = (
        (return_20d > 0).astype(float)
        * (-return_5d).clip(lower=0.0)
        * (1 - result["vp_volume_change_5d"]).clip(lower=0.0)
    )
    result["vp_pullback_depth_in_20d"] = (-return_5d).clip(lower=0.0).div(return_20d.clip(lower=0.02))
    return result


def _add_low_level_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]

    for period in (6, 14, 24):
        result[f"rsi_{period}"] = _rsi(close, period)

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["macd"] = ema12 - ema26
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    result["macd_hist"] = result["macd"] - result["macd_signal"]
    result["macd_hist_slope"] = result["macd_hist"].diff()

    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    result["stoch_k"] = 100 * (close - low_9).div((high_9 - low_9).replace(0.0, np.nan))
    result["stoch_d"] = result["stoch_k"].rolling(3).mean()
    result["stoch_j"] = 3 * result["stoch_k"] - 2 * result["stoch_d"]
    result["stoch_k_minus_d"] = result["stoch_k"] - result["stoch_d"]

    typical = (high + low + close) / 3
    typical_ma = typical.rolling(20).mean()
    mean_dev = (typical - typical_ma).abs().rolling(20).mean()
    result["cci_20"] = (typical - typical_ma).div((0.015 * mean_dev).replace(0.0, np.nan))

    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    result["atr_14"] = _wilder_average(true_range, 14)
    result["atr_pct_14"] = result["atr_14"].div(close.replace(0.0, np.nan))

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=result.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=result.index)
    atr = result["atr_14"].replace(0.0, np.nan)
    result["plus_di_14"] = 100 * _wilder_average(plus_dm, 14).div(atr)
    result["minus_di_14"] = 100 * _wilder_average(minus_dm, 14).div(atr)
    dx = 100 * (result["plus_di_14"] - result["minus_di_14"]).abs().div(
        (result["plus_di_14"] + result["minus_di_14"]).replace(0.0, np.nan)
    )
    result["adx_14"] = _wilder_average(dx, 14)

    highest_14 = high.rolling(14).max()
    lowest_14 = low.rolling(14).min()
    result["williams_r_14"] = -100 * (highest_14 - close).div((highest_14 - lowest_14).replace(0.0, np.nan))

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    result["boll_width_20"] = (upper - lower).div(ma20.replace(0.0, np.nan))
    result["boll_position_20"] = (close - lower).div((upper - lower).replace(0.0, np.nan))
    return result


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _wilder_average(gain, period)
    avg_loss = _wilder_average(loss, period)
    rs = avg_gain.div(avg_loss.replace(0.0, np.nan))
    return 100 - (100 / (1 + rs))


def _wilder_average(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _add_block_features(df: pd.DataFrame, *, window: int, blocks: int) -> pd.DataFrame:
    result = df.copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]
    volume = result["volume"]
    amount = result["amount"]
    ret1 = result["return_1d"]

    volume_sums: list[pd.Series] = []
    amount_sums: list[pd.Series] = []
    for block in range(blocks):
        shift = block * window
        prefix = f"block{window}_{block}"
        result[f"{prefix}_return"] = close.shift(shift).div(close.shift(shift + window).replace(0.0, np.nan)) - 1
        volume_sum = volume.rolling(window).sum().shift(shift)
        amount_sum = amount.rolling(window).sum().shift(shift)
        volume_sums.append(volume_sum)
        amount_sums.append(amount_sum)
        result[f"{prefix}_volume_sum"] = volume_sum
        result[f"{prefix}_amount_sum"] = amount_sum
        block_high = high.rolling(window).max().shift(shift)
        block_low = low.rolling(window).min().shift(shift)
        result[f"{prefix}_drawdown"] = 1 - block_low.div(block_high.replace(0.0, np.nan))
        result[f"{prefix}_volatility"] = ret1.rolling(window).std().shift(shift) * math.sqrt(window)

    for block in range(blocks - 1):
        result[f"block{window}_{block}_vs_{block + 1}_volume_ratio"] = volume_sums[block].div(
            volume_sums[block + 1].replace(0.0, np.nan)
        )
        result[f"block{window}_{block}_vs_{block + 1}_amount_ratio"] = amount_sums[block].div(
            amount_sums[block + 1].replace(0.0, np.nan)
        )
        result[f"block{window}_{block}_vs_{block + 1}_return_diff"] = (
            result[f"block{window}_{block}_return"] - result[f"block{window}_{block + 1}_return"]
        )
    return result


def add_all_horizon_labels(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    for spec in HORIZONS:
        frame = add_path_label(frame, spec)
    ready = pd.Series(True, index=frame.index)
    trade_value = pd.Series(0.0, index=frame.index)
    for spec in HORIZONS:
        value_col = f"value_{spec.name}"
        ready &= frame[value_col].notna()
        ready &= ~frame[f"label_conflict_{spec.name}"].fillna(False).astype(bool)
        trade_value = trade_value + spec.trade_value_weight * pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0)
    frame["trade_value"] = trade_value
    frame.loc[~ready, "trade_value"] = pd.NA
    return frame


def add_path_label(frame: pd.DataFrame, spec: HorizonSpec) -> pd.DataFrame:
    df = frame.copy().sort_values("trade_date").reset_index(drop=True)
    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    suffix = spec.name
    for column in (
        f"entry_open_{suffix}",
        f"period_return_{suffix}",
        f"max_upside_{suffix}",
        f"max_drawdown_{suffix}",
        f"hit_day_{suffix}",
        f"value_{suffix}",
        f"outcome_class_{suffix}",
    ):
        df[column] = pd.NA
    df[f"outcome_{suffix}"] = pd.NA
    df[f"label_conflict_{suffix}"] = False

    for row_index in range(len(df)):
        entry_index = row_index + 1
        exit_index = entry_index + spec.horizon_days - 1
        if entry_index >= len(df) or exit_index >= len(df):
            continue
        entry_open = open_.iloc[entry_index]
        if pd.isna(entry_open) or float(entry_open) <= 0:
            continue

        entry_open_float = float(entry_open)
        up_price = entry_open_float * (1 + spec.upside_target)
        down_price = entry_open_float * (1 - spec.downside_threshold)
        window_high = high.iloc[entry_index : exit_index + 1]
        window_low = low.iloc[entry_index : exit_index + 1]
        window_close = close.iloc[entry_index : exit_index + 1]

        max_upside = float(window_high.max() / entry_open_float - 1)
        max_drawdown = float(1 - window_low.min() / entry_open_float)
        period_return = float(window_close.iloc[-1] / entry_open_float - 1)
        outcome = "neutral"
        outcome_class: int | None = OUTCOME_TO_CLASS[outcome]
        hit_day = spec.horizon_days
        conflict = False

        for offset in range(spec.horizon_days):
            day_high = window_high.iloc[offset]
            day_low = window_low.iloc[offset]
            if pd.isna(day_high) or pd.isna(day_low):
                continue
            hit_up = float(day_high) >= up_price
            hit_down = float(day_low) <= down_price
            if hit_up and hit_down:
                outcome = "conflict"
                outcome_class = None
                hit_day = offset + 1
                conflict = True
                break
            if hit_down:
                outcome = "down"
                outcome_class = OUTCOME_TO_CLASS[outcome]
                hit_day = offset + 1
                break
            if hit_up:
                outcome = "up"
                outcome_class = OUTCOME_TO_CLASS[outcome]
                hit_day = offset + 1
                break

        df.at[row_index, f"entry_open_{suffix}"] = entry_open_float
        df.at[row_index, f"period_return_{suffix}"] = period_return
        df.at[row_index, f"max_upside_{suffix}"] = max_upside
        df.at[row_index, f"max_drawdown_{suffix}"] = max_drawdown
        df.at[row_index, f"hit_day_{suffix}"] = hit_day
        df.at[row_index, f"outcome_{suffix}"] = outcome
        df.at[row_index, f"outcome_class_{suffix}"] = outcome_class if outcome_class is not None else pd.NA
        df.at[row_index, f"label_conflict_{suffix}"] = conflict
        if not conflict:
            df.at[row_index, f"value_{suffix}"] = realized_horizon_value(
                outcome=outcome,
                period_return=period_return,
                max_drawdown=max_drawdown,
                hit_day=hit_day,
                spec=spec,
            )
    return df


def realized_horizon_value(
    *,
    outcome: str,
    period_return: float,
    max_drawdown: float,
    hit_day: int,
    spec: HorizonSpec,
) -> float:
    outcome_score = {"up": 1.0, "down": -1.2, "neutral": 0.0}[outcome]
    final_return_score = 0.30 * float(np.clip(period_return / spec.upside_target, -1.0, 1.0))
    drawdown_penalty = 0.20 * min(max_drawdown / spec.downside_threshold, 2.0)
    speed = 1 - (hit_day / spec.horizon_days)
    if outcome == "up":
        speed_adjustment = 0.20 * speed
    elif outcome == "down":
        speed_adjustment = -0.30 * speed
    else:
        speed_adjustment = 0.0
    return float(outcome_score + final_return_score - drawdown_penalty + speed_adjustment)


def stage1_feature_columns(frame: pd.DataFrame, spec: HorizonSpec) -> list[str]:
    base_by_horizon: dict[str, list[str]] = {
        "5d": [
            "return_1d",
            "return_2d",
            "return_3d",
            "return_5d",
            "intraday_range_pct",
            "body_pct",
            "upper_shadow_pct",
            "lower_shadow_pct",
            "gap_pct",
            "distance_to_ma5",
            "ma5_slope_1d",
            "ma5_slope_5d",
            "ma5_accel",
            "volume_change_2d",
            "volume_change_3d",
            "volume_change_5d",
            "amount_change_2d",
            "amount_change_3d",
            "amount_change_5d",
            "volume_ratio_5",
            "amount_ratio_5",
            "volatility_5d",
            "drawdown_5d",
            "up_days_5",
            "down_days_5",
        ],
        "10d": [
            "return_5d",
            "return_10d",
            "distance_to_ma5",
            "distance_to_ma10",
            "ma5_slope_5d",
            "ma10_slope_5d",
            "ma5_to_ma10",
            "ma5_to_ma10_change_5d",
            "position_in_range_10d",
            "volume_ratio_10",
            "amount_ratio_10",
            "volume_change_10d",
            "amount_change_10d",
            "volatility_10d",
            "drawdown_10d",
            "up_days_10",
            "down_days_10",
        ],
        "20d": [
            "return_10d",
            "return_20d",
            "distance_to_ma20",
            "ma20_slope_1d",
            "ma20_slope_5d",
            "ma20_slope_10d",
            "position_in_range_20d",
            "distance_to_20d_high",
            "distance_to_20d_low",
            "volume_ratio_20",
            "amount_ratio_20",
            "volume_change_20d",
            "amount_change_20d",
            "volatility_20d",
            "drawdown_20d",
            "block20_0_return",
            "block20_1_return",
            "block20_2_return",
            "block20_0_volume_sum",
            "block20_1_volume_sum",
            "block20_2_volume_sum",
            "block20_0_amount_sum",
            "block20_1_amount_sum",
            "block20_2_amount_sum",
            "block20_0_drawdown",
            "block20_1_drawdown",
            "block20_2_drawdown",
            "block20_0_volatility",
            "block20_1_volatility",
            "block20_2_volatility",
            "block20_0_vs_1_volume_ratio",
            "block20_1_vs_2_volume_ratio",
            "block20_0_vs_1_amount_ratio",
            "block20_1_vs_2_amount_ratio",
            "block20_0_vs_1_return_diff",
            "block20_1_vs_2_return_diff",
        ],
        "60d": [
            "return_20d",
            "return_40d",
            "return_60d",
            "distance_to_ma20",
            "distance_to_ma60",
            "ma60_slope_1d",
            "ma60_slope_5d",
            "ma60_slope_10d",
            "ma20_to_ma60",
            "ma20_to_ma60_change_20d",
            "position_in_range_60d",
            "position_in_range_120d",
            "distance_to_60d_high",
            "distance_to_60d_low",
            "distance_to_120d_high",
            "distance_to_120d_low",
            "volume_ratio_60",
            "amount_ratio_60",
            "volume_change_60d",
            "amount_change_60d",
            "volatility_60d",
            "drawdown_60d",
            "block60_0_return",
            "block60_1_return",
            "block60_2_return",
            "block60_0_volume_sum",
            "block60_1_volume_sum",
            "block60_2_volume_sum",
            "block60_0_amount_sum",
            "block60_1_amount_sum",
            "block60_2_amount_sum",
            "block60_0_drawdown",
            "block60_1_drawdown",
            "block60_2_drawdown",
            "block60_0_volatility",
            "block60_1_volatility",
            "block60_2_volatility",
            "block60_0_vs_1_volume_ratio",
            "block60_1_vs_2_volume_ratio",
            "block60_0_vs_1_amount_ratio",
            "block60_1_vs_2_amount_ratio",
            "block60_0_vs_1_return_diff",
            "block60_1_vs_2_return_diff",
        ],
    }
    technical = [
        "rsi_6",
        "rsi_14",
        "rsi_24",
        "macd",
        "macd_signal",
        "macd_hist",
        "macd_hist_slope",
        "stoch_k",
        "stoch_d",
        "stoch_j",
        "stoch_k_minus_d",
        "cci_20",
        "plus_di_14",
        "minus_di_14",
        "adx_14",
        "williams_r_14",
        "boll_width_20",
        "boll_position_20",
        "atr_pct_14",
    ]
    columns = [column for column in [*base_by_horizon[spec.name], *technical] if column in frame.columns]
    return [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]


def resolve_split_dates(
    dataset: pd.DataFrame,
    *,
    train_end: date | None,
    valid_end: date | None,
    test_end: date | None,
) -> tuple[date, date, date]:
    dates = sorted(pd.to_datetime(dataset["trade_date"]).dt.date.unique())
    if len(dates) < 30:
        raise ValueError("Need at least 30 distinct trade dates for train/valid/test splits.")
    if train_end is None:
        train_end = dates[max(int(len(dates) * 0.60) - 1, 0)]
    if valid_end is None:
        valid_end = dates[max(int(len(dates) * 0.80) - 1, dates.index(train_end) + 1)]
    if test_end is None:
        test_end = dates[-1]
    if not train_end < valid_end <= test_end:
        raise ValueError("Split dates must satisfy train_end < valid_end <= test_end.")
    return train_end, valid_end, test_end


def split_dataset(dataset: pd.DataFrame, *, train_end: date, valid_end: date, test_end: date) -> dict[str, pd.DataFrame]:
    frame = dataset.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    train = frame[frame["trade_date"].dt.date <= train_end].reset_index(drop=True)
    valid = frame[(frame["trade_date"].dt.date > train_end) & (frame["trade_date"].dt.date <= valid_end)].reset_index(drop=True)
    test = frame[(frame["trade_date"].dt.date > valid_end) & (frame["trade_date"].dt.date <= test_end)].reset_index(drop=True)
    if train.empty or valid.empty or test.empty:
        raise ValueError("Time split produced an empty train, valid, or test partition.")
    return {"train": train, "valid": valid, "test": test}


def generate_walkforward_windows(
    trade_dates: pd.Series | list[date] | np.ndarray,
    *,
    windows: int,
    train_days: int,
    valid_days: int,
    test_days: int,
    min_train_days: int,
) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(pd.Series(trade_dates), errors="coerce").dropna().dt.date.unique())
    if windows < 1:
        raise ValueError("windows must be at least 1.")
    if min(train_days, valid_days, test_days, min_train_days) < 1:
        raise ValueError("train_days, valid_days, test_days, and min_train_days must be positive.")
    available_train_days = len(dates) - valid_days - test_days
    effective_train_days = min(train_days, available_train_days)
    if effective_train_days < min_train_days:
        return pd.DataFrame()

    total_days = effective_train_days + valid_days + test_days
    max_start = len(dates) - total_days
    if max_start < 0:
        return pd.DataFrame()
    if windows == 1:
        start_indices = [max_start]
    else:
        start_indices = sorted({int(round(value)) for value in np.linspace(0, max_start, num=windows)})

    rows: list[dict[str, object]] = []
    for window_id, start_index in enumerate(start_indices, start=1):
        train_start_index = start_index
        train_end_index = train_start_index + effective_train_days - 1
        valid_start_index = train_end_index + 1
        valid_end_index = valid_start_index + valid_days - 1
        test_start_index = valid_end_index + 1
        test_end_index = test_start_index + test_days - 1
        if test_end_index >= len(dates):
            continue
        rows.append(
            {
                "window_id": window_id,
                "train_start": dates[train_start_index],
                "train_end": dates[train_end_index],
                "valid_start": dates[valid_start_index],
                "valid_end": dates[valid_end_index],
                "test_start": dates[test_start_index],
                "test_end": dates[test_end_index],
                "train_days": effective_train_days,
                "valid_days": valid_days,
                "test_days": test_days,
            }
        )
    return pd.DataFrame(rows)


WALKFORWARD_TOP20_THRESHOLDS: dict[str, float] = {
    "mean_win_rate": 0.70,
    "worst_win_rate": 0.55,
    "mean_avg_return_20d": 0.05,
    "mean_stop_loss_rate_20d": 0.06,
    "mean_bad_risk_rate": 0.15,
    "mean_coverage": 0.20,
}


def summarize_walkforward_topn_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    numeric_metrics = [
        "test_days",
        "allowed_days",
        "coverage",
        "rows",
        "win_rate",
        "avg_return_20d",
        "median_return_20d",
        "take_profit_rate_20d",
        "stop_loss_rate_20d",
        "bad_risk_rate",
        "avg_take_profit_20d",
        "avg_stop_loss_20d",
        "avg_positive_return_20d",
        "avg_negative_return_20d",
    ]
    group_columns = ["model_version", "top_n"] if "model_version" in metrics.columns else ["top_n"]
    rows: list[dict[str, object]] = []
    for keys, group in metrics.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=False))
        row["windows"] = int(group["window_id"].nunique()) if "window_id" in group.columns else int(len(group))
        for column in numeric_metrics:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                row[f"{column}_mean"] = float("nan")
                row[f"{column}_std"] = float("nan")
                row[f"{column}_min"] = float("nan")
                row[f"{column}_max"] = float("nan")
                continue
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=0))
            row[f"{column}_min"] = float(values.min())
            row[f"{column}_max"] = float(values.max())

        if int(row.get("top_n", -1)) == 20:
            mean_win = float(row.get("win_rate_mean", float("nan")))
            min_win = float(row.get("win_rate_min", float("nan")))
            mean_return = float(row.get("avg_return_20d_mean", float("nan")))
            mean_stop_loss = float(row.get("stop_loss_rate_20d_mean", float("nan")))
            mean_bad_risk = float(row.get("bad_risk_rate_mean", float("nan")))
            mean_coverage = float(row.get("coverage_mean", float("nan")))
            row["pass_mean_win_rate"] = int(mean_win >= WALKFORWARD_TOP20_THRESHOLDS["mean_win_rate"])
            row["pass_worst_win_rate"] = int(min_win >= WALKFORWARD_TOP20_THRESHOLDS["worst_win_rate"])
            row["pass_mean_avg_return_20d"] = int(
                mean_return > WALKFORWARD_TOP20_THRESHOLDS["mean_avg_return_20d"]
            )
            row["pass_mean_stop_loss_rate_20d"] = int(
                mean_stop_loss <= WALKFORWARD_TOP20_THRESHOLDS["mean_stop_loss_rate_20d"]
            )
            row["pass_mean_bad_risk_rate"] = int(
                mean_bad_risk <= WALKFORWARD_TOP20_THRESHOLDS["mean_bad_risk_rate"]
            )
            row["pass_mean_coverage"] = int(mean_coverage >= WALKFORWARD_TOP20_THRESHOLDS["mean_coverage"])
            per_window_pass = (
                pd.to_numeric(group.get("win_rate"), errors="coerce").ge(WALKFORWARD_TOP20_THRESHOLDS["mean_win_rate"])
                & pd.to_numeric(group.get("avg_return_20d"), errors="coerce").gt(
                    WALKFORWARD_TOP20_THRESHOLDS["mean_avg_return_20d"]
                )
                & pd.to_numeric(group.get("stop_loss_rate_20d"), errors="coerce").le(
                    WALKFORWARD_TOP20_THRESHOLDS["mean_stop_loss_rate_20d"]
                )
                & pd.to_numeric(group.get("bad_risk_rate"), errors="coerce").le(
                    WALKFORWARD_TOP20_THRESHOLDS["mean_bad_risk_rate"]
                )
                & pd.to_numeric(group.get("coverage"), errors="coerce").ge(
                    WALKFORWARD_TOP20_THRESHOLDS["mean_coverage"]
                )
            )
            row["window_pass_rate"] = float(per_window_pass.mean()) if len(per_window_pass) else float("nan")
            pass_columns = [
                "pass_mean_win_rate",
                "pass_worst_win_rate",
                "pass_mean_avg_return_20d",
                "pass_mean_stop_loss_rate_20d",
                "pass_mean_bad_risk_rate",
                "pass_mean_coverage",
            ]
            row["pass_all_top20_thresholds"] = int(all(int(row[column]) == 1 for column in pass_columns))
        rows.append(row)
    return pd.DataFrame(rows)


def _empty_walkforward_topn_metrics(
    *,
    window: dict[str, object],
    top_n_list: tuple[int, ...],
    model_version: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for top_n in top_n_list:
        rows.append(
            {
                "dataset_split": "test",
                "top_n": top_n,
                "days": 0,
                "rows": 0,
                "win_rate": float("nan"),
                "avg_return_20d": float("nan"),
                "median_return_20d": float("nan"),
                "take_profit_rate_20d": float("nan"),
                "stop_loss_rate_20d": float("nan"),
                "bad_risk_rate": float("nan"),
                "avg_take_profit_20d": float("nan"),
                "avg_stop_loss_20d": float("nan"),
                "avg_positive_return_20d": float("nan"),
                "avg_negative_return_20d": float("nan"),
                "window_id": int(window["window_id"]),
                "model_version": model_version,
                "train_start": window["train_start"],
                "train_end": window["train_end"],
                "valid_start": window["valid_start"],
                "valid_end": window["valid_end"],
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "train_days": int(window["train_days"]),
                "valid_days": int(window["valid_days"]),
                "test_days": int(window["test_days"]),
                "allowed_days": 0,
                "coverage": 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_stage1_oof_predictions(
    frame: pd.DataFrame,
    *,
    feature_columns_by_horizon: dict[str, list[str]],
    max_iter: int,
    folds: int = 3,
) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(frame["trade_date"]).dt.date.unique())
    if len(dates) < 20:
        return pd.DataFrame()
    cut_indices = np.linspace(0.35, 0.90, folds)
    rows: list[pd.DataFrame] = []
    for ratio in cut_indices:
        train_cut_index = min(max(int(len(dates) * ratio), 1), len(dates) - 2)
        train_cut = dates[train_cut_index]
        next_index = min(max(int(len(dates) * (ratio + 0.15)), train_cut_index + 1), len(dates) - 1)
        predict_end = dates[next_index]
        train_frame = frame[pd.to_datetime(frame["trade_date"]).dt.date <= train_cut].copy()
        predict_frame = frame[
            (pd.to_datetime(frame["trade_date"]).dt.date > train_cut)
            & (pd.to_datetime(frame["trade_date"]).dt.date <= predict_end)
        ].copy()
        if train_frame.empty or predict_frame.empty:
            continue
        models = train_stage1_models(
            train_frame,
            feature_columns_by_horizon=feature_columns_by_horizon,
            max_iter=max_iter,
        )
        rows.append(attach_stage1_predictions(predict_frame, models=models, feature_columns_by_horizon=feature_columns_by_horizon))
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True, copy=False)
    result = result.drop_duplicates(subset=["trade_date", "symbol"], keep="last").reset_index(drop=True)
    return add_stage2_derived_features(result)


def build_stage1_oof_predictions_for_horizons(
    frame: pd.DataFrame,
    *,
    feature_columns_by_horizon: dict[str, list[str]],
    horizons: tuple[HorizonSpec, ...],
    max_iter: int,
    folds: int = 3,
    column_prefix: str = "",
) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(frame["trade_date"]).dt.date.unique())
    if len(dates) < 20:
        return pd.DataFrame()
    cut_indices = np.linspace(0.35, 0.90, folds)
    rows: list[pd.DataFrame] = []
    for ratio in cut_indices:
        train_cut_index = min(max(int(len(dates) * ratio), 1), len(dates) - 2)
        train_cut = dates[train_cut_index]
        next_index = min(max(int(len(dates) * (ratio + 0.15)), train_cut_index + 1), len(dates) - 1)
        predict_end = dates[next_index]
        train_frame = frame[pd.to_datetime(frame["trade_date"]).dt.date <= train_cut].copy()
        predict_frame = frame[
            (pd.to_datetime(frame["trade_date"]).dt.date > train_cut)
            & (pd.to_datetime(frame["trade_date"]).dt.date <= predict_end)
        ].copy()
        if train_frame.empty or predict_frame.empty:
            continue
        models = train_stage1_models_for_horizons(
            train_frame,
            feature_columns_by_horizon=feature_columns_by_horizon,
            horizons=horizons,
            max_iter=max_iter,
        )
        rows.append(
            attach_stage1_predictions_for_horizons(
                predict_frame,
                models=models,
                feature_columns_by_horizon=feature_columns_by_horizon,
                horizons=horizons,
                column_prefix=column_prefix,
            )
        )
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True, copy=False)
    result = result.drop_duplicates(subset=["trade_date", "symbol"], keep="last").reset_index(drop=True)
    if column_prefix:
        return add_long_upside_stage2_features(result)
    return add_stage2_derived_features(result)


def train_stage1_models(
    frame: pd.DataFrame,
    *,
    feature_columns_by_horizon: dict[str, list[str]],
    max_iter: int,
) -> dict[str, object]:
    return train_stage1_models_for_horizons(
        frame,
        feature_columns_by_horizon=feature_columns_by_horizon,
        horizons=HORIZONS,
        max_iter=max_iter,
    )


def train_stage1_models_for_horizons(
    frame: pd.DataFrame,
    *,
    feature_columns_by_horizon: dict[str, list[str]],
    horizons: tuple[HorizonSpec, ...],
    max_iter: int,
) -> dict[str, object]:
    models: dict[str, object] = {}
    for spec in horizons:
        label_column = f"outcome_class_{spec.name}"
        feature_columns = feature_columns_by_horizon[spec.name]
        train = frame.dropna(subset=[label_column]).copy()
        X = _model_matrix(train, feature_columns)
        y = pd.to_numeric(train[label_column], errors="coerce").dropna().astype(int)
        X = X.loc[y.index]
        models[spec.name] = fit_stage1_model(X, y, max_iter=max_iter)
    return models


def fit_stage1_model(X: pd.DataFrame, y: pd.Series, *, max_iter: int) -> object:
    unique = sorted(y.dropna().unique().tolist())
    if len(unique) < 2:
        return ConstantClassifier(unique[0] if unique else OUTCOME_TO_CLASS["neutral"])
    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=42,
    )
    model.fit(X, y)
    return model


def fit_stage2_model(frame: pd.DataFrame, *, feature_columns: list[str], max_iter: int) -> object:
    y = pd.to_numeric(frame["trade_value"], errors="coerce")
    valid = y.notna()
    X = _model_matrix(frame.loc[valid], feature_columns)
    y = y.loc[valid]
    if y.nunique(dropna=True) < 2:
        return ConstantRegressor(float(y.mean()) if len(y) else 0.0)
    model = HistGradientBoostingRegressor(
        max_iter=max_iter + 40,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=42,
    )
    model.fit(X, y)
    return model


def attach_stage1_predictions(
    frame: pd.DataFrame,
    *,
    models: dict[str, object],
    feature_columns_by_horizon: dict[str, list[str]],
) -> pd.DataFrame:
    result = attach_stage1_predictions_for_horizons(
        frame,
        models=models,
        feature_columns_by_horizon=feature_columns_by_horizon,
        horizons=HORIZONS,
    )
    return add_stage2_derived_features(result)


def attach_stage1_predictions_for_horizons(
    frame: pd.DataFrame,
    *,
    models: dict[str, object],
    feature_columns_by_horizon: dict[str, list[str]],
    horizons: tuple[HorizonSpec, ...],
    column_prefix: str = "",
) -> pd.DataFrame:
    result = frame.copy()
    for spec in horizons:
        model = models[spec.name]
        features = feature_columns_by_horizon[spec.name]
        X = _model_matrix(result, features)
        probabilities = _predict_stage1_probabilities(model, X)
        suffix = spec.name
        result[f"{column_prefix}up_prob_{suffix}"] = probabilities[:, OUTCOME_TO_CLASS["up"]]
        result[f"{column_prefix}down_prob_{suffix}"] = probabilities[:, OUTCOME_TO_CLASS["down"]]
        result[f"{column_prefix}neutral_prob_{suffix}"] = probabilities[:, OUTCOME_TO_CLASS["neutral"]]
        result[f"{column_prefix}expected_value_{suffix}"] = (
            result[f"{column_prefix}up_prob_{suffix}"] * 1.0
            - result[f"{column_prefix}down_prob_{suffix}"] * 1.2
        )
        result[f"{column_prefix}risk_adjusted_value_{suffix}"] = (
            result[f"{column_prefix}up_prob_{suffix}"] - 1.2 * result[f"{column_prefix}down_prob_{suffix}"]
        )
    return result


def add_long_upside_stage2_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    up20 = _numeric_feature(result, "long_up_prob_20d")
    down20 = _numeric_feature(result, "long_down_prob_20d")
    up60 = _numeric_feature(result, "long_up_prob_60d")
    down60 = _numeric_feature(result, "long_down_prob_60d")
    result["long_stage2_edge_20d"] = up20 - down20
    result["long_stage2_edge_60d"] = up60 - down60
    result["long_stage2_20d_up_without_60d_down"] = up20 * (1 - down60)
    result["long_stage2_60d_trend_confirmation"] = up60 * (1 - down20)
    result["long_stage2_weighted_expected_value"] = sum(
        LONG_UPSIDE_WEIGHTS[spec.name] * _numeric_feature(result, f"long_expected_value_{spec.name}")
        for spec in LONG_UPSIDE_HORIZONS
    )
    result["long_stage2_weighted_down_prob"] = sum(
        LONG_UPSIDE_WEIGHTS[spec.name] * _numeric_feature(result, f"long_down_prob_{spec.name}")
        for spec in LONG_UPSIDE_HORIZONS
    )
    result["long_stage2_weighted_up_prob"] = sum(
        LONG_UPSIDE_WEIGHTS[spec.name] * _numeric_feature(result, f"long_up_prob_{spec.name}")
        for spec in LONG_UPSIDE_HORIZONS
    )
    return result


def _predict_stage1_probabilities(model: object, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(X)
    result = np.zeros((len(X), len(STAGE1_CLASSES)), dtype=float)
    classes = getattr(model, "classes_", np.array(STAGE1_CLASSES))
    for source_position, class_label in enumerate(classes):
        if int(class_label) in STAGE1_CLASSES:
            result[:, int(class_label)] = raw[:, source_position]
    row_sum = result.sum(axis=1)
    missing = row_sum <= 0
    if missing.any():
        result[missing, OUTCOME_TO_CLASS["neutral"]] = 1.0
        row_sum = result.sum(axis=1)
    return result / row_sum.reshape(-1, 1)


def add_stage2_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["stage2_edge_20d"] = result["up_prob_20d"] - result["down_prob_20d"]
    result["stage2_edge_60d"] = result["up_prob_60d"] - result["down_prob_60d"]
    result["stage2_short_down_pressure"] = result["down_prob_5d"] + result["down_prob_10d"]
    result["stage2_20d_up_without_60d_down"] = result["up_prob_20d"] * (1 - result["down_prob_60d"])
    result["stage2_short_up_confirmation"] = result["up_prob_5d"] * result["up_prob_10d"]
    result["stage2_large_risk_suppressor"] = result["down_prob_60d"]
    result["stage2_weighted_expected_value"] = sum(
        spec.trade_value_weight * result[f"expected_value_{spec.name}"] for spec in HORIZONS
    )
    result["stage2_weighted_down_prob"] = sum(spec.trade_value_weight * result[f"down_prob_{spec.name}"] for spec in HORIZONS)
    return result


def realized_upside_value_series(frame: pd.DataFrame, spec: HorizonSpec) -> pd.Series:
    suffix = spec.name
    outcome = frame[f"outcome_{suffix}"].astype(str)
    period_return = pd.to_numeric(frame[f"period_return_{suffix}"], errors="coerce")
    max_upside = pd.to_numeric(frame[f"max_upside_{suffix}"], errors="coerce")
    max_drawdown = pd.to_numeric(frame[f"max_drawdown_{suffix}"], errors="coerce")
    hit_day = pd.to_numeric(frame[f"hit_day_{suffix}"], errors="coerce")

    up_hit_bonus = outcome.eq("up").astype(float)
    final_return_bonus = 0.30 * (period_return / spec.upside_target).clip(-1.0, 1.0)
    max_upside_bonus = 0.20 * (max_upside / spec.upside_target).clip(0.0, 1.5)
    speed_bonus = pd.Series(0.0, index=frame.index, dtype="float64")
    speed_bonus.loc[outcome.eq("up")] = 0.20 * (1 - hit_day.loc[outcome.eq("up")] / spec.horizon_days)
    drawdown_discount = 0.10 * (max_drawdown / spec.downside_threshold).clip(upper=1.5)
    value = up_hit_bonus + final_return_bonus + max_upside_bonus + speed_bonus - drawdown_discount
    value.loc[period_return.isna() | max_upside.isna() | max_drawdown.isna() | hit_day.isna()] = pd.NA
    return value


def add_v4_risk_upside_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    bad_risk = pd.Series(False, index=result.index)
    for spec in HORIZONS:
        bad_risk |= result[f"outcome_{spec.name}"].eq("down")
    bad_risk |= pd.to_numeric(result["max_drawdown_20d"], errors="coerce") > 0.10
    bad_risk |= pd.to_numeric(result["max_drawdown_60d"], errors="coerce") > 0.18
    result["bad_risk"] = bad_risk.astype(float)

    long_upside_value = pd.Series(0.0, index=result.index, dtype="float64")
    ready = pd.Series(True, index=result.index)
    for spec in LONG_UPSIDE_HORIZONS:
        value = realized_upside_value_series(result, spec)
        result[f"long_upside_component_{spec.name}"] = value
        long_upside_value = long_upside_value + LONG_UPSIDE_WEIGHTS[spec.name] * value.fillna(0.0)
        ready &= value.notna()
    result["long_upside_value"] = long_upside_value
    result.loc[~ready, "long_upside_value"] = pd.NA
    return result


def add_v41_long_quality_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    period20 = _numeric_feature(result, "period_return_20d")
    upside20 = _numeric_feature(result, "max_upside_20d")
    drawdown20 = _numeric_feature(result, "max_drawdown_20d")
    period60 = _numeric_feature(result, "period_return_60d")
    upside60 = _numeric_feature(result, "max_upside_60d")
    drawdown60 = _numeric_feature(result, "max_drawdown_60d")

    close_strength20 = _safe_ratio(period20, upside20).clip(-1.0, 1.0)
    capture60 = _safe_ratio(period60, upside60).clip(-1.0, 1.0)
    confirm20 = (period20 / 0.15).clip(-1.0, 1.0)
    trend60 = (
        0.45 * capture60.fillna(0.0)
        + 0.35 * (period60 / 0.30).clip(-1.0, 1.0).fillna(0.0)
        + 0.20 * confirm20.fillna(0.0)
    ).clip(-1.0, 1.0)

    result["long_quality_20d"] = (
        0.45 * (period20 / 0.15).clip(-1.0, 1.5)
        + 0.25 * (upside20 / 0.15).clip(0.0, 1.5)
        + 0.15 * close_strength20
        - 0.15 * (drawdown20 / 0.08).clip(0.0, 1.5)
    )
    result["long_quality_60d"] = (
        0.50 * (period60 / 0.30).clip(-1.0, 1.5)
        + 0.25 * (upside60 / 0.30).clip(0.0, 1.5)
        + 0.15 * trend60
        - 0.10 * (drawdown60 / 0.15).clip(0.0, 1.5)
    )
    valid = (
        period20.notna()
        & upside20.notna()
        & drawdown20.notna()
        & period60.notna()
        & upside60.notna()
        & drawdown60.notna()
    )
    result["long_quality"] = 0.40 * result["long_quality_20d"] + 0.60 * result["long_quality_60d"]
    result.loc[~valid, ["long_quality_20d", "long_quality_60d", "long_quality"]] = pd.NA

    result["long_quality_rank_pct"] = pd.NA
    result["long_quality_grade"] = pd.NA
    candidate_mask = result["action"].eq("candidate") if "action" in result.columns else pd.Series(True, index=result.index)
    for _, day_frame in result[candidate_mask].groupby("trade_date", sort=False):
        values = pd.to_numeric(day_frame["long_quality"], errors="coerce")
        valid_day = values.notna()
        if not valid_day.any():
            continue
        rank_pct = values.loc[valid_day].rank(method="first", pct=True, ascending=True)
        result.loc[rank_pct.index, "long_quality_rank_pct"] = rank_pct
        grade = pd.Series(2, index=rank_pct.index, dtype="int64")
        grade.loc[rank_pct <= 0.15] = 0
        grade.loc[(rank_pct > 0.15) & (rank_pct <= 0.40)] = 1
        grade.loc[(rank_pct > 0.75) & (rank_pct <= 0.90)] = 3
        grade.loc[rank_pct > 0.90] = 4
        result.loc[grade.index, "long_quality_grade"] = grade
    return result


def add_v5_volume_price_extreme_risk_flag(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    high_position = _numeric_feature(result, "vp_20d_range_position") >= 0.72
    high_volume = (
        (_numeric_feature(result, "vp_volume_ratio_1d_to_20d") >= 1.8)
        | (_numeric_feature(result, "vp_amount_ratio_1d_to_20d") >= 1.8)
    )
    long_upper_shadow = _numeric_feature(result, "vp_upper_shadow_1d") >= 0.40
    weak_close = _numeric_feature(result, "vp_close_position_1d") <= 0.45
    bearish_body = _numeric_feature(result, "vp_signed_body_1d") <= -0.25
    below_ma20 = _numeric_feature(result, "distance_to_ma20") < 0
    breakdown = (_numeric_feature(result, "vp_return_1d") <= -0.035) & high_volume & below_ma20
    repeated_weak_volume = _numeric_feature(result, "vp_5d_high_volume_weak_days") >= 2
    distribution = _numeric_feature(result, "vp_20d_distribution_score") >= 0.62
    failed_breakout = _numeric_feature(result, "vp_failed_breakout_1d_flag") >= 0.5

    flag = (
        (high_position & high_volume & long_upper_shadow & weak_close)
        | (high_position & high_volume & bearish_body)
        | breakdown
        | (high_position & repeated_weak_volume & distribution)
        | (high_position & failed_breakout)
    )
    result["volume_price_extreme_risk_flag"] = flag.fillna(False).astype(bool)

    reasons = pd.Series("none", index=result.index, dtype="object")
    reasons.loc[high_position & high_volume & long_upper_shadow & weak_close] = "high_volume_upper_shadow"
    reasons.loc[high_position & high_volume & bearish_body] = "high_volume_bearish"
    reasons.loc[breakdown] = "high_volume_breakdown"
    reasons.loc[high_position & repeated_weak_volume & distribution] = "repeated_high_volume_weak_close"
    reasons.loc[high_position & failed_breakout] = "high_volume_failed_breakout"
    result["volume_price_extreme_risk_reason"] = reasons
    return result


def add_v5_volume_price_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = add_v5_volume_price_extreme_risk_flag(frame)
    if "bad_risk" not in result.columns and "outcome_20d" in result.columns:
        result = add_v4_risk_upside_labels(result)

    period20 = _numeric_feature(result, "period_return_20d")
    upside20 = _numeric_feature(result, "max_upside_20d")
    drawdown20 = _numeric_feature(result, "max_drawdown_20d")
    period60 = _numeric_feature(result, "period_return_60d")
    drawdown60 = _numeric_feature(result, "max_drawdown_60d")
    bad_risk = _numeric_feature(result, "bad_risk")

    risk_label = (
        bad_risk.eq(1)
        | result.get("outcome_20d", pd.Series("", index=result.index)).astype(str).eq("down")
        | drawdown20.gt(0.10)
        | (period20.lt(-0.02) & drawdown20.gt(0.06))
        | (period60.lt(-0.04) & drawdown60.gt(0.12))
    )
    risk_ready = period20.notna() & drawdown20.notna()
    result["volume_price_risk_label"] = risk_label.astype(float)
    result.loc[~risk_ready, "volume_price_risk_label"] = pd.NA

    close_strength20 = _safe_ratio(period20, upside20).clip(-1.0, 1.0)
    quality = (
        0.50 * (period20 / 0.15).clip(-1.0, 1.5)
        + 0.20 * (upside20 / 0.15).clip(0.0, 1.5)
        + 0.15 * (period60 / 0.30).clip(-1.0, 1.2).fillna(0.0)
        + 0.10 * close_strength20.fillna(0.0)
        - 0.25 * (drawdown20 / 0.08).clip(0.0, 1.8)
        - 0.10 * (drawdown60 / 0.15).clip(0.0, 1.5).fillna(0.0)
        - 0.35 * result["volume_price_risk_label"].fillna(0.0)
    )
    result["volume_price_quality_value"] = quality
    result.loc[~risk_ready | upside20.isna(), "volume_price_quality_value"] = pd.NA
    result["volume_price_quality_rank_pct"] = pd.NA
    result["volume_price_quality_grade"] = pd.NA
    for _, day_frame in result.groupby("trade_date", sort=False):
        values = pd.to_numeric(day_frame["volume_price_quality_value"], errors="coerce")
        valid_day = values.notna()
        if not valid_day.any():
            continue
        rank_pct = values.loc[valid_day].rank(method="first", pct=True, ascending=True)
        result.loc[rank_pct.index, "volume_price_quality_rank_pct"] = rank_pct
        grade = pd.Series(2, index=rank_pct.index, dtype="int64")
        grade.loc[rank_pct <= 0.15] = 0
        grade.loc[(rank_pct > 0.15) & (rank_pct <= 0.40)] = 1
        grade.loc[(rank_pct > 0.75) & (rank_pct <= 0.90)] = 3
        grade.loc[rank_pct > 0.90] = 4
        result.loc[grade.index, "volume_price_quality_grade"] = grade
    return result


def volume_price_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "volume_price_extreme_risk_flag",
        "volume_price_risk_label",
        "volume_price_quality_value",
        "volume_price_quality_rank_pct",
        "volume_price_quality_grade",
    }
    columns = [
        column
        for column in frame.columns
        if column.startswith("vp_")
        and column not in excluded
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return columns


def fit_volume_price_risk_model(frame: pd.DataFrame, *, feature_columns: list[str], max_iter: int) -> object:
    y = pd.to_numeric(frame["volume_price_risk_label"], errors="coerce")
    valid = y.notna()
    X = _model_matrix(frame.loc[valid], feature_columns)
    y = y.loc[valid].astype(int)
    if y.nunique(dropna=True) < 2:
        return ConstantClassifier(int(y.iloc[0]) if len(y) else 0)
    positive_rate = float(y.mean())
    sample_weight = pd.Series(1.0, index=y.index, dtype="float64")
    if 0 < positive_rate < 1:
        sample_weight.loc[y.eq(1)] = min(4.0, 0.5 / positive_rate)
        sample_weight.loc[y.eq(0)] = min(4.0, 0.5 / (1.0 - positive_rate))
    model = HistGradientBoostingClassifier(
        max_iter=max(max_iter, 30),
        learning_rate=0.06,
        max_leaf_nodes=23,
        l2_regularization=0.05,
        random_state=51,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def fit_volume_price_quality_model(frame: pd.DataFrame, *, feature_columns: list[str], max_iter: int) -> object:
    y = pd.to_numeric(frame["volume_price_quality_value"], errors="coerce")
    valid = y.notna()
    X = _model_matrix(frame.loc[valid], feature_columns)
    y = y.loc[valid]
    if y.nunique(dropna=True) < 2:
        return ConstantRegressor(float(y.mean()) if len(y) else 0.0)
    model = HistGradientBoostingRegressor(
        max_iter=max(max_iter + 30, 40),
        learning_rate=0.05,
        max_leaf_nodes=23,
        l2_regularization=0.05,
        random_state=52,
    )
    model.fit(X, y)
    return model


def score_volume_price_submodels(
    frame: pd.DataFrame,
    *,
    risk_model: object,
    quality_model: object,
    feature_columns: list[str],
) -> pd.DataFrame:
    result = add_v5_volume_price_labels(frame)
    X = _model_matrix(result, feature_columns)
    result["volume_price_risk_score"] = _predict_positive_probability(risk_model, X)
    result["volume_price_quality_score"] = quality_model.predict(X)
    result["volume_price_quality_score_pct"] = _daily_score_percentile(
        result, "volume_price_quality_score", higher_is_better=True
    )
    result["volume_price_risk_score_pct"] = _daily_score_percentile(
        result, "volume_price_risk_score", higher_is_better=False
    )
    return result


def v5_fusion_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        "opportunity_score",
        "risk_score",
        "long_upside_score",
        "volume_price_risk_score",
        "volume_price_quality_score",
        "volume_price_quality_score_pct",
        "volume_price_risk_score_pct",
        "down_prob_20d",
        "down_prob_60d",
        "stage2_weighted_down_prob",
        "long_stage2_weighted_up_prob",
        "long_stage2_weighted_down_prob",
        "long_stage2_weighted_expected_value",
        "vp_close_position_1d",
        "vp_upper_shadow_1d",
        "vp_signed_body_1d",
        "vp_volume_ratio_1d_to_20d",
        "vp_amount_ratio_1d_to_20d",
        "vp_5d_price_volume_confirm",
        "vp_5d_volume_without_price",
        "vp_5d_shrink_pullback_score",
        "vp_5d_high_volume_weak_days",
        "vp_20d_accumulation_score",
        "vp_20d_distribution_score",
        "vp_20d_up_down_volume_ratio",
        "vp_5d_vs_20d_return_accel",
        "vp_5d_vs_20d_volume_accel",
        "vp_volume_accel_without_price",
        "vp_short_shrink_after_strength",
        "vp_pullback_depth_in_20d",
    ]
    return [column for column in columns if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column])]


def add_v5_fusion_target(frame: pd.DataFrame) -> pd.DataFrame:
    result = add_v5_volume_price_labels(frame)
    long_value = _numeric_feature(result, "long_upside_value")
    volume_quality = _numeric_feature(result, "volume_price_quality_value")
    bad_risk = _numeric_feature(result, "bad_risk")
    volume_risk = _numeric_feature(result, "volume_price_risk_label")
    result["v5_fusion_value"] = (
        0.55 * long_value.fillna(0.0)
        + 0.45 * volume_quality.fillna(0.0)
        - 0.80 * bad_risk.fillna(0.0)
        - 0.30 * volume_risk.fillna(0.0)
    )
    ready = long_value.notna() & volume_quality.notna()
    result.loc[~ready, "v5_fusion_value"] = pd.NA
    return result


def fit_v5_fusion_model(frame: pd.DataFrame, *, feature_columns: list[str], max_iter: int) -> object:
    y = pd.to_numeric(frame["v5_fusion_value"], errors="coerce")
    valid = y.notna()
    train = frame.loc[valid].copy()
    y = y.loc[valid]
    if train.empty or y.nunique(dropna=True) < 2:
        return ConstantRegressor(float(y.mean()) if len(y) else 0.0)
    X = _model_matrix(train, feature_columns)
    sample_weight = pd.Series(1.0, index=train.index, dtype="float64")
    if "action" in train.columns:
        sample_weight.loc[train["action"].eq("candidate")] = 1.4
    model = HistGradientBoostingRegressor(
        max_iter=max(max_iter + 40, 50),
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=53,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def score_v5_fusion_frame(frame: pd.DataFrame, *, fusion_model: object, fusion_features: list[str]) -> pd.DataFrame:
    result = add_v5_fusion_target(frame)
    X = _model_matrix(result, fusion_features)
    result["final_score_v5"] = fusion_model.predict(X)
    result["buy_score_v5"] = 100 * _daily_score_percentile(result, "final_score_v5", higher_is_better=True)
    result["model_version_v5"] = "v5_volume_price_fusion"
    return result


def apply_v5_decision(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "action" not in result.columns:
        result["action"] = "candidate"
    result["pre_v5_action"] = result["action"]
    extreme = result.get("volume_price_extreme_risk_flag", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    block = result["action"].eq("candidate") & extreme
    result.loc[block, "action"] = "avoid"
    result.loc[block, "final_action"] = "avoid"
    if "risk_gate_reason" not in result.columns:
        result["risk_gate_reason"] = "passed"
    reason = result.get("volume_price_extreme_risk_reason", pd.Series("extreme_volume_price_risk", index=result.index))
    result.loc[block, "risk_gate_reason"] = "volume_price_" + reason.loc[block].astype(str)
    result["action_rank_v5"] = np.select(
        [result["action"].eq("candidate"), result["action"].eq("no_trade")],
        [0, 1],
        default=2,
    )
    return result


def v51_candidate_mask(frame: pd.DataFrame) -> pd.Series:
    action = frame.get("action", pd.Series("candidate", index=frame.index)).astype(str)
    permission = frame.get("trade_permission", pd.Series("allow", index=frame.index)).astype(str)
    extreme = frame.get("volume_price_extreme_risk_flag", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    return action.eq("candidate") & permission.eq("allow") & ~extreme


def add_v51_candidate_rank_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "bad_risk" not in result.columns and "outcome_20d" in result.columns:
        result = add_v4_risk_upside_labels(result)
    eligible = v51_candidate_mask(result)

    period20 = _numeric_feature(result, "period_return_20d")
    period60 = _numeric_feature(result, "period_return_60d")
    drawdown20 = _numeric_feature(result, "max_drawdown_20d")
    bad_risk = _numeric_feature(result, "bad_risk")
    outcome20 = result.get("outcome_20d", pd.Series("", index=result.index)).astype(str)
    take_profit20 = outcome20.eq("up").astype(float)
    stop_loss20 = outcome20.eq("down").astype(float)

    value = (
        0.45 * (period20 / 0.15).clip(-1.0, 1.5)
        + 0.20 * take_profit20
        - 0.30 * stop_loss20
        - 0.25 * (drawdown20 / 0.08).clip(0.0, 2.0)
        + 0.15 * (period60 / 0.30).clip(-1.0, 1.2).fillna(0.0)
        - 0.35 * bad_risk.fillna(0.0)
    )
    ready = eligible & period20.notna() & drawdown20.notna() & bad_risk.notna()
    result["v51_candidate_eligible"] = eligible
    result["v51_rank_value"] = pd.NA
    result.loc[ready, "v51_rank_value"] = value.loc[ready]
    result["v51_rank_pct"] = pd.NA
    result["v51_rank_grade"] = pd.NA

    for _, day_frame in result.loc[ready].groupby("trade_date", sort=False):
        values = pd.to_numeric(day_frame["v51_rank_value"], errors="coerce")
        valid = values.notna()
        if not valid.any():
            continue
        rank_pct = values.loc[valid].rank(method="first", pct=True, ascending=True)
        result.loc[rank_pct.index, "v51_rank_pct"] = rank_pct
        grade = pd.Series(2, index=rank_pct.index, dtype="int64")
        grade.loc[rank_pct <= 0.15] = 0
        grade.loc[(rank_pct > 0.15) & (rank_pct <= 0.35)] = 1
        grade.loc[(rank_pct > 0.75) & (rank_pct <= 0.90)] = 3
        grade.loc[rank_pct > 0.90] = 4
        result.loc[grade.index, "v51_rank_grade"] = grade
    return result


def add_v51_cross_sectional_rank_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rank_columns = [
        "long_upside_score",
        "risk_score",
        "opportunity_score",
        "final_score_v42",
        "buy_score_v42",
        "down_prob_20d",
        "down_prob_60d",
        "stage2_weighted_down_prob",
        "volume_price_risk_score",
        "volume_price_quality_score",
        "volume_price_risk_score_pct",
        "volume_price_quality_score_pct",
        "final_score_v5",
        "buy_score_v5",
        "vp_close_position_1d",
        "vp_upper_shadow_1d",
        "vp_signed_body_1d",
        "vp_volume_ratio_1d_to_20d",
        "vp_amount_ratio_1d_to_20d",
        "vp_5d_price_volume_confirm",
        "vp_5d_volume_without_price",
        "vp_5d_shrink_pullback_score",
        "vp_20d_accumulation_score",
        "vp_20d_distribution_score",
        "vp_5d_vs_20d_return_accel",
        "vp_5d_vs_20d_volume_accel",
        "vp_volume_accel_without_price",
        "vp_short_shrink_after_strength",
    ]
    for column in rank_columns:
        if column not in result.columns:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        rank = pd.Series(np.nan, index=result.index, dtype="float64")
        for _, day_index in result.groupby("trade_date", sort=False).groups.items():
            day_values = values.loc[day_index]
            valid = day_values.notna()
            if valid.any():
                rank.loc[day_values.loc[valid].index] = day_values.loc[valid].rank(method="first", pct=True)
        result[f"v51_cs_rank_{column}"] = rank
    return result


def v51_candidate_ranker_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        "long_upside_score",
        "risk_score",
        "opportunity_score",
        "final_score_v42",
        "buy_score_v42",
        "down_prob_20d",
        "down_prob_60d",
        "stage2_weighted_down_prob",
        "long_stage2_weighted_up_prob",
        "long_stage2_weighted_down_prob",
        "long_stage2_weighted_expected_value",
        "volume_price_risk_score",
        "volume_price_quality_score",
        "volume_price_risk_score_pct",
        "volume_price_quality_score_pct",
        "final_score_v5",
        "buy_score_v5",
        "vp_close_position_1d",
        "vp_signed_body_1d",
        "vp_upper_shadow_1d",
        "vp_lower_shadow_1d",
        "vp_return_1d",
        "vp_volume_ratio_1d_to_20d",
        "vp_amount_ratio_1d_to_20d",
        "vp_high_volume_upper_shadow_flag",
        "vp_high_volume_bearish_flag",
        "vp_low_volume_stabilization_flag",
        "vp_failed_breakout_1d_flag",
        "vp_return_5d",
        "vp_volume_change_5d",
        "vp_amount_change_5d",
        "vp_5d_up_down_volume_ratio",
        "vp_5d_high_volume_weak_days",
        "vp_5d_upper_shadow_pressure",
        "vp_5d_lower_shadow_support",
        "vp_5d_volume_concentration",
        "vp_5d_price_volume_confirm",
        "vp_5d_volume_without_price",
        "vp_5d_shrink_pullback_score",
        "vp_return_20d",
        "vp_volume_change_20d",
        "vp_amount_change_20d",
        "vp_20d_range_position",
        "vp_20d_up_down_volume_ratio",
        "vp_20d_high_volume_weak_days",
        "vp_20d_high_volume_strong_days",
        "vp_20d_volume_trend",
        "vp_20d_accumulation_score",
        "vp_20d_distribution_score",
        "vp_5d_vs_20d_return_accel",
        "vp_5d_vs_20d_volume_accel",
        "vp_volume_accel_without_price",
        "vp_short_shrink_after_strength",
        "vp_pullback_depth_in_20d",
    ]
    columns.extend([column for column in frame.columns if column.startswith("v51_cs_rank_")])
    seen: set[str] = set()
    result: list[str] = []
    for column in columns:
        if column in frame.columns and column not in seen and pd.api.types.is_numeric_dtype(frame[column]):
            result.append(column)
            seen.add(column)
    return result


def fit_v51_candidate_ranker_model(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    max_iter: int,
) -> tuple[object, str]:
    train = add_v51_cross_sectional_rank_features(add_v51_candidate_rank_labels(frame))
    train = train[train["v51_candidate_eligible"].fillna(False).astype(bool)].copy()
    y_grade = pd.to_numeric(train["v51_rank_grade"], errors="coerce")
    valid = y_grade.notna()
    train = train.loc[valid].copy()
    y_grade = y_grade.loc[valid].astype(int)
    if train.empty or y_grade.nunique(dropna=True) < 2:
        fallback = pd.to_numeric(train.get("v51_rank_pct", pd.Series(dtype=float)), errors="coerce")
        return ConstantRegressor(float(fallback.mean()) if len(fallback.dropna()) else 0.5), "constant"

    train["trade_date"] = pd.to_datetime(train["trade_date"])
    train = train.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    y_grade = pd.to_numeric(train["v51_rank_grade"], errors="coerce").astype(int)
    X = _model_matrix(train, feature_columns)
    groups = train.groupby("trade_date", sort=False).size().astype(int).tolist()

    if LGBMRanker is not None and len(train) >= 50 and len(groups) >= 5:
        kwargs = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "n_estimators": max(max_iter + 120, 60),
            "learning_rate": 0.035,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "random_state": 61,
            "verbosity": -1,
            "label_gain": [0, 1, 3, 7, 15],
        }
        try:
            model = LGBMRanker(**kwargs)
        except TypeError:  # pragma: no cover - version compatibility guard.
            kwargs.pop("label_gain", None)
            model = LGBMRanker(**kwargs)
        model.fit(X, y_grade, group=groups)
        return model, "lightgbm_lambdarank"

    target = pd.to_numeric(train["v51_rank_pct"], errors="coerce").fillna(0.5)
    sample_weight = pd.Series(1.0, index=train.index, dtype="float64")
    sample_weight.loc[y_grade.isin([0, 4])] = 1.7
    sample_weight.loc[y_grade.eq(3)] = 1.3
    model = HistGradientBoostingRegressor(
        max_iter=max(max_iter + 70, 50),
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.04,
        random_state=62,
    )
    model.fit(X, target, sample_weight=sample_weight)
    return model, "hist_gradient_boosting_regressor"


def score_v51_candidate_ranker_frame(
    frame: pd.DataFrame,
    *,
    ranker_model: object,
    feature_columns: list[str],
) -> pd.DataFrame:
    result = add_v51_cross_sectional_rank_features(add_v51_candidate_rank_labels(frame))
    X = _model_matrix(result, feature_columns)
    result["candidate_rank_score_v51"] = ranker_model.predict(X)
    result["candidate_rank_score_pct_v51"] = _daily_score_percentile(
        result,
        "candidate_rank_score_v51",
        higher_is_better=True,
    )
    result["rank_source_v51"] = "candidate_ranker"
    result["model_version_v51"] = "v51_candidate_ranker"
    result["action_rank_v51"] = np.select(
        [result["action"].eq("candidate"), result["action"].eq("no_trade")],
        [0, 1],
        default=2,
    )
    return result


def apply_v51_blend_score(frame: pd.DataFrame, params: dict[str, object]) -> pd.DataFrame:
    result = frame.copy()
    weight = float(params.get("blend_weight", 1.0))
    v51_pct = pd.to_numeric(result.get("candidate_rank_score_pct_v51", pd.Series(0.5, index=result.index)), errors="coerce")
    if "opportunity_rank_score_pct" in result.columns:
        baseline_pct = pd.to_numeric(result["opportunity_rank_score_pct"], errors="coerce")
    elif "buy_score_v42" in result.columns:
        baseline_pct = pd.to_numeric(result["buy_score_v42"], errors="coerce") / 100
    else:
        baseline_pct = _daily_score_percentile(result, "final_score_v42", higher_is_better=True)
    result["v51_blend_weight"] = weight
    result["final_score_v51_raw"] = pd.to_numeric(result["candidate_rank_score_v51"], errors="coerce")
    result["final_score_v51"] = weight * v51_pct.fillna(0.5) + (1 - weight) * baseline_pct.fillna(0.5)
    result["buy_score_v51"] = 100 * _daily_score_percentile(result, "final_score_v51", higher_is_better=True)
    result["rank_source_v51"] = np.where(weight >= 0.999, "candidate_ranker", "candidate_ranker_v42_blend")
    return result


def select_v51_blend_params(
    scored_valid: pd.DataFrame,
    *,
    top_n_list: tuple[int, ...],
) -> tuple[dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    selection_top_n = 20 if 20 in set(top_n_list) else min(top_n_list)
    for weight in (0.50, 0.65, 0.80, 1.00):
        applied = apply_v51_blend_score(scored_valid, {"blend_weight": weight})
        top_rows = _topn_rows_by_score(
            applied,
            split_name="valid",
            top_n_list=top_n_list,
            score_column="final_score_v51",
            model_version="v51_candidate_ranker",
        )
        for row in top_rows:
            row["blend_weight"] = weight
            row["objective"] = (
                1.00 * row.get("avg_return_20d", 0.0)
                + 0.35 * row.get("win_rate", 0.0)
                + 0.25 * row.get("take_profit_rate_20d", 0.0)
                - 0.90 * row.get("stop_loss_rate_20d", 0.0)
                - 0.45 * row.get("bad_risk_rate", 0.0)
                - 0.50 * row.get("avg_max_drawdown_20d", 0.0)
            )
            rows.append(row)
    grid = pd.DataFrame(rows)
    if grid.empty:
        return {"blend_weight": 1.0, "selected_on": "fallback_ranker_only"}, grid
    selection = grid[grid["top_n"].eq(selection_top_n)].copy()
    selected = selection.sort_values(
        ["objective", "avg_return_20d", "win_rate", "stop_loss_rate_20d"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return {
        "blend_weight": float(selected["blend_weight"]),
        "selected_on": f"valid_top{selection_top_n}",
        "objective": float(selected["objective"]),
    }, grid


def build_v42_opportunity_frame(
    scored: pd.DataFrame,
    *,
    score_column: str = "long_upside_score",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if scored.empty or "trade_date" not in scored.columns:
        return pd.DataFrame()

    for trade_date, day_frame in scored.groupby("trade_date", sort=True):
        day = day_frame.copy()
        risk_candidates = day[day["action"].eq("candidate")].copy() if "action" in day.columns else day.copy()
        row: dict[str, object] = {
            "trade_date": pd.Timestamp(trade_date),
            "dataset_split": day["dataset_split"].iloc[0] if "dataset_split" in day.columns and not day.empty else "",
            "universe_rows": int(len(day)),
            "candidate_count": int(len(risk_candidates)),
            "candidate_fraction": float(len(risk_candidates) / len(day)) if len(day) else 0.0,
        }

        candidate_stat_columns = [
            "risk_score",
            "down_prob_20d",
            "down_prob_60d",
            "stage2_weighted_down_prob",
            score_column,
            "return_20d",
            "return_60d",
            "macd_hist",
            "distance_to_ma20",
            "distance_to_ma60",
        ]
        for column in candidate_stat_columns:
            _add_v42_distribution_stats(row, risk_candidates, column, prefix=f"cand_{column}")

        market_stat_columns = [
            "return_20d",
            "return_60d",
            "drawdown_20d",
            "drawdown_60d",
            "volatility_20d",
            "volatility_60d",
            "macd_hist",
            "distance_to_ma20",
            "distance_to_ma60",
        ]
        for column in market_stat_columns:
            _add_v42_distribution_stats(row, day, column, prefix=f"market_{column}", quantiles=False)

        row["market_share_above_ma20"] = _safe_share_positive(day, "distance_to_ma20")
        row["market_share_above_ma60"] = _safe_share_positive(day, "distance_to_ma60")
        row["market_share_positive_return_20d"] = _safe_share_positive(day, "return_20d")
        row["market_share_positive_return_60d"] = _safe_share_positive(day, "return_60d")
        row["market_share_positive_macd_hist"] = _safe_share_positive(day, "macd_hist")
        row["candidate_share_above_ma20"] = _safe_share_positive(risk_candidates, "distance_to_ma20")
        row["candidate_share_above_ma60"] = _safe_share_positive(risk_candidates, "distance_to_ma60")
        row["candidate_share_positive_return_20d"] = _safe_share_positive(risk_candidates, "return_20d")
        row["candidate_share_positive_return_60d"] = _safe_share_positive(risk_candidates, "return_60d")

        top20 = _v42_daily_outcome_summary(risk_candidates, top_n=20, score_column=score_column)
        top50 = _v42_daily_outcome_summary(risk_candidates, top_n=50, score_column=score_column)
        for key, value in top20.items():
            row[f"top20_{key}"] = value
        for key, value in top50.items():
            row[f"top50_{key}"] = value

        good = (
            row["candidate_count"] >= 20
            and _finite_or(row.get("top20_avg_return_20d"), -math.inf) > 0.0
            and _finite_or(row.get("top20_win_rate_20d"), -math.inf) >= 0.35
            and _finite_or(row.get("top20_stop_loss_rate_20d"), math.inf) <= 0.08
            and _finite_or(row.get("top50_avg_return_20d"), -math.inf) > -0.005
            and _finite_or(row.get("top50_stop_loss_rate_20d"), math.inf) <= 0.10
        )
        row["good_opportunity_day"] = int(good)
        row["opportunity_quality"] = (
            1.00 * _finite_or(row.get("top20_avg_return_20d"), 0.0)
            + 0.45 * _finite_or(row.get("top20_win_rate_20d"), 0.0)
            + 0.25 * _finite_or(row.get("top20_take_profit_rate_20d"), 0.0)
            - 0.85 * _finite_or(row.get("top20_stop_loss_rate_20d"), 0.0)
            - 0.70 * _finite_or(row.get("top20_avg_max_drawdown_20d"), 0.0)
            + 0.30 * _finite_or(row.get("top50_avg_return_20d"), 0.0)
            - 0.35 * _finite_or(row.get("top50_stop_loss_rate_20d"), 0.0)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _add_v42_distribution_stats(
    row: dict[str, object],
    frame: pd.DataFrame,
    column: str,
    *,
    prefix: str,
    quantiles: bool = True,
) -> None:
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna() if column in frame.columns else pd.Series(dtype=float)
    if values.empty:
        row[f"{prefix}_mean"] = float("nan")
        row[f"{prefix}_median"] = float("nan")
        row[f"{prefix}_std"] = float("nan")
        if quantiles:
            for label in ("p10", "p25", "p75", "p90"):
                row[f"{prefix}_{label}"] = float("nan")
            row[f"{prefix}_p90_minus_p10"] = float("nan")
            row[f"{prefix}_top_minus_median"] = float("nan")
        return
    row[f"{prefix}_mean"] = float(values.mean())
    row[f"{prefix}_median"] = float(values.median())
    row[f"{prefix}_std"] = float(values.std(ddof=0))
    if quantiles:
        q10 = float(values.quantile(0.10))
        q25 = float(values.quantile(0.25))
        q75 = float(values.quantile(0.75))
        q90 = float(values.quantile(0.90))
        row[f"{prefix}_p10"] = q10
        row[f"{prefix}_p25"] = q25
        row[f"{prefix}_p75"] = q75
        row[f"{prefix}_p90"] = q90
        row[f"{prefix}_p90_minus_p10"] = q90 - q10
        row[f"{prefix}_top_minus_median"] = float(values.max() - values.median())


def _safe_share_positive(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.notna()
    if not valid.any():
        return float("nan")
    return float((values.loc[valid] > 0).mean())


def _v42_daily_outcome_summary(candidates: pd.DataFrame, *, top_n: int, score_column: str) -> dict[str, object]:
    if candidates.empty or score_column not in candidates.columns:
        return {
            "selected_count": 0,
            "avg_return_20d": float("nan"),
            "win_rate_20d": float("nan"),
            "take_profit_rate_20d": float("nan"),
            "stop_loss_rate_20d": float("nan"),
            "avg_max_drawdown_20d": float("nan"),
            "avg_trade_value": float("nan"),
        }
    subset = candidates.sort_values(score_column, ascending=False).head(top_n).copy()
    period_return = pd.to_numeric(subset.get("period_return_20d", pd.Series(dtype=float)), errors="coerce")
    valid_return = period_return.notna()
    return {
        "selected_count": int(len(subset)),
        "avg_return_20d": _safe_mean(period_return),
        "win_rate_20d": float((period_return.loc[valid_return] > 0).mean()) if valid_return.any() else float("nan"),
        "take_profit_rate_20d": _outcome_rate(subset, "20d", "up"),
        "stop_loss_rate_20d": _outcome_rate(subset, "20d", "down"),
        "avg_max_drawdown_20d": _safe_mean(subset["max_drawdown_20d"]) if "max_drawdown_20d" in subset.columns else float("nan"),
        "avg_trade_value": _safe_mean(subset["trade_value"]) if "trade_value" in subset.columns else float("nan"),
    }


def _finite_or(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def opportunity_gate_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded_prefixes = ("top20_", "top50_")
    excluded = {
        "trade_date",
        "dataset_split",
        "good_opportunity_day",
        "opportunity_quality",
        "opportunity_score",
        "trade_permission",
    }
    columns: list[str] = []
    for column in frame.columns:
        if column in excluded or any(column.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def fit_v42_opportunity_gate_model(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    max_iter: int,
) -> object:
    y = pd.to_numeric(frame["good_opportunity_day"], errors="coerce")
    valid = y.notna()
    train = frame.loc[valid].copy()
    y = y.loc[valid].astype(int)
    if train.empty or y.nunique(dropna=True) < 2:
        return ConstantClassifier(int(y.iloc[0]) if len(y) else 0)
    X = _model_matrix(train, feature_columns)
    positive_rate = float((y == 1).mean())
    sample_weight = pd.Series(1.0, index=train.index, dtype="float64")
    if 0 < positive_rate < 1:
        sample_weight.loc[y.eq(1)] = min(5.0, 0.5 / positive_rate)
        sample_weight.loc[y.eq(0)] = min(5.0, 0.5 / (1.0 - positive_rate))
    model = HistGradientBoostingClassifier(
        max_iter=max(max_iter + 40, 40),
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        random_state=47,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def score_v42_opportunity_gate_frame(
    frame: pd.DataFrame,
    *,
    opportunity_model: object,
    opportunity_features: list[str],
) -> pd.DataFrame:
    result = frame.copy()
    X = _model_matrix(result, opportunity_features)
    result["opportunity_score"] = _predict_positive_probability(opportunity_model, X)
    return result


def fit_v42_conditional_ranker_model(
    frame: pd.DataFrame,
    opportunity_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    max_iter: int,
) -> tuple[object, str, str]:
    good_dates = set(
        pd.to_datetime(
            opportunity_frame.loc[pd.to_numeric(opportunity_frame["good_opportunity_day"], errors="coerce").eq(1), "trade_date"]
        )
        .dt.normalize()
        .tolist()
    )
    scoped = frame[pd.to_datetime(frame["trade_date"]).dt.normalize().isin(good_dates)].copy()
    candidate_scoped = scoped[scoped["action"].eq("candidate")] if "action" in scoped.columns else scoped
    enough_rows = len(candidate_scoped) >= 50
    enough_labels = (
        pd.to_numeric(candidate_scoped.get("long_quality_grade", pd.Series(dtype=float)), errors="coerce").nunique(dropna=True) >= 2
    )
    enough_days = pd.to_datetime(candidate_scoped["trade_date"]).nunique() >= 5 if not candidate_scoped.empty else False
    if not (enough_rows and enough_labels and enough_days):
        scoped = frame.copy()
        scope = "all_candidate_fallback"
    else:
        scope = "good_opportunity_days"
    model, engine = fit_long_quality_ranker_model(scoped, feature_columns=feature_columns, max_iter=max_iter)
    return model, engine, scope


def score_v42_ranker_frame(
    frame: pd.DataFrame,
    *,
    ranker_model: object,
    ranker_features: list[str],
) -> pd.DataFrame:
    result = add_v41_cross_sectional_rank_features(frame.copy())
    X = _model_matrix(result, ranker_features)
    result["opportunity_rank_score"] = ranker_model.predict(X)
    result["opportunity_rank_score_pct"] = _daily_score_percentile(result, "opportunity_rank_score", higher_is_better=True)
    result["final_score_v42"] = pd.to_numeric(result["opportunity_rank_score"], errors="coerce")
    result["buy_score_v42"] = 100 * result["opportunity_rank_score_pct"]
    return result


def score_v42_v4_rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    score = pd.to_numeric(result.get("long_upside_score", pd.Series(np.nan, index=result.index)), errors="coerce")
    result["opportunity_rank_score"] = score
    result["opportunity_rank_score_pct"] = _daily_score_percentile(result, "opportunity_rank_score", higher_is_better=True)
    result["final_score_v42"] = score
    result["buy_score_v42"] = 100 * result["opportunity_rank_score_pct"]
    result["rank_source_v42"] = "v4_long_upside_score"
    return result


def apply_v42_opportunity_decision(
    frame: pd.DataFrame,
    opportunity_frame: pd.DataFrame,
    params: dict[str, object],
) -> pd.DataFrame:
    result = frame.copy()
    threshold = float(params.get("opportunity_threshold", 0.0))
    daily = opportunity_frame.loc[:, [column for column in ("trade_date", "opportunity_score", "good_opportunity_day", "opportunity_quality") if column in opportunity_frame.columns]].copy()
    if daily.empty:
        result["opportunity_score"] = float("nan")
        result["trade_permission"] = "no_trade"
    else:
        daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.normalize()
        daily["trade_permission"] = np.where(
            pd.to_numeric(daily["opportunity_score"], errors="coerce") >= threshold,
            "allow",
            "no_trade",
        )
        result["_merge_trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
        result = result.merge(daily, left_on="_merge_trade_date", right_on="trade_date", how="left", suffixes=("", "_opportunity"))
        if "trade_date_opportunity" in result.columns:
            result = result.drop(columns=["trade_date_opportunity"])
        result = result.drop(columns=["_merge_trade_date"])
    result["opportunity_threshold"] = threshold
    result["trade_permission"] = result["trade_permission"].fillna("no_trade")
    result["risk_candidate_action"] = result["action"] if "action" in result.columns else "candidate"
    risk_pass = result["risk_candidate_action"].eq("candidate")
    allow = result["trade_permission"].eq("allow")
    result["action"] = np.where(risk_pass & allow, "candidate", np.where(risk_pass, "no_trade", "avoid"))
    result["final_action"] = result["action"]
    result["action_rank"] = np.select(
        [result["action"].eq("candidate"), result["action"].eq("no_trade")],
        [0, 1],
        default=2,
    )
    result["action_rank_v42"] = result["action_rank"]
    return result


def apply_v42_opportunity_threshold_to_daily(
    opportunity_frame: pd.DataFrame,
    params: dict[str, object],
) -> pd.DataFrame:
    result = opportunity_frame.copy()
    threshold = float(params.get("opportunity_threshold", 0.0))
    score = (
        pd.to_numeric(result["opportunity_score"], errors="coerce")
        if "opportunity_score" in result.columns
        else pd.Series(np.nan, index=result.index, dtype="float64")
    )
    result["opportunity_threshold"] = threshold
    result["trade_permission"] = np.where(score >= threshold, "allow", "no_trade")
    return result


def fit_risk_filter_model(frame: pd.DataFrame, *, feature_columns: list[str], max_iter: int) -> object:
    y = pd.to_numeric(frame["bad_risk"], errors="coerce")
    valid = y.notna()
    X = _model_matrix(frame.loc[valid], feature_columns)
    y = y.loc[valid].astype(int)
    if y.nunique(dropna=True) < 2:
        return ConstantClassifier(int(y.iloc[0]) if len(y) else 0)
    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=42,
    )
    model.fit(X, y)
    return model


def fit_long_upside_model(frame: pd.DataFrame, *, feature_columns: list[str], max_iter: int) -> object:
    y = pd.to_numeric(frame["long_upside_value"], errors="coerce")
    valid = y.notna()
    X = _model_matrix(frame.loc[valid], feature_columns)
    y = y.loc[valid]
    if y.nunique(dropna=True) < 2:
        return ConstantRegressor(float(y.mean()) if len(y) else 0.0)
    model = HistGradientBoostingRegressor(
        max_iter=max_iter + 40,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.03,
        random_state=45,
    )
    model.fit(X, y)
    return model


def fit_long_quality_ranker_model(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    max_iter: int,
) -> tuple[object, str]:
    train = frame[frame.get("action", pd.Series("candidate", index=frame.index)).eq("candidate")].copy()
    train = add_v41_cross_sectional_rank_features(train)
    y = pd.to_numeric(train["long_quality_grade"], errors="coerce")
    valid = y.notna()
    train = train.loc[valid].copy()
    y = y.loc[valid].astype(int)
    if train.empty or y.nunique(dropna=True) < 2:
        fallback_target = pd.to_numeric(train.get("long_quality_rank_pct", pd.Series(dtype=float)), errors="coerce")
        return ConstantRegressor(float(fallback_target.mean()) if len(fallback_target.dropna()) else 0.0), "constant"

    train["trade_date"] = pd.to_datetime(train["trade_date"])
    train = train.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    y = pd.to_numeric(train["long_quality_grade"], errors="coerce").astype(int)
    X = _model_matrix(train, feature_columns)
    groups = train.groupby("trade_date", sort=False).size().astype(int).tolist()

    if LGBMRanker is not None and len(train) >= 50 and len(groups) >= 5:
        kwargs = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "n_estimators": max(max_iter + 100, 40),
            "learning_rate": 0.04,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "random_state": 46,
            "verbosity": -1,
            "label_gain": [0, 1, 3, 7, 15],
        }
        try:
            model = LGBMRanker(**kwargs)
        except TypeError:  # pragma: no cover - version compatibility guard.
            kwargs.pop("label_gain", None)
            model = LGBMRanker(**kwargs)
        model.fit(X, y, group=groups)
        return model, "lightgbm_lambdarank"

    target = pd.to_numeric(train["long_quality_rank_pct"], errors="coerce").fillna(0.5)
    sample_weight = pd.Series(1.0, index=train.index, dtype="float64")
    sample_weight.loc[y.isin([0, 4])] = 1.6
    sample_weight.loc[y.eq(3)] = 1.3
    model = HistGradientBoostingRegressor(
        max_iter=max(max_iter + 60, 30),
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.03,
        random_state=46,
    )
    model.fit(X, target, sample_weight=sample_weight)
    return model, "hist_gradient_boosting_regressor"


def score_v4_risk_upside_full_frame(
    frame: pd.DataFrame,
    *,
    risk_stage1_models: dict[str, object],
    long_upside_stage1_models: dict[str, object],
    risk_model: object,
    long_upside_model: object,
    risk_feature_columns_by_horizon: dict[str, list[str]],
    long_upside_feature_columns_by_horizon: dict[str, list[str]],
    risk_features: list[str],
    long_upside_features: list[str],
) -> pd.DataFrame:
    with_risk_stage1 = attach_stage1_predictions(
        frame,
        models=risk_stage1_models,
        feature_columns_by_horizon=risk_feature_columns_by_horizon,
    )
    with_long_stage1 = attach_stage1_predictions_for_horizons(
        with_risk_stage1,
        models=long_upside_stage1_models,
        feature_columns_by_horizon=long_upside_feature_columns_by_horizon,
        horizons=LONG_UPSIDE_HORIZONS,
        column_prefix="long_",
    )
    with_labels = (
        add_v4_risk_upside_labels(with_long_stage1)
        if "long_upside_value" not in with_long_stage1.columns and "outcome_20d" in with_long_stage1.columns
        else with_long_stage1
    )
    return score_v4_risk_upside_raw_frame(
        with_labels,
        risk_model=risk_model,
        long_upside_model=long_upside_model,
        risk_features=risk_features,
        long_upside_features=long_upside_features,
    )


def score_v4_risk_upside_raw_frame(
    frame: pd.DataFrame,
    *,
    risk_model: object,
    long_upside_model: object,
    risk_features: list[str],
    long_upside_features: list[str],
) -> pd.DataFrame:
    result = add_stage2_derived_features(frame.copy())
    result = add_long_upside_stage2_features(result)
    risk_X = _model_matrix(result, risk_features)
    long_X = _model_matrix(result, long_upside_features)
    result["risk_score"] = _predict_positive_probability(risk_model, risk_X)
    result["long_upside_score"] = long_upside_model.predict(long_X)
    risk_columns = {spec.name: result[f"down_prob_{spec.name}"] for spec in HORIZONS}
    result["top_risk_horizon"] = pd.DataFrame(risk_columns, index=result.index).idxmax(axis=1)
    upside_columns = {spec.name: result[f"long_up_prob_{spec.name}"] for spec in LONG_UPSIDE_HORIZONS}
    result["top_upside_horizon"] = pd.DataFrame(upside_columns, index=result.index).idxmax(axis=1)
    return result


def score_v41_risk_filter_full_frame(
    frame: pd.DataFrame,
    *,
    stage1_models: dict[str, object],
    risk_model: object,
    feature_columns_by_horizon: dict[str, list[str]],
    risk_features: list[str],
) -> pd.DataFrame:
    with_stage1 = attach_stage1_predictions(frame, models=stage1_models, feature_columns_by_horizon=feature_columns_by_horizon)
    with_labels = (
        add_v4_risk_upside_labels(with_stage1)
        if "bad_risk" not in with_stage1.columns and "outcome_20d" in with_stage1.columns
        else with_stage1
    )
    return score_v41_risk_filter_raw_frame(with_labels, risk_model=risk_model, risk_features=risk_features)


def score_v41_risk_filter_raw_frame(
    frame: pd.DataFrame,
    *,
    risk_model: object,
    risk_features: list[str],
) -> pd.DataFrame:
    result = add_stage2_derived_features(frame.copy())
    X = _model_matrix(result, risk_features)
    result["risk_score"] = _predict_positive_probability(risk_model, X)
    risk_columns = {spec.name: result[f"down_prob_{spec.name}"] for spec in HORIZONS}
    result["top_risk_horizon"] = pd.DataFrame(risk_columns, index=result.index).idxmax(axis=1)
    upside_columns = {spec.name: result[f"up_prob_{spec.name}"] for spec in LONG_UPSIDE_HORIZONS}
    result["top_upside_horizon"] = pd.DataFrame(upside_columns, index=result.index).idxmax(axis=1)
    return result


def score_long_quality_ranker_frame(
    frame: pd.DataFrame,
    *,
    ranker_model: object,
    ranker_features: list[str],
) -> pd.DataFrame:
    result = add_v41_cross_sectional_rank_features(frame.copy())
    X = _model_matrix(result, ranker_features)
    result["long_quality_score"] = ranker_model.predict(X)
    result["long_quality_score_pct"] = _daily_score_percentile(result, "long_quality_score", higher_is_better=True)
    result["final_score_v41"] = pd.to_numeric(result["long_quality_score"], errors="coerce")
    result["buy_score_v41"] = 100 * result["long_quality_score_pct"]
    result["action_rank"] = np.where(result["action"].eq("candidate"), 0, 1)
    result["action_rank_v41"] = result["action_rank"]
    return result


def _predict_positive_probability(model: object, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = getattr(model, "classes_", np.array([0, 1]))
    for position, class_label in enumerate(classes):
        if int(class_label) == 1:
            return probabilities[:, position]
    return np.zeros(len(X), dtype=float)


def select_v41_risk_gate_params(
    scored_valid: pd.DataFrame, *, top_n_list: tuple[int, ...]
) -> tuple[dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    min_candidates = min(10, max(top_n_list)) if top_n_list else 10
    for risk_percentile in (0.30, 0.40, 0.50):
        for risk_cap in (0.36, 0.42, None):
            for down20_max in (0.32, 0.36):
                for down60_max in (0.24, 0.28):
                    for weighted_down_max in (0.30, 0.34):
                        params = {
                            "risk_percentile": risk_percentile,
                            "risk_score_max": risk_cap,
                            "down20_max": down20_max,
                            "down60_max": down60_max,
                            "weighted_down_max": weighted_down_max,
                            "min_candidates_per_day": min_candidates,
                        }
                        candidate_scored = apply_v41_risk_gate_decision(scored_valid, params)
                        candidates = candidate_scored[candidate_scored["action"].eq("candidate")]
                        row = _metric_row("valid", candidates, top_n=None)
                        metric_defaults = {
                            "avg_return_20d": 0.0,
                            "take_profit_rate_20d": 0.0,
                            "stop_loss_rate_20d": 1.0,
                            "avg_max_drawdown_20d": 1.0,
                            "bad_risk_rate": 1.0,
                        }
                        for metric_name, default_value in metric_defaults.items():
                            metric_value = row.get(metric_name, default_value)
                            row[metric_name] = default_value if pd.isna(metric_value) else float(metric_value)
                        row["candidate_rows"] = int(len(candidates))
                        row["candidate_days"] = int(candidates["trade_date"].nunique()) if "trade_date" in candidates.columns else 0
                        row["bad_risk_rate"] = _safe_mean(candidates["bad_risk"]) if "bad_risk" in candidates.columns else float("nan")
                        if pd.isna(row["bad_risk_rate"]):
                            row["bad_risk_rate"] = 1.0
                        row["avg_risk_score"] = _safe_mean(candidates["risk_score"]) if "risk_score" in candidates.columns else float("nan")
                        row["risk_percentile"] = risk_percentile
                        row["risk_score_max"] = risk_cap if risk_cap is not None else "none"
                        row["down20_max"] = down20_max
                        row["down60_max"] = down60_max
                        row["weighted_down_max"] = weighted_down_max
                        row["min_candidates_per_day"] = min_candidates
                        row["meets_stop_constraint"] = row.get("stop_loss_rate_20d", math.inf) <= 0.12
                        row["meets_bad_risk_constraint"] = row.get("bad_risk_rate", math.inf) <= 0.20
                        row["meets_drawdown_constraint"] = row.get("avg_max_drawdown_20d", math.inf) <= 0.04
                        row["meets_count_constraint"] = row.get("candidate_rows", 0) >= min_candidates * max(row.get("candidate_days", 0), 1)
                        row["constraints_satisfied"] = bool(
                            row["meets_stop_constraint"]
                            and row["meets_bad_risk_constraint"]
                            and row["meets_drawdown_constraint"]
                            and row["meets_count_constraint"]
                        )
                        row["objective"] = (
                            0.20 * row.get("avg_return_20d", 0.0)
                            + 0.15 * row.get("take_profit_rate_20d", 0.0)
                            - 0.80 * row.get("stop_loss_rate_20d", 0.0)
                            - 0.70 * row.get("bad_risk_rate", 0.0)
                            - 0.70 * row.get("avg_max_drawdown_20d", 0.0)
                        )
                        rows.append(row)

    grid = pd.DataFrame(rows)
    default_params: dict[str, object] = {
        "risk_percentile": 0.30,
        "risk_score_max": 0.36,
        "down20_max": 0.32,
        "down60_max": 0.24,
        "weighted_down_max": 0.30,
        "min_candidates_per_day": min_candidates,
        "constraints_satisfied": False,
    }
    if grid.empty:
        return default_params, grid

    constrained = grid[grid["constraints_satisfied"]].copy()
    if constrained.empty:
        grid["constraint_violation"] = (
            (grid["stop_loss_rate_20d"] - 0.12).clip(lower=0) * 3.0
            + (grid["bad_risk_rate"] - 0.20).clip(lower=0) * 2.0
            + (grid["avg_max_drawdown_20d"] - 0.04).clip(lower=0) * 4.0
        )
        selected = grid.sort_values(
            ["constraint_violation", "objective", "bad_risk_rate", "stop_loss_rate_20d"],
            ascending=[True, False, True, True],
        ).iloc[0]
        constraints_satisfied = False
    else:
        selected = constrained.sort_values(
            ["objective", "bad_risk_rate", "stop_loss_rate_20d", "avg_max_drawdown_20d"],
            ascending=[False, True, True, True],
        ).iloc[0]
        constraints_satisfied = True

    risk_score_max = selected["risk_score_max"]
    if isinstance(risk_score_max, str) and risk_score_max == "none":
        risk_score_max = None
    else:
        risk_score_max = float(risk_score_max)
    return {
        "risk_percentile": float(selected["risk_percentile"]),
        "risk_score_max": risk_score_max,
        "down20_max": float(selected["down20_max"]),
        "down60_max": float(selected["down60_max"]),
        "weighted_down_max": float(selected["weighted_down_max"]),
        "min_candidates_per_day": int(selected["min_candidates_per_day"]),
        "constraints_satisfied": constraints_satisfied,
        "selected_on": "valid_candidate_pool",
    }, grid


def apply_v41_risk_gate_decision(frame: pd.DataFrame, params: dict[str, object]) -> pd.DataFrame:
    result = frame.copy()
    result["action"] = "avoid"
    result["risk_action"] = "block"
    result["final_action"] = "avoid"
    result["risk_gate_reason"] = "not_evaluated"
    result["applied_risk_percentile"] = pd.NA

    base_percentile = float(params.get("risk_percentile", 0.30))
    percentile_grid = tuple(value for value in (0.30, 0.40, 0.50) if value >= base_percentile)
    if not percentile_grid:
        percentile_grid = (base_percentile,)
    min_candidates = int(params.get("min_candidates_per_day", 10))

    for _, day_frame in result.groupby("trade_date", sort=False):
        selected_mask = pd.Series(False, index=day_frame.index)
        applied_percentile = percentile_grid[-1]
        for percentile in percentile_grid:
            candidate_mask = _v41_risk_candidate_mask(day_frame, params, risk_percentile=percentile)
            selected_mask = candidate_mask
            applied_percentile = percentile
            if int(candidate_mask.sum()) >= min_candidates:
                break
        result.loc[day_frame.index, "action"] = np.where(selected_mask, "candidate", "avoid")
        result.loc[day_frame.index, "risk_action"] = np.where(selected_mask, "pass", "block")
        result.loc[day_frame.index, "final_action"] = np.where(selected_mask, "candidate", "avoid")
        result.loc[day_frame.index, "risk_gate_reason"] = _v41_risk_gate_reasons(
            day_frame,
            selected_mask,
            params,
            risk_percentile=applied_percentile,
        )
        result.loc[day_frame.index, "applied_risk_percentile"] = applied_percentile

    risk_score = pd.to_numeric(result["risk_score"], errors="coerce")
    result["risk_tier"] = np.select(
        [risk_score <= 0.35, risk_score <= 0.55],
        ["low", "medium"],
        default="high",
    )
    result["action_rank"] = np.where(result["action"].eq("candidate"), 0, 1)
    return result


def _v41_risk_candidate_mask(frame: pd.DataFrame, params: dict[str, object], *, risk_percentile: float) -> pd.Series:
    risk_score = pd.to_numeric(frame["risk_score"], errors="coerce")
    down20 = pd.to_numeric(frame["down_prob_20d"], errors="coerce")
    down60 = pd.to_numeric(frame["down_prob_60d"], errors="coerce")
    weighted_down = pd.to_numeric(frame["stage2_weighted_down_prob"], errors="coerce")
    risk_rank = risk_score.rank(method="first", pct=True)
    mask = risk_rank <= risk_percentile
    risk_score_max = params.get("risk_score_max")
    if risk_score_max is not None:
        mask &= risk_score <= float(risk_score_max)
    down20_max = float(params.get("down20_max", 0.32))
    down60_max = float(params.get("down60_max", 0.24))
    weighted_down_max = float(params.get("weighted_down_max", 0.30))
    mask &= down20 <= down20_max
    mask &= down60 <= down60_max
    mask &= weighted_down <= weighted_down_max
    double_high = (down20 >= min(down20_max, 0.32)) & (down60 >= min(down60_max, 0.24))
    mask &= ~double_high
    mask &= risk_score.notna() & down20.notna() & down60.notna() & weighted_down.notna()
    return mask.fillna(False)


def _v41_risk_gate_reasons(
    frame: pd.DataFrame, candidate_mask: pd.Series, params: dict[str, object], *, risk_percentile: float
) -> pd.Series:
    risk_score = pd.to_numeric(frame["risk_score"], errors="coerce")
    down20 = pd.to_numeric(frame["down_prob_20d"], errors="coerce")
    down60 = pd.to_numeric(frame["down_prob_60d"], errors="coerce")
    weighted_down = pd.to_numeric(frame["stage2_weighted_down_prob"], errors="coerce")
    risk_rank = risk_score.rank(method="first", pct=True)
    down20_max = float(params.get("down20_max", 0.32))
    down60_max = float(params.get("down60_max", 0.24))
    weighted_down_max = float(params.get("weighted_down_max", 0.30))
    risk_score_max = params.get("risk_score_max")

    reasons = pd.Series("passed", index=frame.index, dtype="object")
    missing = risk_score.isna() | down20.isna() | down60.isna() | weighted_down.isna()
    reasons.loc[missing] = "missing_score"
    if risk_score_max is not None:
        reasons.loc[~missing & (risk_score > float(risk_score_max))] = "risk_score_cap"
    reasons.loc[~missing & (risk_rank > risk_percentile)] = "risk_percentile"
    reasons.loc[~missing & (down20 > down20_max)] = "down20_cap"
    reasons.loc[~missing & (down60 > down60_max)] = "down60_cap"
    reasons.loc[~missing & (weighted_down > weighted_down_max)] = "weighted_down_cap"
    double_high = (down20 >= min(down20_max, 0.32)) & (down60 >= min(down60_max, 0.24))
    reasons.loc[~missing & double_high] = "double_down_risk"
    reasons.loc[candidate_mask] = "passed"
    return reasons


def _numeric_feature(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _linear_score(series: pd.Series, low: float, high: float) -> pd.Series:
    if math.isclose(high, low):
        return pd.Series(0.0, index=series.index, dtype="float64")
    return ((series - low) / (high - low)).clip(0.0, 1.0).fillna(0.0)


def _triangle_score(series: pd.Series, low: float, peak: float, high: float) -> pd.Series:
    left = _linear_score(series, low, peak)
    right = _linear_score(high - series, 0.0, high - peak)
    return pd.concat([left, right], axis=1).min(axis=1).clip(0.0, 1.0).fillna(0.0)


def _daily_score_percentile(frame: pd.DataFrame, column: str, *, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(np.nan, index=frame.index)
    result = pd.Series(0.5, index=frame.index, dtype="float64")
    for _, day_index in frame.groupby("trade_date", sort=False).groups.items():
        day_values = values.loc[day_index]
        valid = day_values.notna()
        if not valid.any():
            continue
        ranked = day_values.loc[valid].rank(method="first", pct=True, ascending=True)
        if not higher_is_better:
            ranked = 1.0 - ranked + (1.0 / len(ranked))
        result.loc[ranked.index] = ranked
    return result.clip(0.0, 1.0)


def _topn_rows_by_score(
    scored: pd.DataFrame,
    *,
    split_name: str,
    top_n_list: tuple[int, ...],
    score_column: str,
    model_version: str | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for top_n in top_n_list:
        daily_rows: list[dict[str, object]] = []
        for _, day_frame in scored.groupby("trade_date", sort=True):
            candidates = day_frame[day_frame["action"].eq("candidate")].sort_values(score_column, ascending=False)
            subset = candidates.head(top_n)
            if subset.empty:
                continue
            metric_subset = subset.copy()
            metric_subset["__selection_score"] = metric_subset[score_column]
            row = _metric_row(split_name, metric_subset, top_n=top_n)
            row["candidate_count"] = int(len(candidates))
            row["selected_count"] = int(len(subset))
            row["bad_risk_rate"] = _safe_mean(subset["bad_risk"]) if "bad_risk" in subset.columns else float("nan")
            row["avg_risk_score"] = _safe_mean(subset["risk_score"]) if "risk_score" in subset.columns else float("nan")
            row["avg_alpha_grade"] = _safe_mean(subset["alpha_grade"]) if "alpha_grade" in subset.columns else float("nan")
            row["avg_alpha_rank_score"] = (
                _safe_mean(subset["alpha_rank_score"]) if "alpha_rank_score" in subset.columns else float("nan")
            )
            row["avg_long_upside_score"] = (
                _safe_mean(subset["long_upside_score"]) if "long_upside_score" in subset.columns else float("nan")
            )
            row["avg_long_quality"] = _safe_mean(subset["long_quality"]) if "long_quality" in subset.columns else float("nan")
            row["avg_long_quality_grade"] = (
                _safe_mean(subset["long_quality_grade"]) if "long_quality_grade" in subset.columns else float("nan")
            )
            row["avg_long_quality_score"] = (
                _safe_mean(subset["long_quality_score"]) if "long_quality_score" in subset.columns else float("nan")
            )
            row["avg_opportunity_score"] = (
                _safe_mean(subset["opportunity_score"]) if "opportunity_score" in subset.columns else float("nan")
            )
            row["avg_opportunity_rank_score"] = (
                _safe_mean(subset["opportunity_rank_score"]) if "opportunity_rank_score" in subset.columns else float("nan")
            )
            if model_version is not None:
                row["model_version"] = model_version
            daily_rows.append(row)
        if not daily_rows:
            continue
        daily = pd.DataFrame(daily_rows)
        row = {"dataset_split": split_name, "top_n": top_n, "days": len(daily)}
        if model_version is not None:
            row["model_version"] = model_version
        for column in daily.columns:
            if column not in {"dataset_split", "top_n", "model_version"}:
                row[column] = pd.to_numeric(daily[column], errors="coerce").mean()
        rows.append(row)
    return rows


def evaluate_risk_filter_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, group in scored.groupby("dataset_split", sort=False):
        y = pd.to_numeric(group["bad_risk"], errors="coerce")
        score = pd.to_numeric(group["risk_score"], errors="coerce")
        valid = y.notna() & score.notna()
        if not valid.any():
            continue
        yv = y[valid].astype(int)
        sv = score[valid]
        pred = (sv >= 0.5).astype(int)
        row = {
            "dataset_split": split_name,
            "rows": int(valid.sum()),
            "bad_risk_rate": float(yv.mean()),
            "risk_accuracy_0p5": float(accuracy_score(yv, pred)),
            "avg_risk_score": float(sv.mean()),
        }
        try:
            row["risk_auc"] = float(roc_auc_score(yv, sv))
        except ValueError:
            row["risk_auc"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_long_upside_model_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, group in scored.groupby("dataset_split", sort=False):
        for group_name, subset in [
            ("all", group),
            ("realized_risk_ok", group[pd.to_numeric(group["bad_risk"], errors="coerce").eq(0)]),
            ("candidate", group[group["action"].eq("candidate")]),
        ]:
            y = pd.to_numeric(subset["long_upside_value"], errors="coerce")
            pred = pd.to_numeric(subset["long_upside_score"], errors="coerce")
            valid = y.notna() & pred.notna()
            if not valid.any():
                continue
            yv = y[valid]
            pv = pred[valid]
            rows.append(
                {
                    "dataset_split": split_name,
                    "action_group": group_name,
                    "rows": int(valid.sum()),
                    "avg_long_upside_value": float(yv.mean()),
                    "avg_long_upside_score": float(pv.mean()),
                    "mae": float(mean_absolute_error(yv, pv)),
                    "rmse": float(math.sqrt(mean_squared_error(yv, pv))),
                    "pearson_corr": float(yv.corr(pv)) if len(yv) > 1 else float("nan"),
                    "spearman_corr": float(yv.corr(pv, method="spearman")) if len(yv) > 1 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def evaluate_long_quality_ranker_metrics(scored: pd.DataFrame, *, top_n_list: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eval_at = tuple(sorted(set(top_n_list)))
    for split_name, split_frame in scored.groupby("dataset_split", sort=False):
        candidates = split_frame[split_frame["action"].eq("candidate")].copy()
        if candidates.empty:
            continue
        y = pd.to_numeric(candidates["long_quality"], errors="coerce")
        pred = pd.to_numeric(candidates["long_quality_score"], errors="coerce")
        valid = y.notna() & pred.notna()
        base_row: dict[str, object] = {
            "dataset_split": split_name,
            "action_group": "candidate",
            "rows": int(valid.sum()),
            "avg_long_quality": float(y[valid].mean()) if valid.any() else float("nan"),
            "avg_long_quality_score": float(pred[valid].mean()) if valid.any() else float("nan"),
            "pearson_corr": float(y[valid].corr(pred[valid])) if valid.sum() > 1 else float("nan"),
            "spearman_corr": float(y[valid].corr(pred[valid], method="spearman")) if valid.sum() > 1 else float("nan"),
        }
        for top_n in eval_at:
            scores: list[float] = []
            for _, day_frame in candidates.groupby("trade_date", sort=False):
                label = pd.to_numeric(day_frame["long_quality_grade"], errors="coerce")
                score = pd.to_numeric(day_frame["long_quality_score"], errors="coerce")
                day_valid = label.notna() & score.notna()
                if day_valid.sum() < 2:
                    continue
                try:
                    scores.append(
                        float(
                            ndcg_score(
                                [label[day_valid].to_numpy(dtype=float)],
                                [score[day_valid].to_numpy(dtype=float)],
                                k=min(top_n, int(day_valid.sum())),
                            )
                        )
                    )
                except ValueError:
                    continue
            base_row[f"ndcg_at_{top_n}"] = float(np.mean(scores)) if scores else float("nan")
        rows.append(base_row)
    return pd.DataFrame(rows)


def select_v42_opportunity_threshold(
    scored_valid: pd.DataFrame,
    opportunity_valid: pd.DataFrame,
    *,
    top_n_list: tuple[int, ...],
) -> tuple[dict[str, object], pd.DataFrame]:
    if opportunity_valid.empty or "opportunity_score" not in opportunity_valid.columns:
        return {"opportunity_threshold": 0.0, "selected_on": "fallback_allow_all", "coverage_days": 1.0}, pd.DataFrame()

    scores = pd.to_numeric(opportunity_valid["opportunity_score"], errors="coerce").dropna()
    if scores.empty:
        return {"opportunity_threshold": 0.0, "selected_on": "fallback_allow_all", "coverage_days": 1.0}, pd.DataFrame()

    threshold_values = {0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80}
    for quantile in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        threshold_values.add(float(scores.quantile(quantile)))
    thresholds = sorted(value for value in threshold_values if math.isfinite(value))
    rows: list[dict[str, object]] = []
    evaluable_days = int(opportunity_valid["trade_date"].nunique())
    selection_top_n = 20 if 20 in set(top_n_list) else min(top_n_list)

    for threshold in thresholds:
        params = {"opportunity_threshold": float(threshold)}
        applied = apply_v42_opportunity_decision(scored_valid, opportunity_valid, params)
        allowed_days = int(
            pd.to_datetime(
                applied.loc[applied["trade_permission"].eq("allow"), "trade_date"],
                errors="coerce",
            ).nunique()
        )
        coverage = float(allowed_days / evaluable_days) if evaluable_days else 0.0
        topn_rows = _topn_rows_by_score(
            applied,
            split_name="valid",
            top_n_list=top_n_list,
            score_column="final_score_v42",
            model_version=None,
        )
        if not topn_rows:
            rows.append(
                {
                    "dataset_split": "valid",
                    "top_n": selection_top_n,
                    "opportunity_threshold": float(threshold),
                    "coverage_days": coverage,
                    "allowed_days": allowed_days,
                    "avg_return_20d": 0.0,
                    "win_rate": 0.0,
                    "take_profit_rate_20d": 0.0,
                    "stop_loss_rate_20d": 1.0,
                    "avg_max_drawdown_20d": 1.0,
                    "objective": -999.0,
                    "constraints_satisfied": False,
                }
            )
            continue
        for row in topn_rows:
            row["opportunity_threshold"] = float(threshold)
            row["coverage_days"] = coverage
            row["allowed_days"] = allowed_days
            row["meets_coverage_constraint"] = coverage >= 0.30
            row["meets_stop_constraint"] = row.get("stop_loss_rate_20d", math.inf) <= 0.07
            row["objective"] = (
                1.00 * row.get("avg_return_20d", 0.0)
                + 0.35 * row.get("win_rate", 0.0)
                + 0.20 * row.get("take_profit_rate_20d", 0.0)
                - 0.85 * row.get("stop_loss_rate_20d", 0.0)
                - 0.55 * row.get("avg_max_drawdown_20d", 0.0)
            )
            row["constraints_satisfied"] = bool(row["meets_coverage_constraint"] and row["meets_stop_constraint"])
            rows.append(row)

    grid = pd.DataFrame(rows)
    if grid.empty:
        return {"opportunity_threshold": 0.0, "selected_on": "fallback_allow_all", "coverage_days": 1.0}, grid

    selection = grid[grid["top_n"].eq(selection_top_n)].copy()
    constrained = selection[selection.get("constraints_satisfied", False).eq(True)].copy()
    if constrained.empty:
        selection["coverage_penalty"] = (0.30 - selection["coverage_days"]).clip(lower=0) * 0.50
        selection["stop_penalty"] = (selection["stop_loss_rate_20d"] - 0.07).clip(lower=0) * 1.00 if "stop_loss_rate_20d" in selection.columns else 0.0
        selection["selection_score"] = selection["objective"] - selection["coverage_penalty"] - selection["stop_penalty"]
        selected = selection.sort_values(
            ["selection_score", "coverage_days", "avg_return_20d"],
            ascending=[False, False, False],
        ).iloc[0]
        selected_on = f"valid_top{selection_top_n}_unconstrained"
    else:
        selected = constrained.sort_values(
            ["objective", "avg_return_20d", "win_rate", "stop_loss_rate_20d"],
            ascending=[False, False, False, True],
        ).iloc[0]
        selected_on = f"valid_top{selection_top_n}"

    return {
        "opportunity_threshold": float(selected["opportunity_threshold"]),
        "selected_on": selected_on,
        "coverage_days": float(selected.get("coverage_days", float("nan"))),
        "allowed_days": int(selected.get("allowed_days", 0)),
    }, grid


def evaluate_v42_split_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, split_frame in scored.groupby("dataset_split", sort=False):
        for group_name, group in [
            ("all", split_frame),
            ("candidate", split_frame[split_frame["action"].eq("candidate")]),
            ("no_trade", split_frame[split_frame["action"].eq("no_trade")]),
            ("avoid", split_frame[split_frame["action"].eq("avoid")]),
        ]:
            row = _metric_row(split_name, group, top_n=None)
            row["action_group"] = group_name
            row["bad_risk_rate"] = _safe_mean(group["bad_risk"]) if "bad_risk" in group.columns else float("nan")
            row["avg_risk_score"] = _safe_mean(group["risk_score"]) if "risk_score" in group.columns else float("nan")
            row["avg_opportunity_score"] = (
                _safe_mean(group["opportunity_score"]) if "opportunity_score" in group.columns else float("nan")
            )
            row["avg_long_quality"] = _safe_mean(group["long_quality"]) if "long_quality" in group.columns else float("nan")
            row["avg_opportunity_rank_score"] = (
                _safe_mean(group["opportunity_rank_score"]) if "opportunity_rank_score" in group.columns else float("nan")
            )
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_v42_topn_metrics(scored: pd.DataFrame, *, top_n_list: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, split_frame in scored.groupby("dataset_split", sort=False):
        rows.extend(
            _topn_rows_by_score(
                split_frame,
                split_name=split_name,
                top_n_list=top_n_list,
                score_column="final_score_v42",
                model_version=None,
            )
        )
    return pd.DataFrame(rows)


def evaluate_v42_opportunity_metrics(opportunity_scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if opportunity_scored.empty:
        return pd.DataFrame()
    for split_name, group in opportunity_scored.groupby("dataset_split", sort=False):
        y = pd.to_numeric(group.get("good_opportunity_day", pd.Series(dtype=float)), errors="coerce")
        score = pd.to_numeric(group.get("opportunity_score", pd.Series(dtype=float)), errors="coerce")
        permission = group.get("trade_permission", pd.Series("no_trade", index=group.index)).astype(str)
        valid = y.notna()
        allowed = permission.eq("allow")
        row: dict[str, object] = {
            "dataset_split": split_name,
            "days": int(len(group)),
            "allowed_days": int(allowed.sum()),
            "coverage_days": float(allowed.mean()) if len(group) else float("nan"),
            "good_opportunity_rate": float(y.loc[valid].mean()) if valid.any() else float("nan"),
            "avg_opportunity_score": _safe_mean(score),
            "avg_quality_allowed": _safe_mean(group.loc[allowed, "opportunity_quality"]) if "opportunity_quality" in group.columns else float("nan"),
            "avg_quality_blocked": _safe_mean(group.loc[~allowed, "opportunity_quality"]) if "opportunity_quality" in group.columns else float("nan"),
            "top20_return_allowed": _safe_mean(group.loc[allowed, "top20_avg_return_20d"]) if "top20_avg_return_20d" in group.columns else float("nan"),
            "top20_return_blocked": _safe_mean(group.loc[~allowed, "top20_avg_return_20d"]) if "top20_avg_return_20d" in group.columns else float("nan"),
        }
        if valid.any():
            predicted_good = allowed.loc[valid]
            actual_good = y.loc[valid].eq(1)
            true_positive = int((predicted_good & actual_good).sum())
            predicted_positive = int(predicted_good.sum())
            actual_positive = int(actual_good.sum())
            row["good_day_precision"] = float(true_positive / predicted_positive) if predicted_positive else float("nan")
            row["good_day_recall"] = float(true_positive / actual_positive) if actual_positive else float("nan")
            if score.loc[valid].nunique(dropna=True) > 1 and y.loc[valid].nunique(dropna=True) > 1:
                row["opportunity_auc"] = float(roc_auc_score(y.loc[valid].astype(int), score.loc[valid]))
            else:
                row["opportunity_auc"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_v42_ranker_metrics(scored: pd.DataFrame, *, top_n_list: tuple[int, ...]) -> pd.DataFrame:
    renamed = scored.copy()
    if "opportunity_rank_score" in renamed.columns:
        renamed["long_quality_score"] = renamed["opportunity_rank_score"]
    return evaluate_long_quality_ranker_metrics(renamed, top_n_list=top_n_list)


def compare_v42_topn_metrics(
    *,
    baseline_scored: pd.DataFrame,
    v42_scored: pd.DataFrame,
    hybrid_v4_scored: pd.DataFrame | None = None,
    top_n_list: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    baseline = baseline_scored.copy()
    baseline["final_score_v42"] = pd.to_numeric(baseline["long_upside_score"], errors="coerce")
    baseline["action"] = np.where(baseline.get("risk_action", baseline["action"]).eq("pass"), "candidate", baseline["action"])
    baseline_rows: list[dict[str, object]] = []
    for split_name, split_frame in baseline.groupby("dataset_split", sort=False):
        baseline_rows.extend(
            _topn_rows_by_score(
                split_frame,
                split_name=split_name,
                top_n_list=top_n_list,
                score_column="final_score_v42",
                model_version="v4_baseline_same_risk_gate",
            )
        )
    if baseline_rows:
        rows.append(pd.DataFrame(baseline_rows))
    v42_rows: list[dict[str, object]] = []
    for split_name, split_frame in v42_scored.groupby("dataset_split", sort=False):
        v42_rows.extend(
            _topn_rows_by_score(
                split_frame,
                split_name=split_name,
                top_n_list=top_n_list,
                score_column="final_score_v42",
                model_version="v42_opportunity_gated",
            )
        )
    if v42_rows:
        rows.append(pd.DataFrame(v42_rows))
    if hybrid_v4_scored is not None and not hybrid_v4_scored.empty:
        hybrid_rows: list[dict[str, object]] = []
        for split_name, split_frame in hybrid_v4_scored.groupby("dataset_split", sort=False):
            hybrid_rows.extend(
                _topn_rows_by_score(
                    split_frame,
                    split_name=split_name,
                    top_n_list=top_n_list,
                    score_column="final_score_v42",
                    model_version="v42_gate_v4_rank",
                )
            )
        if hybrid_rows:
            rows.append(pd.DataFrame(hybrid_rows))
    return pd.concat(rows, ignore_index=True, copy=False) if rows else pd.DataFrame()


def evaluate_v5_topn_metrics(scored: pd.DataFrame, *, top_n_list: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, split_frame in scored.groupby("dataset_split", sort=False):
        rows.extend(
            _topn_rows_by_score(
                split_frame,
                split_name=split_name,
                top_n_list=top_n_list,
                score_column="final_score_v5",
                model_version=None,
            )
        )
    return pd.DataFrame(rows)


def evaluate_volume_price_risk_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if scored.empty:
        return pd.DataFrame()
    for split_name, split_frame in scored.groupby("dataset_split", sort=False):
        for group_name, group in [
            ("all", split_frame),
            ("candidate", split_frame[split_frame["pre_v5_action"].eq("candidate")] if "pre_v5_action" in split_frame.columns else split_frame),
            ("v5_candidate", split_frame[split_frame["action"].eq("candidate")] if "action" in split_frame.columns else split_frame),
        ]:
            y = pd.to_numeric(group.get("volume_price_risk_label", pd.Series(dtype=float)), errors="coerce")
            score = pd.to_numeric(group.get("volume_price_risk_score", pd.Series(dtype=float)), errors="coerce")
            valid = y.notna() & score.notna()
            row: dict[str, object] = {
                "dataset_split": split_name,
                "action_group": group_name,
                "rows": int(valid.sum()),
                "volume_price_risk_rate": float(y.loc[valid].mean()) if valid.any() else float("nan"),
                "avg_volume_price_risk_score": _safe_mean(score),
                "extreme_risk_rate": (
                    float(group["volume_price_extreme_risk_flag"].fillna(False).astype(bool).mean())
                    if "volume_price_extreme_risk_flag" in group.columns and not group.empty
                    else float("nan")
                ),
            }
            if valid.any():
                pred = score.loc[valid].ge(0.5).astype(int)
                yv = y.loc[valid].astype(int)
                row["risk_accuracy_0p5"] = float(accuracy_score(yv, pred))
                positive_pred = pred.eq(1)
                row["risk_precision_0p5"] = (
                    float(yv.loc[positive_pred].mean()) if positive_pred.any() else float("nan")
                )
                actual_positive = yv.eq(1)
                row["risk_recall_0p5"] = (
                    float(pred.loc[actual_positive].mean()) if actual_positive.any() else float("nan")
                )
                try:
                    row["risk_auc"] = float(roc_auc_score(yv, score.loc[valid]))
                except ValueError:
                    row["risk_auc"] = float("nan")
                if "volume_price_extreme_risk_flag" in group.columns:
                    extreme = group.loc[valid, "volume_price_extreme_risk_flag"].fillna(False).astype(bool)
                    row["extreme_risk_precision"] = float(yv.loc[extreme].mean()) if extreme.any() else float("nan")
                    row["extreme_risk_recall"] = float(extreme.loc[actual_positive].mean()) if actual_positive.any() else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_volume_price_quality_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if scored.empty:
        return pd.DataFrame()
    for split_name, split_frame in scored.groupby("dataset_split", sort=False):
        for group_name, group in [
            ("all", split_frame),
            ("candidate", split_frame[split_frame["pre_v5_action"].eq("candidate")] if "pre_v5_action" in split_frame.columns else split_frame),
            ("v5_candidate", split_frame[split_frame["action"].eq("candidate")] if "action" in split_frame.columns else split_frame),
        ]:
            y = pd.to_numeric(group.get("volume_price_quality_value", pd.Series(dtype=float)), errors="coerce")
            pred = pd.to_numeric(group.get("volume_price_quality_score", pd.Series(dtype=float)), errors="coerce")
            valid = y.notna() & pred.notna()
            if not valid.any():
                rows.append(
                    {
                        "dataset_split": split_name,
                        "action_group": group_name,
                        "rows": 0,
                        "avg_volume_price_quality_value": float("nan"),
                        "avg_volume_price_quality_score": float("nan"),
                    }
                )
                continue
            yv = y.loc[valid]
            pv = pred.loc[valid]
            row: dict[str, object] = {
                "dataset_split": split_name,
                "action_group": group_name,
                "rows": int(valid.sum()),
                "avg_volume_price_quality_value": float(yv.mean()),
                "avg_volume_price_quality_score": float(pv.mean()),
                "mae": float(mean_absolute_error(yv, pv)),
                "rmse": float(math.sqrt(mean_squared_error(yv, pv))),
                "pearson_corr": float(yv.corr(pv)) if len(yv) > 1 else float("nan"),
                "spearman_corr": float(yv.corr(pv, method="spearman")) if len(yv) > 1 else float("nan"),
                "avg_return_20d": _safe_mean(group.loc[valid, "period_return_20d"]) if "period_return_20d" in group.columns else float("nan"),
                "avg_drawdown_20d": _safe_mean(group.loc[valid, "max_drawdown_20d"]) if "max_drawdown_20d" in group.columns else float("nan"),
            }
            grade = pd.to_numeric(group.get("volume_price_quality_grade", pd.Series(dtype=float)), errors="coerce")
            if grade.loc[valid].nunique(dropna=True) > 1 and pv.nunique(dropna=True) > 1:
                try:
                    row["ndcg_at_20"] = float(
                        ndcg_score(
                            [grade.loc[valid].fillna(0).to_numpy(dtype=float)],
                            [pv.to_numpy(dtype=float)],
                            k=min(20, int(valid.sum())),
                        )
                    )
                except ValueError:
                    row["ndcg_at_20"] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def compare_v5_topn_metrics(
    baseline_scored: pd.DataFrame,
    v5_scored: pd.DataFrame,
    *,
    top_n_list: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    baseline_rows: list[dict[str, object]] = []
    for split_name, split_frame in baseline_scored.groupby("dataset_split", sort=False):
        baseline_rows.extend(
            _topn_rows_by_score(
                split_frame,
                split_name=split_name,
                top_n_list=top_n_list,
                score_column="final_score_v42",
                model_version="v42_gate_v4_rank",
            )
        )
    if baseline_rows:
        rows.append(pd.DataFrame(baseline_rows))

    v5_rows: list[dict[str, object]] = []
    for split_name, split_frame in v5_scored.groupby("dataset_split", sort=False):
        v5_rows.extend(
            _topn_rows_by_score(
                split_frame,
                split_name=split_name,
                top_n_list=top_n_list,
                score_column="final_score_v5",
                model_version="v5_volume_price_fusion",
            )
        )
    if v5_rows:
        rows.append(pd.DataFrame(v5_rows))
    return pd.concat(rows, ignore_index=True, copy=False) if rows else pd.DataFrame()


def evaluate_v51_topn_metrics(scored: pd.DataFrame, *, top_n_list: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, split_frame in scored.groupby("dataset_split", sort=False):
        rows.extend(
            _topn_rows_by_score(
                split_frame,
                split_name=split_name,
                top_n_list=top_n_list,
                score_column="final_score_v51",
                model_version=None,
            )
        )
    return pd.DataFrame(rows)


def evaluate_v51_ranker_metrics(scored: pd.DataFrame, *, top_n_list: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eval_at = tuple(sorted(set(top_n_list)))
    if scored.empty:
        return pd.DataFrame()
    for split_name, split_frame in scored.groupby("dataset_split", sort=False):
        candidates = split_frame[split_frame.get("v51_candidate_eligible", pd.Series(False, index=split_frame.index)).fillna(False).astype(bool)]
        y = pd.to_numeric(candidates.get("v51_rank_value", pd.Series(dtype=float)), errors="coerce")
        pred = pd.to_numeric(candidates.get("candidate_rank_score_v51", pd.Series(dtype=float)), errors="coerce")
        valid = y.notna() & pred.notna()
        row: dict[str, object] = {
            "dataset_split": split_name,
            "action_group": "v51_candidate",
            "rows": int(valid.sum()),
            "avg_v51_rank_value": float(y.loc[valid].mean()) if valid.any() else float("nan"),
            "avg_candidate_rank_score_v51": float(pred.loc[valid].mean()) if valid.any() else float("nan"),
            "pearson_corr": float(y.loc[valid].corr(pred.loc[valid])) if valid.sum() > 1 else float("nan"),
            "spearman_corr": float(y.loc[valid].corr(pred.loc[valid], method="spearman")) if valid.sum() > 1 else float("nan"),
        }
        grade = pd.to_numeric(candidates.get("v51_rank_grade", pd.Series(dtype=float)), errors="coerce")
        for top_n in eval_at:
            scores: list[float] = []
            for _, day_frame in candidates.groupby("trade_date", sort=False):
                label = pd.to_numeric(day_frame.get("v51_rank_grade", pd.Series(dtype=float)), errors="coerce")
                score = pd.to_numeric(day_frame.get("candidate_rank_score_v51", pd.Series(dtype=float)), errors="coerce")
                day_valid = label.notna() & score.notna()
                if day_valid.sum() < 2:
                    continue
                try:
                    scores.append(
                        float(
                            ndcg_score(
                                [label.loc[day_valid].to_numpy(dtype=float)],
                                [score.loc[day_valid].to_numpy(dtype=float)],
                                k=min(top_n, int(day_valid.sum())),
                            )
                        )
                    )
                except ValueError:
                    continue
            row[f"ndcg_at_{top_n}"] = float(np.mean(scores)) if scores else float("nan")
        if valid.any() and grade.loc[valid].notna().any():
            row["avg_grade"] = float(grade.loc[valid].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def compare_v51_topn_metrics(
    *,
    baseline_scored: pd.DataFrame,
    v5_scored: pd.DataFrame,
    v51_scored: pd.DataFrame,
    top_n_list: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    baseline_rows: list[dict[str, object]] = []
    for split_name, split_frame in baseline_scored.groupby("dataset_split", sort=False):
        baseline_rows.extend(
            _topn_rows_by_score(
                split_frame,
                split_name=split_name,
                top_n_list=top_n_list,
                score_column="final_score_v42",
                model_version="v42_gate_v4_rank",
            )
        )
    if baseline_rows:
        rows.append(pd.DataFrame(baseline_rows))

    v5_rows: list[dict[str, object]] = []
    for split_name, split_frame in v5_scored.groupby("dataset_split", sort=False):
        v5_rows.extend(
            _topn_rows_by_score(
                split_frame,
                split_name=split_name,
                top_n_list=top_n_list,
                score_column="final_score_v5",
                model_version="v5_volume_price_fusion",
            )
        )
    if v5_rows:
        rows.append(pd.DataFrame(v5_rows))

    v51_rows: list[dict[str, object]] = []
    for split_name, split_frame in v51_scored.groupby("dataset_split", sort=False):
        v51_rows.extend(
            _topn_rows_by_score(
                split_frame,
                split_name=split_name,
                top_n_list=top_n_list,
                score_column="final_score_v51",
                model_version="v51_candidate_ranker",
            )
        )
    if v51_rows:
        rows.append(pd.DataFrame(v51_rows))
    return pd.concat(rows, ignore_index=True, copy=False) if rows else pd.DataFrame()


def stage2_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for spec in HORIZONS:
        suffix = spec.name
        columns.extend(
            [
                f"up_prob_{suffix}",
                f"down_prob_{suffix}",
                f"neutral_prob_{suffix}",
                f"expected_value_{suffix}",
                f"risk_adjusted_value_{suffix}",
            ]
        )
    columns.extend(
        [
            "stage2_edge_20d",
            "stage2_edge_60d",
            "stage2_short_down_pressure",
            "stage2_20d_up_without_60d_down",
            "stage2_short_up_confirmation",
            "stage2_large_risk_suppressor",
            "stage2_weighted_expected_value",
            "stage2_weighted_down_prob",
        ]
    )
    return [column for column in columns if column in frame.columns]


def long_upside_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for spec in LONG_UPSIDE_HORIZONS:
        suffix = spec.name
        columns.extend(
            [
                f"long_up_prob_{suffix}",
                f"long_down_prob_{suffix}",
                f"long_neutral_prob_{suffix}",
                f"long_expected_value_{suffix}",
                f"long_risk_adjusted_value_{suffix}",
            ]
        )
    columns.extend(
        [
            "long_stage2_edge_20d",
            "long_stage2_edge_60d",
            "long_stage2_20d_up_without_60d_down",
            "long_stage2_60d_trend_confirmation",
            "long_stage2_weighted_expected_value",
            "long_stage2_weighted_down_prob",
            "long_stage2_weighted_up_prob",
            "return_20d",
            "return_40d",
            "return_60d",
            "return_120d",
            "distance_to_ma20",
            "distance_to_ma60",
            "distance_to_ma120",
            "ma20_slope_1d",
            "ma60_slope_1d",
            "ma20_slope_5d",
            "ma60_slope_5d",
            "ma20_slope_20d",
            "ma60_slope_20d",
            "ma20_to_ma60",
            "ma20_to_ma60_change_20d",
            "position_in_range_20d",
            "position_in_range_60d",
            "position_in_range_120d",
            "distance_to_60d_high",
            "distance_to_60d_low",
            "distance_to_120d_high",
            "distance_to_120d_low",
            "volume_change_20d",
            "volume_change_60d",
            "amount_change_20d",
            "amount_change_60d",
            "volume_ratio_20",
            "volume_ratio_60",
            "amount_ratio_20",
            "amount_ratio_60",
            "volatility_20d",
            "volatility_60d",
            "drawdown_20d",
            "drawdown_60d",
            "up_days_20",
            "up_days_60",
            "down_days_20",
            "down_days_60",
            "block20_0_return",
            "block20_1_return",
            "block20_2_return",
            "block20_0_volume_sum",
            "block20_1_volume_sum",
            "block20_2_volume_sum",
            "block20_0_amount_sum",
            "block20_1_amount_sum",
            "block20_2_amount_sum",
            "block20_0_vs_1_volume_ratio",
            "block20_1_vs_2_volume_ratio",
            "block20_0_vs_1_amount_ratio",
            "block20_1_vs_2_amount_ratio",
            "block20_0_vs_1_return_diff",
            "block20_1_vs_2_return_diff",
            "block60_0_return",
            "block60_1_return",
            "block60_2_return",
            "block60_0_volume_sum",
            "block60_1_volume_sum",
            "block60_2_volume_sum",
            "block60_0_amount_sum",
            "block60_1_amount_sum",
            "block60_2_amount_sum",
            "block60_0_vs_1_volume_ratio",
            "block60_1_vs_2_volume_ratio",
            "block60_0_vs_1_amount_ratio",
            "block60_1_vs_2_amount_ratio",
            "block60_0_vs_1_return_diff",
            "block60_1_vs_2_return_diff",
            "rsi_6",
            "rsi_14",
            "rsi_24",
            "macd",
            "macd_signal",
            "macd_hist",
            "macd_hist_slope",
            "stoch_k",
            "stoch_d",
            "stoch_j",
            "stoch_k_minus_d",
            "cci_20",
            "plus_di_14",
            "minus_di_14",
            "adx_14",
            "williams_r_14",
            "boll_width_20",
            "boll_position_20",
            "atr_pct_14",
        ]
    )
    seen: set[str] = set()
    result: list[str] = []
    for column in columns:
        if column in frame.columns and column not in seen and pd.api.types.is_numeric_dtype(frame[column]):
            result.append(column)
            seen.add(column)
    return result


def add_v41_cross_sectional_rank_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rank_columns = [
        "risk_score",
        "down_prob_20d",
        "down_prob_60d",
        "stage2_weighted_down_prob",
        "up_prob_20d",
        "up_prob_60d",
        "return_20d",
        "return_40d",
        "return_60d",
        "return_120d",
        "distance_to_ma20",
        "distance_to_ma60",
        "ma20_slope_5d",
        "ma60_slope_5d",
        "ma20_to_ma60",
        "position_in_range_20d",
        "position_in_range_60d",
        "volume_ratio_20",
        "volume_ratio_60",
        "amount_ratio_20",
        "amount_ratio_60",
        "volatility_20d",
        "volatility_60d",
        "drawdown_20d",
        "drawdown_60d",
        "block20_0_return",
        "block60_0_return",
        "block60_1_return",
        "block60_2_return",
        "block60_0_vs_1_volume_ratio",
        "block60_0_vs_1_amount_ratio",
        "macd_hist",
        "macd_hist_slope",
        "rsi_14",
        "stoch_k_minus_d",
        "adx_14",
        "atr_pct_14",
    ]
    for column in rank_columns:
        if column not in result.columns:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        rank = pd.Series(np.nan, index=result.index, dtype="float64")
        for _, day_index in result.groupby("trade_date", sort=False).groups.items():
            day_values = values.loc[day_index]
            valid = day_values.notna()
            if valid.any():
                rank.loc[day_values.loc[valid].index] = day_values.loc[valid].rank(method="first", pct=True)
        result[f"cs_rank_{column}"] = rank
    return result


def long_quality_ranker_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = stage2_feature_columns(frame)
    columns.extend(
        [
            "risk_score",
            "applied_risk_percentile",
            "return_5d",
            "return_10d",
            "return_20d",
            "return_40d",
            "return_60d",
            "return_120d",
            "intraday_range_pct",
            "body_pct",
            "upper_shadow_pct",
            "lower_shadow_pct",
            "gap_pct",
            "distance_to_ma20",
            "distance_to_ma60",
            "distance_to_ma120",
            "ma20_slope_1d",
            "ma60_slope_1d",
            "ma20_slope_5d",
            "ma60_slope_5d",
            "ma20_slope_20d",
            "ma60_slope_20d",
            "ma20_to_ma60",
            "ma20_to_ma60_change_20d",
            "position_in_range_20d",
            "position_in_range_60d",
            "position_in_range_120d",
            "distance_to_60d_high",
            "distance_to_60d_low",
            "distance_to_120d_high",
            "distance_to_120d_low",
            "volume_change_20d",
            "volume_change_60d",
            "amount_change_20d",
            "amount_change_60d",
            "volume_ratio_20",
            "volume_ratio_60",
            "amount_ratio_20",
            "amount_ratio_60",
            "volatility_20d",
            "volatility_60d",
            "drawdown_20d",
            "drawdown_60d",
            "up_days_20",
            "up_days_60",
            "down_days_20",
            "down_days_60",
            "block20_0_return",
            "block20_1_return",
            "block20_2_return",
            "block20_0_volume_sum",
            "block20_1_volume_sum",
            "block20_2_volume_sum",
            "block20_0_amount_sum",
            "block20_1_amount_sum",
            "block20_2_amount_sum",
            "block20_0_vs_1_volume_ratio",
            "block20_1_vs_2_volume_ratio",
            "block20_0_vs_1_amount_ratio",
            "block20_1_vs_2_amount_ratio",
            "block20_0_vs_1_return_diff",
            "block20_1_vs_2_return_diff",
            "block60_0_return",
            "block60_1_return",
            "block60_2_return",
            "block60_0_volume_sum",
            "block60_1_volume_sum",
            "block60_2_volume_sum",
            "block60_0_amount_sum",
            "block60_1_amount_sum",
            "block60_2_amount_sum",
            "block60_0_vs_1_volume_ratio",
            "block60_1_vs_2_volume_ratio",
            "block60_0_vs_1_amount_ratio",
            "block60_1_vs_2_amount_ratio",
            "block60_0_vs_1_return_diff",
            "block60_1_vs_2_return_diff",
            "rsi_6",
            "rsi_14",
            "rsi_24",
            "macd",
            "macd_signal",
            "macd_hist",
            "macd_hist_slope",
            "stoch_k",
            "stoch_d",
            "stoch_j",
            "stoch_k_minus_d",
            "cci_20",
            "plus_di_14",
            "minus_di_14",
            "adx_14",
            "williams_r_14",
            "boll_width_20",
            "boll_position_20",
            "atr_pct_14",
        ]
    )
    columns.extend([column for column in frame.columns if column.startswith("cs_rank_")])
    seen: set[str] = set()
    result: list[str] = []
    for column in columns:
        if column in frame.columns and column not in seen and pd.api.types.is_numeric_dtype(frame[column]):
            result.append(column)
            seen.add(column)
    return result


def _metric_row(split_name: str, frame: pd.DataFrame, top_n: int | None) -> dict[str, object]:
    y_true = pd.to_numeric(frame["trade_value"], errors="coerce")
    prediction_column = next(
        (
            column
            for column in (
                "__selection_score",
                "trade_value_pred",
                "final_score_v5",
                "final_score_v42",
                "opportunity_rank_score",
                "final_score_v41",
                "long_quality_score",
                "final_score_v4_risk_upside",
                "long_upside_score",
                "final_score_v4",
                "alpha_rank_score",
                "final_score",
                "upside_score",
                "trade_value",
            )
            if column in frame.columns
        ),
        "trade_value",
    )
    y_pred = pd.to_numeric(frame[prediction_column], errors="coerce")
    valid = y_true.notna() & y_pred.notna()
    if not valid.any():
        return {"dataset_split": split_name, "top_n": top_n, "rows": 0}
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    returns_20d = pd.to_numeric(frame.loc[valid, "period_return_20d"], errors="coerce")
    positive_returns_20d = returns_20d[returns_20d > 0]
    negative_returns_20d = returns_20d[returns_20d < 0]
    row: dict[str, object] = {
        "dataset_split": split_name,
        "top_n": top_n if top_n is not None else "all",
        "rows": int(len(y_true)),
        "direction_accuracy": float(accuracy_score((y_true > 0).astype(int), (y_pred > 0).astype(int))),
        "win_rate": float((y_true > 0).mean()),
        "avg_trade_value": float(y_true.mean()),
        "avg_pred_trade_value": float(y_pred.mean()),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson_corr": float(y_true.corr(y_pred)) if len(y_true) > 1 else np.nan,
        "spearman_corr": float(y_true.corr(y_pred, method="spearman")) if len(y_true) > 1 else np.nan,
        "avg_return_20d": _safe_mean(returns_20d),
        "median_return_20d": float(returns_20d.median()) if returns_20d.notna().any() else float("nan"),
        "avg_positive_return_20d": _safe_mean(positive_returns_20d),
        "avg_negative_return_20d": _safe_mean(negative_returns_20d),
        "avg_max_drawdown_20d": _safe_mean(frame.loc[valid, "max_drawdown_20d"]),
        "take_profit_rate_20d": _outcome_rate(frame.loc[valid], "20d", "up"),
        "stop_loss_rate_20d": _outcome_rate(frame.loc[valid], "20d", "down"),
        "neutral_rate_20d": _outcome_rate(frame.loc[valid], "20d", "neutral"),
        "avg_take_profit_20d": _outcome_mean(frame.loc[valid], "20d", "up", "period_return"),
        "avg_stop_loss_20d": _outcome_mean(frame.loc[valid], "20d", "down", "period_return"),
    }
    for spec in HORIZONS:
        suffix = spec.name
        row[f"take_profit_rate_{suffix}"] = _outcome_rate(frame.loc[valid], suffix, "up")
        row[f"stop_loss_rate_{suffix}"] = _outcome_rate(frame.loc[valid], suffix, "down")
        row[f"avg_return_{suffix}"] = _safe_mean(frame.loc[valid, f"period_return_{suffix}"])
        row[f"avg_drawdown_{suffix}"] = _safe_mean(frame.loc[valid, f"max_drawdown_{suffix}"])
    return row


def prepare_opportunity_ranker_prediction_report(scored: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rank",
        "trade_date",
        "symbol",
        "name",
        "action",
        "risk_candidate_action",
        "risk_action",
        "final_action",
        "trade_permission",
        "opportunity_score",
        "opportunity_threshold",
        "opportunity_quality",
        "risk_tier",
        "risk_gate_reason",
        "risk_score",
        "long_upside_score",
        "opportunity_rank_score",
        "opportunity_rank_score_pct",
        "final_score_v42",
        "buy_score_v42",
        "rank_source_v42",
        "top_risk_horizon",
        "top_upside_horizon",
        "stage2_weighted_down_prob",
    ]
    for spec in HORIZONS:
        columns.extend([f"up_prob_{spec.name}", f"down_prob_{spec.name}", f"neutral_prob_{spec.name}"])
    explanation = [
        "close",
        "distance_to_ma20",
        "distance_to_ma60",
        "ma20_slope_5d",
        "ma60_slope_5d",
        "return_5d",
        "return_10d",
        "return_20d",
        "return_40d",
        "return_60d",
        "return_120d",
        "volume_ratio_20",
        "volume_ratio_60",
        "amount_ratio_20",
        "amount_ratio_60",
        "position_in_range_20d",
        "position_in_range_60d",
        "block20_0_return",
        "block60_0_return",
        "block60_1_return",
        "block60_2_return",
        "block60_0_vs_1_volume_ratio",
        "block60_0_vs_1_amount_ratio",
        "macd_hist",
        "macd_hist_slope",
        "rsi_14",
        "stoch_k",
        "stoch_d",
        "stoch_j",
        "atr_pct_14",
    ]
    columns.extend([column for column in explanation if column in scored.columns])
    return scored.loc[:, [column for column in columns if column in scored.columns]].copy()


def prepare_v5_prediction_report(scored: pd.DataFrame) -> pd.DataFrame:
    report = scored.copy()
    if "model_version" not in report.columns:
        report["model_version"] = report.get("model_version_v5", "v5_volume_price_fusion")
    columns = [
        "rank",
        "model_version",
        "trade_date",
        "symbol",
        "name",
        "action",
        "pre_v5_action",
        "risk_candidate_action",
        "risk_action",
        "final_action",
        "trade_permission",
        "risk_tier",
        "risk_gate_reason",
        "opportunity_score",
        "opportunity_threshold",
        "risk_score",
        "long_upside_score",
        "opportunity_rank_score",
        "final_score_v42",
        "buy_score_v42",
        "rank_source_v42",
        "volume_price_extreme_risk_flag",
        "volume_price_extreme_risk_reason",
        "volume_price_risk_score",
        "volume_price_risk_score_pct",
        "volume_price_quality_score",
        "volume_price_quality_score_pct",
        "final_score_v5",
        "buy_score_v5",
        "top_risk_horizon",
        "top_upside_horizon",
        "stage2_weighted_down_prob",
    ]
    for spec in HORIZONS:
        columns.extend([f"up_prob_{spec.name}", f"down_prob_{spec.name}", f"neutral_prob_{spec.name}"])
    explanation = [
        "close",
        "distance_to_ma20",
        "distance_to_ma60",
        "return_5d",
        "return_20d",
        "return_60d",
        "volume_ratio_20",
        "amount_ratio_20",
        "vp_close_position_1d",
        "vp_signed_body_1d",
        "vp_upper_shadow_1d",
        "vp_lower_shadow_1d",
        "vp_return_1d",
        "vp_gap_1d",
        "vp_volume_ratio_1d_to_20d",
        "vp_amount_ratio_1d_to_20d",
        "vp_high_volume_upper_shadow_flag",
        "vp_high_volume_bearish_flag",
        "vp_failed_breakout_1d_flag",
        "vp_return_5d",
        "vp_volume_change_5d",
        "vp_amount_change_5d",
        "vp_5d_up_down_volume_ratio",
        "vp_5d_high_volume_weak_days",
        "vp_5d_upper_shadow_pressure",
        "vp_5d_lower_shadow_support",
        "vp_5d_price_volume_confirm",
        "vp_5d_volume_without_price",
        "vp_5d_shrink_pullback_score",
        "vp_return_20d",
        "vp_volume_change_20d",
        "vp_amount_change_20d",
        "vp_20d_range_position",
        "vp_20d_up_down_volume_ratio",
        "vp_20d_accumulation_score",
        "vp_20d_distribution_score",
        "vp_5d_vs_20d_return_accel",
        "vp_5d_vs_20d_volume_accel",
        "vp_volume_accel_without_price",
        "vp_short_shrink_after_strength",
        "vp_pullback_depth_in_20d",
        "macd_hist",
        "rsi_14",
        "stoch_k",
        "stoch_d",
        "stoch_j",
        "atr_pct_14",
    ]
    columns.extend([column for column in explanation if column in report.columns])
    return report.loc[:, [column for column in columns if column in report.columns]].copy()


def prepare_v51_prediction_report(scored: pd.DataFrame) -> pd.DataFrame:
    report = scored.copy()
    if "model_version" not in report.columns:
        report["model_version"] = report.get("model_version_v51", "v51_candidate_ranker")
    columns = [
        "rank",
        "model_version",
        "trade_date",
        "symbol",
        "name",
        "action",
        "pre_v5_action",
        "risk_candidate_action",
        "risk_action",
        "final_action",
        "trade_permission",
        "risk_tier",
        "risk_gate_reason",
        "opportunity_score",
        "opportunity_threshold",
        "risk_score",
        "long_upside_score",
        "opportunity_rank_score",
        "final_score_v42",
        "buy_score_v42",
        "rank_source_v42",
        "volume_price_extreme_risk_flag",
        "volume_price_extreme_risk_reason",
        "volume_price_risk_score",
        "volume_price_quality_score",
        "final_score_v5",
        "buy_score_v5",
        "candidate_rank_score_v51",
        "candidate_rank_score_pct_v51",
        "final_score_v51_raw",
        "final_score_v51",
        "buy_score_v51",
        "v51_blend_weight",
        "rank_source_v51",
        "top_risk_horizon",
        "top_upside_horizon",
        "stage2_weighted_down_prob",
    ]
    for spec in HORIZONS:
        columns.extend([f"up_prob_{spec.name}", f"down_prob_{spec.name}", f"neutral_prob_{spec.name}"])
    explanation = [
        "close",
        "distance_to_ma20",
        "distance_to_ma60",
        "return_5d",
        "return_20d",
        "return_60d",
        "volume_ratio_20",
        "amount_ratio_20",
        "vp_close_position_1d",
        "vp_signed_body_1d",
        "vp_upper_shadow_1d",
        "vp_return_1d",
        "vp_volume_ratio_1d_to_20d",
        "vp_amount_ratio_1d_to_20d",
        "vp_5d_price_volume_confirm",
        "vp_5d_volume_without_price",
        "vp_5d_shrink_pullback_score",
        "vp_20d_range_position",
        "vp_20d_up_down_volume_ratio",
        "vp_20d_accumulation_score",
        "vp_20d_distribution_score",
        "vp_5d_vs_20d_return_accel",
        "vp_5d_vs_20d_volume_accel",
        "vp_volume_accel_without_price",
        "vp_short_shrink_after_strength",
        "macd_hist",
        "rsi_14",
        "stoch_k",
        "stoch_d",
        "stoch_j",
        "atr_pct_14",
    ]
    columns.extend([column for column in explanation if column in report.columns])
    return report.loc[:, [column for column in columns if column in report.columns]].copy()


def format_metric_table(frame: pd.DataFrame, *, limit: int | None = None) -> str:
    if frame.empty:
        return "No metrics."
    display = frame.head(limit).copy() if limit else frame.copy()
    for column in display.columns:
        if column in {"dataset_split", "top_n", "horizon"}:
            continue
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    return display.to_string(index=False)


def format_opportunity_ranker_prediction_table(frame: pd.DataFrame, *, limit: int) -> str:
    if frame.empty:
        return "No predictions."
    columns = [
        "rank",
        "symbol",
        "name",
        "action",
        "trade_permission",
        "risk_tier",
        "risk_gate_reason",
        "opportunity_score",
        "opportunity_threshold",
        "risk_score",
        "long_upside_score",
        "opportunity_rank_score",
        "final_score_v42",
        "buy_score_v42",
        "rank_source_v42",
        "top_risk_horizon",
        "top_upside_horizon",
        "up_prob_20d",
        "down_prob_20d",
        "up_prob_60d",
        "down_prob_60d",
    ]
    display = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit).copy()
    for column in display.columns:
        if column in {
            "rank",
            "symbol",
            "name",
            "action",
            "trade_permission",
            "risk_tier",
            "risk_gate_reason",
            "rank_source_v42",
            "top_risk_horizon",
            "top_upside_horizon",
        }:
            continue
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    return display.to_string(index=False)


def format_volume_price_fusion_prediction_table(frame: pd.DataFrame, *, limit: int) -> str:
    if frame.empty:
        return "No predictions."
    columns = [
        "rank",
        "symbol",
        "name",
        "action",
        "pre_v5_action",
        "trade_permission",
        "risk_tier",
        "risk_gate_reason",
        "opportunity_score",
        "risk_score",
        "long_upside_score",
        "volume_price_risk_score",
        "volume_price_quality_score",
        "final_score_v5",
        "buy_score_v5",
        "final_score_v42",
        "buy_score_v42",
        "volume_price_extreme_risk_flag",
        "volume_price_extreme_risk_reason",
    ]
    display = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit).copy()
    text_columns = {
        "rank",
        "symbol",
        "name",
        "action",
        "pre_v5_action",
        "trade_permission",
        "risk_tier",
        "risk_gate_reason",
        "volume_price_extreme_risk_flag",
        "volume_price_extreme_risk_reason",
    }
    for column in display.columns:
        if column in text_columns:
            continue
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    return display.to_string(index=False)


def format_candidate_ranker_prediction_table(frame: pd.DataFrame, *, limit: int) -> str:
    if frame.empty:
        return "No predictions."
    columns = [
        "rank",
        "symbol",
        "name",
        "action",
        "trade_permission",
        "risk_tier",
        "risk_gate_reason",
        "opportunity_score",
        "risk_score",
        "long_upside_score",
        "volume_price_risk_score",
        "volume_price_quality_score",
        "final_score_v5",
        "candidate_rank_score_v51",
        "final_score_v51",
        "buy_score_v51",
        "v51_blend_weight",
        "rank_source_v51",
        "volume_price_extreme_risk_flag",
        "volume_price_extreme_risk_reason",
    ]
    display = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit).copy()
    text_columns = {
        "rank",
        "symbol",
        "name",
        "action",
        "trade_permission",
        "risk_tier",
        "risk_gate_reason",
        "rank_source_v51",
        "volume_price_extreme_risk_flag",
        "volume_price_extreme_risk_reason",
    }
    for column in display.columns:
        if column in text_columns:
            continue
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    return display.to_string(index=False)


def _outcome_rate(frame: pd.DataFrame, suffix: str, outcome: str) -> float:
    column = f"outcome_{suffix}"
    if column not in frame.columns or frame.empty:
        return float("nan")
    return float(frame[column].eq(outcome).mean())


def _outcome_mean(frame: pd.DataFrame, suffix: str, outcome: str, metric: str) -> float:
    outcome_column = f"outcome_{suffix}"
    metric_column = f"{metric}_{suffix}"
    if outcome_column not in frame.columns or metric_column not in frame.columns:
        return float("nan")
    subset = frame[frame[outcome_column].eq(outcome)]
    return _safe_mean(subset[metric_column])


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _model_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.reindex(columns=columns).copy()
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def _clean_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric_columns = result.select_dtypes(include=["number", "bool"]).columns
    result.loc[:, numeric_columns] = result.loc[:, numeric_columns].replace([np.inf, -np.inf], np.nan)
    return result


def _downcast_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    float_columns = result.select_dtypes(include=["float64"]).columns
    for column in float_columns:
        result[column] = pd.to_numeric(result[column], downcast="float")
    int_columns = result.select_dtypes(include=["int64"]).columns
    for column in int_columns:
        result[column] = pd.to_numeric(result[column], downcast="integer")
    return result
