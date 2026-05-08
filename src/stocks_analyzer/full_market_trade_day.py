from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
import json
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - depends on optional local package.
    LGBMClassifier = None

from .full_market_panel import full_market_report_dir
from .storage import DailyBarsReadError, Storage


TRADE_DAY_GATE_MODEL_VERSION = "trade_day_gate_phase7_v1"
TRADE_DAY_PROGRESS_INTERVAL = 500
DEFAULT_TRADE_DAY_MODEL_NAMES = (
    "always_allow",
    "rule_market_gate",
    "logistic_regression",
    "linear_discriminant_analysis",
    "naive_bayes",
    "lightgbm_classifier",
)
TRADE_DAY_FEATURE_COLUMNS = (
    "equal_return_1d",
    "equal_return_5d",
    "equal_return_20d",
    "equal_return_60d",
    "equal_above_ma20",
    "equal_above_ma60",
    "equal_above_ma120",
    "equal_ma20_slope_5d",
    "equal_ma60_slope_10d",
    "equal_drawdown_20d",
    "equal_drawdown_60d",
    "equal_volatility_20d",
    "equal_volatility_60d",
    "amount_weight_return_1d",
    "amount_weight_return_5d",
    "amount_weight_return_20d",
    "amount_weight_return_60d",
    "amount_weight_above_ma20",
    "amount_weight_above_ma60",
    "amount_weight_above_ma120",
    "amount_weight_ma20_slope_5d",
    "amount_weight_ma60_slope_10d",
    "amount_weight_drawdown_20d",
    "amount_weight_drawdown_60d",
    "amount_weight_volatility_20d",
    "amount_weight_volatility_60d",
    "breadth_above_ma20",
    "breadth_above_ma60",
    "breadth_above_ma120",
    "breadth_positive_return_5d",
    "breadth_positive_return_20d",
    "advancing_ratio_1d",
    "declining_ratio_1d",
    "new_high_20d_ratio",
    "new_low_20d_ratio",
    "new_high_60d_ratio",
    "new_low_60d_ratio",
    "limit_up_ratio",
    "limit_down_ratio",
    "amount_ratio_5d_20d",
    "amount_ratio_20d_60d",
    "cross_section_return_mean_1d",
    "cross_section_return_median_1d",
    "cross_section_return_dispersion_1d",
    "cross_section_volatility_median_20d",
    "cross_section_max_drawdown_median_20d",
)


@dataclass(slots=True)
class TradeDayGateValidationResult:
    dataset: pd.DataFrame
    feature_audit: pd.DataFrame
    windows: pd.DataFrame
    metrics: pd.DataFrame
    deciles: pd.DataFrame
    filter_impact: pd.DataFrame
    summary: pd.DataFrame
    report_dir: Path
    dataset_path: Path
    feature_audit_path: Path
    windows_path: Path
    metrics_path: Path
    deciles_path: Path
    filter_impact_path: Path
    summary_path: Path
    config_path: Path


@dataclass(slots=True)
class TradeDayGateTrainResult:
    model_path: Path
    metadata_path: Path
    model_name: str
    train_rows: int
    train_start: str
    train_end: str
    selected_threshold: float
    feature_columns: tuple[str, ...]


@dataclass(slots=True)
class TradeDayGatePredictionResult:
    prediction: pd.DataFrame
    output_path: Path
    artifact_path: Path


def build_trade_day_feature_frame(
    *,
    storage: Storage,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    min_stock_count: int = 500,
) -> pd.DataFrame:
    universe = storage.load_universe().copy()
    if limit is not None:
        universe = universe.head(max(int(limit), 0)).copy()

    rows: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []
    instruments = universe.to_dict("records")
    total_symbols = len(instruments)
    for index, instrument in enumerate(instruments, start=1):
        _log_progress("Trade-day feature panel", index, total_symbols)
        symbol = str(instrument.get("symbol", "")).zfill(6)
        name = str(instrument.get("name", ""))
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError) as exc:
            skipped.append({"symbol": symbol, "name": name, "reason": type(exc).__name__})
            continue
        frame = _symbol_trade_day_frame(bars, symbol=symbol)
        if start_date is not None:
            frame = frame[frame["trade_date"].dt.date >= start_date]
        if end_date is not None:
            frame = frame[frame["trade_date"].dt.date <= end_date]
        if frame.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "no_trade_day_feature_rows"})
            continue
        rows.append(frame)

    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    features = aggregate_trade_day_features(panel, min_stock_count=min_stock_count)
    logging.info(
        "Trade-day feature frame complete: symbols=%s rows=%s skipped=%s",
        total_symbols,
        len(features),
        len(skipped),
    )
    return features


def aggregate_trade_day_features(panel: pd.DataFrame, *, min_stock_count: int = 500) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for trade_date, day in panel.groupby("trade_date", sort=True):
        daily_return = pd.to_numeric(day["daily_return"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = daily_return.between(-0.5, 1.0)
        day = day.loc[valid].copy()
        daily_return = daily_return.loc[valid]
        if day.empty:
            continue
        stock_count = int(day["symbol"].nunique())
        if stock_count < min_stock_count:
            continue
        amount = pd.to_numeric(day["amount"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "trade_date": pd.Timestamp(trade_date),
                "stock_count": stock_count,
                "equal_return_1d": float(daily_return.mean()),
                "cross_section_return_mean_1d": float(daily_return.mean()),
                "cross_section_return_median_1d": float(daily_return.median()),
                "cross_section_return_dispersion_1d": float(daily_return.std(ddof=0)),
                "amount_weight_return_1d": _weighted_return(daily_return=daily_return, amount=amount),
                "total_amount": float(amount.sum()),
                "breadth_above_ma20": _share(day["above_ma20"]),
                "breadth_above_ma60": _share(day["above_ma60"]),
                "breadth_above_ma120": _share(day["above_ma120"]),
                "breadth_positive_return_5d": _share(pd.to_numeric(day["return_5d"], errors="coerce").gt(0)),
                "breadth_positive_return_20d": _share(pd.to_numeric(day["return_20d"], errors="coerce").gt(0)),
                "advancing_ratio_1d": float(daily_return.gt(0).mean()),
                "declining_ratio_1d": float(daily_return.lt(0).mean()),
                "new_high_20d_ratio": _share(day["new_high_20d"]),
                "new_low_20d_ratio": _share(day["new_low_20d"]),
                "new_high_60d_ratio": _share(day["new_high_60d"]),
                "new_low_60d_ratio": _share(day["new_low_60d"]),
                "limit_up_ratio": _share(day["limit_up_like"]),
                "limit_down_ratio": _share(day["limit_down_like"]),
                "cross_section_volatility_median_20d": _median(day["volatility_20d"]),
                "cross_section_max_drawdown_median_20d": _median(day["max_drawdown_20d"]),
            }
        )
    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    result["synthetic_equal_weight_index"] = result["equal_return_1d"].fillna(0.0).add(1.0).cumprod()
    result["synthetic_amount_weight_index"] = result["amount_weight_return_1d"].fillna(0.0).add(1.0).cumprod()
    result["amount_ratio_5d_20d"] = result["total_amount"].rolling(5, min_periods=5).mean().div(
        result["total_amount"].rolling(20, min_periods=20).mean()
    )
    result["amount_ratio_20d_60d"] = result["total_amount"].rolling(20, min_periods=20).mean().div(
        result["total_amount"].rolling(60, min_periods=60).mean()
    )
    _add_index_features(result, source_column="synthetic_equal_weight_index", prefix="equal")
    _add_index_features(result, source_column="synthetic_amount_weight_index", prefix="amount_weight")
    return result.replace([np.inf, -np.inf], np.nan)


def add_trade_day_labels(
    features: pd.DataFrame,
    *,
    horizon_days: int = 5,
    drawdown_threshold: float = -0.03,
    return_threshold: float = -0.02,
    market_source_column: str = "synthetic_equal_weight_index",
) -> pd.DataFrame:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    if market_source_column not in features.columns:
        raise ValueError(f"Trade-day features do not contain market source column: {market_source_column}")
    result = features.copy()
    market = pd.to_numeric(result[market_source_column], errors="coerce")
    future_return = market.shift(-horizon_days).div(market).sub(1.0)
    future_drawdown = _future_min_return(market, horizon=horizon_days)
    result["future_market_return"] = future_return
    result["future_market_max_drawdown"] = future_drawdown
    result["bad_buy_day"] = (
        future_drawdown.le(float(drawdown_threshold)) | future_return.le(float(return_threshold))
    ).astype(float)
    result["label_horizon_days"] = int(horizon_days)
    result["drawdown_threshold"] = float(drawdown_threshold)
    result["return_threshold"] = float(return_threshold)
    result["market_source_column"] = market_source_column
    result["config_id"] = _trade_day_config_id(
        horizon_days=horizon_days,
        drawdown_threshold=drawdown_threshold,
        return_threshold=return_threshold,
    )
    result = result.dropna(subset=["bad_buy_day", "future_market_return", "future_market_max_drawdown"]).copy()
    return result.reset_index(drop=True)


def build_trade_day_feature_audit(features: pd.DataFrame, *, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    rows = []
    for column in feature_columns:
        values = pd.to_numeric(features[column], errors="coerce") if column in features.columns else pd.Series(dtype=float)
        rows.append(
            {
                "feature": column,
                "missing_rate": float(values.isna().mean()) if len(values) else np.nan,
                "mean": float(values.mean()) if values.notna().any() else np.nan,
                "std": float(values.std()) if values.notna().any() else np.nan,
                "min": float(values.min()) if values.notna().any() else np.nan,
                "max": float(values.max()) if values.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def validate_trade_day_gate(
    *,
    storage: Storage,
    project_root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    min_stock_count: int = 500,
    train_days: int = 1000,
    valid_days: int = 250,
    step_days: int = 250,
    max_windows: int | None = None,
    horizon_days_grid: tuple[int, ...] = (5, 10),
    drawdown_threshold_grid: tuple[float, ...] = (-0.02, -0.03, -0.05),
    return_threshold_grid: tuple[float, ...] = (-0.01, -0.02, -0.03),
    market_source_column: str = "synthetic_equal_weight_index",
    model_names: tuple[str, ...] = DEFAULT_TRADE_DAY_MODEL_NAMES,
    filter_rates: tuple[float, ...] = (0.2, 0.3),
    min_training_rows: int = 200,
    allow_short_sample: bool = False,
) -> TradeDayGateValidationResult:
    features = build_trade_day_feature_frame(
        storage=storage,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        min_stock_count=min_stock_count,
    )
    if features.empty:
        raise RuntimeError("Trade-day gate validation has no feature rows.")
    feature_columns = _available_feature_columns(features)
    if not feature_columns:
        raise RuntimeError("Trade-day gate validation has no usable feature columns.")
    feature_audit = build_trade_day_feature_audit(features, feature_columns=feature_columns)

    dataset_parts: list[pd.DataFrame] = []
    windows_parts: list[pd.DataFrame] = []
    metrics_parts: list[pd.DataFrame] = []
    decile_parts: list[pd.DataFrame] = []
    filter_parts: list[pd.DataFrame] = []
    for horizon_days in horizon_days_grid:
        for drawdown_threshold in drawdown_threshold_grid:
            for return_threshold in return_threshold_grid:
                config_id = _trade_day_config_id(
                    horizon_days=horizon_days,
                    drawdown_threshold=drawdown_threshold,
                    return_threshold=return_threshold,
                )
                logging.info("Trade-day gate validation config started: %s", config_id)
                dataset = add_trade_day_labels(
                    features,
                    horizon_days=horizon_days,
                    drawdown_threshold=drawdown_threshold,
                    return_threshold=return_threshold,
                    market_source_column=market_source_column,
                )
                dataset = dataset.dropna(subset=["bad_buy_day", *feature_columns]).copy()
                if dataset.empty:
                    logging.warning("Trade-day gate validation config has no labeled rows: %s", config_id)
                    continue
                windows = build_trade_day_walkforward_windows(
                    dataset,
                    train_days=train_days,
                    valid_days=valid_days,
                    step_days=step_days,
                    embargo_days=horizon_days,
                    max_windows=max_windows,
                )
                if windows.empty:
                    if allow_short_sample:
                        logging.warning("Trade-day gate validation config has no windows: %s", config_id)
                        continue
                    raise RuntimeError(f"No trade-day gate walk-forward windows can be built for {config_id}.")
                if not allow_short_sample and len(windows) < 3:
                    raise RuntimeError(
                        f"Trade-day gate validation needs at least 3 windows for {config_id}; got {len(windows)}."
                    )
                dataset_parts.append(dataset)
                windows = windows.copy()
                windows["config_id"] = config_id
                windows["horizon_days"] = int(horizon_days)
                windows["drawdown_threshold"] = float(drawdown_threshold)
                windows["return_threshold"] = float(return_threshold)
                windows_parts.append(windows)

                for window_index, window in enumerate(windows.to_dict("records"), start=1):
                    window_id = str(window["window_id"])
                    logging.info(
                        "Trade-day gate window %s/%s started: %s %s",
                        window_index,
                        len(windows),
                        config_id,
                        window_id,
                    )
                    train = _window_slice(dataset, start=window["train_start"], end=window["train_end"])
                    valid = _window_slice(dataset, start=window["valid_start"], end=window["valid_end"])
                    result = _fit_score_trade_day_models(
                        train=train,
                        valid=valid,
                        feature_columns=feature_columns,
                        model_names=model_names,
                        min_training_rows=min_training_rows,
                    )
                    metrics = _attach_window_config(result["metrics"], window)
                    scored = _attach_window_config(result["scored"], window)
                    metrics_parts.append(metrics)
                    decile_parts.append(build_trade_day_decile_report(scored))
                    filter_parts.append(build_trade_day_filter_impact(scored, filter_rates=filter_rates))
                    logging.info("Trade-day gate window complete: %s %s", config_id, window_id)

    dataset_frame = pd.concat(dataset_parts, ignore_index=True) if dataset_parts else pd.DataFrame()
    windows_frame = pd.concat(windows_parts, ignore_index=True) if windows_parts else pd.DataFrame()
    metrics_frame = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()
    deciles_frame = pd.concat(decile_parts, ignore_index=True) if decile_parts else pd.DataFrame()
    filter_impact_frame = pd.concat(filter_parts, ignore_index=True) if filter_parts else pd.DataFrame()
    summary = summarize_trade_day_gate(metrics_frame, deciles_frame, filter_impact_frame)

    report_dir = full_market_report_dir(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = report_dir / "trade_day_gate_dataset.csv"
    feature_audit_path = report_dir / "trade_day_gate_feature_audit.csv"
    windows_path = report_dir / "trade_day_gate_windows.csv"
    metrics_path = report_dir / "trade_day_gate_metrics.csv"
    deciles_path = report_dir / "trade_day_gate_decile_report.csv"
    filter_impact_path = report_dir / "trade_day_gate_filter_impact.csv"
    summary_path = report_dir / "trade_day_gate_summary.csv"
    config_path = report_dir / "trade_day_gate_config.json"
    dataset_frame.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    feature_audit.to_csv(feature_audit_path, index=False, encoding="utf-8-sig")
    windows_frame.to_csv(windows_path, index=False, encoding="utf-8-sig")
    metrics_frame.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    deciles_frame.to_csv(deciles_path, index=False, encoding="utf-8-sig")
    filter_impact_frame.to_csv(filter_impact_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    config_path.write_text(
        json.dumps(
            {
                "start_date": start_date.isoformat() if start_date else "",
                "end_date": end_date.isoformat() if end_date else "",
                "limit": limit,
                "min_stock_count": int(min_stock_count),
                "train_days": int(train_days),
                "valid_days": int(valid_days),
                "step_days": int(step_days),
                "max_windows": max_windows,
                "horizon_days_grid": [int(value) for value in horizon_days_grid],
                "drawdown_threshold_grid": [float(value) for value in drawdown_threshold_grid],
                "return_threshold_grid": [float(value) for value in return_threshold_grid],
                "market_source_column": market_source_column,
                "model_names": list(model_names),
                "filter_rates": [float(value) for value in filter_rates],
                "min_training_rows": int(min_training_rows),
                "allow_short_sample": bool(allow_short_sample),
                "feature_columns": list(feature_columns),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return TradeDayGateValidationResult(
        dataset=dataset_frame,
        feature_audit=feature_audit,
        windows=windows_frame,
        metrics=metrics_frame,
        deciles=deciles_frame,
        filter_impact=filter_impact_frame,
        summary=summary,
        report_dir=report_dir,
        dataset_path=dataset_path,
        feature_audit_path=feature_audit_path,
        windows_path=windows_path,
        metrics_path=metrics_path,
        deciles_path=deciles_path,
        filter_impact_path=filter_impact_path,
        summary_path=summary_path,
        config_path=config_path,
    )


def train_trade_day_gate_model(
    *,
    storage: Storage,
    project_root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    min_stock_count: int = 500,
    model_name: str = "naive_bayes",
    horizon_days: int = 10,
    drawdown_threshold: float = -0.02,
    return_threshold: float = -0.01,
    market_source_column: str = "synthetic_equal_weight_index",
    block_rate: float = 0.2,
    min_training_rows: int = 200,
) -> TradeDayGateTrainResult:
    features = build_trade_day_feature_frame(
        storage=storage,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        min_stock_count=min_stock_count,
    )
    feature_columns = _available_feature_columns(features)
    dataset = add_trade_day_labels(
        features,
        horizon_days=horizon_days,
        drawdown_threshold=drawdown_threshold,
        return_threshold=return_threshold,
        market_source_column=market_source_column,
    )
    dataset = dataset.dropna(subset=["bad_buy_day", *feature_columns]).copy()
    if len(dataset) < min_training_rows:
        raise RuntimeError(f"Insufficient trade-day gate training rows: {len(dataset)}")
    if dataset["bad_buy_day"].astype(int).nunique() < 2 and model_name not in {"always_allow", "rule_market_gate"}:
        raise RuntimeError("Trade-day gate model training requires both bad and good buy-day labels.")
    fitted = _fit_trade_day_model(dataset, feature_columns=feature_columns, model_name=model_name)
    train_score = _score_trade_day_model(dataset, model=fitted, feature_columns=feature_columns, model_name=model_name)
    selected_threshold = float(pd.Series(train_score).quantile(1.0 - float(block_rate)))
    trade_dates = pd.to_datetime(dataset["trade_date"], errors="coerce").dropna()
    train_start = trade_dates.min().date().isoformat()
    train_end = trade_dates.max().date().isoformat()
    artifact = {
        "model_version": TRADE_DAY_GATE_MODEL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "model": fitted,
        "feature_columns": tuple(feature_columns),
        "label_config": {
            "horizon_days": int(horizon_days),
            "drawdown_threshold": float(drawdown_threshold),
            "return_threshold": float(return_threshold),
            "market_source_column": market_source_column,
        },
        "train_config": {
            "start_date": start_date.isoformat() if start_date else "",
            "end_date": end_date.isoformat() if end_date else "",
            "limit": limit,
            "min_stock_count": int(min_stock_count),
            "min_training_rows": int(min_training_rows),
            "block_rate": float(block_rate),
        },
        "train_rows": int(len(dataset)),
        "train_start": train_start,
        "train_end": train_end,
        "bad_day_rate": float(dataset["bad_buy_day"].mean()),
        "selected_threshold": selected_threshold,
        "allowed_day_coverage_on_training": float(pd.Series(train_score).lt(selected_threshold).mean()),
    }
    model_path, metadata_path = save_trade_day_gate_model_artifact(project_root, artifact)
    return TradeDayGateTrainResult(
        model_path=model_path,
        metadata_path=metadata_path,
        model_name=model_name,
        train_rows=int(len(dataset)),
        train_start=train_start,
        train_end=train_end,
        selected_threshold=selected_threshold,
        feature_columns=tuple(feature_columns),
    )


def predict_trade_day_gate(
    *,
    storage: Storage,
    project_root: Path,
    trade_date: date,
    output: Path | None = None,
    limit: int | None = None,
) -> TradeDayGatePredictionResult:
    artifact = load_trade_day_gate_model_artifact(project_root)
    train_config = artifact.get("train_config", {})
    features = build_trade_day_feature_frame(
        storage=storage,
        start_date=None,
        end_date=trade_date,
        limit=limit if limit is not None else train_config.get("limit"),
        min_stock_count=int(train_config.get("min_stock_count", 500)),
    )
    feature_columns = tuple(artifact["feature_columns"])
    row = _latest_trade_day_prediction_row(features, trade_date)
    if row.empty:
        raise RuntimeError(f"No trade-day gate feature row on or before {trade_date.isoformat()}")
    row = row.tail(1).dropna(subset=list(feature_columns)).copy()
    if row.empty:
        raise RuntimeError(f"Trade-day gate feature row has missing model features for {trade_date.isoformat()}")
    feature_trade_date = pd.Timestamp(row.iloc[0]["trade_date"]).date().isoformat()
    score = float(
        _score_trade_day_model(
            row,
            model=artifact.get("model"),
            feature_columns=feature_columns,
            model_name=str(artifact["model_name"]),
        )[0]
    )
    threshold = float(artifact["selected_threshold"])
    permission = "allow" if score < threshold else "no_trade"
    reason = "buy_day_risk_score_below_threshold" if permission == "allow" else "buy_day_risk_score_ge_threshold"
    record: dict[str, Any] = {
        "trade_date": trade_date.isoformat(),
        "feature_trade_date": feature_trade_date,
        "buy_day_risk_score": score,
        "selected_threshold": threshold,
        "trade_permission": permission,
        "suggested_action": "candidate_allowed" if permission == "allow" else "observation_only",
        "reason": reason,
        "model_name": artifact["model_name"],
        "model_version": artifact["model_version"],
    }
    for column in feature_columns:
        record[column] = float(row.iloc[0][column])
    prediction = pd.DataFrame([record])
    output_path = output if output is not None else trade_day_gate_prediction_path(project_root, trade_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output_path, index=False, encoding="utf-8-sig")
    return TradeDayGatePredictionResult(
        prediction=prediction,
        output_path=output_path,
        artifact_path=trade_day_gate_model_path(project_root),
    )


def build_trade_day_walkforward_windows(
    dataset: pd.DataFrame,
    *,
    train_days: int,
    valid_days: int,
    step_days: int,
    embargo_days: int,
    max_windows: int | None = None,
) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    trade_dates = pd.Series(pd.to_datetime(dataset["trade_date"], errors="coerce").dropna().unique()).sort_values().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    start_index = 0
    while True:
        train_start_index = start_index
        train_end_index = train_start_index + int(train_days) - 1
        valid_start_index = train_end_index + 1 + max(int(embargo_days), 0)
        valid_end_index = valid_start_index + int(valid_days) - 1
        if valid_end_index >= len(trade_dates):
            break
        rows.append(
            {
                "window_id": f"wf_{len(rows) + 1:02d}",
                "train_start": pd.Timestamp(trade_dates.iloc[train_start_index]).date().isoformat(),
                "train_end": pd.Timestamp(trade_dates.iloc[train_end_index]).date().isoformat(),
                "valid_start": pd.Timestamp(trade_dates.iloc[valid_start_index]).date().isoformat(),
                "valid_end": pd.Timestamp(trade_dates.iloc[valid_end_index]).date().isoformat(),
                "train_days": int(train_days),
                "valid_days": int(valid_days),
                "embargo_days": int(embargo_days),
            }
        )
        if max_windows is not None and len(rows) >= max_windows:
            break
        start_index += int(step_days)
    return pd.DataFrame(rows)


def build_trade_day_decile_report(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = [column for column in ("config_id", "window_id", "model_name") if column in scored.columns]
    for key, group in scored.groupby(group_columns, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        key_map = dict(zip(group_columns, key_values, strict=False))
        frame = group.copy()
        frame["buy_day_risk_decile"] = pd.qcut(frame["buy_day_risk_score"].rank(method="first"), 10, labels=False, duplicates="drop")
        for decile, decile_frame in frame.groupby("buy_day_risk_decile", sort=True):
            rows.append(
                {
                    **key_map,
                    "buy_day_risk_decile": int(decile),
                    "rows": int(len(decile_frame)),
                    "bad_buy_day_rate": _safe_mean(decile_frame["bad_buy_day"]),
                    "avg_future_market_return": _safe_mean(decile_frame["future_market_return"]),
                    "avg_future_market_max_drawdown": _safe_mean(decile_frame["future_market_max_drawdown"]),
                }
            )
    return pd.DataFrame(rows)


def build_trade_day_filter_impact(scored: pd.DataFrame, *, filter_rates: tuple[float, ...] = (0.2, 0.3)) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = [column for column in ("config_id", "window_id", "model_name") if column in scored.columns]
    for key, group in scored.groupby(group_columns, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        key_map = dict(zip(group_columns, key_values, strict=False))
        frame = group.dropna(subset=["buy_day_risk_score", "bad_buy_day", "future_market_return", "future_market_max_drawdown"]).copy()
        if frame.empty:
            continue
        ordered = frame.sort_values(["buy_day_risk_score", "trade_date"], ascending=[False, True])
        baseline_bad = _safe_mean(ordered["bad_buy_day"])
        baseline_return = _safe_mean(ordered["future_market_return"])
        baseline_drawdown = _safe_mean(ordered["future_market_max_drawdown"])
        for filter_rate in filter_rates:
            rate = float(filter_rate)
            blocked_rows = max(1, int(np.ceil(len(ordered) * rate)))
            blocked = ordered.iloc[:blocked_rows].copy()
            allowed = ordered.iloc[blocked_rows:].copy()
            chronological_dates = frame.sort_values("trade_date")["trade_date"]
            rows.append(
                {
                    **key_map,
                    "filter_rate": rate,
                    "rows": int(len(ordered)),
                    "allowed_rows": int(len(allowed)),
                    "blocked_rows": int(len(blocked)),
                    "allowed_day_coverage": float(len(allowed) / len(ordered)),
                    "blocked_day_coverage": float(len(blocked) / len(ordered)),
                    "baseline_bad_buy_day_rate": baseline_bad,
                    "bad_buy_day_rate_allowed": _safe_mean(allowed["bad_buy_day"]),
                    "bad_buy_day_rate_blocked": _safe_mean(blocked["bad_buy_day"]),
                    "future_return_mean_baseline": baseline_return,
                    "future_return_mean_allowed": _safe_mean(allowed["future_market_return"]),
                    "future_return_mean_blocked": _safe_mean(blocked["future_market_return"]),
                    "future_return_delta_allowed": _safe_mean(allowed["future_market_return"]) - baseline_return,
                    "future_max_drawdown_mean_baseline": baseline_drawdown,
                    "future_max_drawdown_mean_allowed": _safe_mean(allowed["future_market_max_drawdown"]),
                    "future_max_drawdown_mean_blocked": _safe_mean(blocked["future_market_max_drawdown"]),
                    "future_max_drawdown_delta_allowed": _safe_mean(allowed["future_market_max_drawdown"]) - baseline_drawdown,
                    "max_consecutive_no_trade_days": _max_consecutive_blocked(chronological_dates, blocked["trade_date"]),
                }
            )
    return pd.DataFrame(rows)


def summarize_trade_day_gate(metrics: pd.DataFrame, deciles: pd.DataFrame, filter_impact: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (config_id, model_name), group in metrics.groupby(["config_id", "model_name"], sort=False):
        successful = group[group["error"].fillna("").eq("")].copy()
        decile_checks = _trade_day_decile_checks(deciles, config_id=str(config_id), model_name=str(model_name))
        filter_group = (
            filter_impact[
                filter_impact["config_id"].astype(str).eq(str(config_id))
                & filter_impact["model_name"].astype(str).eq(str(model_name))
            ]
            if not filter_impact.empty
            else pd.DataFrame()
        )
        rows.append(
            {
                "config_id": config_id,
                "model_name": model_name,
                "windows": int(group["window_id"].nunique()) if "window_id" in group.columns else int(len(group)),
                "successful_windows": int(len(successful)),
                "avg_pr_auc": _safe_mean(successful.get("pr_auc", pd.Series(dtype=float))),
                "avg_pr_auc_baseline": _safe_mean(successful.get("pr_auc_baseline", pd.Series(dtype=float))),
                "avg_roc_auc": _safe_mean(successful.get("roc_auc", pd.Series(dtype=float))),
                "pr_auc_beat_baseline_rate": _mean_bool(successful["pr_auc"].gt(successful["pr_auc_baseline"])) if not successful.empty else 0.0,
                "top_decile_higher_bad_day_rate": _safe_mean(decile_checks.get("top_decile_higher_bad_day", pd.Series(dtype=float))),
                "top_decile_worse_drawdown_rate": _safe_mean(decile_checks.get("top_decile_worse_drawdown", pd.Series(dtype=float))),
                "avg_allowed_day_coverage": _safe_mean(filter_group.get("allowed_day_coverage", pd.Series(dtype=float))),
                "avg_future_return_delta_allowed": _safe_mean(filter_group.get("future_return_delta_allowed", pd.Series(dtype=float))),
                "avg_future_max_drawdown_delta_allowed": _safe_mean(filter_group.get("future_max_drawdown_delta_allowed", pd.Series(dtype=float))),
                "phase7_pass": _phase7_pass(successful, decile_checks, filter_group),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["phase7_pass", "avg_future_max_drawdown_delta_allowed", "avg_pr_auc"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return summary


def trade_day_gate_model_dir(project_root: Path) -> Path:
    return project_root / "data" / "ml" / "full_market_trade_day_gate"


def trade_day_gate_model_path(project_root: Path) -> Path:
    return trade_day_gate_model_dir(project_root) / "trade_day_gate_model.pkl"


def trade_day_gate_metadata_path(project_root: Path) -> Path:
    return trade_day_gate_model_dir(project_root) / "trade_day_gate_model_metadata.json"


def save_trade_day_gate_model_artifact(project_root: Path, artifact: dict[str, Any]) -> tuple[Path, Path]:
    model_dir = trade_day_gate_model_dir(project_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = trade_day_gate_model_path(project_root)
    metadata_path = trade_day_gate_metadata_path(project_root)
    with model_path.open("wb") as file:
        pickle.dump(artifact, file)
    metadata = {key: value for key, value in artifact.items() if key != "model"}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_path, metadata_path


def load_trade_day_gate_model_artifact(project_root: Path) -> dict[str, Any]:
    model_path = trade_day_gate_model_path(project_root)
    if not model_path.exists():
        raise FileNotFoundError(f"Trade-day gate model artifact not found: {model_path}")
    with model_path.open("rb") as file:
        return pickle.load(file)


def trade_day_gate_prediction_path(project_root: Path, trade_date: date) -> Path:
    return full_market_report_dir(project_root) / f"trade_day_gate_prediction_{trade_date.isoformat()}.csv"


def format_trade_day_gate_prediction_table(prediction: pd.DataFrame) -> str:
    if prediction.empty:
        return "No trade-day gate prediction."
    columns = [
        "trade_date",
        "feature_trade_date",
        "buy_day_risk_score",
        "selected_threshold",
        "trade_permission",
        "suggested_action",
        "reason",
        "model_name",
    ]
    frame = prediction.loc[:, [column for column in columns if column in prediction.columns]].copy()
    for column in ("buy_day_risk_score", "selected_threshold"):
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: f"{float(value):.6f}")
    return frame.to_string(index=False)


def _symbol_trade_day_frame(bars: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    frame = bars.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        values = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
        frame[column] = pd.to_numeric(values, errors="coerce")
    close = frame["close"].where(frame["close"].gt(0))
    prev_close = close.shift(1)
    daily_return = close.div(prev_close).sub(1.0).replace([np.inf, -np.inf], np.nan)
    return_5d = close.pct_change(5, fill_method=None)
    return_20d = close.pct_change(20, fill_method=None)
    log_return = np.log(close / prev_close).replace([np.inf, -np.inf], np.nan)
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    ma120 = close.rolling(120, min_periods=120).mean()
    high20 = frame["high"].rolling(20, min_periods=20).max()
    low20 = frame["low"].rolling(20, min_periods=20).min()
    high60 = frame["high"].rolling(60, min_periods=60).max()
    low60 = frame["low"].rolling(60, min_periods=60).min()
    rolling_max20 = close.rolling(20, min_periods=20).max()
    locked = frame["high"].sub(frame["low"]).abs().le((close.abs() * 0.0005).fillna(0.0))
    result = pd.DataFrame(
        {
            "trade_date": frame["trade_date"],
            "symbol": str(symbol).zfill(6),
            "daily_return": daily_return,
            "return_5d": return_5d,
            "return_20d": return_20d,
            "amount": frame["amount"],
            "above_ma20": close.gt(ma20),
            "above_ma60": close.gt(ma60),
            "above_ma120": close.gt(ma120),
            "new_high_20d": close.ge(high20),
            "new_low_20d": close.le(low20),
            "new_high_60d": close.ge(high60),
            "new_low_60d": close.le(low60),
            "limit_up_like": locked & daily_return.ge(0.095),
            "limit_down_like": locked & daily_return.le(-0.095),
            "volatility_20d": log_return.rolling(20, min_periods=20).std(),
            "max_drawdown_20d": close.div(rolling_max20).sub(1.0),
        }
    )
    return result.dropna(subset=["daily_return"]).replace([np.inf, -np.inf], np.nan)


def _add_index_features(frame: pd.DataFrame, *, source_column: str, prefix: str) -> None:
    index_value = pd.to_numeric(frame[source_column], errors="coerce")
    log_return = np.log(index_value / index_value.shift(1)).replace([np.inf, -np.inf], np.nan)
    for window in (5, 20, 60):
        frame[f"{prefix}_return_{window}d"] = index_value.pct_change(window, fill_method=None)
    frame[f"{prefix}_above_ma20"] = index_value.gt(index_value.rolling(20, min_periods=20).mean()).astype(float)
    frame[f"{prefix}_above_ma60"] = index_value.gt(index_value.rolling(60, min_periods=60).mean()).astype(float)
    frame[f"{prefix}_above_ma120"] = index_value.gt(index_value.rolling(120, min_periods=120).mean()).astype(float)
    ma20 = index_value.rolling(20, min_periods=20).mean()
    ma60 = index_value.rolling(60, min_periods=60).mean()
    frame[f"{prefix}_ma20_slope_5d"] = ma20.div(ma20.shift(5)).sub(1.0)
    frame[f"{prefix}_ma60_slope_10d"] = ma60.div(ma60.shift(10)).sub(1.0)
    frame[f"{prefix}_drawdown_20d"] = index_value.div(index_value.rolling(20, min_periods=20).max()).sub(1.0)
    frame[f"{prefix}_drawdown_60d"] = index_value.div(index_value.rolling(60, min_periods=60).max()).sub(1.0)
    frame[f"{prefix}_volatility_20d"] = log_return.rolling(20, min_periods=20).std()
    frame[f"{prefix}_volatility_60d"] = log_return.rolling(60, min_periods=60).std()


def _fit_score_trade_day_models(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: tuple[str, ...],
    model_names: tuple[str, ...],
    min_training_rows: int,
) -> dict[str, pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    scored_parts: list[pd.DataFrame] = []
    if len(train) < min_training_rows or valid.empty:
        return {
            "metrics": _empty_trade_day_metric_rows(model_names, train_rows=len(train), valid_rows=len(valid), error="insufficient_window_rows"),
            "scored": pd.DataFrame(),
        }
    for model_name in model_names:
        try:
            model = _fit_trade_day_model(train, feature_columns=feature_columns, model_name=model_name)
            score = _score_trade_day_model(valid, model=model, feature_columns=feature_columns, model_name=model_name)
            metrics_rows.append(
                _trade_day_metric_row(
                    valid["bad_buy_day"].astype(int),
                    score,
                    model_name=model_name,
                    feature_count=len(feature_columns),
                    train_rows=len(train),
                )
            )
            scored = valid.loc[
                :,
                [
                    "trade_date",
                    "bad_buy_day",
                    "future_market_return",
                    "future_market_max_drawdown",
                    "config_id",
                    "label_horizon_days",
                    "drawdown_threshold",
                    "return_threshold",
                ],
            ].copy()
            scored["model_name"] = model_name
            scored["buy_day_risk_score"] = score
            scored_parts.append(scored)
        except Exception as exc:
            metrics_rows.append(
                {
                    "model_name": model_name,
                    "feature_count": int(len(feature_columns)),
                    "train_rows": int(len(train)),
                    "rows": int(len(valid)),
                    "bad_day_rate": _safe_mean(valid.get("bad_buy_day", pd.Series(dtype=float))),
                    "accuracy": np.nan,
                    "bad_day_precision": np.nan,
                    "bad_day_recall": np.nan,
                    "bad_day_f1": np.nan,
                    "roc_auc": np.nan,
                    "pr_auc": np.nan,
                    "pr_auc_baseline": _safe_mean(valid.get("bad_buy_day", pd.Series(dtype=float))),
                    "brier": np.nan,
                    "true_negative": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "true_positive": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            logging.warning("Trade-day gate model failed: %s error=%s", model_name, exc)
    return {
        "metrics": pd.DataFrame(metrics_rows),
        "scored": pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame(),
    }


def _fit_trade_day_model(frame: pd.DataFrame, *, feature_columns: tuple[str, ...], model_name: str) -> Any:
    if model_name in {"always_allow", "rule_market_gate"}:
        return None
    model = _trade_day_models((model_name,))[model_name]
    X = frame.loc[:, feature_columns]
    y = frame["bad_buy_day"].astype(int)
    if y.nunique() < 2:
        raise RuntimeError("Trade-day gate training requires both label classes.")
    return model.fit(X, y)


def _score_trade_day_model(
    frame: pd.DataFrame,
    *,
    model: Any,
    feature_columns: tuple[str, ...],
    model_name: str,
) -> np.ndarray:
    if model_name == "always_allow":
        return np.zeros(len(frame), dtype=float)
    if model_name == "rule_market_gate":
        return _rule_market_gate_score(frame)
    X = frame.loc[:, feature_columns]
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "steps"):
            classes = getattr(model.steps[-1][1], "classes_", None)
        if classes is not None and int(classes[0]) == 1:
            return np.ones(len(frame), dtype=float)
        return np.zeros(len(frame), dtype=float)
    return proba[:, 1]


def _trade_day_models(model_names: tuple[str, ...]) -> dict[str, Any]:
    available: dict[str, Any] = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "linear_discriminant_analysis": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "naive_bayes": GaussianNB(),
    }
    if LGBMClassifier is not None:
        available["lightgbm_classifier"] = LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )
    unknown = [model_name for model_name in model_names if model_name not in available]
    if unknown:
        raise ValueError(f"Unknown trade-day gate model names: {', '.join(unknown)}")
    return {model_name: available[model_name] for model_name in model_names}


def _rule_market_gate_score(frame: pd.DataFrame) -> np.ndarray:
    allow = (
        pd.to_numeric(frame.get("equal_above_ma20", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0).gt(0)
        & pd.to_numeric(frame.get("equal_ma20_slope_5d", pd.Series(np.nan, index=frame.index)), errors="coerce").gt(0)
        & pd.to_numeric(frame.get("breadth_above_ma20", pd.Series(np.nan, index=frame.index)), errors="coerce").ge(0.45)
        & pd.to_numeric(frame.get("limit_down_ratio", pd.Series(np.nan, index=frame.index)), errors="coerce").le(0.03)
    )
    return np.where(allow.to_numpy(dtype=bool), 0.0, 1.0)


def _trade_day_metric_row(y_true: pd.Series, score: np.ndarray, *, model_name: str, feature_count: int, train_rows: int) -> dict[str, Any]:
    pred = score >= 0.5
    y = y_true.astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred.astype(int), labels=[0, 1]).ravel()
    return {
        "model_name": model_name,
        "feature_count": int(feature_count),
        "train_rows": int(train_rows),
        "rows": int(len(y)),
        "bad_day_rate": float(y.mean()) if len(y) else np.nan,
        "accuracy": float(accuracy_score(y, pred)) if len(y) else np.nan,
        "bad_day_precision": float(precision_score(y, pred, zero_division=0)),
        "bad_day_recall": float(recall_score(y, pred, zero_division=0)),
        "bad_day_f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y, score),
        "pr_auc": _safe_pr_auc(y, score),
        "pr_auc_baseline": float(y.mean()) if len(y) else np.nan,
        "brier": float(brier_score_loss(y, np.clip(score, 0.0, 1.0))) if len(y) else np.nan,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "error": "",
    }


def _empty_trade_day_metric_rows(model_names: tuple[str, ...], *, train_rows: int, valid_rows: int, error: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": model_name,
                "feature_count": 0,
                "train_rows": int(train_rows),
                "rows": int(valid_rows),
                "bad_day_rate": np.nan,
                "accuracy": np.nan,
                "bad_day_precision": np.nan,
                "bad_day_recall": np.nan,
                "bad_day_f1": np.nan,
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "pr_auc_baseline": np.nan,
                "brier": np.nan,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0,
                "true_positive": 0,
                "error": error,
            }
            for model_name in model_names
        ]
    )


def _attach_window_config(frame: pd.DataFrame, window: dict[str, Any]) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    for key, value in window.items():
        result[key] = value
    return result


def _window_slice(dataset: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(dataset["trade_date"], errors="coerce").dt.date
    return dataset[(dates >= date.fromisoformat(str(start))) & (dates <= date.fromisoformat(str(end)))].copy()


def _trade_day_decile_checks(deciles: pd.DataFrame, *, config_id: str, model_name: str) -> pd.DataFrame:
    if deciles.empty:
        return pd.DataFrame()
    frame = deciles[
        deciles["config_id"].astype(str).eq(config_id)
        & deciles["model_name"].astype(str).eq(model_name)
    ].copy()
    rows = []
    for window_id, group in frame.groupby("window_id", sort=False):
        bottom = group[group["buy_day_risk_decile"].eq(group["buy_day_risk_decile"].min())]
        top = group[group["buy_day_risk_decile"].eq(group["buy_day_risk_decile"].max())]
        if bottom.empty or top.empty:
            continue
        rows.append(
            {
                "window_id": window_id,
                "top_decile_higher_bad_day": bool(top.iloc[0]["bad_buy_day_rate"] > bottom.iloc[0]["bad_buy_day_rate"]),
                "top_decile_worse_drawdown": bool(
                    top.iloc[0]["avg_future_market_max_drawdown"] < bottom.iloc[0]["avg_future_market_max_drawdown"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _phase7_pass(metrics: pd.DataFrame, decile_checks: pd.DataFrame, filter_group: pd.DataFrame) -> bool:
    if metrics.empty or decile_checks.empty or filter_group.empty:
        return False
    pr_rate = _mean_bool(metrics["pr_auc"].gt(metrics["pr_auc_baseline"]))
    bad_rate = _mean_bool(decile_checks["top_decile_higher_bad_day"])
    drawdown_decile_rate = _mean_bool(decile_checks["top_decile_worse_drawdown"])
    coverage_ok = _safe_mean(filter_group["allowed_day_coverage"]) >= 0.5
    drawdown_ok = _safe_mean(filter_group["future_max_drawdown_delta_allowed"]) > 0
    return_ok = _safe_mean(filter_group["future_return_delta_allowed"]) >= -0.001
    return bool(pr_rate >= 0.70 and bad_rate >= 0.70 and drawdown_decile_rate >= 0.70 and coverage_ok and drawdown_ok and return_ok)


def _available_feature_columns(features: pd.DataFrame) -> tuple[str, ...]:
    columns = [column for column in TRADE_DAY_FEATURE_COLUMNS if column in features.columns]
    usable = []
    for column in columns:
        values = pd.to_numeric(features[column], errors="coerce")
        if values.notna().any():
            usable.append(column)
    return tuple(usable)


def _latest_trade_day_prediction_row(features: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    if features.empty or "trade_date" not in features.columns:
        return pd.DataFrame()
    dates = pd.to_datetime(features["trade_date"], errors="coerce")
    eligible = features[dates.dt.date <= trade_date].copy()
    if eligible.empty:
        return pd.DataFrame()
    return eligible.sort_values("trade_date").tail(1).copy()


def _future_min_return(index_value: pd.Series, *, horizon: int) -> pd.Series:
    values = []
    for offset in range(1, int(horizon) + 1):
        values.append(index_value.shift(-offset).div(index_value).sub(1.0))
    return pd.concat(values, axis=1).min(axis=1) if values else pd.Series(np.nan, index=index_value.index)


def _weighted_return(*, daily_return: pd.Series, amount: pd.Series) -> float:
    amount_sum = float(amount.sum())
    if amount_sum > 0:
        return float((daily_return * amount).sum() / amount_sum)
    return float(daily_return.mean())


def _share(values: pd.Series) -> float:
    if values.empty:
        return np.nan
    return float(pd.Series(values).fillna(False).astype(bool).mean())


def _median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else np.nan


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else np.nan


def _mean_bool(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(values.fillna(False).astype(bool).mean())


def _safe_roc_auc(y_true: pd.Series, score: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, score))
    except ValueError:
        return np.nan


def _safe_pr_auc(y_true: pd.Series, score: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return float(y_true.mean()) if len(y_true) else np.nan
    try:
        return float(average_precision_score(y_true, score))
    except ValueError:
        return np.nan


def _max_consecutive_blocked(all_dates: pd.Series, blocked_dates: pd.Series) -> int:
    blocked = set(pd.to_datetime(blocked_dates, errors="coerce").dropna().dt.normalize())
    max_run = 0
    current = 0
    for value in pd.to_datetime(all_dates, errors="coerce").dropna().dt.normalize():
        if value in blocked:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return int(max_run)


def _trade_day_config_id(*, horizon_days: int, drawdown_threshold: float, return_threshold: float) -> str:
    return f"h{int(horizon_days)}_dd{abs(float(drawdown_threshold)):g}_ret{abs(float(return_threshold)):g}"


def _log_progress(stage_name: str, current: int, total: int) -> None:
    if total <= 0:
        return
    if current == 1 or current % TRADE_DAY_PROGRESS_INTERVAL == 0 or current == total:
        logging.info("%s progress: %s/%s", stage_name, current, total)
