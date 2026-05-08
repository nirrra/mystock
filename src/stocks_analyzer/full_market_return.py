from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # LightGBM is the reference model for Qlib Alpha158 benchmark.
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - depends on optional local package.
    LGBMRegressor = None

from .full_market_alpha158 import build_alpha158_feature_frame, build_alpha158_return_panel
from .full_market_panel import full_market_report_dir
from .storage import DailyBarsReadError, Storage


QLIB_ALPHA158_LGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "learning_rate": 0.2,
    "max_depth": 8,
    "num_leaves": 210,
    "colsample_bytree": 0.8879,
    "subsample": 0.8789,
    "reg_alpha": 205.6999,
    "reg_lambda": 580.9768,
    "n_jobs": 1,
    "random_state": 42,
    "verbosity": -1,
}


@dataclass(slots=True)
class Alpha158QlibReturnValidationResult:
    feature_audit: pd.DataFrame
    signal_metrics: pd.DataFrame
    daily_ic: pd.DataFrame
    deciles: pd.DataFrame
    topk_daily: pd.DataFrame
    topk_summary: pd.DataFrame
    report_dir: Path
    feature_audit_path: Path
    signal_metrics_path: Path
    daily_ic_path: Path
    deciles_path: Path
    topk_daily_path: Path
    topk_summary_path: Path
    config_path: Path


@dataclass(slots=True)
class Alpha158QlibReturnTrainResult:
    model_path: Path
    metadata_path: Path
    train_rows: int
    train_start: str
    train_end: str
    feature_columns: tuple[str, ...]


@dataclass(slots=True)
class Alpha158QlibReturnPredictionResult:
    predictions: pd.DataFrame
    skipped: pd.DataFrame
    output_path: Path
    artifact_path: Path


def train_alpha158_qlib_return_model(
    *,
    storage: Storage,
    project_root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    min_training_rows: int = 200,
) -> Alpha158QlibReturnTrainResult:
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is required for Qlib Alpha158 return deployment training.")
    logging.info("Alpha158 Qlib return deployment panel build started")
    panel = build_alpha158_return_panel(storage=storage, start_date=start_date, end_date=end_date, limit=limit)
    dataset = panel.dataset
    if len(dataset) < min_training_rows:
        raise RuntimeError(f"Insufficient Alpha158 Qlib return training rows: {len(dataset)}")
    if not panel.feature_columns:
        raise RuntimeError("Alpha158 Qlib return training has no feature columns.")

    logging.info(
        "Alpha158 Qlib return deployment model fit started: rows=%s features=%s",
        len(dataset),
        len(panel.feature_columns),
    )
    model = LGBMRegressor(**QLIB_ALPHA158_LGBM_PARAMS)
    fitted = model.fit(dataset.loc[:, panel.feature_columns], dataset["LABEL0"])

    trade_dates = pd.to_datetime(dataset["trade_date"], errors="coerce").dropna()
    train_start = trade_dates.min().date().isoformat()
    train_end = trade_dates.max().date().isoformat()
    artifact = {
        "model_version": "alpha158_qlib_return_phase4_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": "lightgbm_regressor",
        "model": fitted,
        "feature_columns": tuple(panel.feature_columns),
        "reference": "Qlib examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml",
        "handler": "Qlib Alpha158",
        "label": "Ref($close, -2)/Ref($close, -1) - 1",
        "learn_processors": ["DropnaLabel", "CSZScoreNorm(label)"],
        "feature_processors": [],
        "local_deviations": [
            "local parquet daily bars are used instead of Qlib binary provider",
            "local universe is current repository universe instead of csi300",
        ],
        "lightgbm_params": QLIB_ALPHA158_LGBM_PARAMS,
        "train_config": {
            "start_date": start_date.isoformat() if start_date else "",
            "end_date": end_date.isoformat() if end_date else "",
            "limit": limit,
            "min_training_rows": int(min_training_rows),
        },
        "train_rows": int(len(dataset)),
        "train_start": train_start,
        "train_end": train_end,
        "feature_count": int(len(panel.feature_columns)),
        "skipped_symbols": int(len(panel.skipped)),
    }
    model_path, metadata_path = save_alpha158_qlib_return_model_artifact(project_root, artifact)
    logging.info("Alpha158 Qlib return deployment model saved: %s", model_path)
    return Alpha158QlibReturnTrainResult(
        model_path=model_path,
        metadata_path=metadata_path,
        train_rows=int(len(dataset)),
        train_start=train_start,
        train_end=train_end,
        feature_columns=tuple(panel.feature_columns),
    )


def alpha158_qlib_return_model_dir(project_root: Path) -> Path:
    return project_root / "data" / "ml" / "full_market_alpha158_return"


def alpha158_qlib_return_model_path(project_root: Path) -> Path:
    return alpha158_qlib_return_model_dir(project_root) / "alpha158_qlib_return_model.pkl"


def alpha158_qlib_return_metadata_path(project_root: Path) -> Path:
    return alpha158_qlib_return_model_dir(project_root) / "alpha158_qlib_return_model_metadata.json"


def save_alpha158_qlib_return_model_artifact(project_root: Path, artifact: dict[str, Any]) -> tuple[Path, Path]:
    model_dir = alpha158_qlib_return_model_dir(project_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = alpha158_qlib_return_model_path(project_root)
    metadata_path = alpha158_qlib_return_metadata_path(project_root)
    with model_path.open("wb") as file:
        pickle.dump(artifact, file)
    metadata = {key: value for key, value in artifact.items() if key != "model"}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_path, metadata_path


def load_alpha158_qlib_return_model_artifact(project_root: Path) -> dict[str, Any]:
    model_path = alpha158_qlib_return_model_path(project_root)
    if not model_path.exists():
        raise FileNotFoundError(f"Alpha158 Qlib return model artifact not found: {model_path}")
    with model_path.open("rb") as file:
        artifact = pickle.load(file)
    return artifact


def predict_alpha158_qlib_return(
    *,
    storage: Storage,
    project_root: Path,
    trade_date: date,
    output: Path | None = None,
    limit: int | None = None,
) -> Alpha158QlibReturnPredictionResult:
    artifact = load_alpha158_qlib_return_model_artifact(project_root)
    feature_columns = tuple(artifact["feature_columns"])
    universe = storage.load_universe().copy()
    if limit is not None:
        universe = universe.head(max(int(limit), 0)).copy()

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, object]] = []
    instruments = universe.to_dict("records")
    total_symbols = len(instruments)
    for index, instrument in enumerate(instruments, start=1):
        _log_prediction_progress("Alpha158 Qlib return prediction", index, total_symbols)
        symbol = str(instrument.get("symbol", "")).zfill(6)
        name = str(instrument.get("name", ""))
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError) as exc:
            skipped.append({"symbol": symbol, "name": name, "reason": type(exc).__name__})
            continue
        bars = bars.copy()
        bars["trade_date"] = pd.to_datetime(bars["trade_date"], errors="coerce")
        bars = bars.dropna(subset=["trade_date"])
        bars = bars[bars["trade_date"].dt.date <= trade_date].copy()
        frame = build_alpha158_feature_frame(bars, symbol=symbol, name=name)
        if frame.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "empty_alpha158_feature_frame"})
            continue
        row = _latest_prediction_row(frame, trade_date)
        if row.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "no_feature_date_on_or_before_trade_date"})
            continue
        row = row.dropna(subset=list(feature_columns))
        if row.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "no_feature_row"})
            continue
        feature_trade_date = pd.Timestamp(row.iloc[0]["trade_date"]).date().isoformat()
        return_score = float(artifact["model"].predict(row.loc[:, feature_columns])[0])
        record: dict[str, Any] = {
            "trade_date": trade_date.isoformat(),
            "feature_trade_date": feature_trade_date,
            "symbol": symbol,
            "name": name,
            "return_score": return_score,
            "prediction_scope": "full_market_daily",
            "model_name": artifact["model_name"],
            "model_version": artifact["model_version"],
        }
        for column in feature_columns:
            record[column] = float(row.iloc[0][column])
        rows.append(record)

    predictions = pd.DataFrame(rows)
    if not predictions.empty:
        predictions = predictions.sort_values(["return_score", "symbol"], ascending=[False, True]).reset_index(drop=True)
    skipped_frame = pd.DataFrame(skipped)
    output_path = output if output is not None else alpha158_qlib_return_predictions_path(project_root, trade_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
    skipped_path = output_path.with_name(f"{output_path.stem}_skipped.csv")
    skipped_frame.to_csv(skipped_path, index=False, encoding="utf-8-sig")
    logging.info(
        "Alpha158 Qlib return predictions saved: rows=%s skipped=%s output=%s",
        len(predictions),
        len(skipped_frame),
        output_path,
    )
    return Alpha158QlibReturnPredictionResult(
        predictions=predictions,
        skipped=skipped_frame,
        output_path=output_path,
        artifact_path=alpha158_qlib_return_model_path(project_root),
    )


def alpha158_qlib_return_predictions_path(project_root: Path, trade_date: date) -> Path:
    return full_market_report_dir(project_root) / f"alpha158_qlib_return_predictions_{trade_date.isoformat()}.csv"


def format_alpha158_qlib_return_prediction_table(predictions: pd.DataFrame, *, top_n: int = 20) -> str:
    if predictions.empty:
        return "No Alpha158 Qlib return predictions."
    columns = ["trade_date", "feature_trade_date", "symbol", "name", "return_score", "model_name"]
    available = [column for column in columns if column in predictions.columns]
    frame = predictions.loc[:, available].head(max(int(top_n), 0)).copy()
    if "return_score" in frame.columns:
        frame["return_score"] = frame["return_score"].map(lambda value: f"{float(value):.6f}")
    return frame.to_string(index=False)


def validate_alpha158_qlib_return(
    *,
    storage: Storage,
    project_root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    train_end: date,
    valid_end: date,
    limit: int | None = None,
    topk: int = 50,
    n_drop: int = 5,
    min_training_rows: int = 200,
) -> Alpha158QlibReturnValidationResult:
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is required for Qlib Alpha158 return reproduction.")
    logging.info("Alpha158 Qlib return panel build started")
    panel = build_alpha158_return_panel(storage=storage, start_date=start_date, end_date=end_date, limit=limit)
    dataset = panel.dataset
    if dataset.empty:
        raise RuntimeError("Alpha158 Qlib return validation has no labeled rows.")
    if not panel.feature_columns:
        raise RuntimeError("Alpha158 Qlib return validation has no feature columns.")

    train = _date_slice(dataset, start=start_date, end=train_end)
    valid = _date_slice(dataset, start=_next_date(train_end), end=valid_end)
    test = _date_slice(dataset, start=_next_date(valid_end), end=end_date)
    if len(train) < min_training_rows:
        raise RuntimeError(f"Insufficient Alpha158 Qlib return training rows: {len(train)}")
    if valid.empty or test.empty:
        raise RuntimeError("Alpha158 Qlib return validation requires non-empty valid and test segments.")

    logging.info(
        "Alpha158 Qlib return rows: train=%s valid=%s test=%s features=%s",
        len(train),
        len(valid),
        len(test),
        len(panel.feature_columns),
    )
    model = LGBMRegressor(**QLIB_ALPHA158_LGBM_PARAMS)
    model.fit(train.loc[:, panel.feature_columns], train["LABEL0"])

    scored_parts = []
    for split_name, split in (("valid", valid), ("test", test)):
        scored = split.loc[:, ["trade_date", "symbol", "name", "LABEL0", "LABEL0_raw"]].copy()
        scored["split"] = split_name
        scored["score"] = model.predict(split.loc[:, panel.feature_columns])
        scored_parts.append(scored)
    scored_frame = pd.concat(scored_parts, ignore_index=True)

    daily_ic = build_alpha158_daily_ic(scored_frame)
    signal_metrics = summarize_alpha158_signal_metrics(scored_frame, daily_ic)
    deciles = build_alpha158_decile_report(scored_frame)
    topk_daily = build_alpha158_topk_dropout_report(scored_frame, topk=topk, n_drop=n_drop)
    topk_summary = summarize_alpha158_topk_report(topk_daily)

    report_dir = full_market_report_dir(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    feature_audit_path = report_dir / "alpha158_qlib_return_feature_audit.csv"
    signal_metrics_path = report_dir / "alpha158_qlib_return_signal_metrics.csv"
    daily_ic_path = report_dir / "alpha158_qlib_return_daily_ic.csv"
    deciles_path = report_dir / "alpha158_qlib_return_decile_report.csv"
    topk_daily_path = report_dir / "alpha158_qlib_return_topk_daily.csv"
    topk_summary_path = report_dir / "alpha158_qlib_return_topk_summary.csv"
    config_path = report_dir / "alpha158_qlib_return_config.json"
    panel.feature_audit.to_csv(feature_audit_path, index=False, encoding="utf-8-sig")
    signal_metrics.to_csv(signal_metrics_path, index=False, encoding="utf-8-sig")
    daily_ic.to_csv(daily_ic_path, index=False, encoding="utf-8-sig")
    deciles.to_csv(deciles_path, index=False, encoding="utf-8-sig")
    topk_daily.to_csv(topk_daily_path, index=False, encoding="utf-8-sig")
    topk_summary.to_csv(topk_summary_path, index=False, encoding="utf-8-sig")
    config_path.write_text(
        json.dumps(
            {
                "reference": "Qlib examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml",
                "handler": "Qlib Alpha158",
                "label": "Ref($close, -2)/Ref($close, -1) - 1",
                "learn_processors": ["DropnaLabel", "CSZScoreNorm(label)"],
                "feature_processors": [],
                "local_deviations": [
                    "local parquet daily bars are used instead of Qlib binary provider",
                    "local universe is current repository universe instead of csi300",
                    "date segments are shifted to available local 2015-2026 history",
                    "TopKDropout report is a close-to-close local approximation, not Qlib exchange engine",
                ],
                "start_date": start_date.isoformat() if start_date else "",
                "end_date": end_date.isoformat() if end_date else "",
                "train": ["", train_end.isoformat()],
                "valid": [_next_date(train_end).isoformat(), valid_end.isoformat()],
                "test": [_next_date(valid_end).isoformat(), end_date.isoformat() if end_date else ""],
                "limit": limit,
                "topk": int(topk),
                "n_drop": int(n_drop),
                "min_training_rows": int(min_training_rows),
                "feature_count": int(len(panel.feature_columns)),
                "skipped_symbols": int(len(panel.skipped)),
                "lightgbm_params": QLIB_ALPHA158_LGBM_PARAMS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return Alpha158QlibReturnValidationResult(
        feature_audit=panel.feature_audit,
        signal_metrics=signal_metrics,
        daily_ic=daily_ic,
        deciles=deciles,
        topk_daily=topk_daily,
        topk_summary=topk_summary,
        report_dir=report_dir,
        feature_audit_path=feature_audit_path,
        signal_metrics_path=signal_metrics_path,
        daily_ic_path=daily_ic_path,
        deciles_path=deciles_path,
        topk_daily_path=topk_daily_path,
        topk_summary_path=topk_summary_path,
        config_path=config_path,
    )


def build_alpha158_daily_ic(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, trade_date), group in scored.groupby(["split", "trade_date"], sort=True):
        valid = group[["score", "LABEL0_raw"]].dropna()
        if len(valid) < 2:
            continue
        if valid["score"].nunique(dropna=True) < 2 or valid["LABEL0_raw"].nunique(dropna=True) < 2:
            continue
        rows.append(
            {
                "split": split,
                "trade_date": trade_date,
                "rows": int(len(valid)),
                "ic": float(valid["score"].corr(valid["LABEL0_raw"], method="pearson")),
                "rank_ic": float(valid["score"].corr(valid["LABEL0_raw"], method="spearman")),
            }
        )
    return pd.DataFrame(rows)


def summarize_alpha158_signal_metrics(scored: pd.DataFrame, daily_ic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in scored.groupby("split", sort=False):
        y_true = pd.to_numeric(group["LABEL0"], errors="coerce")
        y_pred = pd.to_numeric(group["score"], errors="coerce")
        valid = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
        ic_rows = daily_ic[daily_ic["split"].eq(split)] if not daily_ic.empty else pd.DataFrame()
        rows.append(
            {
                "split": split,
                "rows": int(len(valid)),
                "days": int(group["trade_date"].nunique()),
                "mse_qlib_label": float(np.mean((valid["y_true"] - valid["y_pred"]) ** 2)) if not valid.empty else np.nan,
                "mae_qlib_label": float(np.mean(np.abs(valid["y_true"] - valid["y_pred"]))) if not valid.empty else np.nan,
                "mean_ic": _safe_mean(ic_rows.get("ic", pd.Series(dtype=float))),
                "icir": _safe_icir(ic_rows.get("ic", pd.Series(dtype=float))),
                "positive_ic_rate": _positive_rate(ic_rows.get("ic", pd.Series(dtype=float))),
                "mean_rank_ic": _safe_mean(ic_rows.get("rank_ic", pd.Series(dtype=float))),
                "rank_icir": _safe_icir(ic_rows.get("rank_ic", pd.Series(dtype=float))),
                "positive_rank_ic_rate": _positive_rate(ic_rows.get("rank_ic", pd.Series(dtype=float))),
                "avg_daily_count": _safe_mean(ic_rows.get("rows", pd.Series(dtype=float))),
            }
        )
    return pd.DataFrame(rows)


def build_alpha158_decile_report(scored: pd.DataFrame) -> pd.DataFrame:
    frame = scored.copy()
    frame["score_pct"] = frame.groupby(["split", "trade_date"])["score"].rank(pct=True, method="first")
    frame["score_decile"] = np.ceil(frame["score_pct"].mul(10)).sub(1).clip(0, 9).astype(int)
    rows = []
    for (split, decile), group in frame.groupby(["split", "score_decile"], sort=True):
        label = pd.to_numeric(group["LABEL0_raw"], errors="coerce")
        rows.append(
            {
                "split": split,
                "score_decile": int(decile),
                "rows": int(len(group)),
                "avg_label_return": float(label.mean()),
                "median_label_return": float(label.median()),
                "hit_rate": float(label.gt(0).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_alpha158_topk_dropout_report(
    scored: pd.DataFrame,
    *,
    topk: int = 50,
    n_drop: int = 5,
    open_cost: float = 0.0005,
    close_cost: float = 0.0015,
) -> pd.DataFrame:
    rows = []
    for split, split_frame in scored.groupby("split", sort=False):
        holdings: list[str] = []
        for trade_date, day in split_frame.groupby("trade_date", sort=True):
            ranked = day.dropna(subset=["score", "LABEL0_raw"]).sort_values("score", ascending=False)
            if ranked.empty:
                continue
            score_by_symbol = ranked.set_index("symbol")["score"]
            available_symbols = set(score_by_symbol.index)
            holdings = [symbol for symbol in holdings if symbol in available_symbols]
            previous_holdings = set(holdings)
            if not holdings:
                holdings = ranked.head(topk)["symbol"].tolist()
            else:
                current_scores = score_by_symbol.reindex(holdings).dropna().sort_values()
                sell_count = min(max(n_drop, 0), len(current_scores))
                sold = set(current_scores.head(sell_count).index)
                holdings = [symbol for symbol in holdings if symbol not in sold]
                for symbol in ranked["symbol"]:
                    if len(holdings) >= topk:
                        break
                    if symbol not in holdings:
                        holdings.append(symbol)
            selected = ranked[ranked["symbol"].isin(holdings)]
            gross_return = float(pd.to_numeric(selected["LABEL0_raw"], errors="coerce").mean())
            current_holdings = set(holdings)
            buys = len(current_holdings - previous_holdings)
            sells = len(previous_holdings - current_holdings)
            cost = (buys * open_cost + sells * close_cost) / max(topk, 1)
            rows.append(
                {
                    "split": split,
                    "trade_date": trade_date,
                    "topk": int(topk),
                    "n_drop": int(n_drop),
                    "holding_count": int(len(selected)),
                    "buys": int(buys),
                    "sells": int(sells),
                    "turnover": float((buys + sells) / max(topk, 1)),
                    "gross_return": gross_return,
                    "cost": float(cost),
                    "net_return": gross_return - cost,
                }
            )
    return pd.DataFrame(rows)


def summarize_alpha158_topk_report(topk_daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in topk_daily.groupby("split", sort=False):
        net = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        gross = pd.to_numeric(group["gross_return"], errors="coerce").dropna()
        rows.append(
            {
                "split": split,
                "days": int(len(group)),
                "avg_gross_return": float(gross.mean()) if not gross.empty else np.nan,
                "avg_net_return": float(net.mean()) if not net.empty else np.nan,
                "annualized_net_return": _annualized_return(net),
                "information_ratio": _information_ratio(net),
                "max_drawdown": _max_drawdown(net),
                "hit_rate": float(net.gt(0).mean()) if not net.empty else np.nan,
                "avg_turnover": _safe_mean(group.get("turnover", pd.Series(dtype=float))),
                "avg_holding_count": _safe_mean(group.get("holding_count", pd.Series(dtype=float))),
            }
        )
    return pd.DataFrame(rows)


def _date_slice(frame: pd.DataFrame, *, start: date | None, end: date | None) -> pd.DataFrame:
    result = frame
    if start is not None:
        result = result[result["trade_date"].dt.date >= start]
    if end is not None:
        result = result[result["trade_date"].dt.date <= end]
    return result.copy()


def _next_date(value: date) -> date:
    return (pd.Timestamp(value) + pd.Timedelta(days=1)).date()


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else np.nan


def _safe_icir(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return np.nan
    std = numeric.std(ddof=0)
    if std == 0 or pd.isna(std):
        return np.nan
    return float(numeric.mean() / std)


def _positive_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.gt(0).mean()) if not numeric.empty else np.nan


def _log_prediction_progress(stage_name: str, current: int, total: int) -> None:
    if total <= 0:
        return
    if current == 1 or current % 500 == 0 or current == total:
        logging.info("%s progress: %s/%s", stage_name, current, total)


def _latest_prediction_row(frame: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    if frame.empty or "trade_date" not in frame.columns:
        return pd.DataFrame()
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    eligible = frame[dates.dt.date <= trade_date].copy()
    if eligible.empty:
        return pd.DataFrame()
    return eligible.sort_values("trade_date").tail(1).copy()


def _annualized_return(returns: pd.Series) -> float:
    numeric = pd.to_numeric(returns, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    equity = float((1.0 + numeric).prod())
    if equity <= 0:
        return -1.0
    return float(equity ** (252 / len(numeric)) - 1.0)


def _information_ratio(returns: pd.Series) -> float:
    numeric = pd.to_numeric(returns, errors="coerce").dropna()
    if len(numeric) < 2:
        return np.nan
    std = numeric.std(ddof=0)
    if std == 0 or pd.isna(std):
        return np.nan
    return float(numeric.mean() / std * np.sqrt(252))


def _max_drawdown(returns: pd.Series) -> float:
    numeric = pd.to_numeric(returns, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    equity = (1.0 + numeric).cumprod()
    drawdown = equity.div(equity.cummax()).sub(1.0)
    return float(drawdown.min())
