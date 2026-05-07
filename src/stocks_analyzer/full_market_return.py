from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # LightGBM is the reference model for Qlib Alpha158 benchmark.
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - depends on optional local package.
    LGBMRegressor = None

from .full_market_alpha158 import build_alpha158_return_panel
from .full_market_panel import full_market_report_dir
from .storage import Storage


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
