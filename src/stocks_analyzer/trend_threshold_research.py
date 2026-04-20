from __future__ import annotations

import math
import logging
from datetime import date

import pandas as pd

from .models import AppConfig
from .storage import Storage
from .trend_backtest import backtest_signal_returns
from .trend_indicator_scores import build_next_open_entries, scan_indicator_scored_entries


RESEARCH_METRICS = [
    "buy_score",
    "trend_base_score",
    "price_action_score",
    "macd_score",
    "volume_score",
    "volume_price_divergence_score",
    "boll_score",
    "rsi_score",
    "kdj_score",
    "atr_score",
    "positive_indicator_count",
]

RESEARCH_DATASET_COLUMNS = [
    "dataset_split",
    "trade_date",
    "planned_entry_date",
    "entry_date",
    "exit_date",
    "holding_days",
    "symbol",
    "name",
    "signal_type",
    "setup_type",
    "entry_score",
    "trend_score",
    "trend_base_score",
    "price_action_score",
    "macd_score",
    "volume_score",
    "volume_price_divergence_score",
    "boll_score",
    "rsi_score",
    "kdj_score",
    "atr_score",
    "buy_score",
    "positive_indicator_count",
    "return_pct",
    "max_drawdown_pct",
    "max_upside_pct",
    "min_return_pct",
    "entry_note",
    "entry_timing",
]


def build_threshold_research_dataset(
    storage: Storage,
    config: AppConfig,
    *,
    start_date: date,
    end_date: date,
    sample_mode: str = "monthly",
    train_end_date: date | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    logging.info(
        "Threshold research started: start=%s end=%s sample_mode=%s%s",
        start_date.isoformat(),
        end_date.isoformat(),
        sample_mode,
        f" train_end={train_end_date.isoformat()}" if train_end_date is not None else "",
    )
    logging.info("Threshold research stage: scanning scored setup samples")
    scored = scan_indicator_scored_entries(
        storage,
        config,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        progress_callback=_log_threshold_research_progress,
    )
    logging.info("Threshold research stage: building next-open entries")
    entries = build_next_open_entries(scored)
    logging.info("Threshold research stage: sampling entries")
    entries = sample_threshold_research_entries(entries, sample_mode=sample_mode)
    if entries.empty:
        return pd.DataFrame(columns=RESEARCH_DATASET_COLUMNS)

    logging.info("Threshold research stage: loading daily history for %s symbols", entries["symbol"].astype(str).nunique())
    daily_history = _load_daily_history_map(storage, entries["symbol"].astype(str).tolist())
    logging.info("Threshold research stage: backtesting next-open samples")
    backtest = backtest_signal_returns(entries, daily_history, config.trend_backtest, entry_timing="next_open")
    if backtest.empty:
        return pd.DataFrame(columns=RESEARCH_DATASET_COLUMNS)

    merge_columns = [
        "trade_date",
        "planned_entry_date",
        "symbol",
        "signal_type",
        "setup_type",
        "name",
        "entry_score",
        "trend_score",
        "trend_base_score",
        "price_action_score",
        "macd_score",
        "volume_score",
        "volume_price_divergence_score",
        "boll_score",
        "rsi_score",
        "kdj_score",
        "atr_score",
        "buy_score",
        "positive_indicator_count",
        "entry_timing",
    ]
    metadata = entries.loc[:, merge_columns].copy()
    metadata["trade_date"] = pd.to_datetime(metadata["trade_date"])
    metadata["planned_entry_date"] = pd.to_datetime(metadata["planned_entry_date"])
    metadata["symbol"] = metadata["symbol"].astype(str).str.zfill(6)

    dataset = backtest.merge(
        metadata,
        on=["trade_date", "symbol", "signal_type"],
        how="left",
        suffixes=("", "_score"),
    )
    dataset["setup_type"] = dataset["setup_type"].fillna(dataset["signal_type"])
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    dataset["planned_entry_date"] = pd.to_datetime(dataset["planned_entry_date"])
    dataset["entry_date"] = pd.to_datetime(dataset["entry_date"])
    dataset["exit_date"] = pd.to_datetime(dataset["exit_date"])
    dataset["dataset_split"] = _resolve_dataset_split(dataset["trade_date"], train_end_date)
    if train_end_date is not None:
        all_period = dataset.copy()
        all_period["dataset_split"] = "all_period"
        dataset = pd.concat([all_period, dataset], ignore_index=True)
    else:
        dataset["dataset_split"] = "all_period"

    logging.info("Threshold research finished: generated %s labeled rows", len(dataset))
    return dataset.reindex(columns=RESEARCH_DATASET_COLUMNS)


def sample_threshold_research_entries(entries: pd.DataFrame, *, sample_mode: str) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame(columns=list(entries.columns))

    mode = sample_mode.lower()
    frame = entries.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    if mode == "daily":
        return frame.reset_index(drop=True)

    if mode == "weekly":
        bucket = frame["trade_date"].dt.to_period("W-FRI")
    elif mode == "monthly":
        bucket = frame["trade_date"].dt.to_period("M")
    else:
        raise ValueError(f"Unsupported sample mode: {sample_mode}")

    sampled_dates = frame.groupby(bucket, sort=True)["trade_date"].max().drop_duplicates().tolist()
    return frame[frame["trade_date"].isin(sampled_dates)].sort_values(["trade_date", "buy_score", "symbol"], ascending=[True, False, True]).reset_index(drop=True)


def summarize_indicator_distributions(
    dataset: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()

    metric_names = metrics or RESEARCH_METRICS
    rows: list[dict[str, object]] = []
    for dataset_split in sorted(dataset["dataset_split"].dropna().unique().tolist()):
        split_frame = dataset[dataset["dataset_split"] == dataset_split]
        for holding_days in sorted(split_frame["holding_days"].dropna().astype(int).unique().tolist()):
            holding_frame = split_frame[split_frame["holding_days"] == holding_days]
            for signal_scope, scope_frame in _iter_signal_scopes(holding_frame):
                if scope_frame.empty:
                    continue
                groups = _label_groups(scope_frame)
                for label_group, label_frame in groups.items():
                    for metric in metric_names:
                        series = pd.to_numeric(label_frame.get(metric), errors="coerce").dropna()
                        if series.empty:
                            continue
                        rows.append(
                            {
                                "dataset_split": dataset_split,
                                "holding_days": int(holding_days),
                                "signal_scope": signal_scope,
                                "label_group": label_group,
                                "metric": metric,
                                "sample_count": int(len(series)),
                                "mean": round(float(series.mean()), 4),
                                "median": round(float(series.median()), 4),
                                "min": round(float(series.min()), 4),
                                "p20": round(float(series.quantile(0.2)), 4),
                                "p40": round(float(series.quantile(0.4)), 4),
                                "p50": round(float(series.quantile(0.5)), 4),
                                "p60": round(float(series.quantile(0.6)), 4),
                                "p80": round(float(series.quantile(0.8)), 4),
                                "max": round(float(series.max()), 4),
                            }
                        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["dataset_split", "holding_days", "signal_scope", "metric", "label_group"]).reset_index(drop=True)


def derive_threshold_candidates(distributions: pd.DataFrame) -> pd.DataFrame:
    if distributions.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_columns = ["dataset_split", "holding_days", "signal_scope", "metric"]
    for keys, frame in distributions.groupby(group_columns, dropna=False):
        stats = {str(record["label_group"]): record for record in frame.to_dict("records")}
        all_stats = stats.get("all")
        strong_stats = stats.get("strong")
        weak_stats = stats.get("weak")
        if all_stats is None or strong_stats is None or weak_stats is None:
            continue

        separation_gap = round(float(strong_stats["median"]) - float(weak_stats["median"]), 4)
        candidates = {
            "loose": max(float(all_stats["p50"]), float(weak_stats["p60"])),
            "balanced": max(float(all_stats["p50"]), (float(strong_stats["p20"]) + float(weak_stats["p80"])) / 2),
            "strict": max(float(all_stats["p50"]), float(strong_stats["p20"])),
        }
        for candidate_type, threshold in candidates.items():
            normalized = _normalize_threshold(metric=str(keys[3]), threshold=threshold)
            rows.append(
                {
                    "dataset_split": keys[0],
                    "holding_days": int(keys[1]),
                    "signal_scope": keys[2],
                    "metric": keys[3],
                    "candidate_type": candidate_type,
                    "threshold": normalized,
                    "all_sample_count": int(all_stats["sample_count"]),
                    "strong_sample_count": int(strong_stats["sample_count"]),
                    "weak_sample_count": int(weak_stats["sample_count"]),
                    "all_median": float(all_stats["median"]),
                    "strong_median": float(strong_stats["median"]),
                    "weak_median": float(weak_stats["median"]),
                    "strong_p20": float(strong_stats["p20"]),
                    "weak_p80": float(weak_stats["p80"]),
                    "separation_gap": separation_gap,
                }
            )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    return result.drop_duplicates(subset=["dataset_split", "holding_days", "signal_scope", "metric", "candidate_type", "threshold"]).sort_values(
        ["dataset_split", "holding_days", "signal_scope", "metric", "candidate_type"]
    ).reset_index(drop=True)


def evaluate_threshold_candidates(dataset: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty or candidates.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for candidate in candidates.to_dict("records"):
        subset = _filter_research_subset(
            dataset,
            dataset_split=str(candidate["dataset_split"]),
            holding_days=int(candidate["holding_days"]),
            signal_scope=str(candidate["signal_scope"]),
        )
        if subset.empty:
            continue

        metric = str(candidate["metric"])
        threshold = float(candidate["threshold"])
        filtered = subset[pd.to_numeric(subset[metric], errors="coerce") >= threshold].copy()
        rows.append(_summarize_filtered_subset(filtered, base=subset, metadata=candidate, evaluation_kind="single_metric"))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["dataset_split", "holding_days", "signal_scope", "metric", "candidate_type"]).reset_index(drop=True)


def build_combo_threshold_candidates(candidates: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()

    balanced = candidates[candidates["candidate_type"] == "balanced"].copy()
    if balanced.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_columns = ["dataset_split", "holding_days", "signal_scope"]
    for keys, frame in balanced.groupby(group_columns, dropna=False):
        threshold_map = {str(record["metric"]): record for record in frame.to_dict("records")}
        buy = threshold_map.get("buy_score")
        if buy is None:
            continue

        buy_threshold = float(buy["threshold"])
        rows.append(
            _combo_row(
                keys=keys,
                combo_name="candidate_buy_only",
                buy_score_min=buy_threshold,
            )
        )

        trend_base = threshold_map.get("trend_base_score")
        if trend_base is not None:
            rows.append(
                _combo_row(
                    keys=keys,
                    combo_name="candidate_buy_plus_trend_base",
                    buy_score_min=buy_threshold,
                    trend_base_score_min=float(trend_base["threshold"]),
                )
            )

        price_action = threshold_map.get("price_action_score")
        if price_action is not None:
            rows.append(
                _combo_row(
                    keys=keys,
                    combo_name="candidate_buy_plus_price_action",
                    buy_score_min=buy_threshold,
                    price_action_score_min=float(price_action["threshold"]),
                )
            )

        macd = threshold_map.get("macd_score")
        if macd is not None:
            rows.append(
                _combo_row(
                    keys=keys,
                    combo_name="candidate_buy_plus_macd",
                    buy_score_min=buy_threshold,
                    macd_score_min=float(macd["threshold"]),
                )
            )

        indicator_count = threshold_map.get("positive_indicator_count")
        if indicator_count is not None:
            rows.append(
                _combo_row(
                    keys=keys,
                    combo_name="candidate_buy_plus_indicator_count",
                    buy_score_min=buy_threshold,
                    positive_indicator_count_min=int(indicator_count["threshold"]),
                )
            )

        rows.append(
            _combo_row(
                keys=keys,
                combo_name="current_default_rules",
                buy_score_min=float(config.trend_entry_rules.buy_score_min),
                trend_base_score_min=float(config.trend_entry_rules.trend_base_score_min),
                price_action_score_min=float(config.trend_entry_rules.price_action_score_min),
                macd_score_min=float(config.trend_entry_rules.macd_score_min),
                positive_indicator_count_min=int(config.trend_entry_rules.positive_indicator_count_min),
            )
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).drop_duplicates().sort_values(["dataset_split", "holding_days", "signal_scope", "combo_name"]).reset_index(drop=True)


def evaluate_combo_thresholds(dataset: pd.DataFrame, combo_candidates: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty or combo_candidates.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for combo in combo_candidates.to_dict("records"):
        subset = _filter_research_subset(
            dataset,
            dataset_split=str(combo["dataset_split"]),
            holding_days=int(combo["holding_days"]),
            signal_scope=str(combo["signal_scope"]),
        )
        if subset.empty:
            continue

        filtered = subset.copy()
        for metric in ["buy_score", "trend_base_score", "price_action_score", "macd_score"]:
            threshold = combo.get(f"{metric}_min")
            if threshold is None or pd.isna(threshold):
                continue
            filtered = filtered[pd.to_numeric(filtered[metric], errors="coerce") >= float(threshold)]
        indicator_count_min = combo.get("positive_indicator_count_min")
        if indicator_count_min is not None and not pd.isna(indicator_count_min):
            filtered = filtered[pd.to_numeric(filtered["positive_indicator_count"], errors="coerce") >= int(indicator_count_min)]

        rows.append(_summarize_filtered_subset(filtered, base=subset, metadata=combo, evaluation_kind="combo"))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["dataset_split", "holding_days", "signal_scope", "combo_name"]).reset_index(drop=True)


def build_default_threshold_candidates(
    candidates: pd.DataFrame,
    combo_candidates: pd.DataFrame,
    combo_evaluation: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty or combo_evaluation.empty:
        return pd.DataFrame()

    evaluated = combo_evaluation[combo_evaluation["signal_scope"] != "all"].copy()
    if evaluated.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    grouped = evaluated.groupby(["dataset_split", "holding_days", "signal_scope"], dropna=False)
    for keys, frame in grouped:
        recommended = _choose_recommended_combo(frame)
        if recommended is None:
            continue

        current_default = frame[frame["combo_name"] == "current_default_rules"]
        current_default_row = current_default.iloc[0] if not current_default.empty else None
        balanced = candidates[
            (candidates["dataset_split"] == keys[0])
            & (candidates["holding_days"] == keys[1])
            & (candidates["signal_scope"] == keys[2])
            & (candidates["candidate_type"] == "balanced")
        ].copy()
        balanced_map = {
            str(record["metric"]): record["threshold"]
            for record in balanced.to_dict("records")
        }
        rows.append(
            {
                "dataset_split": keys[0],
                "holding_days": int(keys[1]),
                "signal_scope": keys[2],
                "recommended_combo_name": recommended["combo_name"],
                "buy_score_min": recommended.get("buy_score_min"),
                "trend_base_score_min": recommended.get("trend_base_score_min"),
                "price_action_score_min": recommended.get("price_action_score_min"),
                "macd_score_min": recommended.get("macd_score_min"),
                "positive_indicator_count_min": recommended.get("positive_indicator_count_min"),
                "balanced_buy_score_min": balanced_map.get("buy_score"),
                "balanced_trend_base_score_min": balanced_map.get("trend_base_score"),
                "balanced_price_action_score_min": balanced_map.get("price_action_score"),
                "balanced_macd_score_min": balanced_map.get("macd_score"),
                "balanced_positive_indicator_count_min": balanced_map.get("positive_indicator_count"),
                "selected_count": int(recommended["selected_count"]),
                "coverage": float(recommended["coverage"]),
                "win_rate": float(recommended["win_rate"]) if recommended.get("win_rate") is not None else None,
                "avg_return_pct": float(recommended["avg_return_pct"]) if recommended.get("avg_return_pct") is not None else None,
                "current_default_selected_count": int(current_default_row["selected_count"]) if current_default_row is not None else None,
                "current_default_coverage": float(current_default_row["coverage"]) if current_default_row is not None else None,
                "current_default_win_rate": float(current_default_row["win_rate"]) if current_default_row is not None else None,
                "current_default_avg_return_pct": float(current_default_row["avg_return_pct"]) if current_default_row is not None else None,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["dataset_split", "holding_days", "signal_scope"]).reset_index(drop=True)


def _iter_signal_scopes(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    scopes: list[tuple[str, pd.DataFrame]] = [("all", frame)]
    signal_types = sorted(frame["signal_type"].dropna().astype(str).unique().tolist()) if "signal_type" in frame.columns else []
    for signal_type in signal_types:
        scopes.append((signal_type, frame[frame["signal_type"] == signal_type].copy()))
    return scopes


def _label_groups(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    strong_cutoff = float(frame["return_pct"].quantile(0.8))
    bottom_cutoff = float(frame["return_pct"].quantile(0.2))
    return {
        "all": frame,
        "strong": frame[frame["return_pct"] >= strong_cutoff].copy(),
        "weak": frame[frame["return_pct"] < 0].copy(),
        "bottom": frame[frame["return_pct"] <= bottom_cutoff].copy(),
    }


def _normalize_threshold(*, metric: str, threshold: float) -> float | int:
    if metric == "positive_indicator_count":
        return int(math.ceil(threshold))
    return round(float(threshold), 4)


def _filter_research_subset(
    dataset: pd.DataFrame,
    *,
    dataset_split: str,
    holding_days: int,
    signal_scope: str,
) -> pd.DataFrame:
    subset = dataset[(dataset["dataset_split"] == dataset_split) & (dataset["holding_days"] == holding_days)].copy()
    if signal_scope != "all":
        subset = subset[subset["signal_type"] == signal_scope].copy()
    return subset.reset_index(drop=True)


def _summarize_filtered_subset(
    filtered: pd.DataFrame,
    *,
    base: pd.DataFrame,
    metadata: dict[str, object],
    evaluation_kind: str,
) -> dict[str, object]:
    selected_count = int(len(filtered))
    base_count = int(len(base))
    summary = {
        **metadata,
        "evaluation_kind": evaluation_kind,
        "base_sample_count": base_count,
        "selected_count": selected_count,
        "coverage": round(0.0 if base_count == 0 else selected_count / base_count, 4),
        "win_rate": round(float((filtered["return_pct"] > 0).mean()), 4) if selected_count else None,
        "avg_return_pct": round(float(filtered["return_pct"].mean()), 4) if selected_count else None,
        "median_return_pct": round(float(filtered["return_pct"].median()), 4) if selected_count else None,
        "avg_max_drawdown_pct": round(float(filtered["max_drawdown_pct"].mean()), 4) if selected_count else None,
        "avg_max_upside_pct": round(float(filtered["max_upside_pct"].mean()), 4) if selected_count else None,
        "avg_buy_score": round(float(filtered["buy_score"].mean()), 4) if selected_count and "buy_score" in filtered.columns else None,
    }
    return summary


def _combo_row(
    *,
    keys: tuple[object, object, object],
    combo_name: str,
    buy_score_min: float | None = None,
    trend_base_score_min: float | None = None,
    price_action_score_min: float | None = None,
    macd_score_min: float | None = None,
    positive_indicator_count_min: int | None = None,
) -> dict[str, object]:
    return {
        "dataset_split": keys[0],
        "holding_days": int(keys[1]),
        "signal_scope": keys[2],
        "combo_name": combo_name,
        "buy_score_min": round(float(buy_score_min), 4) if buy_score_min is not None else None,
        "trend_base_score_min": round(float(trend_base_score_min), 4) if trend_base_score_min is not None else None,
        "price_action_score_min": round(float(price_action_score_min), 4) if price_action_score_min is not None else None,
        "macd_score_min": round(float(macd_score_min), 4) if macd_score_min is not None else None,
        "positive_indicator_count_min": int(positive_indicator_count_min) if positive_indicator_count_min is not None else None,
    }


def _resolve_dataset_split(trade_dates: pd.Series, train_end_date: date | None) -> pd.Series:
    if train_end_date is None:
        return pd.Series(["all_period"] * len(trade_dates), index=trade_dates.index)
    normalized = pd.to_datetime(trade_dates)
    return normalized.dt.date.map(lambda value: "in_sample" if value <= train_end_date else "out_of_sample")


def _load_daily_history_map(storage: Storage, symbols: list[str]) -> dict[str, pd.DataFrame]:
    history: dict[str, pd.DataFrame] = {}
    for symbol in sorted({str(symbol).zfill(6) for symbol in symbols}):
        try:
            history[symbol] = storage.load_daily_bars(symbol)
        except FileNotFoundError:
            continue
    return history


def _log_threshold_research_progress(current: int, total: int) -> None:
    if total <= 0:
        return
    if current == 1 or current == total or current % 100 == 0:
        logging.info("Threshold research scan progress: %s/%s", current, total)


def _choose_recommended_combo(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None

    eligible = frame[(frame["coverage"] >= 0.10) & (frame["selected_count"] >= 20)].copy()
    candidate_pool = eligible if not eligible.empty else frame.copy()
    ranked = candidate_pool.sort_values(
        ["avg_return_pct", "win_rate", "coverage", "selected_count"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    if ranked.empty:
        return None
    return ranked.iloc[0]
