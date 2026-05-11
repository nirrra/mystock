from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ROLLING_PHASE_STRATEGIES = (
    "random_top20",
    "centered_risk_top20",
    "rolling5_phase4_mean_top20",
    "rolling5_mean_top20",
    "rolling5_stable_top20",
)

ROLLING_PHASE_BASE_COLUMNS = (
    "symbol",
    "name",
    "signal_date",
    "entry_date",
    "entry_open",
    "phase1_score_100",
    "phase2_score_100",
    "phase4_score_100",
    "phase1_center_score",
    "phase2_center_score",
    "centered_risk_score",
)


@dataclass(slots=True)
class RollingPhaseScoreValidationResult:
    rolling_deciles: pd.DataFrame
    strategy_trades: pd.DataFrame
    strategy_summary: pd.DataFrame
    daily_counts: pd.DataFrame
    output_dir: Path
    rolling_deciles_path: Path
    strategy_trades_path: Path
    strategy_summary_path: Path
    daily_counts_path: Path


def validate_rolling_phase_scores(
    *,
    strict_dir: Path,
    output_dir: Path | None = None,
    window: int = 5,
    min_periods: int | None = None,
    horizons: tuple[int, ...] = (5, 10, 20, 60),
    strategies: tuple[str, ...] = DEFAULT_ROLLING_PHASE_STRATEGIES,
    top_n: int = 20,
    phase1_min_score: float = 40.0,
    phase2_min_score: float = 50.0,
    phase4_min_score: float = 70.0,
    std_penalty_phase1: float = 0.05,
    std_penalty_phase2: float = 0.05,
    std_penalty_phase4: float = 0.10,
) -> RollingPhaseScoreValidationResult:
    if window <= 0:
        raise ValueError("window must be positive")
    effective_min_periods = window if min_periods is None else int(min_periods)
    if effective_min_periods <= 0 or effective_min_periods > window:
        raise ValueError("min_periods must be in [1, window]")
    if any(int(horizon) <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive")
    unknown = sorted(set(strategies) - set(DEFAULT_ROLLING_PHASE_STRATEGIES))
    if unknown:
        raise ValueError(f"Unsupported rolling phase strategies: {unknown}")

    strict_root = Path(strict_dir)
    panel_path = strict_root / "oos_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Missing strict OOS panel: {panel_path}. "
            "Run validate-strict-mixed-score first, or pass --strict-dir to an existing OOS directory."
        )

    output_root = output_dir or strict_root / f"rolling{window}_phase_score_validation"
    output_root.mkdir(parents=True, exist_ok=True)

    panel = _read_oos_panel(panel_path, horizons=horizons)
    panel = add_rolling_phase_features(
        panel,
        window=window,
        min_periods=effective_min_periods,
        std_penalty_phase1=std_penalty_phase1,
        std_penalty_phase2=std_penalty_phase2,
        std_penalty_phase4=std_penalty_phase4,
    )
    rolling_deciles = build_rolling_phase_decile_report(panel, horizons=horizons)
    strategy_trades, daily_counts = build_rolling_phase_strategy_trades(
        panel,
        strategies=strategies,
        horizons=horizons,
        top_n=top_n,
        phase1_min_score=phase1_min_score,
        phase2_min_score=phase2_min_score,
        phase4_min_score=phase4_min_score,
    )
    strategy_summary = summarize_rolling_phase_strategy_trades(
        strategy_trades,
        daily_counts=daily_counts,
        signal_days=int(panel["signal_date"].nunique()) if "signal_date" in panel.columns else 0,
    )

    rolling_deciles_path = output_root / "rolling_phase_decile_report.csv"
    strategy_trades_path = output_root / "rolling_phase_strategy_trades.csv"
    strategy_summary_path = output_root / "rolling_phase_strategy_summary.csv"
    daily_counts_path = output_root / "rolling_phase_daily_counts.csv"
    rolling_deciles.to_csv(rolling_deciles_path, index=False, encoding="utf-8-sig")
    strategy_trades.to_csv(strategy_trades_path, index=False, encoding="utf-8-sig")
    strategy_summary.to_csv(strategy_summary_path, index=False, encoding="utf-8-sig")
    daily_counts.to_csv(daily_counts_path, index=False, encoding="utf-8-sig")

    return RollingPhaseScoreValidationResult(
        rolling_deciles=rolling_deciles,
        strategy_trades=strategy_trades,
        strategy_summary=strategy_summary,
        daily_counts=daily_counts,
        output_dir=output_root,
        rolling_deciles_path=rolling_deciles_path,
        strategy_trades_path=strategy_trades_path,
        strategy_summary_path=strategy_summary_path,
        daily_counts_path=daily_counts_path,
    )


def add_rolling_phase_features(
    panel: pd.DataFrame,
    *,
    window: int,
    min_periods: int,
    std_penalty_phase1: float,
    std_penalty_phase2: float,
    std_penalty_phase4: float,
) -> pd.DataFrame:
    required = {"symbol", "signal_date", "phase1_score_100", "phase2_score_100", "phase4_score_100"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"OOS panel missing required columns: {sorted(missing)}")
    result = panel.copy()
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    result["_signal_ts"] = pd.to_datetime(result["signal_date"], errors="coerce")
    result = result.dropna(subset=["_signal_ts"]).sort_values(["symbol", "_signal_ts"]).reset_index(drop=True)
    grouped = result.groupby("symbol", sort=False)
    for phase in ("phase1", "phase2", "phase4"):
        source = f"{phase}_score_100"
        values = pd.to_numeric(result[source], errors="coerce")
        result[source] = values
        result[f"{phase}_5d_mean"] = grouped[source].transform(
            lambda series: pd.to_numeric(series, errors="coerce").rolling(window, min_periods=min_periods).mean()
        )
        result[f"{phase}_5d_std"] = grouped[source].transform(
            lambda series: pd.to_numeric(series, errors="coerce").rolling(window, min_periods=min_periods).std(ddof=0)
        )
        result[f"{phase}_5d_delta"] = result[source] - result[f"{phase}_5d_mean"]

    result["phase1_5d_center_score"] = _centered_phase_score(result["phase1_5d_mean"])
    result["phase2_5d_center_score"] = _centered_phase_score(result["phase2_5d_mean"])
    result["rolling5_mean_score"] = (
        pd.to_numeric(result["phase4_5d_mean"], errors="coerce")
        + 0.08 * pd.to_numeric(result["phase1_5d_center_score"], errors="coerce")
        + 0.12 * pd.to_numeric(result["phase2_5d_center_score"], errors="coerce")
    ).round(4)
    result["rolling5_stable_score"] = (
        pd.to_numeric(result["rolling5_mean_score"], errors="coerce")
        - float(std_penalty_phase1) * pd.to_numeric(result["phase1_5d_std"], errors="coerce")
        - float(std_penalty_phase2) * pd.to_numeric(result["phase2_5d_std"], errors="coerce")
        - float(std_penalty_phase4) * pd.to_numeric(result["phase4_5d_std"], errors="coerce")
    ).round(4)

    if "phase1_center_score" not in result.columns:
        result["phase1_center_score"] = _centered_phase_score(result["phase1_score_100"])
    if "phase2_center_score" not in result.columns:
        result["phase2_center_score"] = _centered_phase_score(result["phase2_score_100"])
    if "centered_risk_score" not in result.columns:
        result["centered_risk_score"] = (
            pd.to_numeric(result["phase4_score_100"], errors="coerce")
            + 0.08 * pd.to_numeric(result["phase1_center_score"], errors="coerce")
            + 0.12 * pd.to_numeric(result["phase2_center_score"], errors="coerce")
        ).round(4)
    return result.drop(columns=["_signal_ts"], errors="ignore")


def build_rolling_phase_decile_report(panel: pd.DataFrame, *, horizons: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    score_columns = [
        "phase1_score_100",
        "phase2_score_100",
        "phase4_score_100",
        "phase1_5d_mean",
        "phase2_5d_mean",
        "phase4_5d_mean",
        "phase1_5d_std",
        "phase2_5d_std",
        "phase4_5d_std",
        "phase1_5d_delta",
        "phase2_5d_delta",
        "phase4_5d_delta",
        "rolling5_mean_score",
        "rolling5_stable_score",
        "centered_risk_score",
    ]
    for score_column in score_columns:
        if score_column not in panel.columns:
            continue
        frame = panel.dropna(subset=[score_column]).copy()
        if frame.empty:
            continue
        frame["score_decile"] = frame.groupby("signal_date", group_keys=False)[score_column].transform(_daily_decile)
        for horizon in horizons:
            return_column = f"return_{int(horizon)}d"
            profit_column = f"max_profit_{int(horizon)}d"
            drawdown_column = f"max_drawdown_{int(horizon)}d"
            if return_column not in frame.columns:
                continue
            grouped = frame.dropna(subset=["score_decile", return_column]).groupby("score_decile", dropna=False)
            for decile, group in grouped:
                returns = pd.to_numeric(group[return_column], errors="coerce")
                max_profit = pd.to_numeric(group[profit_column], errors="coerce") if profit_column in group else pd.Series(dtype=float)
                max_drawdown = pd.to_numeric(group[drawdown_column], errors="coerce") if drawdown_column in group else pd.Series(dtype=float)
                rows.append(
                    {
                        "score_column": score_column,
                        "horizon": int(horizon),
                        "score_decile": int(decile),
                        "rows": int(len(group)),
                        "avg_return": float(returns.mean()),
                        "median_return": float(returns.median()),
                        "win_rate": float(returns.gt(0).mean()),
                        "avg_max_profit": float(max_profit.mean()) if not max_profit.empty else math.nan,
                        "median_max_profit": float(max_profit.median()) if not max_profit.empty else math.nan,
                        "avg_max_drawdown": float(max_drawdown.mean()) if not max_drawdown.empty else math.nan,
                        "median_max_drawdown": float(max_drawdown.median()) if not max_drawdown.empty else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_rolling_phase_strategy_trades(
    panel: pd.DataFrame,
    *,
    strategies: tuple[str, ...],
    horizons: tuple[int, ...],
    top_n: int,
    phase1_min_score: float,
    phase2_min_score: float,
    phase4_min_score: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    for signal_date, day in panel.groupby("signal_date", sort=True):
        for strategy in strategies:
            selected = _select_rolling_phase_strategy(
                day,
                strategy=strategy,
                signal_date=date.fromisoformat(str(signal_date)),
                top_n=top_n,
                phase1_min_score=phase1_min_score,
                phase2_min_score=phase2_min_score,
                phase4_min_score=phase4_min_score,
            )
            count_rows.append({"strategy": strategy, "signal_date": signal_date, "candidate_count": int(len(selected))})
            for selected_rank, candidate in enumerate(selected.to_dict("records"), start=1):
                for horizon in horizons:
                    raw_return = candidate.get(f"return_{int(horizon)}d")
                    if pd.isna(raw_return):
                        continue
                    rows.append(
                        {
                            "strategy": strategy,
                            "signal_date": signal_date,
                            "symbol": candidate.get("symbol"),
                            "name": candidate.get("name", ""),
                            "selected_rank": selected_rank,
                            "horizon": int(horizon),
                            "phase1_score_100": candidate.get("phase1_score_100"),
                            "phase2_score_100": candidate.get("phase2_score_100"),
                            "phase4_score_100": candidate.get("phase4_score_100"),
                            "phase1_5d_mean": candidate.get("phase1_5d_mean"),
                            "phase2_5d_mean": candidate.get("phase2_5d_mean"),
                            "phase4_5d_mean": candidate.get("phase4_5d_mean"),
                            "phase1_5d_std": candidate.get("phase1_5d_std"),
                            "phase2_5d_std": candidate.get("phase2_5d_std"),
                            "phase4_5d_std": candidate.get("phase4_5d_std"),
                            "phase1_5d_delta": candidate.get("phase1_5d_delta"),
                            "phase2_5d_delta": candidate.get("phase2_5d_delta"),
                            "phase4_5d_delta": candidate.get("phase4_5d_delta"),
                            "centered_risk_score": candidate.get("centered_risk_score"),
                            "rolling5_mean_score": candidate.get("rolling5_mean_score"),
                            "rolling5_stable_score": candidate.get("rolling5_stable_score"),
                            "entry_date": candidate.get("entry_date"),
                            "entry_open": candidate.get("entry_open"),
                            "raw_return": raw_return,
                            "max_profit": candidate.get(f"max_profit_{int(horizon)}d"),
                            "max_drawdown": candidate.get(f"max_drawdown_{int(horizon)}d"),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(count_rows)


def summarize_rolling_phase_strategy_trades(
    trades: pd.DataFrame,
    *,
    daily_counts: pd.DataFrame,
    signal_days: int,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    grouped = trades.groupby(["strategy", "horizon"], dropna=False)
    summary = grouped.agg(
        trade_count=("symbol", "count"),
        signal_days_with_trades=("signal_date", "nunique"),
        avg_raw_return=("raw_return", "mean"),
        median_raw_return=("raw_return", "median"),
        raw_win_rate=("raw_return", lambda values: float(pd.to_numeric(values, errors="coerce").gt(0).mean())),
        avg_max_profit=("max_profit", "mean"),
        median_max_profit=("max_profit", "median"),
        avg_max_drawdown=("max_drawdown", "mean"),
        median_max_drawdown=("max_drawdown", "median"),
        avg_phase1_5d_mean=("phase1_5d_mean", "mean"),
        avg_phase2_5d_mean=("phase2_5d_mean", "mean"),
        avg_phase4_5d_mean=("phase4_5d_mean", "mean"),
        avg_phase1_5d_std=("phase1_5d_std", "mean"),
        avg_phase2_5d_std=("phase2_5d_std", "mean"),
        avg_phase4_5d_std=("phase4_5d_std", "mean"),
    ).reset_index()
    gains = (
        trades[pd.to_numeric(trades["raw_return"], errors="coerce").gt(0)]
        .groupby(["strategy", "horizon"])["raw_return"]
        .mean()
        .rename("avg_gain")
    )
    losses = (
        trades[pd.to_numeric(trades["raw_return"], errors="coerce").lt(0)]
        .groupby(["strategy", "horizon"])["raw_return"]
        .mean()
        .rename("avg_loss")
    )
    summary = summary.merge(gains, on=["strategy", "horizon"], how="left").merge(losses, on=["strategy", "horizon"], how="left")
    summary["payoff_ratio"] = summary["avg_gain"] / summary["avg_loss"].abs()
    if not daily_counts.empty:
        no_candidate = daily_counts.groupby("strategy")["candidate_count"].apply(
            lambda values: float(pd.to_numeric(values, errors="coerce").eq(0).mean())
        )
        summary = summary.merge(no_candidate.rename("no_candidate_day_rate"), on="strategy", how="left")
    else:
        summary["no_candidate_day_rate"] = math.nan
    summary["signal_days"] = int(signal_days)
    return summary.sort_values(["horizon", "avg_raw_return"], ascending=[True, False]).reset_index(drop=True)


def format_rolling_phase_strategy_summary(summary: pd.DataFrame, *, top_n: int = 80) -> str:
    if summary.empty:
        return "No rolling phase strategy summary rows."
    columns = [
        "strategy",
        "horizon",
        "trade_count",
        "avg_raw_return",
        "median_raw_return",
        "raw_win_rate",
        "avg_gain",
        "avg_loss",
        "payoff_ratio",
        "avg_max_profit",
        "median_max_profit",
        "avg_max_drawdown",
        "median_max_drawdown",
        "avg_phase1_5d_mean",
        "avg_phase2_5d_mean",
        "avg_phase4_5d_mean",
        "avg_phase1_5d_std",
        "avg_phase2_5d_std",
        "avg_phase4_5d_std",
        "no_candidate_day_rate",
    ]
    available = [column for column in columns if column in summary.columns]
    return summary.loc[:, available].head(top_n).to_string(index=False)


def _read_oos_panel(panel_path: Path, *, horizons: tuple[int, ...]) -> pd.DataFrame:
    header = pd.read_csv(panel_path, nrows=0)
    existing = set(header.columns)
    columns: list[str] = [column for column in ROLLING_PHASE_BASE_COLUMNS if column in existing]
    for horizon in horizons:
        for prefix in ("return", "max_profit", "max_drawdown"):
            column = f"{prefix}_{int(horizon)}d"
            if column in existing:
                columns.append(column)
    columns = list(dict.fromkeys(columns))
    required = {"symbol", "signal_date", "phase1_score_100", "phase2_score_100", "phase4_score_100"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"OOS panel missing required columns: {sorted(missing)}")
    return pd.read_csv(panel_path, usecols=columns)


def _select_rolling_phase_strategy(
    day: pd.DataFrame,
    *,
    strategy: str,
    signal_date: date,
    top_n: int,
    phase1_min_score: float,
    phase2_min_score: float,
    phase4_min_score: float,
) -> pd.DataFrame:
    base = day.copy()
    if strategy == "random_top20":
        base["random_score"] = base["symbol"].map(lambda symbol: _stable_random_score(signal_date, symbol))
        return base.sort_values(["random_score", "symbol"], ascending=[False, True]).head(max(int(top_n), 0))
    if strategy == "centered_risk_top20":
        selected = base[
            pd.to_numeric(base["phase1_score_100"], errors="coerce").ge(float(phase1_min_score))
            & pd.to_numeric(base["phase2_score_100"], errors="coerce").ge(float(phase2_min_score))
            & pd.to_numeric(base["phase4_score_100"], errors="coerce").ge(float(phase4_min_score))
            & pd.to_numeric(base["centered_risk_score"], errors="coerce").notna()
        ].copy()
        return selected.sort_values(
            ["centered_risk_score", "phase4_score_100", "phase1_center_score", "phase2_center_score", "symbol"],
            ascending=[False, False, False, False, True],
        ).head(max(int(top_n), 0))
    if strategy == "rolling5_phase4_mean_top20":
        selected = _rolling5_base_filter(base, phase1_min_score=phase1_min_score, phase2_min_score=phase2_min_score, phase4_min_score=phase4_min_score)
        return selected.sort_values(["phase4_5d_mean", "phase4_score_100", "symbol"], ascending=[False, False, True]).head(max(int(top_n), 0))
    if strategy == "rolling5_mean_top20":
        selected = _rolling5_base_filter(base, phase1_min_score=phase1_min_score, phase2_min_score=phase2_min_score, phase4_min_score=phase4_min_score)
        return selected.sort_values(
            ["rolling5_mean_score", "phase4_5d_mean", "phase4_score_100", "symbol"],
            ascending=[False, False, False, True],
        ).head(max(int(top_n), 0))
    if strategy == "rolling5_stable_top20":
        selected = _rolling5_base_filter(base, phase1_min_score=phase1_min_score, phase2_min_score=phase2_min_score, phase4_min_score=phase4_min_score)
        return selected.sort_values(
            ["rolling5_stable_score", "rolling5_mean_score", "phase4_5d_mean", "symbol"],
            ascending=[False, False, False, True],
        ).head(max(int(top_n), 0))
    raise ValueError(f"Unsupported strategy: {strategy}")


def _rolling5_base_filter(
    base: pd.DataFrame,
    *,
    phase1_min_score: float,
    phase2_min_score: float,
    phase4_min_score: float,
) -> pd.DataFrame:
    return base[
        pd.to_numeric(base["phase1_5d_mean"], errors="coerce").ge(float(phase1_min_score))
        & pd.to_numeric(base["phase2_5d_mean"], errors="coerce").ge(float(phase2_min_score))
        & pd.to_numeric(base["phase4_5d_mean"], errors="coerce").ge(float(phase4_min_score))
        & pd.to_numeric(base["rolling5_mean_score"], errors="coerce").notna()
    ].copy()


def _centered_phase_score(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return (100.0 - 2.0 * (numeric - 80.0).abs()).clip(lower=0.0, upper=100.0).round(4)


def _daily_decile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = numeric.dropna()
    if valid.empty:
        return result
    if len(valid) < 10:
        rank = valid.rank(method="first")
        decile = np.floor((rank - 1.0) / len(valid) * 10.0).astype(int).clip(0, 9)
    else:
        decile = pd.qcut(valid.rank(method="first"), 10, labels=False, duplicates="drop")
    result.loc[valid.index] = pd.Series(decile, index=valid.index).astype("Int64")
    return result


def _stable_random_score(signal_date: date, symbol: str) -> float:
    digest = hashlib.blake2b(f"{signal_date.isoformat()}:{str(symbol).zfill(6)}".encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)
