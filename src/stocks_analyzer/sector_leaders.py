from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import logging
import math
import pandas as pd

from .phase_display import normalize_symbol
from .sector_membership import load_sector_membership, sector_performance_dir


SECTOR_LEADER_COLUMNS = [
    "trade_date",
    "sector_type",
    "sector_name",
    "sector_label",
    "member_count",
    "valid_member_count",
    "symbol",
    "name",
    "leader_type",
    "leader_rank",
    "leader_score",
    "leader_tags",
    "long_term_leader_score",
    "swing_leader_score",
    "combined_leader_score",
    "is_dual_leader",
    "two_year_return_pct",
    "sector_two_year_return_pct",
    "excess_return_vs_sector_pct",
    "outperform_sector_ratio",
    "new_high_count_120d",
    "distance_to_high_pct",
    "max_drawdown_pct",
    "recovery_from_drawdown_low_pct",
    "amount_share_pct",
    "swing_count",
    "swing_lead_count",
    "swing_lead_ratio",
    "avg_swing_event_score",
    "avg_swing_excess_return_pct",
    "best_swing_rank",
    "recent_swing_event_score",
    "membership_source_updated_at",
]
SECTOR_LEADER_SCORE_COLUMNS = [
    "trade_date",
    "sector_type",
    "sector_name",
    "sector_label",
    "member_count",
    "valid_member_count",
    "symbol",
    "name",
    "long_term_rank",
    "swing_rank",
    "combined_rank",
    "leader_tags",
    "long_term_leader_score",
    "swing_leader_score",
    "combined_leader_score",
    "is_dual_leader",
    "two_year_return_pct",
    "sector_two_year_return_pct",
    "excess_return_vs_sector_pct",
    "outperform_sector_ratio",
    "new_high_count_120d",
    "distance_to_high_pct",
    "max_drawdown_pct",
    "recovery_from_drawdown_low_pct",
    "amount_share_pct",
    "swing_count",
    "swing_lead_count",
    "swing_lead_ratio",
    "avg_swing_event_score",
    "avg_swing_excess_return_pct",
    "best_swing_rank",
    "recent_swing_event_score",
    "membership_source_updated_at",
]
SECTOR_LEADER_SUMMARY_COLUMNS = [
    "trade_date",
    "sector_type",
    "sector_name",
    "sector_label",
    "member_count",
    "valid_member_count",
    "swing_count",
    "long_term_top5",
    "swing_top5",
    "dual_leaders",
    "leader_report_note",
]
SKIPPED_SECTOR_LEADER_COLUMNS = [
    "trade_date",
    "sector_type",
    "sector_name",
    "sector_label",
    "member_count",
    "reason",
]


@dataclass(frozen=True)
class SectorLeaderAnalysisResult:
    trade_date: date | None
    output_path: Path
    summary_path: Path
    skipped_path: Path
    all_scores_path: Path
    row_count: int
    all_score_row_count: int
    summary_row_count: int
    skipped_count: int
    symbol_count: int
    sector_count: int


@dataclass(frozen=True)
class SectorSwing:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    start_pos: int
    end_pos: int
    return_pct: float


def sector_leaders_path(project_root: Path, trade_date: date) -> Path:
    return sector_performance_dir(project_root) / f"sector_leaders_{trade_date.isoformat()}.csv"


def sector_leaders_summary_path(project_root: Path, trade_date: date) -> Path:
    return sector_performance_dir(project_root) / f"sector_leaders_summary_{trade_date.isoformat()}.csv"


def sector_leader_scores_all_path(project_root: Path, trade_date: date) -> Path:
    return sector_performance_dir(project_root) / f"sector_leader_scores_all_{trade_date.isoformat()}.csv"


def skipped_sector_leaders_path(project_root: Path, trade_date: date) -> Path:
    return sector_performance_dir(project_root) / f"skipped_sector_leaders_{trade_date.isoformat()}.csv"


def analyze_sector_leaders(
    *,
    project_root: Path,
    trade_date: date | None = None,
    daily_dir: Path | None = None,
    lookback_days: int = 504,
    min_history_days: int = 180,
    min_valid_members: int = 5,
    top_n: int = 10,
    sector_type: str = "all",
    sector_name: str | None = None,
    output: Path | None = None,
    progress: bool = False,
) -> SectorLeaderAnalysisResult:
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if min_history_days <= 0:
        raise ValueError("min_history_days must be positive")
    if min_history_days > lookback_days:
        raise ValueError("min_history_days cannot exceed lookback_days")
    if min_valid_members <= 0:
        raise ValueError("min_valid_members must be positive")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if sector_type not in {"all", "industry", "concept"}:
        raise ValueError("sector_type must be one of: all, industry, concept")

    daily_root = daily_dir if daily_dir is not None else project_root / "data" / "daily"
    members = _prepare_leader_membership(
        project_root=project_root,
        sector_type=sector_type,
        sector_name=sector_name,
    )
    if members.empty:
        output_path = output or sector_performance_dir(project_root) / "sector_leaders_empty.csv"
        summary_path = output_path.with_name(output_path.stem + "_summary.csv")
        skipped_path = output_path.with_name(output_path.stem + "_skipped.csv")
        all_scores_path = output_path.with_name(output_path.stem + "_all_scores.csv")
        _write_empty_outputs(
            output_path=output_path,
            summary_path=summary_path,
            skipped_path=skipped_path,
            all_scores_path=all_scores_path,
        )
        return SectorLeaderAnalysisResult(
            trade_date=trade_date,
            output_path=output_path,
            summary_path=summary_path,
            skipped_path=skipped_path,
            all_scores_path=all_scores_path,
            row_count=0,
            all_score_row_count=0,
            summary_row_count=0,
            skipped_count=0,
            symbol_count=0,
            sector_count=0,
        )

    symbols = sorted(members["symbol"].unique())
    stock_history = _load_stock_history(
        daily_root=daily_root,
        symbols=symbols,
        trade_date=trade_date,
        lookback_days=lookback_days,
        progress=progress,
    )
    if stock_history.empty:
        resolved_date = trade_date
        output_path, summary_path, skipped_path, all_scores_path = _resolve_output_paths(project_root, resolved_date, output)
        _write_empty_outputs(
            output_path=output_path,
            summary_path=summary_path,
            skipped_path=skipped_path,
            all_scores_path=all_scores_path,
        )
        return SectorLeaderAnalysisResult(
            trade_date=resolved_date,
            output_path=output_path,
            summary_path=summary_path,
            skipped_path=skipped_path,
            all_scores_path=all_scores_path,
            row_count=0,
            all_score_row_count=0,
            summary_row_count=0,
            skipped_count=0,
            symbol_count=0,
            sector_count=members["sector_key"].nunique(),
        )

    resolved_trade_date = stock_history["trade_date"].max().date()
    all_dates = pd.Index(sorted(stock_history["trade_date"].unique()))
    if len(all_dates) > lookback_days:
        all_dates = all_dates[-lookback_days:]
        stock_history = stock_history[stock_history["trade_date"].isin(all_dates)].copy()

    history_by_symbol = {symbol: frame.sort_values("trade_date") for symbol, frame in stock_history.groupby("symbol")}
    sector_info = _build_sector_info(members)
    leader_rows: list[dict[str, object]] = []
    all_score_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    grouped = members.groupby("sector_key", sort=True)
    for index, (sector_key, group) in enumerate(grouped, start=1):
        if progress and (index == 1 or index % 100 == 0 or index == len(grouped)):
            logging.info("Sector leader analysis progress: %s/%s", index, len(grouped))
        info = sector_info.loc[sector_key]
        sector_symbols = sorted(group["symbol"].unique())
        result = _analyze_one_sector(
            info=info,
            members=group,
            sector_symbols=sector_symbols,
            history_by_symbol=history_by_symbol,
            all_dates=all_dates,
            min_history_days=min_history_days,
            min_valid_members=min_valid_members,
            top_n=top_n,
            trade_date=resolved_trade_date,
        )
        if result.skipped_reason:
            skipped_rows.append(
                {
                    "trade_date": resolved_trade_date.isoformat(),
                    "sector_type": info["sector_type"],
                    "sector_name": info["sector_name"],
                    "sector_label": info["sector_label"],
                    "member_count": int(info["member_count"]),
                    "reason": result.skipped_reason,
                }
            )
            continue
        leader_rows.extend(result.leader_rows)
        all_score_rows.extend(result.all_score_rows)
        summary_rows.append(result.summary_row)

    output_path, summary_path, skipped_path, all_scores_path = _resolve_output_paths(project_root, resolved_trade_date, output)
    leader_frame = pd.DataFrame(leader_rows, columns=SECTOR_LEADER_COLUMNS)
    all_scores_frame = pd.DataFrame(all_score_rows, columns=SECTOR_LEADER_SCORE_COLUMNS)
    summary_frame = pd.DataFrame(summary_rows, columns=SECTOR_LEADER_SUMMARY_COLUMNS)
    skipped_frame = pd.DataFrame(skipped_rows, columns=SKIPPED_SECTOR_LEADER_COLUMNS)
    if not leader_frame.empty:
        leader_frame = leader_frame.sort_values(
            ["sector_type", "sector_name", "leader_type", "leader_rank", "symbol"],
            kind="stable",
        ).reset_index(drop=True)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values(["sector_type", "sector_name"], kind="stable").reset_index(drop=True)
    if not all_scores_frame.empty:
        all_scores_frame = all_scores_frame.sort_values(
            ["sector_type", "sector_name", "combined_rank", "symbol"],
            kind="stable",
        ).reset_index(drop=True)
    for path, frame in (
        (output_path, leader_frame),
        (all_scores_path, all_scores_frame),
        (summary_path, summary_frame),
        (skipped_path, skipped_frame),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")

    return SectorLeaderAnalysisResult(
        trade_date=resolved_trade_date,
        output_path=output_path,
        summary_path=summary_path,
        skipped_path=skipped_path,
        all_scores_path=all_scores_path,
        row_count=len(leader_frame),
        all_score_row_count=len(all_scores_frame),
        summary_row_count=len(summary_frame),
        skipped_count=len(skipped_frame),
        symbol_count=stock_history["symbol"].nunique(),
        sector_count=members["sector_key"].nunique(),
    )


@dataclass(frozen=True)
class _OneSectorResult:
    leader_rows: list[dict[str, object]]
    all_score_rows: list[dict[str, object]]
    summary_row: dict[str, object]
    skipped_reason: str | None = None


def _analyze_one_sector(
    *,
    info: pd.Series,
    members: pd.DataFrame,
    sector_symbols: list[str],
    history_by_symbol: dict[str, pd.DataFrame],
    all_dates: pd.Index,
    min_history_days: int,
    min_valid_members: int,
    top_n: int,
    trade_date: date,
) -> _OneSectorResult:
    valid_frames: list[pd.DataFrame] = []
    name_by_symbol = members.drop_duplicates("symbol").set_index("symbol")["name"].to_dict()
    updated_at_by_symbol = members.drop_duplicates("symbol").set_index("symbol")["updated_at"].to_dict()
    for symbol in sector_symbols:
        frame = history_by_symbol.get(symbol)
        if frame is None or len(frame) < min_history_days:
            continue
        data = frame.copy()
        data["name"] = name_by_symbol.get(symbol, "")
        data["updated_at"] = updated_at_by_symbol.get(symbol, pd.NA)
        valid_frames.append(data)
    if len(valid_frames) < min_valid_members:
        return _OneSectorResult([], [], {}, "insufficient_valid_members")

    sector_stock = pd.concat(valid_frames, ignore_index=True)
    sector_returns = (
        sector_stock.groupby("trade_date", sort=True)
        .agg(sector_return_pct=("return_pct", "mean"), valid_count=("return_pct", "count"), total_amount=("amount", "sum"))
        .reindex(all_dates)
    )
    sector_returns = sector_returns[sector_returns["valid_count"].fillna(0) >= min_valid_members].copy()
    if len(sector_returns) < min_history_days:
        return _OneSectorResult([], [], {}, "insufficient_sector_history")
    sector_ret = pd.to_numeric(sector_returns["sector_return_pct"], errors="coerce").fillna(0.0)
    sector_index = (1.0 + sector_ret / 100.0).cumprod() * 100.0
    sector_index.index = pd.to_datetime(sector_index.index)
    sector_total_return_pct = _series_total_return_pct(sector_index)
    sector_total_amount = float(pd.to_numeric(sector_returns["total_amount"], errors="coerce").fillna(0.0).sum())
    stock_metrics = _build_stock_metric_frame(
        sector_stock=sector_stock,
        sector_index=sector_index,
        sector_ret=sector_ret,
        sector_total_return_pct=sector_total_return_pct,
        sector_total_amount=sector_total_amount,
        min_history_days=min_history_days,
    )
    if len(stock_metrics) < min_valid_members:
        return _OneSectorResult([], [], {}, "insufficient_stock_metrics")

    stock_metrics = _add_long_term_scores(stock_metrics)
    swings = detect_sector_swings(sector_index)
    stock_metrics = _add_swing_scores(
        stock_metrics=stock_metrics,
        sector_stock=sector_stock,
        sector_index=sector_index,
        swings=swings,
        event_top_n=10,
    )
    stock_metrics["combined_leader_score"] = (
        0.55 * stock_metrics["long_term_leader_score"] + 0.45 * stock_metrics["swing_leader_score"]
    )
    stock_metrics["combined_leader_score"] = stock_metrics["combined_leader_score"].clip(0.0, 100.0)

    long_ranked = stock_metrics.sort_values(
        ["long_term_leader_score", "combined_leader_score", "symbol"],
        ascending=[False, False, True],
        kind="stable",
    ).copy()
    swing_ranked = stock_metrics.sort_values(
        ["swing_leader_score", "combined_leader_score", "symbol"],
        ascending=[False, False, True],
        kind="stable",
    ).copy()
    long_top_symbols = set(long_ranked.head(5)["symbol"].astype(str))
    swing_top_symbols = set(swing_ranked.head(5)["symbol"].astype(str))
    dual_symbols = long_top_symbols & swing_top_symbols
    stock_metrics["is_dual_leader"] = stock_metrics["symbol"].isin(dual_symbols)
    stock_metrics["leader_tags"] = stock_metrics.apply(
        lambda row: _leader_tags(row, long_top_symbols=long_top_symbols, swing_top_symbols=swing_top_symbols),
        axis=1,
    )
    stock_metrics = _add_leader_ranks(stock_metrics)
    long_ranked = stock_metrics.set_index("symbol").loc[long_ranked["symbol"]].reset_index()
    swing_ranked = stock_metrics.set_index("symbol").loc[swing_ranked["symbol"]].reset_index()

    leader_rows: list[dict[str, object]] = []
    leader_rows.extend(
        _leader_rows_for_type(
            ranked=long_ranked,
            info=info,
            trade_date=trade_date,
            leader_type="long_term",
            score_column="long_term_leader_score",
            top_n=top_n,
            member_count=int(info["member_count"]),
            valid_member_count=len(stock_metrics),
            swing_count=len(swings),
        )
    )
    leader_rows.extend(
        _leader_rows_for_type(
            ranked=swing_ranked,
            info=info,
            trade_date=trade_date,
            leader_type="swing",
            score_column="swing_leader_score",
            top_n=top_n,
            member_count=int(info["member_count"]),
            valid_member_count=len(stock_metrics),
            swing_count=len(swings),
        )
    )
    all_score_rows = _leader_score_rows_for_all(
        ranked=stock_metrics.sort_values(["combined_rank", "symbol"], ascending=[True, True], kind="stable"),
        info=info,
        trade_date=trade_date,
        member_count=int(info["member_count"]),
        valid_member_count=len(stock_metrics),
        swing_count=len(swings),
    )
    summary_row = {
        "trade_date": trade_date.isoformat(),
        "sector_type": info["sector_type"],
        "sector_name": info["sector_name"],
        "sector_label": info["sector_label"],
        "member_count": int(info["member_count"]),
        "valid_member_count": len(stock_metrics),
        "swing_count": len(swings),
        "long_term_top5": _format_top_list(long_ranked.head(5), "long_term_leader_score"),
        "swing_top5": _format_top_list(swing_ranked.head(5), "swing_leader_score"),
        "dual_leaders": _format_top_list(stock_metrics[stock_metrics["symbol"].isin(dual_symbols)], "combined_leader_score"),
        "leader_report_note": "current_membership_backfilled;survivorship_bias_possible",
    }
    return _OneSectorResult(leader_rows, all_score_rows, summary_row)


def _prepare_leader_membership(*, project_root: Path, sector_type: str, sector_name: str | None) -> pd.DataFrame:
    members = load_sector_membership(project_root=project_root)
    if members.empty:
        return pd.DataFrame(columns=["symbol", "name", "sector_key", "sector_type", "sector_name", "sector_label", "updated_at"])
    frame = members.copy()
    frame["symbol"] = frame["symbol"].map(normalize_symbol)
    frame = frame[frame["symbol"].astype(str).str.len().eq(6)].copy()
    frame["sector_type"] = frame["sector_type"].astype(str).str.strip()
    frame["sector_name"] = frame["sector_name"].astype(str).str.strip()
    frame["sector_label"] = frame["sector_label"].astype(str).str.strip()
    frame["name"] = frame["name"].astype(str).str.strip()
    if sector_type != "all":
        frame = frame[frame["sector_type"].eq(sector_type)].copy()
    if sector_name:
        pattern = str(sector_name).strip()
        frame = frame[frame["sector_name"].str.contains(pattern, case=False, regex=False, na=False)].copy()
    frame = frame[
        frame["sector_type"].isin(["industry", "concept"])
        & frame["sector_name"].ne("")
        & frame["sector_label"].ne("")
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "name", "sector_key", "sector_type", "sector_name", "sector_label", "updated_at"])
    frame["sector_key"] = frame["sector_type"] + "\x1f" + frame["sector_label"]
    frame = frame.drop_duplicates(["symbol", "sector_key"], keep="first")
    return frame.loc[:, ["symbol", "name", "sector_key", "sector_type", "sector_name", "sector_label", "updated_at"]]


def _load_stock_history(
    *,
    daily_root: Path,
    symbols: list[str],
    trade_date: date | None,
    lookback_days: int,
    progress: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    read_days = lookback_days + 5
    for index, symbol in enumerate(symbols, start=1):
        if progress and (index == 1 or index % 500 == 0 or index == len(symbols)):
            logging.info("Sector leader daily load progress: %s/%s", index, len(symbols))
        frame = _read_symbol_daily_history(daily_root=daily_root, symbol=symbol, trade_date=trade_date, read_days=read_days)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["symbol", "trade_date", "return_pct", "close", "amount"])
    result = pd.concat(frames, ignore_index=True)
    result = result.dropna(subset=["trade_date", "return_pct", "close"])
    result = result.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return result


def _read_symbol_daily_history(*, daily_root: Path, symbol: str, trade_date: date | None, read_days: int) -> pd.DataFrame:
    target = daily_root / f"{normalize_symbol(symbol)}.parquet"
    columns = ["trade_date", "close", "amount", "pct_change"]
    if not target.exists():
        return pd.DataFrame(columns=["symbol", "trade_date", "return_pct", "close", "amount"])
    try:
        frame = pd.read_parquet(target, columns=columns)
    except Exception:
        try:
            frame = pd.read_parquet(target)
        except Exception as exc:
            logging.warning("Failed to read daily bars for sector leaders %s: %s", symbol, exc)
            return pd.DataFrame(columns=["symbol", "trade_date", "return_pct", "close", "amount"])
    if frame.empty or "trade_date" not in frame.columns or "close" not in frame.columns:
        return pd.DataFrame(columns=["symbol", "trade_date", "return_pct", "close", "amount"])
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce").fillna(0.0).clip(lower=0.0) if "amount" in data.columns else 0.0
    data = data.dropna(subset=["trade_date", "close"]).sort_values("trade_date")
    if trade_date is not None:
        data = data[data["trade_date"].dt.date <= trade_date].copy()
    if data.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", "return_pct", "close", "amount"])
    if len(data) > read_days:
        data = data.tail(read_days).copy()
    raw_pct = pd.to_numeric(data["pct_change"], errors="coerce") if "pct_change" in data.columns else pd.Series(pd.NA, index=data.index)
    computed_pct = data["close"].pct_change() * 100.0
    data["return_pct"] = raw_pct.fillna(computed_pct)
    data["symbol"] = normalize_symbol(symbol)
    data = data.dropna(subset=["return_pct"])
    return data.loc[:, ["symbol", "trade_date", "return_pct", "close", "amount"]]


def _build_sector_info(members: pd.DataFrame) -> pd.DataFrame:
    return (
        members.groupby("sector_key", sort=True)
        .agg(
            sector_type=("sector_type", "first"),
            sector_name=("sector_name", "first"),
            sector_label=("sector_label", "first"),
            member_count=("symbol", "nunique"),
        )
        .sort_values(["sector_type", "sector_name"])
    )


def _build_stock_metric_frame(
    *,
    sector_stock: pd.DataFrame,
    sector_index: pd.Series,
    sector_ret: pd.Series,
    sector_total_return_pct: float,
    sector_total_amount: float,
    min_history_days: int,
) -> pd.DataFrame:
    sector_dates = pd.Index(sector_index.index)
    sector_ret = sector_ret.reindex(sector_dates).fillna(0.0)
    rows: list[dict[str, object]] = []
    for symbol, group in sector_stock.groupby("symbol", sort=True):
        stock = group.drop_duplicates("trade_date", keep="last").set_index("trade_date").sort_index()
        stock = stock.reindex(sector_dates)
        valid = stock.dropna(subset=["close", "return_pct"]).copy()
        if len(valid) < min_history_days:
            continue
        close = pd.to_numeric(valid["close"], errors="coerce").dropna()
        if len(close) < min_history_days or float(close.iloc[0]) <= 0:
            continue
        return_pct = (float(close.iloc[-1]) / float(close.iloc[0]) - 1.0) * 100.0
        common_ret = pd.to_numeric(valid["return_pct"], errors="coerce").dropna()
        common_sector_ret = sector_ret.reindex(common_ret.index).dropna()
        common = pd.concat([common_ret.rename("stock"), common_sector_ret.rename("sector")], axis=1).dropna()
        outperform_ratio = float((common["stock"] > common["sector"]).mean()) if not common.empty else math.nan
        amount_sum = float(pd.to_numeric(valid["amount"], errors="coerce").fillna(0.0).clip(lower=0.0).sum())
        amount_share_pct = amount_sum / sector_total_amount * 100.0 if sector_total_amount > 0 else math.nan
        drawdown = _max_drawdown_pct(close)
        trough_value = _max_drawdown_trough_value(close)
        recovery = (float(close.iloc[-1]) / trough_value - 1.0) * 100.0 if trough_value and trough_value > 0 else math.nan
        high = float(close.max())
        distance_to_high = (float(close.iloc[-1]) / high - 1.0) * 100.0 if high > 0 else math.nan
        new_high_count = _new_high_count(close, window=120)
        rows.append(
            {
                "symbol": symbol,
                "name": str(group["name"].dropna().iloc[0]) if "name" in group.columns and not group["name"].dropna().empty else "",
                "two_year_return_pct": return_pct,
                "sector_two_year_return_pct": sector_total_return_pct,
                "excess_return_vs_sector_pct": return_pct - sector_total_return_pct,
                "outperform_sector_ratio": outperform_ratio,
                "new_high_count_120d": new_high_count,
                "distance_to_high_pct": distance_to_high,
                "max_drawdown_pct": drawdown,
                "recovery_from_drawdown_low_pct": recovery,
                "amount_share_pct": amount_share_pct,
                "updated_at": str(group["updated_at"].dropna().iloc[0]) if "updated_at" in group.columns and not group["updated_at"].dropna().empty else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _add_long_term_scores(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["return_2y_score"] = _percentile_score(result["two_year_return_pct"], higher_better=True)
    result["excess_2y_score"] = _percentile_score(result["excess_return_vs_sector_pct"], higher_better=True)
    result["outperform_ratio_score"] = _percentile_score(result["outperform_sector_ratio"], higher_better=True)
    result["new_high_count_score"] = _percentile_score(result["new_high_count_120d"], higher_better=True)
    result["distance_to_high_score"] = _percentile_score(result["distance_to_high_pct"], higher_better=True)
    result["new_high_score"] = 0.6 * result["new_high_count_score"] + 0.4 * result["distance_to_high_score"]
    result["liquidity_share_score"] = _percentile_score(_winsorize(result["amount_share_pct"]), higher_better=True)
    result["max_drawdown_score"] = _percentile_score(result["max_drawdown_pct"], higher_better=True)
    result["recovery_score"] = _percentile_score(result["recovery_from_drawdown_low_pct"], higher_better=True)
    result["drawdown_quality_score"] = 0.7 * result["max_drawdown_score"] + 0.3 * result["recovery_score"]
    result["long_term_leader_score"] = (
        0.25 * result["return_2y_score"]
        + 0.25 * result["excess_2y_score"]
        + 0.15 * result["outperform_ratio_score"]
        + 0.10 * result["new_high_score"]
        + 0.15 * result["liquidity_share_score"]
        + 0.10 * result["drawdown_quality_score"]
    )
    result["long_term_leader_score"] = result["long_term_leader_score"].clip(0.0, 100.0)
    return result


def detect_sector_swings(
    sector_index: pd.Series,
    *,
    min_length: int = 5,
    max_length: int = 60,
    low_to_high_threshold: float = 0.08,
    rolling_window: int = 20,
    rolling_return_threshold: float = 0.05,
) -> list[SectorSwing]:
    clean = pd.to_numeric(sector_index, errors="coerce").dropna()
    if len(clean) <= min_length:
        return []
    values = clean.to_numpy(dtype=float)
    candidates: list[SectorSwing] = []
    for start in range(0, max(0, len(values) - min_length)):
        stop = min(len(values), start + max_length + 1)
        if stop <= start + min_length:
            continue
        future = values[start + min_length : stop]
        if len(future) == 0 or values[start] <= 0:
            continue
        best_offset = int(future.argmax()) + min_length
        end = start + best_offset
        ret = values[end] / values[start] - 1.0
        if ret >= low_to_high_threshold:
            candidates.append(_make_swing(clean, start, end, ret * 100.0))
    if len(values) > rolling_window:
        for start in range(0, len(values) - rolling_window):
            end = start + rolling_window
            if values[start] <= 0:
                continue
            ret = values[end] / values[start] - 1.0
            if ret >= rolling_return_threshold:
                candidates.append(_make_swing(clean, start, end, ret * 100.0))
    return _dedupe_overlapping_swings(candidates)


def _add_swing_scores(
    *,
    stock_metrics: pd.DataFrame,
    sector_stock: pd.DataFrame,
    sector_index: pd.Series,
    swings: list[SectorSwing],
    event_top_n: int,
) -> pd.DataFrame:
    result = stock_metrics.copy()
    result["swing_count"] = len(swings)
    if not swings:
        for column in (
            "swing_lead_count",
            "swing_lead_ratio",
            "avg_swing_event_score",
            "avg_swing_excess_return_pct",
            "best_swing_rank",
            "recent_swing_event_score",
            "swing_leader_score",
        ):
            result[column] = 0.0
        result["best_swing_rank"] = pd.NA
        return result

    event_rows: list[pd.DataFrame] = []
    stock_by_symbol = {symbol: group.set_index("trade_date").sort_index() for symbol, group in sector_stock.groupby("symbol", sort=True)}
    for event_index, swing in enumerate(swings, start=1):
        event_frame = _score_one_swing_event(
            stock_metrics=stock_metrics,
            stock_by_symbol=stock_by_symbol,
            sector_index=sector_index,
            swing=swing,
            event_index=event_index,
        )
        if not event_frame.empty:
            event_rows.append(event_frame)
    if not event_rows:
        return _add_swing_scores(
            stock_metrics=stock_metrics,
            sector_stock=sector_stock,
            sector_index=sector_index,
            swings=[],
            event_top_n=event_top_n,
        )
    events = pd.concat(event_rows, ignore_index=True)
    events["is_swing_lead"] = events["event_rank"] <= event_top_n
    aggregate = (
        events.groupby("symbol", sort=True)
        .agg(
            swing_lead_count=("is_swing_lead", "sum"),
            avg_swing_event_score=("swing_event_score", lambda x: _mean_top_values(x, top_n=3)),
            avg_swing_excess_return_pct=("total_swing_excess_pct", "mean"),
            best_swing_rank=("event_rank", "min"),
            recent_swing_event_score=("swing_event_score", lambda x: float(x.iloc[-1]) if len(x) else 0.0),
        )
        .reset_index()
    )
    result = result.merge(aggregate, on="symbol", how="left")
    result["swing_lead_count"] = result["swing_lead_count"].fillna(0).astype(float)
    result["swing_lead_ratio"] = result["swing_lead_count"] / max(1, len(swings))
    result["avg_swing_event_score"] = result["avg_swing_event_score"].fillna(0.0)
    result["avg_swing_excess_return_pct"] = result["avg_swing_excess_return_pct"].fillna(0.0)
    result["best_swing_rank"] = result["best_swing_rank"]
    result["recent_swing_event_score"] = result["recent_swing_event_score"].fillna(0.0)
    result["swing_lead_ratio_score"] = (result["swing_lead_ratio"] * 100.0).clip(0.0, 100.0)
    result["avg_swing_excess_score"] = _percentile_score(result["avg_swing_excess_return_pct"], higher_better=True)
    result["best_swing_rank_score"] = _percentile_score(result["best_swing_rank"], higher_better=False).fillna(0.0)
    result.loc[result["best_swing_rank"].isna(), "best_swing_rank_score"] = 0.0
    result["swing_leader_score"] = (
        0.35 * result["avg_swing_event_score"]
        + 0.25 * result["swing_lead_ratio_score"]
        + 0.20 * result["avg_swing_excess_score"]
        + 0.10 * result["best_swing_rank_score"]
        + 0.10 * result["recent_swing_event_score"]
    )
    result["swing_leader_score"] = result["swing_leader_score"].fillna(0.0).clip(0.0, 100.0)
    return result


def _score_one_swing_event(
    *,
    stock_metrics: pd.DataFrame,
    stock_by_symbol: dict[str, pd.DataFrame],
    sector_index: pd.Series,
    swing: SectorSwing,
    event_index: int,
) -> pd.DataFrame:
    dates = pd.Index(sector_index.index)
    early_pos = min(swing.end_pos, swing.start_pos + 5)
    sector_start = float(sector_index.iloc[swing.start_pos])
    sector_early = float(sector_index.iloc[early_pos])
    sector_end = float(sector_index.iloc[swing.end_pos])
    sector_early_return = (sector_early / sector_start - 1.0) * 100.0 if sector_start > 0 else math.nan
    sector_total_return = (sector_end / sector_start - 1.0) * 100.0 if sector_start > 0 else math.nan
    rows: list[dict[str, object]] = []
    for symbol in stock_metrics["symbol"].astype(str):
        stock = stock_by_symbol.get(symbol)
        if stock is None:
            continue
        stock = stock.reindex(dates)
        start_close = _safe_float(stock["close"].iloc[swing.start_pos])
        early_close = _safe_float(stock["close"].iloc[early_pos])
        end_close = _safe_float(stock["close"].iloc[swing.end_pos])
        if start_close is None or early_close is None or end_close is None or start_close <= 0:
            continue
        early_return = (early_close / start_close - 1.0) * 100.0
        total_return = (end_close / start_close - 1.0) * 100.0
        event_amount = pd.to_numeric(stock["amount"].iloc[swing.start_pos : swing.end_pos + 1], errors="coerce").dropna()
        pre_start = max(0, swing.start_pos - 20)
        pre_amount = pd.to_numeric(stock["amount"].iloc[pre_start : swing.start_pos], errors="coerce").dropna()
        volume_expansion = event_amount.mean() / pre_amount.mean() if not event_amount.empty and not pre_amount.empty and pre_amount.mean() > 0 else math.nan
        rows.append(
            {
                "event_index": event_index,
                "symbol": symbol,
                "early_return_pct": early_return,
                "early_excess_pct": early_return - sector_early_return,
                "total_swing_return_pct": total_return,
                "total_swing_excess_pct": total_return - sector_total_return,
                "volume_expansion": volume_expansion,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["early_return_score"] = _percentile_score(frame["early_return_pct"], higher_better=True)
    frame["early_excess_score"] = _percentile_score(frame["early_excess_pct"], higher_better=True)
    frame["total_swing_return_score"] = _percentile_score(frame["total_swing_return_pct"], higher_better=True)
    frame["total_swing_excess_score"] = _percentile_score(frame["total_swing_excess_pct"], higher_better=True)
    frame["volume_expansion_score"] = _percentile_score(_winsorize(frame["volume_expansion"]), higher_better=True)
    frame["swing_event_score"] = (
        0.25 * frame["early_return_score"]
        + 0.20 * frame["early_excess_score"]
        + 0.25 * frame["total_swing_return_score"]
        + 0.20 * frame["total_swing_excess_score"]
        + 0.10 * frame["volume_expansion_score"]
    )
    frame = frame.sort_values(["swing_event_score", "symbol"], ascending=[False, True], kind="stable").reset_index(drop=True)
    frame["event_rank"] = frame.index + 1
    return frame


def _leader_rows_for_type(
    *,
    ranked: pd.DataFrame,
    info: pd.Series,
    trade_date: date,
    leader_type: str,
    score_column: str,
    top_n: int,
    member_count: int,
    valid_member_count: int,
    swing_count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, row in enumerate(ranked.head(top_n).to_dict("records"), start=1):
        rows.append(
            {
                "trade_date": trade_date.isoformat(),
                "sector_type": info["sector_type"],
                "sector_name": info["sector_name"],
                "sector_label": info["sector_label"],
                "member_count": member_count,
                "valid_member_count": valid_member_count,
                "symbol": f'="{row["symbol"]}"',
                "name": row.get("name", ""),
                "leader_type": leader_type,
                "leader_rank": rank,
                "leader_score": _round_or_na(row.get(score_column), 4),
                "leader_tags": row.get("leader_tags", ""),
                "long_term_leader_score": _round_or_na(row.get("long_term_leader_score"), 4),
                "swing_leader_score": _round_or_na(row.get("swing_leader_score"), 4),
                "combined_leader_score": _round_or_na(row.get("combined_leader_score"), 4),
                "is_dual_leader": bool(row.get("is_dual_leader", False)),
                "two_year_return_pct": _round_or_na(row.get("two_year_return_pct"), 4),
                "sector_two_year_return_pct": _round_or_na(row.get("sector_two_year_return_pct"), 4),
                "excess_return_vs_sector_pct": _round_or_na(row.get("excess_return_vs_sector_pct"), 4),
                "outperform_sector_ratio": _round_or_na(row.get("outperform_sector_ratio"), 6),
                "new_high_count_120d": _int_or_na(row.get("new_high_count_120d")),
                "distance_to_high_pct": _round_or_na(row.get("distance_to_high_pct"), 4),
                "max_drawdown_pct": _round_or_na(row.get("max_drawdown_pct"), 4),
                "recovery_from_drawdown_low_pct": _round_or_na(row.get("recovery_from_drawdown_low_pct"), 4),
                "amount_share_pct": _round_or_na(row.get("amount_share_pct"), 6),
                "swing_count": swing_count,
                "swing_lead_count": int(row.get("swing_lead_count") or 0),
                "swing_lead_ratio": _round_or_na(row.get("swing_lead_ratio"), 6),
                "avg_swing_event_score": _round_or_na(row.get("avg_swing_event_score"), 4),
                "avg_swing_excess_return_pct": _round_or_na(row.get("avg_swing_excess_return_pct"), 4),
                "best_swing_rank": _int_or_na(row.get("best_swing_rank")),
                "recent_swing_event_score": _round_or_na(row.get("recent_swing_event_score"), 4),
                "membership_source_updated_at": row.get("updated_at", pd.NA),
            }
        )
    return rows


def _add_leader_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["long_term_rank"] = (
        result.sort_values(["long_term_leader_score", "combined_leader_score", "symbol"], ascending=[False, False, True], kind="stable")
        .reset_index()
        .reset_index()
        .set_index("index")["level_0"]
        .add(1)
    )
    result["swing_rank"] = (
        result.sort_values(["swing_leader_score", "combined_leader_score", "symbol"], ascending=[False, False, True], kind="stable")
        .reset_index()
        .reset_index()
        .set_index("index")["level_0"]
        .add(1)
    )
    result["combined_rank"] = (
        result.sort_values(["combined_leader_score", "long_term_leader_score", "symbol"], ascending=[False, False, True], kind="stable")
        .reset_index()
        .reset_index()
        .set_index("index")["level_0"]
        .add(1)
    )
    return result


def _leader_score_rows_for_all(
    *,
    ranked: pd.DataFrame,
    info: pd.Series,
    trade_date: date,
    member_count: int,
    valid_member_count: int,
    swing_count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in ranked.to_dict("records"):
        rows.append(
            {
                "trade_date": trade_date.isoformat(),
                "sector_type": info["sector_type"],
                "sector_name": info["sector_name"],
                "sector_label": info["sector_label"],
                "member_count": member_count,
                "valid_member_count": valid_member_count,
                "symbol": f'="{row["symbol"]}"',
                "name": row.get("name", ""),
                "long_term_rank": _int_or_na(row.get("long_term_rank")),
                "swing_rank": _int_or_na(row.get("swing_rank")),
                "combined_rank": _int_or_na(row.get("combined_rank")),
                "leader_tags": row.get("leader_tags", ""),
                "long_term_leader_score": _round_or_na(row.get("long_term_leader_score"), 4),
                "swing_leader_score": _round_or_na(row.get("swing_leader_score"), 4),
                "combined_leader_score": _round_or_na(row.get("combined_leader_score"), 4),
                "is_dual_leader": bool(row.get("is_dual_leader", False)),
                "two_year_return_pct": _round_or_na(row.get("two_year_return_pct"), 4),
                "sector_two_year_return_pct": _round_or_na(row.get("sector_two_year_return_pct"), 4),
                "excess_return_vs_sector_pct": _round_or_na(row.get("excess_return_vs_sector_pct"), 4),
                "outperform_sector_ratio": _round_or_na(row.get("outperform_sector_ratio"), 6),
                "new_high_count_120d": _int_or_na(row.get("new_high_count_120d")),
                "distance_to_high_pct": _round_or_na(row.get("distance_to_high_pct"), 4),
                "max_drawdown_pct": _round_or_na(row.get("max_drawdown_pct"), 4),
                "recovery_from_drawdown_low_pct": _round_or_na(row.get("recovery_from_drawdown_low_pct"), 4),
                "amount_share_pct": _round_or_na(row.get("amount_share_pct"), 6),
                "swing_count": swing_count,
                "swing_lead_count": int(row.get("swing_lead_count") or 0),
                "swing_lead_ratio": _round_or_na(row.get("swing_lead_ratio"), 6),
                "avg_swing_event_score": _round_or_na(row.get("avg_swing_event_score"), 4),
                "avg_swing_excess_return_pct": _round_or_na(row.get("avg_swing_excess_return_pct"), 4),
                "best_swing_rank": _int_or_na(row.get("best_swing_rank")),
                "recent_swing_event_score": _round_or_na(row.get("recent_swing_event_score"), 4),
                "membership_source_updated_at": row.get("updated_at", pd.NA),
            }
        )
    return rows


def _leader_tags(row: pd.Series, *, long_top_symbols: set[str], swing_top_symbols: set[str]) -> str:
    tags: list[str] = []
    symbol = str(row["symbol"])
    if symbol in long_top_symbols and symbol in swing_top_symbols:
        tags.append("双重龙头")
    elif symbol in long_top_symbols:
        tags.append("长期核心")
    elif symbol in swing_top_symbols:
        tags.append("波段先锋")
    if float(row.get("long_term_leader_score", 0.0) or 0.0) >= 70:
        tags.append("长期核心候选")
    if float(row.get("swing_leader_score", 0.0) or 0.0) >= 70:
        tags.append("波段活跃候选")
    return "/".join(dict.fromkeys(tags))


def _format_top_list(frame: pd.DataFrame, score_column: str) -> str:
    if frame.empty:
        return ""
    parts: list[str] = []
    for row in frame.head(5).to_dict("records"):
        score = row.get(score_column)
        score_text = "" if pd.isna(score) else f"{float(score):.1f}"
        parts.append(f'{row.get("symbol")} {row.get("name", "")}({score_text})')
    return " / ".join(parts)


def _make_swing(series: pd.Series, start: int, end: int, return_pct: float) -> SectorSwing:
    return SectorSwing(
        start_date=pd.Timestamp(series.index[start]),
        end_date=pd.Timestamp(series.index[end]),
        start_pos=start,
        end_pos=end,
        return_pct=float(return_pct),
    )


def _dedupe_overlapping_swings(candidates: list[SectorSwing]) -> list[SectorSwing]:
    selected: list[SectorSwing] = []
    for candidate in sorted(candidates, key=lambda item: item.return_pct, reverse=True):
        if any(_overlap(candidate, existing) for existing in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: item.start_pos)


def _overlap(left: SectorSwing, right: SectorSwing) -> bool:
    return left.start_pos <= right.end_pos and right.start_pos <= left.end_pos


def _percentile_score(values: pd.Series, *, higher_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    result = pd.Series(50.0, index=values.index, dtype="float64")
    if valid.empty:
        return result
    if valid.nunique(dropna=True) <= 1:
        result.loc[valid.index] = 50.0 if len(valid) > 1 else 100.0
        return result
    ranks = valid.rank(method="average", ascending=higher_better)
    if not higher_better:
        ranks = valid.rank(method="average", ascending=False)
    score = (ranks - 1) / (len(valid) - 1) * 100.0
    result.loc[valid.index] = score
    return result


def _winsorize(values: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return numeric
    lo = valid.quantile(lower)
    hi = valid.quantile(upper)
    return numeric.clip(lower=lo, upper=hi)


def _new_high_count(close: pd.Series, *, window: int) -> int:
    clean = pd.to_numeric(close, errors="coerce").dropna()
    if len(clean) < window:
        return 0
    rolling_high = clean.rolling(window=window, min_periods=window).max()
    return int((clean >= rolling_high).fillna(False).sum())


def _max_drawdown_pct(close: pd.Series) -> float:
    clean = pd.to_numeric(close, errors="coerce").dropna()
    if clean.empty:
        return math.nan
    running_max = clean.cummax()
    drawdown = clean / running_max - 1.0
    return float(drawdown.min() * 100.0)


def _max_drawdown_trough_value(close: pd.Series) -> float | None:
    clean = pd.to_numeric(close, errors="coerce").dropna()
    if clean.empty:
        return None
    running_max = clean.cummax()
    drawdown = clean / running_max - 1.0
    if drawdown.dropna().empty:
        return float(clean.iloc[-1])
    trough_index = drawdown.idxmin()
    value = clean.loc[trough_index]
    if pd.isna(value):
        return None
    return float(value)


def _series_total_return_pct(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 2 or float(clean.iloc[0]) == 0:
        return 0.0
    return (float(clean.iloc[-1]) / float(clean.iloc[0]) - 1.0) * 100.0


def _mean_top_values(values: pd.Series, *, top_n: int) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    return float(clean.sort_values(ascending=False).head(top_n).mean())


def _safe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _round_or_na(value: object, digits: int = 4) -> object:
    if pd.isna(value):
        return pd.NA
    try:
        number = float(value)
    except (TypeError, ValueError):
        return pd.NA
    if not math.isfinite(number):
        return pd.NA
    return round(number, digits)


def _int_or_na(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    try:
        return int(value)
    except (TypeError, ValueError):
        return pd.NA


def _resolve_output_paths(project_root: Path, trade_date: date | None, output: Path | None) -> tuple[Path, Path, Path, Path]:
    if output is not None:
        return (
            output,
            output.with_name(output.stem + "_summary.csv"),
            output.with_name(output.stem + "_skipped.csv"),
            output.with_name(output.stem + "_all_scores.csv"),
        )
    if trade_date is None:
        base = sector_performance_dir(project_root)
        return (
            base / "sector_leaders_empty.csv",
            base / "sector_leaders_summary_empty.csv",
            base / "skipped_sector_leaders_empty.csv",
            base / "sector_leader_scores_all_empty.csv",
        )
    return (
        sector_leaders_path(project_root, trade_date),
        sector_leaders_summary_path(project_root, trade_date),
        skipped_sector_leaders_path(project_root, trade_date),
        sector_leader_scores_all_path(project_root, trade_date),
    )


def _write_empty_outputs(*, output_path: Path, summary_path: Path, skipped_path: Path, all_scores_path: Path) -> None:
    for path, columns in (
        (output_path, SECTOR_LEADER_COLUMNS),
        (all_scores_path, SECTOR_LEADER_SCORE_COLUMNS),
        (summary_path, SECTOR_LEADER_SUMMARY_COLUMNS),
        (skipped_path, SKIPPED_SECTOR_LEADER_COLUMNS),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8-sig")
