from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .atr import ATR_WATCHLIST_FIELD_MAP
from .concern_sectors import concern_sector_members_path, read_concern_sector_members, read_stock_concern_sectors
from .daily_returns import read_full_market_daily_returns
from .full_market_return import alpha158_qlib_return_predictions_path
from .full_market_risk import barrier_risk_predictions_path, tail_risk_predictions_path
from .phase4_rolling import PHASE4_ROLLING_COLUMNS, merge_phase4_rolling_frame
from .phase_display import normalize_symbol
from .position_sizing import RECOMMENDED_POSITION_PERCENT_FIELD, add_recommended_position_percent
from .sector_watchlist import load_sector_watchlist, watchlist_sectors_path
from .watchlist import (
    LIMIT_UP_DAILY_RETURN_THRESHOLD,
    _prepare_atr_frame,
    _prepare_macd_frame,
    _prepare_pattern_groups,
    _prepare_phase4_predictions,
    _prepare_phase_risk_predictions,
    add_centered_risk_scores,
)


PUBLIC_MIN_PHASE1 = 20.0
PUBLIC_MIN_PHASE2 = 20.0
ROUTE_MIN_PHASE1 = 40.0
ROUTE_MIN_PHASE2 = 40.0
ROUTE_SECTOR_TOP_N = 20
SECTOR_LEADER_POOL_TOP_N = 5


@dataclass(frozen=True)
class RouteWatchlistResult:
    trade_date: date
    a1_path: Path
    a2_path: Path
    b_path: Path
    sector_leader_pool_path: Path
    a1_count: int
    a2_count: int
    b_count: int
    sector_count: int
    sector_leader_count: int


def watchlist_a1_recent_mainline_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "watchlists" / f"watchlist_a1_recent_mainline_{trade_date.isoformat()}.json"


def watchlist_a2_rotation_expected_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "watchlists" / f"watchlist_a2_rotation_expected_{trade_date.isoformat()}.json"


def watchlist_b_pattern_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "watchlists" / f"watchlist_b_pattern_{trade_date.isoformat()}.json"


def watchlist_sector_leader_pool_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "watchlists" / f"watchlist_sector_leader_pool_{trade_date.isoformat()}.json"


def find_latest_sector_leader_pool_before(*, project_root: Path, trade_date: date) -> tuple[date, Path]:
    candidates: list[tuple[date, Path]] = []
    for path in (project_root / "reports" / "watchlists").glob("watchlist_sector_leader_pool_*.json"):
        parsed = _parse_date_from_name(path.name, "watchlist_sector_leader_pool")
        if parsed is not None and parsed < trade_date:
            candidates.append((parsed, path))
    if not candidates:
        raise FileNotFoundError(f"No sector leader pool found before {trade_date.isoformat()}")
    return max(candidates, key=lambda item: item[0])


def write_route_watchlists_from_files(*, project_root: Path, trade_date: date) -> RouteWatchlistResult:
    payloads = build_route_watchlists_from_files(project_root=project_root, trade_date=trade_date)
    a1_path = _write_payload(watchlist_a1_recent_mainline_path(project_root, trade_date), payloads["a1"], csv_kind="stocks")
    a2_path = _write_payload(watchlist_a2_rotation_expected_path(project_root, trade_date), payloads["a2"], csv_kind="stocks")
    b_path = _write_payload(watchlist_b_pattern_path(project_root, trade_date), payloads["b"], csv_kind="stocks")
    pool_path = _write_payload(watchlist_sector_leader_pool_path(project_root, trade_date), payloads["sector_leader_pool"], csv_kind="pool")
    return RouteWatchlistResult(
        trade_date=trade_date,
        a1_path=a1_path,
        a2_path=a2_path,
        b_path=b_path,
        sector_leader_pool_path=pool_path,
        a1_count=len(payloads["a1"].get("candidates", [])),
        a2_count=len(payloads["a2"].get("candidates", [])),
        b_count=len(payloads["b"].get("candidates", [])),
        sector_count=len(payloads["sector_leader_pool"].get("sectors", [])),
        sector_leader_count=len(payloads["sector_leader_pool"].get("candidates", [])),
    )


def write_sector_leader_pool_payload(*, project_root: Path, trade_date: date, payload: dict[str, object]) -> Path:
    return _write_payload(watchlist_sector_leader_pool_path(project_root, trade_date), payload, csv_kind="pool")


def build_route_watchlists_from_files(*, project_root: Path, trade_date: date) -> dict[str, dict[str, object]]:
    pattern_path = project_root / "reports" / "patterns" / f"patterns_all_{trade_date.isoformat()}.csv"
    macd_path = project_root / "reports" / "macd" / f"macd_{trade_date.isoformat()}.csv"
    atr_path = project_root / "reports" / "atr" / f"atr_{trade_date.isoformat()}.csv"
    sector_payload = load_sector_watchlist(project_root=project_root, trade_date=trade_date)
    concern_members = read_concern_sector_members(project_root=project_root, trade_date=trade_date)
    stock_concerns = read_stock_concern_sectors(project_root=project_root, trade_date=trade_date)
    score_frame = build_stock_score_frame(
        project_root=project_root,
        trade_date=trade_date,
        phase1_predictions=_read_csv(tail_risk_predictions_path(project_root, trade_date)),
        phase2_predictions=_read_csv(barrier_risk_predictions_path(project_root, trade_date)),
        phase4_predictions=_read_csv(alpha158_qlib_return_predictions_path(project_root, trade_date)),
        pattern_frame=_read_csv(pattern_path),
        daily_returns=read_full_market_daily_returns(project_root=project_root, trade_date=trade_date),
        macd_frame=_read_csv(macd_path),
        atr_frame=_read_csv(atr_path),
        stock_concerns=stock_concerns,
    )
    return build_route_watchlists(
        trade_date=trade_date,
        project_root=project_root,
        stock_scores=score_frame,
        sector_payload=sector_payload,
        concern_members=concern_members,
        source_files={
            "watchlist_sectors": str(watchlist_sectors_path(project_root, trade_date)),
            "concern_sector_members": str(concern_sector_members_path(project_root, trade_date)),
            "phase1": str(tail_risk_predictions_path(project_root, trade_date)),
            "phase2": str(barrier_risk_predictions_path(project_root, trade_date)),
            "phase4": str(alpha158_qlib_return_predictions_path(project_root, trade_date)),
            "pattern": str(pattern_path),
            "macd": str(macd_path),
            "atr": str(atr_path),
        },
    )


def build_stock_score_frame(
    *,
    project_root: Path,
    trade_date: date,
    phase1_predictions: pd.DataFrame,
    phase2_predictions: pd.DataFrame,
    phase4_predictions: pd.DataFrame,
    pattern_frame: pd.DataFrame,
    daily_returns: pd.DataFrame,
    macd_frame: pd.DataFrame,
    atr_frame: pd.DataFrame,
    stock_concerns: pd.DataFrame,
) -> pd.DataFrame:
    phase1 = _prepare_phase_risk_predictions(
        phase1_predictions,
        score_column="risk_score",
        output_score_column="phase1_risk_score",
        prefix="phase1",
        filter_rate=0.2,
        model_prefix="phase1",
        extra_columns=("log_return_1d",),
    )
    phase2 = _prepare_phase_risk_predictions(
        phase2_predictions,
        score_column="barrier_risk_score",
        output_score_column="phase2_barrier_risk_score",
        prefix="phase2",
        filter_rate=0.2,
        model_prefix="phase2",
        extra_columns=("is_cusum_event",),
    )
    phase4 = _prepare_phase4_predictions(phase4_predictions)
    phase4 = merge_phase4_rolling_frame(phase4, project_root=project_root, trade_date=trade_date)
    macd = _prepare_macd_frame(macd_frame)
    atr = _prepare_atr_frame(atr_frame)
    if not atr.empty:
        atr_source = pd.to_numeric(atr.get("atr_pct_14"), errors="coerce")
        if "ATR%" not in atr.columns:
            atr["ATR%"] = atr_source.round(4)
    pattern_groups = _prepare_pattern_groups(pattern_frame)

    frame = phase1.merge(phase2, on="symbol", how="inner").merge(phase4, on="symbol", how="left")
    for extra in (macd, atr, _prepare_daily_returns(daily_returns), _prepare_stock_concerns(stock_concerns)):
        if not extra.empty and "symbol" in extra.columns:
            frame = frame.merge(extra, on="symbol", how="left")
    if "name" not in frame.columns:
        frame["name"] = ""
    if "phase4_name" in frame.columns:
        frame["name"] = frame["name"].where(frame["name"].astype(str).str.strip().ne(""), frame["phase4_name"])
    if "daily_return_pct" not in frame.columns:
        frame["daily_return_pct"] = pd.NA
    frame["涨幅%"] = pd.to_numeric(frame["daily_return_pct"], errors="coerce").round(4)
    frame["limit_up_excluded_by_daily_return"] = pd.to_numeric(frame["daily_return_pct"], errors="coerce").gt(
        LIMIT_UP_DAILY_RETURN_THRESHOLD * 100.0
    ).fillna(False)
    frame = add_centered_risk_scores(frame)
    frame = add_recommended_position_percent(frame)
    frame["pattern_match"] = frame["symbol"].isin(set(pattern_groups))
    frame["pattern_ids"] = frame["symbol"].map(
        lambda symbol: "/".join(str(value) for value in pattern_groups.get(symbol, pd.DataFrame()).get("pattern_id", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        if symbol in pattern_groups
        else ""
    )
    return frame.drop_duplicates("symbol", keep="first").reset_index(drop=True)


def build_route_watchlists(
    *,
    trade_date: date,
    project_root: Path,
    stock_scores: pd.DataFrame,
    sector_payload: dict[str, object],
    concern_members: pd.DataFrame,
    source_files: dict[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    sectors = _sector_frames(sector_payload)
    short_sectors = sectors["short"].head(ROUTE_SECTOR_TOP_N)
    phase9_sectors = sectors["phase9"].head(ROUTE_SECTOR_TOP_N)
    members = _prepare_concern_members(concern_members)
    stocks = _prepare_route_stock_scores(stock_scores)
    a1 = _build_a_route(
        trade_date=trade_date,
        route="A1",
        route_label="近期强势",
        sectors=short_sectors,
        stocks=stocks,
        members=members,
        sort_columns=["phase9_score_100", "centered_risk_score", "leader_score", "symbol"],
        sort_ascending=[False, False, False, True],
    )
    a2 = _build_a_route(
        trade_date=trade_date,
        route="A2",
        route_label="轮转预期",
        sectors=phase9_sectors,
        stocks=stocks,
        members=members,
        sort_columns=["centered_risk_score", "phase4_score_100", "leader_score", "symbol"],
        sort_ascending=[False, False, False, True],
    )
    b = _build_b_route(
        trade_date=trade_date,
        stocks=stocks,
        members=members,
        short_sectors=short_sectors,
        phase9_sectors=phase9_sectors,
    )
    pool = _build_sector_leader_pool(
        trade_date=trade_date,
        short_sectors=short_sectors,
        phase9_sectors=phase9_sectors,
        members=members,
        stocks=stocks,
    )
    policy = {
        "public_filter": "daily_return_pct <= 9.9 and phase1_score_100 > 20 and phase2_score_100 > 20",
        "a1_route_filter": "public filter only; mixed score and leader score are left for review selection",
        "a2_route_filter": "public filter plus phase1_score_100 > 40 and phase2_score_100 > 40; mixed score and leader score are left for review selection",
        "a1_sector_scope": "long mainline Top100 internal short-mainline Top20",
        "a2_sector_scope": "long mainline Top100 internal phase9 Top20",
        "b_route_filter": "any pattern hit plus public filter; no A1/A2 mixed ranking",
    }
    return {
        "a1": _payload(trade_date=trade_date, route="A1", label="近期强势", candidates=a1, sectors=_sector_records(short_sectors), source_files=source_files, policy=policy),
        "a2": _payload(trade_date=trade_date, route="A2", label="轮转预期", candidates=a2, sectors=_sector_records(phase9_sectors), source_files=source_files, policy=policy),
        "b": _payload(trade_date=trade_date, route="B", label="形态符合", candidates=b, sectors=[], source_files=source_files, policy=policy),
        "sector_leader_pool": {
            "trade_date": trade_date.isoformat(),
            "source_files": source_files or {},
            "selection_policy": {
                "short_sector_top_n": ROUTE_SECTOR_TOP_N,
                "phase9_sector_top_n": ROUTE_SECTOR_TOP_N,
                "leader_top_n_per_sector": SECTOR_LEADER_POOL_TOP_N,
                "source_scope": "daily_sector_leader_pool",
            },
            **pool,
        },
    }


def _build_a_route(
    *,
    trade_date: date,
    route: str,
    route_label: str,
    sectors: pd.DataFrame,
    stocks: pd.DataFrame,
    members: pd.DataFrame,
    sort_columns: list[str],
    sort_ascending: list[bool],
) -> list[dict[str, object]]:
    rows = _join_sector_members(sectors=sectors, members=members, stocks=stocks)
    if rows.empty:
        return []
    mask = rows["_public_filter_pass"]
    if route == "A2":
        mask = mask & rows["phase1_score_100"].gt(ROUTE_MIN_PHASE1) & rows["phase2_score_100"].gt(ROUTE_MIN_PHASE2)
    rows = rows[mask].copy()
    if rows.empty:
        return []
    rows["route"] = route
    rows["route_label"] = route_label
    rows = rows.sort_values(sort_columns, ascending=sort_ascending, na_position="last", kind="stable")
    rows = rows.drop_duplicates("symbol", keep="first").reset_index(drop=True)
    rows["route_rank"] = range(1, len(rows) + 1)
    return [_candidate_record(row, trade_date=trade_date) for _, row in rows.iterrows()]


def _build_b_route(
    *,
    trade_date: date,
    stocks: pd.DataFrame,
    members: pd.DataFrame,
    short_sectors: pd.DataFrame,
    phase9_sectors: pd.DataFrame,
) -> list[dict[str, object]]:
    rows = stocks[stocks["_public_filter_pass"] & stocks["pattern_match"].fillna(False).astype(bool)].copy()
    if rows.empty:
        return []
    matched = _best_route_sector_match(rows=rows, members=members, route_sectors=pd.concat([short_sectors, phase9_sectors], ignore_index=True))
    rows = rows.merge(matched, on="symbol", how="left")
    rows["route"] = "B"
    rows["route_label"] = "形态符合"
    rows["leader_score"] = pd.to_numeric(rows.get("leader_score"), errors="coerce")
    rows = rows.sort_values(["centered_risk_score", "phase4_score_100", "leader_score", "symbol"], ascending=[False, False, False, True], na_position="last")
    rows["route_rank"] = range(1, len(rows) + 1)
    return [_candidate_record(row, trade_date=trade_date) for _, row in rows.iterrows()]


def _build_sector_leader_pool(
    *,
    trade_date: date,
    short_sectors: pd.DataFrame,
    phase9_sectors: pd.DataFrame,
    members: pd.DataFrame,
    stocks: pd.DataFrame,
) -> dict[str, object]:
    sectors = pd.concat([short_sectors.assign(pool_reason="短期强度Top20"), phase9_sectors.assign(pool_reason="P9买入Top20")], ignore_index=True)
    if sectors.empty:
        return {"sector_count": 0, "candidate_count": 0, "sectors": [], "candidates": []}
    sectors = sectors.drop_duplicates(["sector_type", "sector_label"], keep="first").reset_index(drop=True)
    sector_records: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    stock_lookup = stocks.set_index("symbol", drop=False) if not stocks.empty and "symbol" in stocks.columns else pd.DataFrame()
    for _, sector in sectors.iterrows():
        key_members = members[
            members["sector_type"].eq(str(sector["sector_type"]))
            & members["sector_label"].eq(str(sector["sector_label"]))
        ].copy()
        key_members = key_members.sort_values(["leader_score", "symbol"], ascending=[False, True], na_position="last").head(SECTOR_LEADER_POOL_TOP_N)
        leaders: list[dict[str, object]] = []
        for _, member in key_members.iterrows():
            symbol = normalize_symbol(member.get("symbol"))
            stock = stock_lookup.loc[symbol] if symbol in stock_lookup.index else pd.Series(dtype=object)
            item = _pool_candidate_record(trade_date=trade_date, sector=sector, member=member, stock=stock)
            leaders.append(item)
            candidate_rows.append(item)
        sector_records.append({**_sector_record(sector), "pool_reason": sector.get("pool_reason"), "leaders": leaders})
    candidates = _dedupe_pool_candidates(candidate_rows)
    return {
        "sector_count": len(sector_records),
        "candidate_count": len(candidates),
        "sectors": sector_records,
        "candidates": candidates,
    }


def _sector_frames(sector_payload: dict[str, object]) -> dict[str, pd.DataFrame]:
    raw = sector_payload.get("sectors") if isinstance(sector_payload, dict) else []
    frame = pd.DataFrame(raw if isinstance(raw, list) else [])
    if frame.empty:
        columns = ["sector_type", "sector_label", "sector_name", "long_mainline_score_100", "short_mainline_score_100", "phase9_score_100"]
        empty = pd.DataFrame(columns=columns)
        return {"long": empty, "short": empty, "phase9": empty}
    for column in ("sector_type", "sector_label", "sector_name"):
        frame[column] = _series(frame, column).fillna("").astype(str).str.strip()
    for column in ("long_mainline_score_100", "short_mainline_score_100", "phase9_score_100", "sector_avg_pct_change", "sector_up_ratio"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    long_pool = frame[_series(frame, "selected_as_long_mainline", default=False).fillna(False).astype(bool)].copy()
    if long_pool.empty:
        long_pool = frame.sort_values(["long_mainline_score_100", "sector_name"], ascending=[False, True], na_position="last").head(100)
    short = long_pool.sort_values(["short_mainline_score_100", "long_mainline_score_100", "sector_name"], ascending=[False, False, True], na_position="last")
    phase9 = long_pool.sort_values(["phase9_score_100", "long_mainline_score_100", "sector_name"], ascending=[False, False, True], na_position="last")
    return {"long": long_pool.reset_index(drop=True), "short": short.reset_index(drop=True), "phase9": phase9.reset_index(drop=True)}


def _prepare_route_stock_scores(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["symbol"] = result["symbol"].map(normalize_symbol)
    for column in ("phase1_score_100", "phase2_score_100", "phase4_score_100", "centered_risk_score", "涨幅%", "daily_return_pct"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    for column, default in (("pattern_match", False), ("pattern_ids", ""), ("limit_up_excluded_by_daily_return", False)):
        if column not in result.columns:
            result[column] = default
    result["_public_filter_pass"] = (
        result["phase1_score_100"].gt(PUBLIC_MIN_PHASE1)
        & result["phase2_score_100"].gt(PUBLIC_MIN_PHASE2)
        & ~_series(result, "limit_up_excluded_by_daily_return", default=False).fillna(False).astype(bool)
    )
    return result.drop_duplicates("symbol", keep="first")


def _prepare_concern_members(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "name", "sector_type", "sector_name", "sector_label", "leader_score"])
    result = frame.rename(
        columns={
            "编号": "symbol",
            "名称": "name",
            "板块类型": "sector_type",
            "板块名称": "sector_name",
            "板块代码": "sector_label",
            "龙头指数": "leader_score",
            "长期龙头指数": "long_term_leader_score",
            "波段龙头指数": "swing_leader_score",
            "龙头标签": "leader_tags",
        }
    ).copy()
    result["symbol"] = result["symbol"].map(normalize_symbol)
    for column in ("sector_type", "sector_name", "sector_label", "name", "leader_tags"):
        if column not in result.columns:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str).str.strip()
    for column in ("leader_score", "long_term_leader_score", "swing_leader_score"):
        result[column] = pd.to_numeric(result.get(column), errors="coerce")
    return result.dropna(subset=["symbol", "leader_score"]).copy()


def _join_sector_members(*, sectors: pd.DataFrame, members: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    if sectors.empty or members.empty or stocks.empty:
        return pd.DataFrame()
    sector_keys = sectors.loc[:, ["sector_type", "sector_label", "sector_name", "long_mainline_score_100", "short_mainline_score_100", "phase9_score_100", "sector_avg_pct_change", "sector_up_ratio"]].copy()
    rows = members.merge(sector_keys, on=["sector_type", "sector_label"], how="inner", suffixes=("", "_sector"))
    rows["matched_mainline_sector"] = rows["sector_name_sector"].where(_series(rows, "sector_name_sector").notna(), rows["sector_name"])
    rows = rows.merge(stocks, on="symbol", how="inner", suffixes=("", "_stock"))
    if "name_stock" not in rows.columns:
        rows["name_stock"] = ""
    rows["name"] = rows["name_stock"].where(rows["name_stock"].astype(str).str.strip().ne(""), _series(rows, "name"))
    return rows


def _best_route_sector_match(*, rows: pd.DataFrame, members: pd.DataFrame, route_sectors: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or members.empty or route_sectors.empty:
        return pd.DataFrame(columns=["symbol", "matched_mainline_sector", "leader_score", "long_mainline_score_100", "short_mainline_score_100", "phase9_score_100"])
    joined = _join_sector_members(sectors=route_sectors, members=members, stocks=rows.loc[:, ["symbol"]].drop_duplicates())
    if joined.empty:
        return pd.DataFrame(columns=["symbol", "matched_mainline_sector", "leader_score", "long_mainline_score_100", "short_mainline_score_100", "phase9_score_100"])
    joined = joined.sort_values(["leader_score", "phase9_score_100", "short_mainline_score_100"], ascending=[False, False, False], na_position="last")
    return joined.drop_duplicates("symbol", keep="first").loc[:, ["symbol", "matched_mainline_sector", "leader_score", "long_mainline_score_100", "short_mainline_score_100", "phase9_score_100"]]


def _prepare_daily_returns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    result = frame.loc[:, [column for column in ("symbol", "name", "daily_close", "daily_return_pct", "daily_return_1d") if column in frame.columns]].copy()
    result["symbol"] = result["symbol"].map(normalize_symbol)
    return result.drop_duplicates("symbol", keep="first")


def _prepare_stock_concerns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    result = frame.rename(
        columns={
            "编号": "symbol",
            "是否弱势股": "weak_stock",
            "关切板块": "concern_sectors",
            "最高龙头指数": "top_concern_leader_score",
            "关切板块数量": "concern_sector_count",
        }
    ).copy()
    result["symbol"] = result["symbol"].map(normalize_symbol)
    return result.loc[:, [column for column in ("symbol", "weak_stock", "concern_sectors", "top_concern_leader_score", "concern_sector_count") if column in result.columns]].drop_duplicates("symbol", keep="first")


def _candidate_record(row: pd.Series, *, trade_date: date) -> dict[str, object]:
    return {
        "trade_date": trade_date.isoformat(),
        "route": _cell(row.get("route")),
        "route_label": _cell(row.get("route_label")),
        "route_rank": _cell(row.get("route_rank")),
        "symbol": normalize_symbol(row.get("symbol")),
        "name": _cell(row.get("name")),
        "daily_return_pct": _round(row.get("涨幅%", row.get("daily_return_pct")), 4),
        "matched_mainline_sector": _cell(row.get("matched_mainline_sector")),
        "concern_sectors": _cell(row.get("concern_sectors")),
        "leader_score": _round(row.get("leader_score"), 2),
        "long_mainline_score_100": _round(row.get("long_mainline_score_100"), 2),
        "short_mainline_score_100": _round(row.get("short_mainline_score_100"), 2),
        "phase9_score_100": _round(row.get("phase9_score_100"), 2),
        "phase1_score_100": _round(row.get("phase1_score_100"), 2),
        "phase2_score_100": _round(row.get("phase2_score_100"), 2),
        "phase4_score_100": _round(row.get("phase4_score_100"), 2),
        "phase4_5d_mean": _round(row.get("phase4_5d_mean"), 2),
        "phase4_5d_std": _round(row.get("phase4_5d_std"), 2),
        "mixed_score": _round(row.get("centered_risk_score"), 4),
        "pattern_match": bool(row.get("pattern_match", False)),
        "pattern_ids": _cell(row.get("pattern_ids")),
        "ATR%": _round(row.get("ATR%", row.get("atr_pct_14")), 2),
        RECOMMENDED_POSITION_PERCENT_FIELD: _round(row.get(RECOMMENDED_POSITION_PERCENT_FIELD), 2),
        "macd_cross_state": _cell(row.get("macd_cross_state")),
        "macd_divergence_state": _cell(row.get("macd_divergence_state")),
        "volume_price_divergence_state": _cell(row.get("volume_price_divergence_state")),
    }


def _pool_candidate_record(*, trade_date: date, sector: pd.Series, member: pd.Series, stock: pd.Series) -> dict[str, object]:
    symbol = normalize_symbol(member.get("symbol"))
    return {
        "trade_date": trade_date.isoformat(),
        "symbol": symbol,
        "name": _cell(stock.get("name")) or _cell(member.get("name")),
        "source": "sector_leader_pool",
        "pool_route": _cell(sector.get("pool_reason")),
        "matched_mainline_sector": _cell(sector.get("sector_name")),
        "sector_type": _cell(sector.get("sector_type")),
        "sector_label": _cell(sector.get("sector_label")),
        "leader_score": _round(member.get("leader_score"), 2),
        "long_mainline_score_100": _round(sector.get("long_mainline_score_100"), 2),
        "short_mainline_score_100": _round(sector.get("short_mainline_score_100"), 2),
        "phase9_score_100": _round(sector.get("phase9_score_100"), 2),
        "phase1_score_100": _round(stock.get("phase1_score_100"), 2),
        "phase2_score_100": _round(stock.get("phase2_score_100"), 2),
        "phase4_score_100": _round(stock.get("phase4_score_100"), 2),
        "mixed_score": _round(stock.get("centered_risk_score"), 4),
        "concern_sectors": _cell(stock.get("concern_sectors")),
    }


def _dedupe_pool_candidates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_symbol: dict[str, dict[str, object]] = {}
    sector_map: dict[str, list[str]] = {}
    for row in rows:
        symbol = normalize_symbol(row.get("symbol"))
        if not symbol:
            continue
        sector_map.setdefault(symbol, []).append(str(row.get("matched_mainline_sector") or ""))
        current = by_symbol.get(symbol)
        if current is None or _float(row.get("leader_score")) > _float(current.get("leader_score")):
            by_symbol[symbol] = dict(row)
    result = []
    for row in by_symbol.values():
        symbol = normalize_symbol(row.get("symbol"))
        row["source_sectors"] = "/".join(dict.fromkeys(item for item in sector_map.get(symbol, []) if item))
        result.append(row)
    return sorted(result, key=lambda item: (-_float(item.get("leader_score")), normalize_symbol(item.get("symbol"))))


def _payload(
    *,
    trade_date: date,
    route: str,
    label: str,
    candidates: list[dict[str, object]],
    sectors: list[dict[str, object]],
    source_files: dict[str, str] | None,
    policy: dict[str, object],
) -> dict[str, object]:
    return {
        "trade_date": trade_date.isoformat(),
        "route": route,
        "route_label": label,
        "source_files": source_files or {},
        "selection_policy": policy,
        "sector_count": len(sectors),
        "candidate_count": len(candidates),
        "sectors": sectors,
        "candidates": candidates,
    }


def _sector_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [_sector_record(row) for _, row in frame.iterrows()] if not frame.empty else []


def _sector_record(row: pd.Series) -> dict[str, object]:
    return {
        "sector_type": _cell(row.get("sector_type")),
        "sector_name": _cell(row.get("sector_name")),
        "sector_label": _cell(row.get("sector_label")),
        "long_mainline_score_100": _round(row.get("long_mainline_score_100"), 2),
        "short_mainline_score_100": _round(row.get("short_mainline_score_100"), 2),
        "phase9_score_100": _round(row.get("phase9_score_100"), 2),
        "sector_avg_pct_change": _round(row.get("sector_avg_pct_change"), 4),
        "sector_up_ratio": _round(row.get("sector_up_ratio"), 4),
    }


def _write_payload(path: Path, payload: dict[str, object], *, csv_kind: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    if csv_kind == "pool":
        _pool_csv_frame(payload).to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    else:
        _candidate_csv_frame(payload).to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    return path


def _candidate_csv_frame(payload: dict[str, object]) -> pd.DataFrame:
    rows = payload.get("candidates")
    rows = rows if isinstance(rows, list) else []
    output = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        output.append(
            {
                "交易日期": item.get("trade_date"),
                "编号": _excel_symbol(item.get("symbol")),
                "名称": item.get("name"),
                "涨幅%": item.get("daily_return_pct"),
                "来源路线": item.get("route_label"),
                "关切板块": item.get("concern_sectors"),
                "契合主线": item.get("matched_mainline_sector"),
                "龙头指数": item.get("leader_score"),
                "P1": item.get("phase1_score_100"),
                "P2": item.get("phase2_score_100"),
                "P4": item.get("phase4_score_100"),
                "混合分": item.get("mixed_score"),
                "Pattern": item.get("pattern_ids") if item.get("pattern_match") else "",
                "ATR%": item.get("ATR%"),
                "建议仓位": item.get(RECOMMENDED_POSITION_PERCENT_FIELD),
                "长期主线指数": item.get("long_mainline_score_100"),
                "短期主线指数": item.get("short_mainline_score_100"),
                "P9买入分": item.get("phase9_score_100"),
                "P4五日均": item.get("phase4_5d_mean"),
                "P4五日std": item.get("phase4_5d_std"),
                "macd": item.get("macd_cross_state"),
            }
        )
    return pd.DataFrame(output, columns=_candidate_csv_columns())


def _pool_csv_frame(payload: dict[str, object]) -> pd.DataFrame:
    rows = payload.get("candidates")
    rows = rows if isinstance(rows, list) else []
    output = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        output.append(
            {
                "交易日期": item.get("trade_date"),
                "编号": _excel_symbol(item.get("symbol")),
                "名称": item.get("name"),
                "来源路线": item.get("pool_route"),
                "关切板块": item.get("source_sectors") or item.get("concern_sectors"),
                "契合主线": item.get("matched_mainline_sector"),
                "龙头指数": item.get("leader_score"),
                "P1": item.get("phase1_score_100"),
                "P2": item.get("phase2_score_100"),
                "P4": item.get("phase4_score_100"),
                "混合分": item.get("mixed_score"),
                "长期主线指数": item.get("long_mainline_score_100"),
                "短期主线指数": item.get("short_mainline_score_100"),
                "P9买入分": item.get("phase9_score_100"),
            }
        )
    return pd.DataFrame(output)


def _candidate_csv_columns() -> list[str]:
    return [
        "交易日期",
        "编号",
        "名称",
        "涨幅%",
        "来源路线",
        "关切板块",
        "契合主线",
        "龙头指数",
        "P1",
        "P2",
        "P4",
        "混合分",
        "Pattern",
        "ATR%",
        "建议仓位",
        "长期主线指数",
        "短期主线指数",
        "P9买入分",
        "P4五日均",
        "P4五日std",
        "macd",
    ]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _series(frame: pd.DataFrame, name: str, *, default: object = "") -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index, dtype=object)


def _parse_date_from_name(name: str, prefix: str) -> date | None:
    stem = name.removeprefix(prefix + "_").removesuffix(".json")
    try:
        return date.fromisoformat(stem)
    except ValueError:
        return None


def _excel_symbol(symbol: object) -> str:
    return f'="{normalize_symbol(symbol)}"'


def _round(value: object, digits: int) -> object:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return round(number, digits)


def _float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if pd.isna(number):
        return float("-inf")
    return number


def _cell(value: object) -> object:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return value


def _json_default(value: object) -> object:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return str(value)
