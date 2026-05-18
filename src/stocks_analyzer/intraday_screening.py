from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .atr import build_atr_snapshot_row
from .concern_sectors import read_concern_sector_members
from .full_market_return import predict_alpha158_qlib_return
from .full_market_risk import predict_barrier_risk, predict_tail_risk
from .indicators import add_indicators
from .intraday_update import INTRADAY_DATA_INTERFACES, run_intraday_update
from .macd_divergence import summarize_recent_macd_divergence
from .phase4_rolling import PHASE4_ROLLING_COLUMNS, PHASE4_ROLLING_RANK_COLUMNS, merge_phase4_rolling_frame
from .phase_display import normalize_symbol
from .position_sizing import RECOMMENDED_POSITION_PERCENT_FIELD, add_recommended_position_percent
from .sector_phase9 import predict_sector_phase9_buy_score, sector_phase9_model_path, sector_phase9_predictions_path
from .sector_membership import append_sector_display_columns
from .sector_tracking_workbook import write_sector_intraday_tracking_workbook
from .sector_watchlist import build_sector_tracking_payload_from_files
from .storage import DailyBarsReadError, Storage
from .route_watchlists import (
    build_route_watchlists,
    find_latest_sector_leader_pool_before,
    watchlist_sector_leader_pool_path,
    write_sector_leader_pool_payload,
)
from .watchlist import (
    _prepare_atr_frame,
    _prepare_macd_frame,
    _prepare_phase4_predictions,
    _prepare_phase_risk_predictions,
    add_centered_risk_scores,
)

DEFAULT_INTRADAY_REPORT_KEEP_DATES = 10
_DATED_REPORT_PATTERNS = (
    re.compile(r"^intraday_watchlist_a_(\d{4}-\d{2}-\d{2})\.csv$"),
    re.compile(r"^intraday_watchlist_a1_recent_mainline_(\d{4}-\d{2}-\d{2})\.csv$"),
    re.compile(r"^intraday_watchlist_a2_rotation_expected_(\d{4}-\d{2}-\d{2})\.csv$"),
    re.compile(r"^intraday_sector_strength_(\d{4}-\d{2}-\d{2})\.csv$"),
)


@dataclass(slots=True)
class IntradayScreeningResult:
    trade_date: date
    source_pool_path: Path
    output_path: Path
    sector_strength_path: Path | None
    a1_path: Path | None
    a2_path: Path | None
    candidate_count: int
    pool_candidate_count: int
    intraday_updated_count: int
    missing_intraday_symbols: list[str]
    cleaned_report_files: int
    phase1_path: Path
    phase2_path: Path
    phase4_path: Path
    full_market_pool_refreshed: bool = False
    full_market_scanned_count: int = 0


@dataclass(slots=True)
class _AnalysisResult:
    frame: pd.DataFrame
    candidates: list[dict[str, object]]
    intraday: pd.DataFrame
    phase1: pd.DataFrame
    phase2: pd.DataFrame
    phase4: pd.DataFrame
    macd: pd.DataFrame
    atr: pd.DataFrame
    phase1_path: Path
    phase2_path: Path
    phase4_path: Path


def run_intraday_screening(
    *,
    storage: Storage,
    project_root: Path,
    trade_date: date,
    data_interface: str = "sina_raw",
    limit: int | None = None,
    skip_intraday_update: bool = False,
    timeout_seconds: float = 15.0,
    chunk_size: int = 50,
    output: Path | None = None,
    report_keep_dates: int | None = DEFAULT_INTRADAY_REPORT_KEEP_DATES,
    refresh_full_market_pool: bool = False,
) -> IntradayScreeningResult:
    updated_count = 0
    missing_update_symbols: list[str] = []
    full_market_scanned_count = 0

    if refresh_full_market_pool:
        (
            source_pool_path,
            pool_payload,
            updated_count,
            missing_update_symbols,
            full_market_scanned_count,
        ) = _refresh_full_market_intraday_pool(
            storage=storage,
            project_root=project_root,
            trade_date=trade_date,
            data_interface=data_interface,
            limit=limit,
            skip_intraday_update=skip_intraday_update,
            timeout_seconds=timeout_seconds,
            chunk_size=chunk_size,
        )
    else:
        source_pool_path, pool_payload = _load_intraday_pool_for_screening(project_root, trade_date)

    pool_candidates = _previous_pool_candidates(pool_payload, limit=None)
    candidates = pool_candidates
    if limit is not None:
        candidates = candidates[: max(int(limit), 0)]
    symbols = [candidate["symbol"] for candidate in candidates]
    if not symbols:
        raise RuntimeError(f"No sector leader pool symbols found in {source_pool_path}")

    if not skip_intraday_update and not refresh_full_market_pool:
        update_result = run_intraday_update(
            storage=storage,
            project_root=project_root,
            source=data_interface,
            symbols=symbols,
            timeout_seconds=timeout_seconds,
            chunk_size=chunk_size,
        )
        updated_count = len(update_result.updated_symbols)
        missing_update_symbols = update_result.failed_symbols

    output_dir = project_root / "reports" / "intraday_screening"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = storage.paths.intraday_dir / "screening_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    analysis = _run_candidate_analysis(
        storage=storage,
        project_root=project_root,
        trade_date=trade_date,
        candidates=candidates,
        output_dir=cache_dir,
        file_tag="pool",
    )
    output_path = output if output is not None else output_dir / f"intraday_watchlist_a_{trade_date.isoformat()}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _format_intraday_output_for_csv(analysis.frame).to_csv(output_path, index=False, encoding="utf-8-sig")
    sector_strength_path = _save_intraday_sector_strength(
        output_dir=output_dir,
        project_root=project_root,
        trade_date=trade_date,
        frame=analysis.frame,
        source_pool_payload=pool_payload,
    )
    a1_path, a2_path = _save_intraday_route_watchlists(
        output_dir=output_dir,
        trade_date=trade_date,
        frame=analysis.frame,
        sector_strength_path=sector_strength_path,
    )
    cleaned_report_files = _cleanup_intraday_screening_reports(
        output_dir,
        keep_dates=report_keep_dates,
        preserve_paths=(output_path, sector_strength_path, a1_path, a2_path),
    )

    intraday = _load_intraday_snapshot_frame(storage, symbols=symbols)
    missing_intraday = [symbol for symbol in symbols if symbol not in set(intraday.get("symbol", pd.Series(dtype=str)).astype(str))]
    return IntradayScreeningResult(
        trade_date=trade_date,
        source_pool_path=source_pool_path,
        output_path=output_path,
        sector_strength_path=sector_strength_path,
        a1_path=a1_path,
        a2_path=a2_path,
        candidate_count=len(analysis.frame),
        pool_candidate_count=len(pool_candidates),
        intraday_updated_count=updated_count,
        missing_intraday_symbols=sorted(set(missing_update_symbols + missing_intraday)),
        cleaned_report_files=cleaned_report_files,
        phase1_path=analysis.phase1_path,
        phase2_path=analysis.phase2_path,
        phase4_path=analysis.phase4_path,
        full_market_pool_refreshed=refresh_full_market_pool,
        full_market_scanned_count=full_market_scanned_count,
    )


class _IntradayOverlayStorage:
    def __init__(self, *, storage: Storage, universe: pd.DataFrame, trade_date: date) -> None:
        self._storage = storage
        self._universe = universe.copy()
        self._trade_date = trade_date
        self.paths = storage.paths

    def load_universe(self) -> pd.DataFrame:
        return self._universe.copy()

    def load_daily_bars(self, symbol: str) -> pd.DataFrame:
        normalized_symbol = normalize_symbol(symbol)
        daily = self._storage.load_daily_bars(normalized_symbol).copy()
        daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce")
        daily = daily.dropna(subset=["trade_date"])
        daily = daily[daily["trade_date"].dt.date < self._trade_date].copy()

        try:
            intraday = self._storage.load_intraday_bars(normalized_symbol)
        except FileNotFoundError:
            return daily.sort_values("trade_date").reset_index(drop=True)

        provisional = _normalize_intraday_bar_for_daily(intraday, symbol=normalized_symbol, trade_date=self._trade_date)
        if provisional.empty:
            return daily.sort_values("trade_date").reset_index(drop=True)

        merged = pd.concat([daily, provisional], ignore_index=True)
        merged["trade_date"] = pd.to_datetime(merged["trade_date"], errors="coerce")
        merged = merged.dropna(subset=["trade_date"]).drop_duplicates("trade_date", keep="last")
        return merged.sort_values("trade_date").reset_index(drop=True)


def _load_previous_intraday_pool(project_root: Path, trade_date: date) -> tuple[Path, dict[str, object]]:
    _, path = find_latest_sector_leader_pool_before(project_root=project_root, trade_date=trade_date)
    return path, json.loads(path.read_text(encoding="utf-8"))


def _load_intraday_pool_for_screening(project_root: Path, trade_date: date) -> tuple[Path, dict[str, object]]:
    today_path = watchlist_sector_leader_pool_path(project_root, trade_date)
    if today_path.exists():
        payload = json.loads(today_path.read_text(encoding="utf-8"))
        if _is_intraday_full_market_pool(payload):
            return today_path, payload
    return _load_previous_intraday_pool(project_root, trade_date)


def _is_intraday_full_market_pool(payload: dict[str, object]) -> bool:
    policy = payload.get("selection_policy") if isinstance(payload, dict) else {}
    if not isinstance(policy, dict):
        return False
    return str(policy.get("source_scope", "")).strip() == "intraday_full_market"


def _refresh_full_market_intraday_pool(
    *,
    storage: Storage,
    project_root: Path,
    trade_date: date,
    data_interface: str,
    limit: int | None,
    skip_intraday_update: bool,
    timeout_seconds: float,
    chunk_size: int,
) -> tuple[Path, dict[str, object], int, list[str], int]:
    full_market_candidates = _full_market_candidates(storage, limit=limit)
    symbols = [candidate["symbol"] for candidate in full_market_candidates]
    if not symbols:
        raise RuntimeError("No full-market symbols found in local universe.")

    updated_count = 0
    missing_update_symbols: list[str] = []
    if not skip_intraday_update:
        update_result = run_intraday_update(
            storage=storage,
            project_root=project_root,
            source=data_interface,
            symbols=symbols,
            timeout_seconds=timeout_seconds,
            chunk_size=chunk_size,
        )
        updated_count = len(update_result.updated_symbols)
        missing_update_symbols = update_result.failed_symbols

    cache_dir = storage.paths.intraday_dir / "screening_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    analysis = _run_candidate_analysis(
        storage=storage,
        project_root=project_root,
        trade_date=trade_date,
        candidates=full_market_candidates,
        output_dir=cache_dir,
        file_tag="full_market",
    )
    route_inputs = analysis.frame.copy()
    route_inputs["daily_return_pct"] = pd.to_numeric(route_inputs.get("intraday_pct_change"), errors="coerce")
    route_inputs["涨幅%"] = route_inputs["daily_return_pct"]
    route_inputs["limit_up_excluded_by_daily_return"] = route_inputs["daily_return_pct"].gt(9.9).fillna(False)
    sector_payload = _load_latest_sector_watchlist(project_root=project_root, trade_date=trade_date)
    sector_date = _sector_payload_date(sector_payload, fallback=trade_date)
    concern_members = read_concern_sector_members(project_root=project_root, trade_date=sector_date)
    payloads = build_route_watchlists(
        trade_date=trade_date,
        project_root=project_root,
        stock_scores=route_inputs,
        sector_payload=sector_payload,
        concern_members=concern_members,
        source_files={
            "phase1": str(analysis.phase1_path),
            "phase2": str(analysis.phase2_path),
            "phase4": str(analysis.phase4_path),
            "macd": "intraday_overlay",
            "atr": "intraday_overlay",
        },
    )
    payload = payloads["sector_leader_pool"]
    selection_policy = dict(payload.get("selection_policy", {}))
    selection_policy["source_scope"] = "intraday_full_market"
    selection_policy["full_market_scan_symbols"] = int(len(full_market_candidates))
    selection_policy["full_market_pool_date"] = trade_date.isoformat()
    selection_policy["full_market_intraday_updated"] = int(updated_count)
    selection_policy["full_market_intraday_missing"] = int(len(missing_update_symbols))
    payload["selection_policy"] = selection_policy

    target = write_sector_leader_pool_payload(project_root=project_root, trade_date=trade_date, payload=payload)
    written_payload = json.loads(target.read_text(encoding="utf-8"))
    return target, written_payload, updated_count, missing_update_symbols, len(full_market_candidates)


def _full_market_candidates(storage: Storage, *, limit: int | None) -> list[dict[str, object]]:
    universe = storage.load_universe().copy()
    if "symbol" not in universe.columns:
        return []
    universe["symbol"] = universe["symbol"].map(normalize_symbol)
    universe = universe[universe["symbol"].astype(str).str.len().eq(6)].drop_duplicates("symbol", keep="first")
    if "name" not in universe.columns:
        universe["name"] = ""
    if limit is not None:
        universe = universe.head(max(int(limit), 0))
    return [
        {
            "symbol": str(row.get("symbol", "")),
            "name": str(row.get("name", "") or ""),
            "universe_rank": index,
            "prev_source": "full_market_scan",
            "prev_source_tags": "",
            "prev_pattern_match": False,
            "prev_pattern_id": "",
            "prev_pattern_ids": "",
            "prev_patterns": "",
            "prev_reason": "",
        }
        for index, row in enumerate(universe.to_dict("records"), start=1)
    ]


def _load_latest_pattern_frame(*, project_root: Path, trade_date: date) -> tuple[Path | None, pd.DataFrame]:
    pattern_dir = project_root / "reports" / "patterns"
    dated: list[tuple[date, Path]] = []
    for path in pattern_dir.glob("patterns_all_*.csv"):
        match = re.fullmatch(r"patterns_all_(\d{4}-\d{2}-\d{2})\.csv", path.name)
        if not match:
            continue
        try:
            parsed = date.fromisoformat(match.group(1))
        except ValueError:
            continue
        if parsed <= trade_date:
            dated.append((parsed, path))
    if not dated:
        return None, pd.DataFrame()
    _, path = max(dated, key=lambda item: item[0])
    return path, pd.read_csv(path)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _previous_pool_candidates(payload: dict[str, object], *, limit: int | None) -> list[dict[str, object]]:
    raw_candidates = payload.get("candidates") if isinstance(payload, dict) else []
    if not isinstance(raw_candidates, list):
        return []

    extracted_symbols = _extract_pool_symbols(payload)
    order = {symbol: index for index, symbol in enumerate(extracted_symbols)}
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        symbol = normalize_symbol(raw.get("symbol", ""))
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        candidates.append(
            {
                "symbol": symbol,
                "name": str(raw.get("name", "") or ""),
                "prev_rank": order.get(symbol, len(order)) + 1,
                "prev_source": raw.get("source"),
                "prev_source_tags": _jsonish(raw.get("source_tags")),
                "prev_pattern_match": raw.get("pattern_match"),
                "prev_pattern_id": raw.get("pattern_id"),
                "prev_pattern_ids": _jsonish(raw.get("pattern_ids")),
                "prev_patterns": _jsonish(raw.get("patterns")),
                "prev_reason": raw.get("reason"),
                "prev_phase1_score_100": raw.get("phase1_score_100"),
                "prev_phase2_score_100": raw.get("phase2_score_100"),
                "prev_phase4_score_100": raw.get("phase4_score_100"),
                "prev_watchlist_streak": raw.get("连续上榜天数"),
                "pool_route": raw.get("pool_route"),
                "matched_mainline_sector": raw.get("matched_mainline_sector"),
                "source_sectors": raw.get("source_sectors") or raw.get("concern_sectors"),
                "sector_type": raw.get("sector_type"),
                "sector_label": raw.get("sector_label"),
                "leader_score": raw.get("leader_score"),
                "long_mainline_score_100": raw.get("long_mainline_score_100"),
                "short_mainline_score_100": raw.get("short_mainline_score_100"),
                "phase9_score_100": raw.get("phase9_score_100"),
            }
        )
    candidates = sorted(candidates, key=lambda item: int(item["prev_rank"]))
    if limit is not None:
        candidates = candidates[: max(int(limit), 0)]
    return candidates


def _extract_pool_symbols(payload: dict[str, object]) -> list[str]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return []
    symbols: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        symbol = normalize_symbol(item.get("symbol", ""))
        if not symbol or symbol in seen:
            continue
        symbols.append(symbol)
        seen.add(symbol)
    return symbols


def _candidate_universe(candidates: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": candidate["symbol"],
                "name": candidate.get("name", ""),
                "latest_price": pd.NA,
                "volume": pd.NA,
                "amount": pd.NA,
                "turnover_rate": pd.NA,
            }
            for candidate in candidates
        ]
    )


def _run_candidate_analysis(
    *,
    storage: Storage,
    project_root: Path,
    trade_date: date,
    candidates: list[dict[str, object]],
    output_dir: Path,
    file_tag: str,
) -> _AnalysisResult:
    symbols = [candidate["symbol"] for candidate in candidates]
    universe = _candidate_universe(candidates)
    overlay_storage = _IntradayOverlayStorage(storage=storage, universe=universe, trade_date=trade_date)
    phase1_path = output_dir / f"intraday_{file_tag}_tail_risk_predictions_{trade_date.isoformat()}.csv"
    phase2_path = output_dir / f"intraday_{file_tag}_barrier_risk_predictions_{trade_date.isoformat()}.csv"
    phase4_path = output_dir / f"intraday_{file_tag}_alpha158_qlib_return_predictions_{trade_date.isoformat()}.csv"

    phase1 = predict_tail_risk(
        storage=overlay_storage,
        project_root=project_root,
        trade_date=trade_date,
        output=phase1_path,
        latest_only=True,
        feature_lookback_bars=61,
        include_features=False,
        prediction_scope="intraday_screening",
    ).predictions
    phase2 = predict_barrier_risk(
        storage=overlay_storage,
        project_root=project_root,
        trade_date=trade_date,
        output=phase2_path,
        latest_only=True,
        feature_lookback_bars=61,
        include_features=False,
        prediction_scope="intraday_screening",
    ).predictions
    phase4 = predict_alpha158_qlib_return(
        storage=overlay_storage,
        project_root=project_root,
        trade_date=trade_date,
        output=phase4_path,
        latest_only=True,
        feature_lookback_bars=61,
        include_features=False,
        prediction_scope="intraday_screening",
    ).predictions
    macd = _build_intraday_macd_summary(overlay_storage, trade_date=trade_date, symbols=symbols)
    atr = _build_intraday_atr_summary(overlay_storage, trade_date=trade_date, symbols=symbols)
    intraday = _load_intraday_snapshot_frame(storage, symbols=symbols)
    frame = _build_output_frame(
        project_root=project_root,
        trade_date=trade_date,
        candidates=candidates,
        intraday=intraday,
        phase1=phase1,
        phase2=phase2,
        phase4=phase4,
        macd=macd,
        atr=atr,
    )
    return _AnalysisResult(
        frame=frame,
        candidates=candidates,
        intraday=intraday,
        phase1=phase1,
        phase2=phase2,
        phase4=phase4,
        macd=macd,
        atr=atr,
        phase1_path=phase1_path,
        phase2_path=phase2_path,
        phase4_path=phase4_path,
    )


def _concat_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    available = [frame for frame in frames if frame is not None and not frame.empty]
    if not available:
        return pd.DataFrame()
    return pd.concat(available, ignore_index=True)


def _save_intraday_sector_strength(
    *,
    output_dir: Path,
    project_root: Path,
    trade_date: date,
    frame: pd.DataFrame,
    source_pool_payload: dict[str, object],
) -> Path | None:
    sector_payload = _load_latest_sector_watchlist(project_root=project_root, trade_date=trade_date)
    sector_date = _sector_payload_date(sector_payload, fallback=trade_date)
    _ensure_intraday_sector_phase9_predictions(project_root=project_root, trade_date=sector_date)
    tracking_payload = build_sector_tracking_payload_from_files(project_root=project_root, trade_date=sector_date)
    sectors = source_pool_payload.get("sectors") if isinstance(source_pool_payload, dict) else []
    if not isinstance(sectors, list) or not sectors:
        return None
    if frame.empty or "symbol" not in frame.columns:
        return None

    working = frame.copy()
    working["symbol"] = working["symbol"].map(normalize_symbol)
    pct = pd.to_numeric(working.get("intraday_pct_change"), errors="coerce")
    pct_by_symbol = dict(zip(working["symbol"], pct, strict=False))
    rows: list[dict[str, object]] = []
    for item in sectors:
        if not isinstance(item, dict):
            continue
        leaders = item.get("leaders")
        if isinstance(leaders, list) and leaders:
            normalized_symbols = [normalize_symbol(leader.get("symbol")) for leader in leaders if isinstance(leader, dict)]
            leader_names = [str(leader.get("name") or "") for leader in leaders if isinstance(leader, dict)]
        else:
            normalized_symbols = [normalize_symbol(symbol) for symbol in item.get("leader_symbols", []) if normalize_symbol(symbol)] if isinstance(item.get("leader_symbols"), list) else []
            leader_names = item.get("leader_names", []) if isinstance(item.get("leader_names"), list) else []
        leader_pct = [pct_by_symbol[symbol] for symbol in normalized_symbols if symbol in pct_by_symbol and pd.notna(pct_by_symbol[symbol])]
        rows.append(
            {
                "日期": trade_date.isoformat(),
                "板块类型": _sector_type_cn(item.get("sector_type")),
                "板块名称": item.get("sector_name"),
                "长期主线指数": item.get("long_mainline_score_100"),
                "短期主线指数": item.get("short_mainline_score_100"),
                "P9买入分": item.get("phase9_score_100"),
                "来源路线": item.get("pool_reason"),
                "龙头编号": "/".join(normalized_symbols),
                "龙头名称": "/".join(leader_names),
                "龙头盘中平均涨幅%": round(float(pd.Series(leader_pct).mean()), 4) if leader_pct else pd.NA,
                "有效龙头数": len(leader_pct),
            }
        )
    if not rows:
        return None
    target = output_dir / f"intraday_sector_strength_{trade_date.isoformat()}.csv"
    output_frame = pd.DataFrame(rows).sort_values(
        ["龙头盘中平均涨幅%", "短期主线指数"],
        ascending=[False, False],
        na_position="last",
    )
    output_frame["盘中强度排名"] = output_frame["龙头盘中平均涨幅%"].rank(method="min", ascending=False).astype("Int64")
    output_frame["P9排名"] = pd.to_numeric(output_frame["P9买入分"], errors="coerce").rank(method="min", ascending=False).astype("Int64")
    output_frame.to_csv(
        target,
        index=False,
        encoding="utf-8-sig",
    )
    write_sector_intraday_tracking_workbook(
        project_root=project_root,
        trade_date=trade_date,
        sector_payload=tracking_payload,
        intraday_strength=output_frame,
    )
    return target


def _load_latest_sector_watchlist(*, project_root: Path, trade_date: date) -> dict[str, object]:
    candidates: list[tuple[date, Path]] = []
    for path in (project_root / "reports" / "watchlists").glob("watchlist_sectors_*.json"):
        match = re.fullmatch(r"watchlist_sectors_(\d{4}-\d{2}-\d{2})\.json", path.name)
        if not match:
            continue
        try:
            parsed = date.fromisoformat(match.group(1))
        except ValueError:
            continue
        if parsed <= trade_date:
            candidates.append((parsed, path))
    if not candidates:
        return {}
    _, path = max(candidates, key=lambda item: item[0])
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _sector_payload_date(payload: dict[str, object], *, fallback: date) -> date:
    text = str(payload.get("trade_date") or "").strip() if isinstance(payload, dict) else ""
    if text:
        try:
            return date.fromisoformat(text)
        except ValueError:
            pass
    return fallback


def _ensure_intraday_sector_phase9_predictions(*, project_root: Path, trade_date: date) -> Path | None:
    output_path = sector_phase9_predictions_path(project_root, trade_date)
    if output_path.exists():
        return output_path
    if not sector_phase9_model_path(project_root).exists():
        logging.warning("Sector Phase9 model artifact missing; intraday sector P9 scores stay blank: %s", sector_phase9_model_path(project_root))
        return None
    try:
        result = predict_sector_phase9_buy_score(project_root=project_root, trade_date=trade_date)
    except Exception as exc:  # pragma: no cover - defensive around local data/model availability.
        logging.warning("Sector Phase9 intraday prediction failed for %s: %s", trade_date.isoformat(), exc)
        return None
    return result.output_path


def _save_intraday_route_watchlists(
    *,
    output_dir: Path,
    trade_date: date,
    frame: pd.DataFrame,
    sector_strength_path: Path | None,
) -> tuple[Path | None, Path | None]:
    if frame.empty or sector_strength_path is None or not sector_strength_path.exists():
        return None, None
    sectors = pd.read_csv(sector_strength_path)
    if sectors.empty or "板块名称" not in sectors.columns:
        return None, None
    a1_sectors = sectors.sort_values(
        ["龙头盘中平均涨幅%", "P9买入分"],
        ascending=[False, False],
        na_position="last",
    ).head(10)["板块名称"].astype(str).tolist()
    a2_sectors = sectors.sort_values(
        ["P9买入分", "龙头盘中平均涨幅%"],
        ascending=[False, False],
        na_position="last",
    ).head(10)["板块名称"].astype(str).tolist()
    a1 = _select_intraday_a_route(frame, selected_sectors=a1_sectors, route_label="近期强势")
    a2 = _select_intraday_a_route(frame, selected_sectors=a2_sectors, route_label="轮转预期")
    a1_path = output_dir / f"intraday_watchlist_a1_recent_mainline_{trade_date.isoformat()}.csv"
    a2_path = output_dir / f"intraday_watchlist_a2_rotation_expected_{trade_date.isoformat()}.csv"
    _format_intraday_output_for_csv(a1).to_csv(a1_path, index=False, encoding="utf-8-sig")
    _format_intraday_output_for_csv(a2).to_csv(a2_path, index=False, encoding="utf-8-sig")
    return a1_path, a2_path


def _select_intraday_a_route(frame: pd.DataFrame, *, selected_sectors: list[str], route_label: str) -> pd.DataFrame:
    if frame.empty:
        return frame.head(0).copy()
    result = frame.copy()
    sector_text = result.get("matched_mainline_sector", pd.Series("", index=result.index)).fillna("").astype(str)
    source_sectors = result.get("source_sectors", pd.Series("", index=result.index)).fillna("").astype(str)
    selected = set(selected_sectors)
    sector_match = sector_text.isin(selected) | source_sectors.map(
        lambda text: any(item and item in selected for item in str(text).split("/"))
    )
    phase1 = pd.to_numeric(_frame_series(result, "phase1_score_100"), errors="coerce")
    phase2 = pd.to_numeric(_frame_series(result, "phase2_score_100"), errors="coerce")
    mask = (
        sector_match
        & phase1.gt(20)
        & phase2.gt(20)
        & pd.to_numeric(_frame_series(result, "intraday_pct_change"), errors="coerce").le(9.9).fillna(True)
    )
    if route_label == "轮转预期":
        mask = mask & phase1.gt(40) & phase2.gt(40)
    result = result[mask].copy()
    if result.empty:
        return result
    result["intraday_route"] = route_label
    result = _sort_intraday_pool(
        result,
        leading_columns=["phase9_score_100" if route_label == "近期强势" else "intraday_pool_score"],
        leading_ascending=[False],
    )
    return result.reset_index(drop=True)


def _frame_series(frame: pd.DataFrame, column: str, default: object = pd.NA) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _sector_type_cn(value: object) -> str:
    text = str(value or "")
    if text == "industry":
        return "行业"
    if text == "concept":
        return "概念"
    return text


def _add_intraday_pool_score(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    required = {"phase1_score_100", "phase2_score_100", "phase4_score_100"}
    if not required.issubset(result.columns):
        result["intraday_pool_score"] = pd.NA
        return result
    result = add_centered_risk_scores(result)
    result["intraday_pool_score"] = result["centered_risk_score"]
    return result


def _add_intraday_selection_source(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    base_source = result.get("prev_source", pd.Series("", index=result.index)).fillna("").astype(str)
    source = base_source.where(base_source.str.strip().ne(""), "sector_leader_pool")
    result["intraday_selection_source"] = source
    return result


def _sort_intraday_pool(
    frame: pd.DataFrame,
    *,
    leading_columns: list[str] | None = None,
    leading_ascending: list[bool] | None = None,
) -> pd.DataFrame:
    result = _add_intraday_pool_score(frame)
    leading = leading_columns or []
    leading_directions = leading_ascending if leading_ascending is not None else [True] * len(leading)
    sort_columns = [
        *leading,
        "intraday_pool_score",
        "phase4_score_100",
        "phase1_center_score",
        "phase2_center_score",
        "symbol",
    ]
    ascending = [
        *leading_directions,
        False,
        False,
        False,
        False,
        True,
    ]
    available_columns: list[str] = []
    available_ascending: list[bool] = []
    for column, direction in zip(sort_columns, ascending):
        if column in result.columns:
            available_columns.append(column)
            available_ascending.append(direction)
    if not available_columns:
        return result
    return result.sort_values(available_columns, ascending=available_ascending, na_position="last")


def _cleanup_intraday_screening_reports(
    output_dir: Path,
    *,
    keep_dates: int | None,
    preserve_paths: tuple[Path | None, ...],
) -> int:
    preserved = {_resolved(path) for path in preserve_paths if path is not None}
    deleted = 0
    for path in output_dir.glob("intraday_*_predictions_*.csv"):
        if _delete_report_file(path, preserved):
            deleted += 1

    if keep_dates is None:
        return deleted

    keep_count = max(int(keep_dates), 1)
    dated_files: list[tuple[date, Path]] = []
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        report_date = _dated_intraday_report_date(path.name)
        if report_date is not None:
            dated_files.append((report_date, path))
    keep = set(sorted({report_date for report_date, _ in dated_files}, reverse=True)[:keep_count])
    for report_date, path in dated_files:
        if report_date not in keep and _delete_report_file(path, preserved):
            deleted += 1
    return deleted


def _dated_intraday_report_date(filename: str) -> date | None:
    for pattern in _DATED_REPORT_PATTERNS:
        match = pattern.match(filename)
        if match is None:
            continue
        try:
            return date.fromisoformat(match.group(1))
        except ValueError:
            return None
    return None


def _delete_report_file(path: Path, preserved: set[Path]) -> bool:
    if not path.is_file() or _resolved(path) in preserved:
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True


def _resolved(path: Path) -> Path:
    return path.resolve()


def _normalize_intraday_bar_for_daily(intraday: pd.DataFrame, *, symbol: str, trade_date: date) -> pd.DataFrame:
    if intraday.empty:
        return pd.DataFrame()
    frame = intraday.copy()
    frame["trade_date"] = pd.to_datetime(frame.get("trade_date"), errors="coerce")
    frame = frame[frame["trade_date"].dt.date.eq(trade_date)].copy()
    if frame.empty:
        return pd.DataFrame()
    if "quote_datetime" in frame.columns:
        frame["quote_datetime"] = pd.to_datetime(frame["quote_datetime"], errors="coerce")
        frame = frame.sort_values("quote_datetime", na_position="first")
    latest = frame.tail(1).copy()
    latest["symbol"] = symbol
    for column in ("open", "high", "low", "close", "pre_close", "volume", "amount"):
        latest[column] = pd.to_numeric(latest.get(column), errors="coerce")
    latest["change"] = latest["close"].sub(latest["pre_close"])
    latest["pct_change"] = latest["close"].div(latest["pre_close"]).sub(1.0).mul(100.0)
    latest["amplitude"] = latest["high"].sub(latest["low"]).div(latest["pre_close"].replace(0, pd.NA)).mul(100.0)
    latest["turnover"] = float("nan")
    keep = [
        "trade_date",
        "symbol",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "pct_change",
        "change",
        "amplitude",
        "turnover",
    ]
    return latest.loc[:, keep].reset_index(drop=True)


def _build_intraday_macd_summary(storage: _IntradayOverlayStorage, *, trade_date: date, symbols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    universe = storage.load_universe().set_index("symbol")
    for symbol in symbols:
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError):
            continue
        cutoff = bars[pd.to_datetime(bars["trade_date"], errors="coerce").dt.date <= trade_date].reset_index(drop=True)
        if cutoff.empty:
            continue
        macd_frame = _prepare_daily_macd_frame(cutoff)
        latest = macd_frame.iloc[-1]
        divergence = summarize_recent_macd_divergence(macd_frame)
        bullish_volume_divergence, bearish_volume_divergence = _detect_daily_volume_price_divergence(macd_frame)
        rows.append(
            {
                "trade_date": trade_date.isoformat(),
                "symbol": symbol,
                "name": str(universe.loc[symbol].get("name", "")) if symbol in universe.index else "",
                "macd": _safe_float_or_none(latest.get("macd")),
                "macd_signal_line": _safe_float_or_none(latest.get("macd_signal_line")),
                "macd_hist": _safe_float_or_none(latest.get("macd_hist")),
                "macd_cross_state": _describe_macd_cross_state(macd_frame),
                "macd_divergence_state": _describe_macd_divergence_state(divergence),
                "volume_price_divergence_state": _describe_volume_price_divergence_state(
                    bullish_volume_divergence,
                    bearish_volume_divergence,
                ),
                "macd_top_divergence_15d": bool(divergence.get("macd_top_divergence_15d", False)),
                "macd_bottom_divergence_15d": bool(divergence.get("macd_bottom_divergence_15d", False)),
                "macd_top_divergence_signal_date": divergence.get("macd_top_divergence_signal_date"),
                "macd_bottom_divergence_signal_date": divergence.get("macd_bottom_divergence_signal_date"),
                "bullish_volume_price_divergence_flag": bool(bullish_volume_divergence),
                "bearish_volume_price_divergence_flag": bool(bearish_volume_divergence),
            }
        )
    return pd.DataFrame(rows)


def _build_intraday_atr_summary(storage: _IntradayOverlayStorage, *, trade_date: date, symbols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    universe = storage.load_universe().set_index("symbol")
    for symbol in symbols:
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError):
            continue
        cutoff = bars[pd.to_datetime(bars["trade_date"], errors="coerce").dt.date <= trade_date].reset_index(drop=True)
        if cutoff.empty:
            continue
        snapshot = build_atr_snapshot_row(
            cutoff,
            symbol=symbol,
            name=str(universe.loc[symbol].get("name", "")) if symbol in universe.index else "",
            trade_date=trade_date,
        )
        if snapshot is not None:
            rows.append(snapshot)
    return pd.DataFrame(rows)


def _load_intraday_snapshot_frame(storage: Storage, *, symbols: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol in symbols:
        try:
            frame = storage.load_intraday_bars(symbol)
        except FileNotFoundError:
            continue
        if frame.empty:
            continue
        copied = frame.copy()
        copied["symbol"] = copied["symbol"].map(normalize_symbol)
        if "quote_datetime" in copied.columns:
            copied["quote_datetime"] = pd.to_datetime(copied["quote_datetime"], errors="coerce")
            copied = copied.sort_values("quote_datetime", na_position="first")
        rows.append(copied.tail(1))
    if not rows:
        return pd.DataFrame(columns=["symbol"])
    result = pd.concat(rows, ignore_index=True)
    for column in ("trade_date", "quote_datetime"):
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    if "pct_change" not in result.columns:
        result["pct_change"] = pd.NA
    pct_change = pd.to_numeric(result["pct_change"], errors="coerce")
    if {"close", "pre_close"}.issubset(result.columns):
        close = pd.to_numeric(result["close"], errors="coerce")
        pre_close = pd.to_numeric(result["pre_close"], errors="coerce").replace(0, pd.NA)
        derived_pct_change = close.div(pre_close).sub(1.0).mul(100.0)
        result["pct_change"] = pct_change.where(pct_change.notna(), derived_pct_change)
    return result


def _build_output_frame(
    *,
    project_root: Path,
    trade_date: date,
    candidates: list[dict[str, object]],
    intraday: pd.DataFrame,
    phase1: pd.DataFrame,
    phase2: pd.DataFrame,
    phase4: pd.DataFrame,
    macd: pd.DataFrame,
    atr: pd.DataFrame,
) -> pd.DataFrame:
    base = pd.DataFrame(candidates)
    phase1_prepared = _prepare_phase_risk_predictions(
        phase1,
        score_column="risk_score",
        output_score_column="phase1_risk_score",
        prefix="phase1",
        filter_rate=0.2,
        model_prefix="phase1",
    )
    phase2_prepared = _prepare_phase_risk_predictions(
        phase2,
        score_column="barrier_risk_score",
        output_score_column="phase2_barrier_risk_score",
        prefix="phase2",
        filter_rate=0.2,
        model_prefix="phase2",
        extra_columns=("is_cusum_event", "mlfin_daily_vol", "mlfin_cusum_threshold"),
    )
    phase4_prepared = _prepare_phase4_predictions(phase4)
    phase4_prepared = merge_phase4_rolling_frame(phase4_prepared, project_root=project_root, trade_date=trade_date)
    macd_prepared = _prepare_macd_frame(macd)
    atr_prepared = _prepare_atr_frame(atr).rename(columns={"trade_date": "atr_trade_date", "close": "atr_close"})
    intraday_prepared = _prepare_intraday_for_output(intraday)

    result = base.copy()
    for frame in (intraday_prepared, phase1_prepared, phase2_prepared, phase4_prepared, macd_prepared, atr_prepared):
        if frame.empty or "symbol" not in frame.columns:
            continue
        frame = frame.drop(columns=["name"], errors="ignore")
        result = result.merge(frame, on="symbol", how="left")
    result = _add_phase_display_ranks(result)
    result = append_sector_display_columns(result, project_root=project_root)
    result = _add_intraday_pool_score(result)
    result = _add_intraday_selection_source(result)
    result = add_recommended_position_percent(result)
    result = _drop_internal_score_columns(result)
    return _order_output_columns(result)


def _prepare_intraday_for_output(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns:
        return pd.DataFrame(columns=["symbol"])
    result = frame.copy()
    result["symbol"] = result["symbol"].map(normalize_symbol)
    rename_map = {
        "trade_date": "intraday_trade_date",
        "pct_change": "intraday_pct_change",
        "quote_datetime": "intraday_quote_datetime",
        "quote_time": "intraday_quote_time",
        "source": "intraday_source",
        "fetched_at": "intraday_fetched_at",
        "provisional": "intraday_provisional",
    }
    result = result.rename(columns={source: target for source, target in rename_map.items() if source in result.columns})
    for column in ("intraday_trade_date", "intraday_quote_datetime"):
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce").astype("string")
    keep = ["symbol", *rename_map.values()]
    return result.loc[:, [column for column in keep if column in result.columns]].drop_duplicates("symbol", keep="last")


def _add_phase_display_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for phase in ("phase1", "phase2", "phase4"):
        score_column = f"{phase}_score_100"
        rank_column = f"{phase}_rank"
        if score_column not in result.columns:
            result[rank_column] = pd.NA
            continue
        scores = pd.to_numeric(result[score_column], errors="coerce")
        result[rank_column] = scores.rank(method="min", ascending=False).astype("Int64")
    return result


def _drop_internal_score_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[
            "phase1_risk_score",
            "phase1_risk_rank",
            "phase1_risk_percentile",
            "phase2_barrier_risk_score",
            "phase2_risk_rank",
            "phase2_risk_percentile",
            "phase4_return_score",
            "phase4_score_percentile",
        ],
        errors="ignore",
    )


def _order_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    technical_columns = [
        "macd_cross_state",
        "macd_divergence_state",
        "volume_price_divergence_state",
        "macd_top_divergence_15d",
        "macd_bottom_divergence_15d",
        "macd_top_divergence_signal_date",
        "macd_bottom_divergence_signal_date",
        "bullish_volume_price_divergence_flag",
        "bearish_volume_price_divergence_flag",
        "macd",
        "macd_signal_line",
        "macd_hist",
        "atr_trade_date",
        "atr_close",
        "atr_14",
        "atr_stop_loss_1x",
        "atr_stop_loss_2x",
        "atr_take_profit_2x",
        "atr_take_profit_3x",
        "atr_volatility_regime",
    ]
    phase_detail_columns = [
        "intraday_pool_score",
        "phase4_5d_std",
        "centered_risk_score",
        "phase1_center_score",
        "phase2_center_score",
        "phase4_rank",
        *PHASE4_ROLLING_RANK_COLUMNS,
        "phase1_rank",
        "phase2_rank",
        "phase1_excluded_by_top20_risk",
        "phase2_excluded_by_top20_risk",
        "phase2_is_cusum_event",
        "phase1_feature_trade_date",
        "phase2_feature_trade_date",
        "phase4_feature_trade_date",
        "phase1_model_name",
        "phase1_model_version",
        "phase2_mlfin_daily_vol",
        "phase2_mlfin_cusum_threshold",
        "phase2_model_name",
        "phase2_model_version",
        "phase4_name",
        "phase4_model_name",
        "phase4_model_version",
        "prev_rank",
        "universe_rank",
        "intraday_quote_datetime",
        "intraday_quote_time",
        "intraday_fetched_at",
        "intraday_provisional",
    ]
    pattern_detail_columns = [
        "prev_source_tags",
        "prev_reason",
        "prev_watchlist_streak",
        "prev_patterns",
        "prev_phase1_score_100",
        "prev_phase2_score_100",
        "prev_phase4_score_100",
    ]
    preferred = [
        "intraday_trade_date",
        "symbol",
        "name",
        "intraday_selection_source",
        "pool_route",
        "source_sectors",
        "matched_mainline_sector",
        "leader_score",
        "industry_names",
        "concept_names",
        "intraday_pct_change",
        "phase1_score_100",
        "phase2_score_100",
        "phase4_score_100",
        "phase4_5d_mean",
        "phase4_5d_std",
        "prev_pattern_match",
        "prev_pattern_ids",
        "prev_pattern_id",
        "atr_pct_14",
        RECOMMENDED_POSITION_PERCENT_FIELD,
        *technical_columns,
        "prev_source",
        "intraday_source",
        *phase_detail_columns,
        *pattern_detail_columns,
    ]
    seen_columns: set[str] = set()
    columns = []
    for column in preferred:
        if column in frame.columns and column not in seen_columns:
            columns.append(column)
            seen_columns.add(column)
    columns.extend([column for column in frame.columns if column not in columns])
    ordered = frame.loc[:, columns].copy()
    sort_columns = [column for column in ("prev_rank", "phase4_rank", "phase1_rank", "phase2_rank", "symbol") if column in ordered.columns]
    if sort_columns:
        ordered = ordered.sort_values(sort_columns, ascending=[True] * len(sort_columns), na_position="last")
    return ordered.reset_index(drop=True)


def _format_intraday_output_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "日期",
                "编号",
                "股票名称",
                "来源",
                "来源路线",
                "关切板块",
                "契合主线",
                "龙头指数",
                "盘中涨幅%",
                "P1风险质量分",
                "P2交易风险分",
                "P4上涨质量分",
                "P4五日均分",
                "Pattern命中",
                "ATR%",
                RECOMMENDED_POSITION_PERCENT_FIELD,
            ]
        )
    result = frame.copy()
    result = result.loc[
        :,
        [
            column
            for column in result.columns
            if not str(column).startswith(("phase5", "phase7", "phase8", "prev_phase5"))
            and column not in {"today_limit_up_excluded", "today_high_return_vs_prev_close", "today_close_return_vs_prev_close"}
        ],
    ]
    if "symbol" in result.columns:
        result["symbol"] = result["symbol"].map(lambda value: f'="{normalize_symbol(value)}"')
    rename_map = {
        "intraday_trade_date": "日期",
        "symbol": "编号",
        "name": "股票名称",
        "intraday_selection_source": "来源",
        "pool_route": "来源路线",
        "source_sectors": "关切板块",
        "matched_mainline_sector": "契合主线",
        "leader_score": "龙头指数",
        "intraday_pct_change": "盘中涨幅%",
        "phase1_score_100": "P1风险质量分",
        "phase2_score_100": "P2交易风险分",
        "phase4_score_100": "P4上涨质量分",
        "phase4_5d_mean": "P4五日均分",
        "phase4_5d_std": "P4五日std",
        "prev_pattern_match": "Pattern命中",
        "prev_pattern_ids": "Pattern编号",
        "atr_pct_14": "ATR%",
        "macd_cross_state": "macd交叉",
        "macd_divergence_state": "macd背离",
        "volume_price_divergence_state": "量价背离",
        "intraday_pool_score": "P1/P2/P4综合分",
        "centered_risk_score": "P1/P2/P4综合分",
        "industry_names": "行业",
        "concept_names": "概念",
        "prev_source": "上一轮来源",
        "prev_reason": "Pattern理由",
        "prev_patterns": "Pattern详细",
        "intraday_source": "行情源",
        "intraday_quote_datetime": "行情时间",
        "atr_close": "最新价格",
        "atr_trade_date": "ATR日期",
        "atr_14": "ATR14",
        "atr_stop_loss_1x": "1ATR止损参考",
        "atr_stop_loss_2x": "2ATR止损参考",
        "atr_take_profit_2x": "2ATR止盈参考",
        "atr_take_profit_3x": "3ATR止盈参考",
        "atr_volatility_regime": "波动分层",
    }
    result = result.rename(columns={key: value for key, value in rename_map.items() if key in result.columns})
    preferred = [
        "日期",
        "编号",
        "股票名称",
        "来源",
        "来源路线",
        "关切板块",
        "契合主线",
        "龙头指数",
        "盘中涨幅%",
        "P1风险质量分",
        "P2交易风险分",
        "P4上涨质量分",
        "P4五日均分",
        "P4五日std",
        "Pattern命中",
        "Pattern编号",
        "ATR%",
        RECOMMENDED_POSITION_PERCENT_FIELD,
        "macd交叉",
        "macd背离",
        "量价背离",
        "macd_top_divergence_15d",
        "macd_bottom_divergence_15d",
        "bullish_volume_price_divergence_flag",
        "bearish_volume_price_divergence_flag",
        "macd",
        "macd_signal_line",
        "macd_hist",
        "最新价格",
        "ATR14",
        "1ATR止损参考",
        "2ATR止损参考",
        "2ATR止盈参考",
        "3ATR止盈参考",
        "波动分层",
        "行业",
        "概念",
        "P1/P2/P4综合分",
        "上一轮来源",
        "Pattern理由",
        "Pattern详细",
    ]
    ordered = [column for column in preferred if column in result.columns]
    ordered.extend([column for column in result.columns if column not in ordered])
    return result.loc[:, ordered].loc[:, lambda df: ~df.columns.duplicated()].copy()


def _prepare_daily_macd_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy().sort_values("trade_date").reset_index(drop=True)
    if not {"macd_dif", "macd_dea", "macd_hist"}.issubset(frame.columns):
        frame = add_indicators(frame).sort_values("trade_date").reset_index(drop=True)
    if "macd" not in frame.columns and "macd_dif" in frame.columns:
        frame["macd"] = frame["macd_dif"]
    if "macd_signal_line" not in frame.columns and "macd_dea" in frame.columns:
        frame["macd_signal_line"] = frame["macd_dea"]
    return frame


def _describe_macd_cross_state(dataframe: pd.DataFrame) -> str:
    normalized = _prepare_daily_macd_frame(dataframe)
    if normalized.empty or "macd" not in normalized.columns or "macd_signal_line" not in normalized.columns:
        return "unknown"
    recent = normalized.tail(3).reset_index(drop=True)
    recent_cross_up = False
    recent_cross_down = False
    for offset in range(1, len(recent)):
        prev_row = recent.iloc[offset - 1]
        current_row = recent.iloc[offset]
        if pd.isna(prev_row.get("macd")) or pd.isna(prev_row.get("macd_signal_line")):
            continue
        if pd.isna(current_row.get("macd")) or pd.isna(current_row.get("macd_signal_line")):
            continue
        if float(prev_row["macd"]) <= float(prev_row["macd_signal_line"]) and float(current_row["macd"]) > float(current_row["macd_signal_line"]):
            recent_cross_up = True
        if float(prev_row["macd"]) >= float(prev_row["macd_signal_line"]) and float(current_row["macd"]) < float(current_row["macd_signal_line"]):
            recent_cross_down = True
    latest = recent.iloc[-1]
    macd = _safe_float_or_none(latest.get("macd"))
    signal_line = _safe_float_or_none(latest.get("macd_signal_line"))
    if recent_cross_up:
        return "golden_cross"
    if recent_cross_down:
        return "dead_cross"
    if macd is None or signal_line is None:
        return "unknown"
    return "above_signal" if macd >= signal_line else "below_signal"


def _detect_daily_volume_price_divergence(dataframe: pd.DataFrame) -> tuple[bool, bool]:
    if dataframe.empty or len(dataframe) < 6:
        return False, False
    frame = dataframe.copy().sort_values("trade_date").reset_index(drop=True)
    recent = frame.tail(5).reset_index(drop=True)
    previous = frame.iloc[-6]
    latest = recent.iloc[-1]
    recent_avg_volume = pd.to_numeric(recent["volume"], errors="coerce").mean()
    previous_volume = _safe_float_or_none(previous.get("volume"))
    latest_close = _safe_float_or_none(latest.get("close"))
    previous_close = _safe_float_or_none(previous.get("close"))
    if previous_volume is None or latest_close is None or previous_close is None or pd.isna(recent_avg_volume):
        return False, False
    bullish = latest_close > previous_close and float(recent_avg_volume) < previous_volume * 0.9
    bearish = latest_close < previous_close and float(recent_avg_volume) > previous_volume * 1.1
    return bullish, bearish


def _describe_macd_divergence_state(macd_summary: dict[str, object]) -> str:
    if bool(macd_summary.get("macd_bottom_divergence_15d", False)):
        return "bottom_divergence"
    if bool(macd_summary.get("macd_top_divergence_15d", False)):
        return "top_divergence"
    return "none"


def _describe_volume_price_divergence_state(bullish: bool, bearish: bool) -> str:
    if bullish and not bearish:
        return "bullish"
    if bearish and not bullish:
        return "bearish"
    return "none"


def _safe_float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _jsonish(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
