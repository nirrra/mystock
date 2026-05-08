from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from .atr import build_atr_snapshot_row
from .full_market_return import predict_alpha158_qlib_return
from .full_market_risk import predict_barrier_risk, predict_tail_risk
from .indicators import add_indicators
from .intraday_update import INTRADAY_DATA_INTERFACES, run_intraday_update
from .macd_divergence import summarize_recent_macd_divergence
from .phase_display import normalize_symbol
from .storage import DailyBarsReadError, Storage
from .track_stock import DEFAULT_TRACK_STOCK_FILENAME, TRACK_INPUT_SHEET
from .watchlist import (
    _prepare_atr_frame,
    _prepare_macd_frame,
    _prepare_phase4_predictions,
    _prepare_phase_risk_predictions,
    extract_watchlist_symbols,
    find_latest_watchlist_before,
)

DEFAULT_INTRADAY_REPORT_KEEP_DATES = 10
INTRADAY_FOCUS_SIZE = 20
INTRADAY_FOCUS_MIN_PHASE_SCORE = 40.0
INTRADAY_FOCUS_MAX_PCT_CHANGE = 8.0
_DATED_REPORT_PATTERNS = (
    re.compile(r"^intraday_screening_(\d{4}-\d{2}-\d{2})\.csv$"),
    re.compile(r"^intraday_screening_focus_(\d{4}-\d{2}-\d{2})\.csv$"),
    re.compile(r"^intraday_top10_(\d{4}-\d{2}-\d{2})\.csv$"),
    re.compile(r"^intraday_top20_(\d{4}-\d{2}-\d{2})\.csv$"),
    re.compile(r"^intraday_top20_previous_(\d{4}-\d{2}-\d{2})\.csv$"),
    re.compile(r"^intraday_track_stock_(\d{4}-\d{2}-\d{2})\.csv$"),
)


@dataclass(slots=True)
class IntradayScreeningResult:
    trade_date: date
    source_watchlist_path: Path
    output_path: Path
    focus_output_path: Path | None
    top20_path: Path
    track_stock_path: Path | None
    candidate_count: int
    focus_candidate_count: int
    track_stock_count: int
    intraday_updated_count: int
    missing_intraday_symbols: list[str]
    cleaned_report_files: int
    phase1_path: Path
    phase2_path: Path
    phase4_path: Path

    @property
    def top10_path(self) -> Path:
        return self.top20_path


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
    data_interface: str = "eastmoney_direct",
    limit: int | None = None,
    watchlist_only: bool = False,
    skip_intraday_update: bool = False,
    timeout_seconds: float = 15.0,
    chunk_size: int = 50,
    output: Path | None = None,
    report_keep_dates: int | None = DEFAULT_INTRADAY_REPORT_KEEP_DATES,
) -> IntradayScreeningResult:
    source_watchlist_path, watchlist_payload = _load_previous_watchlist(project_root, trade_date)
    previous_candidates = _previous_watchlist_candidates(watchlist_payload, limit=None)
    candidates = previous_candidates if watchlist_only else _full_market_candidates(storage, previous_candidates)
    if limit is not None:
        candidates = candidates[: max(int(limit), 0)]
    tracked_symbols = _load_tracked_symbols(project_root)
    candidates = _merge_tracked_candidates(storage=storage, candidates=candidates, tracked_symbols=tracked_symbols)
    symbols = [candidate["symbol"] for candidate in candidates]
    if not symbols:
        raise RuntimeError(f"No previous-watchlist symbols found in {source_watchlist_path}")

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

    output_dir = project_root / "reports" / "intraday_screening"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = storage.paths.intraday_dir / "screening_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    focus_output_path: Path | None = None
    focus_candidate_count = 0
    focus_analysis: _AnalysisResult | None = None
    remaining_candidates = candidates
    if not watchlist_only:
        focus_candidates = _load_focus_candidates(storage, candidates)
        if focus_candidates:
            focus_analysis = _run_candidate_analysis(
                storage=storage,
                project_root=project_root,
                trade_date=trade_date,
                candidates=focus_candidates,
                output_dir=cache_dir,
                file_tag="focus",
            )
            focus_output_path = output_dir / f"intraday_top20_previous_{trade_date.isoformat()}.csv"
            focus_analysis.frame.to_csv(focus_output_path, index=False, encoding="utf-8-sig")
            focus_candidate_count = len(focus_analysis.frame)
            focus_symbols = {candidate["symbol"] for candidate in focus_candidates}
            remaining_candidates = [candidate for candidate in candidates if candidate["symbol"] not in focus_symbols]

    if focus_analysis is not None:
        if remaining_candidates:
            remaining_analysis = _run_candidate_analysis(
                storage=storage,
                project_root=project_root,
                trade_date=trade_date,
                candidates=remaining_candidates,
                output_dir=cache_dir,
                file_tag="remaining",
            )
            combined_intraday = _concat_frames(focus_analysis.intraday, remaining_analysis.intraday)
            combined_phase1 = _concat_frames(focus_analysis.phase1, remaining_analysis.phase1)
            combined_phase2 = _concat_frames(focus_analysis.phase2, remaining_analysis.phase2)
            combined_phase4 = _concat_frames(focus_analysis.phase4, remaining_analysis.phase4)
            combined_macd = _concat_frames(focus_analysis.macd, remaining_analysis.macd)
            combined_atr = _concat_frames(focus_analysis.atr, remaining_analysis.atr)
            combined = _build_output_frame(
                candidates=candidates,
                intraday=combined_intraday,
                phase1=combined_phase1,
                phase2=combined_phase2,
                phase4=combined_phase4,
                macd=combined_macd,
                atr=combined_atr,
            )
            analysis = _AnalysisResult(
                frame=combined,
                candidates=candidates,
                intraday=combined_intraday,
                phase1=combined_phase1,
                phase2=combined_phase2,
                phase4=combined_phase4,
                macd=combined_macd,
                atr=combined_atr,
                phase1_path=remaining_analysis.phase1_path,
                phase2_path=remaining_analysis.phase2_path,
                phase4_path=remaining_analysis.phase4_path,
            )
        else:
            analysis = focus_analysis
    else:
        analysis = _run_candidate_analysis(
            storage=storage,
            project_root=project_root,
            trade_date=trade_date,
            candidates=candidates,
            output_dir=cache_dir,
            file_tag="all" if not watchlist_only else "watchlist",
        )
    output_path = output if output is not None else output_dir / f"intraday_screening_{trade_date.isoformat()}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis.frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    top20_path = _save_top20_focus(storage=storage, output_dir=output_dir, trade_date=trade_date, frame=analysis.frame)
    track_stock_path = _save_intraday_track_stock(output_dir=output_dir, trade_date=trade_date, frame=analysis.frame)
    cleaned_report_files = _cleanup_intraday_screening_reports(
        output_dir,
        keep_dates=report_keep_dates,
        preserve_paths=(output_path, focus_output_path, top20_path, track_stock_path),
    )

    intraday = _load_intraday_snapshot_frame(storage, symbols=symbols)
    missing_intraday = [symbol for symbol in symbols if symbol not in set(intraday.get("symbol", pd.Series(dtype=str)).astype(str))]
    return IntradayScreeningResult(
        trade_date=trade_date,
        source_watchlist_path=source_watchlist_path,
        output_path=output_path,
        focus_output_path=focus_output_path,
        top20_path=top20_path,
        track_stock_path=track_stock_path,
        candidate_count=len(analysis.frame),
        focus_candidate_count=focus_candidate_count,
        track_stock_count=len(tracked_symbols),
        intraday_updated_count=updated_count,
        missing_intraday_symbols=sorted(set(missing_update_symbols + missing_intraday)),
        cleaned_report_files=cleaned_report_files,
        phase1_path=analysis.phase1_path,
        phase2_path=analysis.phase2_path,
        phase4_path=analysis.phase4_path,
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


def _load_previous_watchlist(project_root: Path, trade_date: date) -> tuple[Path, dict[str, object]]:
    _, path = find_latest_watchlist_before(project_root=project_root, trade_date=trade_date)
    return path, json.loads(path.read_text(encoding="utf-8"))


def _previous_watchlist_candidates(payload: dict[str, object], *, limit: int | None) -> list[dict[str, object]]:
    raw_candidates = payload.get("candidates") if isinstance(payload, dict) else []
    if not isinstance(raw_candidates, list):
        return []

    extracted_symbols = extract_watchlist_symbols(payload)
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
            }
        )
    candidates = sorted(candidates, key=lambda item: int(item["prev_rank"]))
    if limit is not None:
        candidates = candidates[: max(int(limit), 0)]
    return candidates


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


def _full_market_candidates(storage: Storage, previous_candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    universe = storage.load_universe().copy()
    if "symbol" not in universe.columns:
        raise RuntimeError("Universe file lacks symbol column.")
    universe["symbol"] = universe["symbol"].map(normalize_symbol)
    universe = universe[universe["symbol"].astype(str).str.len().gt(0)].drop_duplicates("symbol", keep="first")
    previous_by_symbol = {str(candidate["symbol"]): candidate for candidate in previous_candidates}
    rows: list[dict[str, object]] = []
    for index, row in enumerate(universe.to_dict("records"), start=1):
        symbol = str(row.get("symbol", ""))
        previous = previous_by_symbol.get(symbol, {})
        name = str(row.get("name", "") or previous.get("name", "") or "")
        merged = {
            "symbol": symbol,
            "name": name,
            "prev_rank": previous.get("prev_rank", pd.NA),
            "universe_rank": index,
            "prev_source": previous.get("prev_source", ""),
            "prev_source_tags": previous.get("prev_source_tags", ""),
            "prev_pattern_match": previous.get("prev_pattern_match", False),
            "prev_pattern_id": previous.get("prev_pattern_id", ""),
            "prev_pattern_ids": previous.get("prev_pattern_ids", ""),
            "prev_patterns": previous.get("prev_patterns", ""),
            "prev_reason": previous.get("prev_reason", ""),
            "prev_phase1_score_100": previous.get("prev_phase1_score_100", pd.NA),
            "prev_phase2_score_100": previous.get("prev_phase2_score_100", pd.NA),
            "prev_phase4_score_100": previous.get("prev_phase4_score_100", pd.NA),
            "prev_watchlist_streak": previous.get("prev_watchlist_streak", pd.NA),
        }
        rows.append(merged)
    return rows


def _load_tracked_symbols(project_root: Path) -> list[str]:
    path = project_root / DEFAULT_TRACK_STOCK_FILENAME
    if not path.exists():
        return []
    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
    except OSError:
        return []
    try:
        if TRACK_INPUT_SHEET not in workbook.sheetnames:
            return []
        sheet = workbook[TRACK_INPUT_SHEET]
        header_values = [str(cell.value or "").strip().lower() for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
        symbol_column = 1
        has_symbol_header = False
        for index, value in enumerate(header_values, start=1):
            if value in {"symbol", "code", "股票代码", "代码", "证券代码"}:
                symbol_column = index
                has_symbol_header = True
                break
        start_row = 2 if has_symbol_header else 1
        symbols: list[str] = []
        seen: set[str] = set()
        for row in sheet.iter_rows(min_row=start_row, min_col=symbol_column, max_col=symbol_column, values_only=True):
            symbol = normalize_symbol(row[0] if row else "")
            if not symbol or symbol in seen:
                continue
            symbols.append(symbol)
            seen.add(symbol)
        return symbols
    finally:
        workbook.close()


def _merge_tracked_candidates(
    *,
    storage: Storage,
    candidates: list[dict[str, object]],
    tracked_symbols: list[str],
) -> list[dict[str, object]]:
    tracked = {normalize_symbol(symbol) for symbol in tracked_symbols if normalize_symbol(symbol)}
    if not tracked:
        return candidates
    result = [dict(candidate) for candidate in candidates]
    by_symbol = {str(candidate.get("symbol", "")): candidate for candidate in result}
    for candidate in result:
        candidate["track_stock"] = str(candidate.get("symbol", "")) in tracked

    missing = [symbol for symbol in tracked_symbols if normalize_symbol(symbol) not in by_symbol]
    if not missing:
        return result

    universe_names = _universe_name_lookup(storage)
    for symbol in missing:
        normalized = normalize_symbol(symbol)
        if not normalized:
            continue
        result.append(
            {
                "symbol": normalized,
                "name": universe_names.get(normalized, ""),
                "prev_rank": pd.NA,
                "universe_rank": pd.NA,
                "prev_source": "",
                "prev_source_tags": "",
                "prev_pattern_match": False,
                "prev_pattern_id": "",
                "prev_pattern_ids": "",
                "prev_patterns": "",
                "prev_reason": "",
                "prev_phase1_score_100": pd.NA,
                "prev_phase2_score_100": pd.NA,
                "prev_phase4_score_100": pd.NA,
                "prev_watchlist_streak": pd.NA,
                "track_stock": True,
            }
        )
    return result


def _universe_name_lookup(storage: Storage) -> dict[str, str]:
    try:
        universe = storage.load_universe().copy()
    except FileNotFoundError:
        return {}
    if "symbol" not in universe.columns:
        return {}
    universe["symbol"] = universe["symbol"].map(normalize_symbol)
    return {
        str(row.get("symbol", "")): str(row.get("name", "") or "")
        for row in universe.to_dict("records")
        if str(row.get("symbol", ""))
    }


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
    ).predictions
    phase2 = predict_barrier_risk(
        storage=overlay_storage,
        project_root=project_root,
        trade_date=trade_date,
        output=phase2_path,
    ).predictions
    phase4 = predict_alpha158_qlib_return(
        storage=overlay_storage,
        project_root=project_root,
        trade_date=trade_date,
        output=phase4_path,
    ).predictions

    macd = _build_intraday_macd_summary(overlay_storage, trade_date=trade_date, symbols=symbols)
    atr = _build_intraday_atr_summary(overlay_storage, trade_date=trade_date, symbols=symbols)
    intraday = _load_intraday_snapshot_frame(storage, symbols=symbols)
    frame = _build_output_frame(
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


def _load_focus_candidates(storage: Storage, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    path = _existing_focus_symbols_path(storage)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    symbols = [normalize_symbol(symbol) for symbol in payload.get("symbols", []) if normalize_symbol(symbol)]
    if not symbols:
        return []
    by_symbol = {str(candidate["symbol"]): candidate for candidate in candidates}
    return [by_symbol[symbol] for symbol in symbols if symbol in by_symbol]


def _save_top20_focus(*, storage: Storage, output_dir: Path, trade_date: date, frame: pd.DataFrame) -> Path:
    top20 = _select_top20_focus(frame)
    target = output_dir / f"intraday_top20_{trade_date.isoformat()}.csv"
    top20.to_csv(target, index=False, encoding="utf-8-sig")
    payload = {
        "trade_date": trade_date.isoformat(),
        "source_file": str(target),
        "symbols": top20["symbol"].astype(str).str.zfill(6).tolist() if "symbol" in top20.columns else [],
    }
    focus_path = _focus_symbols_path(storage)
    focus_path.parent.mkdir(parents=True, exist_ok=True)
    focus_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _save_intraday_track_stock(*, output_dir: Path, trade_date: date, frame: pd.DataFrame) -> Path | None:
    track = _select_intraday_track_stock(frame)
    if track.empty:
        return None
    target = output_dir / f"intraday_track_stock_{trade_date.isoformat()}.csv"
    track.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def _select_top20_focus(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.head(0).copy()
    required = {"phase1_score_100", "phase2_score_100", "phase4_score_100", "intraday_pct_change"}
    if not required.issubset(frame.columns):
        return frame.head(0).copy()
    result = frame.copy()
    phase1 = pd.to_numeric(result["phase1_score_100"], errors="coerce")
    phase2 = pd.to_numeric(result["phase2_score_100"], errors="coerce")
    phase4 = pd.to_numeric(result["phase4_score_100"], errors="coerce")
    pct_change = pd.to_numeric(result["intraday_pct_change"], errors="coerce")
    result = result[
        phase1.gt(INTRADAY_FOCUS_MIN_PHASE_SCORE)
        & phase2.gt(INTRADAY_FOCUS_MIN_PHASE_SCORE)
        & pct_change.le(INTRADAY_FOCUS_MAX_PCT_CHANGE)
    ].copy()
    if result.empty:
        selected = result
    else:
        result["_phase4_focus_sort"] = phase4.loc[result.index]
        selected = result.sort_values(["_phase4_focus_sort", "symbol"], ascending=[False, True], na_position="last").head(INTRADAY_FOCUS_SIZE)
    selected = selected.copy()
    selected["intraday_selection_source"] = "top20"
    tracked = _select_intraday_track_stock(frame)
    if not tracked.empty:
        tracked = tracked.copy()
        tracked["intraday_selection_source"] = "track_stock"
        selected_symbols = set(selected["symbol"].astype(str)) if "symbol" in selected.columns else set()
        tracked_only = tracked[~tracked["symbol"].astype(str).isin(selected_symbols)].copy()
        selected.loc[selected["symbol"].astype(str).isin(set(tracked["symbol"].astype(str))), "intraday_selection_source"] = "top20+track_stock"
        selected = _concat_frames(selected, tracked_only)
    if selected.empty:
        return selected.drop(columns=["_phase4_focus_sort"], errors="ignore").reset_index(drop=True)
    selected["_phase4_focus_sort"] = pd.to_numeric(selected.get("phase4_score_100"), errors="coerce")
    selected["_track_focus_sort"] = selected.get("intraday_selection_source", "").astype(str).eq("track_stock").astype(int)
    selected = selected.sort_values(
        ["_track_focus_sort", "_phase4_focus_sort", "symbol"],
        ascending=[True, False, True],
        na_position="last",
    )
    return selected.drop(columns=["_phase4_focus_sort", "_track_focus_sort"], errors="ignore").reset_index(drop=True)


def _select_intraday_track_stock(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "track_stock" not in frame.columns:
        return frame.head(0).copy()
    result = frame[frame["track_stock"].fillna(False).astype(bool)].copy()
    if result.empty:
        return result
    result["_phase4_focus_sort"] = pd.to_numeric(result.get("phase4_score_100"), errors="coerce")
    result = result.sort_values(["_phase4_focus_sort", "symbol"], ascending=[False, True], na_position="last")
    return result.drop(columns=["_phase4_focus_sort"], errors="ignore").reset_index(drop=True)


def _focus_symbols_path(storage: Storage) -> Path:
    return storage.paths.intraday_dir / "focus_top20.json"


def _legacy_focus_symbols_path(storage: Storage) -> Path:
    return storage.paths.intraday_dir / "focus_top10.json"


def _existing_focus_symbols_path(storage: Storage) -> Path:
    path = _focus_symbols_path(storage)
    if path.exists():
        return path
    return _legacy_focus_symbols_path(storage)


def _select_top10_focus(frame: pd.DataFrame) -> pd.DataFrame:
    return _select_top20_focus(frame)


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
    return result


def _build_output_frame(
    *,
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
    preferred = [
        "phase4_rank",
        "phase4_score_100",
        "phase1_rank",
        "phase1_score_100",
        "phase2_rank",
        "phase2_score_100",
        "symbol",
        "name",
        "intraday_selection_source",
        "track_stock",
        "prev_rank",
        "universe_rank",
        "intraday_trade_date",
        "intraday_quote_datetime",
        "intraday_source",
        "intraday_pct_change",
        "phase1_excluded_by_top20_risk",
        "phase1_feature_trade_date",
        "phase2_excluded_by_top20_risk",
        "phase2_is_cusum_event",
        "phase2_feature_trade_date",
        "phase4_feature_trade_date",
        "macd",
        "macd_signal_line",
        "macd_hist",
        "macd_cross_state",
        "macd_divergence_state",
        "volume_price_divergence_state",
        "macd_top_divergence_15d",
        "macd_bottom_divergence_15d",
        "atr_trade_date",
        "atr_close",
        "atr_14",
        "atr_pct_14",
        "atr_stop_loss_1x",
        "atr_stop_loss_2x",
        "atr_take_profit_2x",
        "atr_take_profit_3x",
        "atr_volatility_regime",
        "prev_source",
        "prev_pattern_match",
        "prev_pattern_id",
        "prev_pattern_ids",
        "prev_reason",
        "prev_watchlist_streak",
        "prev_source_tags",
        "prev_patterns",
    ]
    columns = [column for column in preferred if column in frame.columns]
    columns.extend([column for column in frame.columns if column not in columns])
    ordered = frame.loc[:, columns].copy()
    sort_columns = [column for column in ("phase4_rank", "phase1_rank", "phase2_rank", "prev_rank", "universe_rank", "symbol") if column in ordered.columns]
    if sort_columns:
        ordered = ordered.sort_values(sort_columns, ascending=[True] * len(sort_columns), na_position="last")
    return ordered.reset_index(drop=True)


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
