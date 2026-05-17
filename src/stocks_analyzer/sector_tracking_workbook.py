from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


DAILY_SECTOR_TRACKING_WORKBOOK = "sector_mainline_daily_tracking.xlsx"
INTRADAY_SECTOR_TRACKING_WORKBOOK = "sector_mainline_intraday_tracking.xlsx"
LONG_MAINLINE_SHEET = "长线主线"
SHORT_MAINLINE_SHEET = "短线主线"
BUY_SCORE_SHEET = "主线买入分"

DAILY_COLUMNS = [
    "日期",
    "更新时间",
    "板块类型",
    "板块名称",
    "板块代码",
    "长期主线指数",
    "短期主线指数",
    "P9买入分",
    "P9排名",
    "当日涨幅%",
    "成交额加权涨幅%",
    "上涨家数占比",
    "龙头编号",
    "龙头名称",
    "入选原因",
    "成员数",
    "有效成员数",
]

INTRADAY_COLUMNS = [
    "日期",
    "更新时间",
    "板块类型",
    "板块名称",
    "板块代码",
    "长期主线指数",
    "短期主线指数",
    "P9买入分",
    "P9排名",
    "当日涨幅%",
    "成交额加权涨幅%",
    "上涨家数占比",
    "龙头盘中平均涨幅%",
    "有效龙头数",
    "龙头编号",
    "龙头名称",
    "入选原因",
    "成员数",
    "有效成员数",
]


def sector_daily_tracking_workbook_path(project_root: Path) -> Path:
    return project_root / "reports" / "sectors" / DAILY_SECTOR_TRACKING_WORKBOOK


def sector_intraday_tracking_workbook_path(project_root: Path) -> Path:
    return project_root / "reports" / "sectors" / INTRADAY_SECTOR_TRACKING_WORKBOOK


def write_sector_daily_tracking_workbook(
    *,
    project_root: Path,
    trade_date: date,
    sector_payload: dict[str, object],
) -> Path:
    rows = _sector_rows(sector_payload=sector_payload, trade_date=trade_date)
    sheets = _tracking_sheets(rows=rows, columns=DAILY_COLUMNS)
    target = sector_daily_tracking_workbook_path(project_root)
    _write_tracking_workbook(target=target, trade_date=trade_date, sheets=sheets, sort_column_by_sheet=_daily_sort_columns())
    return target


def write_sector_intraday_tracking_workbook(
    *,
    project_root: Path,
    trade_date: date,
    sector_payload: dict[str, object],
    intraday_strength: pd.DataFrame | None = None,
) -> Path:
    intraday_lookup = _intraday_strength_lookup(intraday_strength)
    rows = _sector_rows(
        sector_payload=sector_payload,
        trade_date=trade_date,
        intraday_lookup=intraday_lookup,
        row_date=trade_date.isoformat(),
    )
    sheets = _tracking_sheets(rows=rows, columns=INTRADAY_COLUMNS)
    target = sector_intraday_tracking_workbook_path(project_root)
    _write_tracking_workbook(target=target, trade_date=trade_date, sheets=sheets, sort_column_by_sheet=_intraday_sort_columns())
    return target


def _sector_rows(
    *,
    sector_payload: dict[str, object],
    trade_date: date,
    intraday_lookup: dict[tuple[str, str], dict[str, object]] | None = None,
    row_date: str | None = None,
) -> list[dict[str, object]]:
    raw_rows = sector_payload.get("sectors") if isinstance(sector_payload, dict) else []
    if not isinstance(raw_rows, list):
        raw_rows = []
    payload_date = str(row_date or sector_payload.get("trade_date") or trade_date.isoformat())
    updated_at = datetime.now().isoformat(timespec="seconds")
    lookup = intraday_lookup or {}
    rows: list[dict[str, object]] = []
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        sector_type = _sector_type_cn(item.get("sector_type"))
        sector_name = _cell(item.get("sector_name"))
        intraday = lookup.get((str(sector_type), str(sector_name)), {})
        leader_symbols = item.get("leader_symbols") if isinstance(item.get("leader_symbols"), list) else []
        leader_names = item.get("leader_names") if isinstance(item.get("leader_names"), list) else []
        rows.append(
            {
                "日期": payload_date,
                "更新时间": updated_at,
                "板块类型": sector_type,
                "板块名称": sector_name,
                "板块代码": _cell(item.get("sector_label")),
                "长期主线指数": _cell(item.get("long_mainline_score_100")),
                "短期主线指数": _cell(item.get("short_mainline_score_100")),
                "P9买入分": _cell(item.get("phase9_score_100")),
                "P9排名": _cell(item.get("phase9_rank")),
                "当日涨幅%": _cell(item.get("sector_avg_pct_change")),
                "成交额加权涨幅%": _cell(item.get("sector_amount_weighted_pct_change")),
                "上涨家数占比": _cell(item.get("sector_up_ratio")),
                "龙头盘中平均涨幅%": _cell(intraday.get("龙头盘中平均涨幅%")),
                "有效龙头数": _cell(intraday.get("有效龙头数")),
                "龙头编号": "/".join(str(symbol).zfill(6) for symbol in leader_symbols if str(symbol).strip()),
                "龙头名称": "/".join(str(name) for name in leader_names if str(name).strip()),
                "入选原因": _selection_reason(item),
                "成员数": _cell(item.get("member_count")),
                "有效成员数": _cell(item.get("valid_count")),
                "_long_selected": _bool_cell(item.get("selected_as_long_mainline", True), default=True),
                "_short_selected": _bool_cell(item.get("selected_as_short_mainline", False), default=False),
                "_buy_selected": _bool_cell(item.get("selected_as_phase9_buy", False), default=False),
            }
        )
    return rows


def _tracking_sheets(*, rows: list[dict[str, object]], columns: list[str]) -> dict[str, pd.DataFrame]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        empty = pd.DataFrame(columns=columns)
        return {
            LONG_MAINLINE_SHEET: empty.copy(),
            SHORT_MAINLINE_SHEET: empty.copy(),
            BUY_SCORE_SHEET: empty.copy(),
        }

    all_frame = frame.copy()
    return {
        LONG_MAINLINE_SHEET: _public_columns(all_frame, columns),
        SHORT_MAINLINE_SHEET: _public_columns(all_frame, columns),
        BUY_SCORE_SHEET: _public_columns(all_frame, columns),
    }


def _write_tracking_workbook(
    *,
    target: Path,
    trade_date: date,
    sheets: dict[str, pd.DataFrame],
    sort_column_by_sheet: dict[str, str],
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_existing_sheets(target)
    date_text = trade_date.isoformat()
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        for sheet_name in (LONG_MAINLINE_SHEET, SHORT_MAINLINE_SHEET, BUY_SCORE_SHEET):
            current = sheets.get(sheet_name, pd.DataFrame())
            combined = _replace_date_rows(
                existing=existing.get(sheet_name),
                current=current,
                trade_date=date_text,
            )
            combined = _sort_sheet(combined, score_column=sort_column_by_sheet.get(sheet_name, "长期主线指数"))
            combined.to_excel(writer, index=False, sheet_name=sheet_name)
    _format_workbook(target)


def _read_existing_sheets(target: Path) -> dict[str, pd.DataFrame]:
    if not target.exists():
        return {}
    result: dict[str, pd.DataFrame] = {}
    for sheet_name in (LONG_MAINLINE_SHEET, SHORT_MAINLINE_SHEET, BUY_SCORE_SHEET):
        try:
            result[sheet_name] = pd.read_excel(target, sheet_name=sheet_name, dtype={"板块代码": str, "龙头编号": str})
        except ValueError:
            continue
    return result


def _replace_date_rows(*, existing: pd.DataFrame | None, current: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if existing is None or existing.empty:
        result = current.copy()
    else:
        kept = existing.copy()
        if "日期" in kept.columns:
            parsed_dates = pd.to_datetime(kept["日期"], errors="coerce").dt.date.astype(str)
            raw_dates = kept["日期"].astype(str)
            kept = kept[parsed_dates.ne(trade_date) & raw_dates.ne(trade_date)].copy()
        result = pd.concat([kept, current.copy()], ignore_index=True)
    if result.empty:
        return current.copy()
    return result.drop_duplicates()


def _sort_sheet(frame: pd.DataFrame, *, score_column: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    result["_date_sort"] = pd.to_datetime(result.get("日期"), errors="coerce")
    result["_score_sort"] = pd.to_numeric(result.get(score_column), errors="coerce")
    sort_columns = ["_date_sort", "_score_sort", "板块类型", "板块名称"]
    result = result.sort_values(sort_columns, ascending=[False, False, True, True], na_position="last")
    return result.drop(columns=["_date_sort", "_score_sort"]).reset_index(drop=True)


def _public_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result[columns].copy()


def _intraday_strength_lookup(frame: pd.DataFrame | None) -> dict[tuple[str, str], dict[str, object]]:
    if frame is None or frame.empty:
        return {}
    result: dict[tuple[str, str], dict[str, object]] = {}
    for item in frame.to_dict("records"):
        sector_type = str(item.get("板块类型") or "").strip()
        sector_name = str(item.get("板块名称") or "").strip()
        if not sector_type or not sector_name:
            continue
        result[(sector_type, sector_name)] = item
    return result


def _daily_sort_columns() -> dict[str, str]:
    return {
        LONG_MAINLINE_SHEET: "长期主线指数",
        SHORT_MAINLINE_SHEET: "短期主线指数",
        BUY_SCORE_SHEET: "P9买入分",
    }


def _intraday_sort_columns() -> dict[str, str]:
    return {
        LONG_MAINLINE_SHEET: "长期主线指数",
        SHORT_MAINLINE_SHEET: "短期主线指数",
        BUY_SCORE_SHEET: "P9买入分",
    }


def _format_workbook(target: Path) -> None:
    workbook = load_workbook(target)
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for column_cells in sheet.columns:
            header = str(column_cells[0].value or "")
            max_length = max(len(str(cell.value or "")) for cell in column_cells)
            width = min(max(max_length + 2, len(header) + 2, 10), 28)
            sheet.column_dimensions[column_cells[0].column_letter].width = width
    workbook.save(target)


def _selection_reason(item: dict[str, object]) -> str:
    reasons: list[str] = []
    if item.get("selected_as_long_mainline"):
        reasons.append("长期主线")
    if item.get("selected_as_short_mainline"):
        reasons.append("短期主线")
    if item.get("selected_as_phase9_buy"):
        reasons.append("P9高买入分")
    return "/".join(reasons)


def _sector_type_cn(value: object) -> str:
    text = str(value or "")
    if text == "industry":
        return "行业"
    if text == "concept":
        return "概念"
    return text


def _cell(value: object) -> object:
    if value is None:
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass
    return value


def _bool_cell(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "是"}
    return bool(value)
