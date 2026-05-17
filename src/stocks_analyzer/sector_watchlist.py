from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .sector_leaders import sector_leaders_path
from .sector_membership import sector_performance_path
from .sector_phase9 import build_sector_phase9_panel, sector_phase9_predictions_path


LONG_MAINLINE_TOP_N = 100
SHORT_MAINLINE_TOP_N = 10
PHASE9_BUY_TOP_N = 10
SECTOR_LEADER_TOP_N = 3
MAINLINE_HISTORY_DAYS = 900


def watchlist_sectors_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "watchlists" / f"watchlist_sectors_{trade_date.isoformat()}.json"


def build_sector_watchlist(
    *,
    project_root: Path,
    trade_date: date,
    sector_performance: pd.DataFrame,
    sector_leaders: pd.DataFrame,
    phase9_predictions: pd.DataFrame | None = None,
    mainline_scores: pd.DataFrame | None = None,
    source_files: dict[str, str] | None = None,
) -> dict[str, object]:
    performance = _prepare_sector_performance(sector_performance)
    phase9 = _prepare_phase9_predictions(phase9_predictions)
    mainline = _prepare_mainline_scores(mainline_scores)
    leaders = _prepare_sector_leaders(sector_leaders)

    sectors = _build_sector_score_frame(performance=performance, mainline=mainline, phase9=phase9)
    if sectors.empty:
        records: list[dict[str, object]] = []
    else:
        sectors["selected"] = sectors["selected_as_long_mainline"]
        selected = sectors[sectors["selected"].fillna(False)].copy()
        selected["selection_score"] = selected[
            ["long_mainline_score_100", "short_mainline_score_100", "phase9_score_100"]
        ].max(axis=1, skipna=True)
        selected = selected.sort_values(
            ["long_mainline_rank", "long_mainline_score_100", "short_mainline_rank_in_long_pool", "phase9_rank_in_long_pool", "sector_type", "sector_name"],
            ascending=[True, False, True, True, True, True],
            na_position="last",
        )
        records = [
            _sector_record(row=row, leaders=leaders)
            for _, row in selected.iterrows()
        ]
    strength_summary = _mainline_strength_summary(sectors if not sectors.empty else pd.DataFrame())

    return {
        "trade_date": trade_date.isoformat(),
        "source_files": source_files or {},
        "selection_policy": {
            "long_mainline_top_n": LONG_MAINLINE_TOP_N,
            "short_mainline_top_n": SHORT_MAINLINE_TOP_N,
            "phase9_buy_top_n": PHASE9_BUY_TOP_N,
            "leader_top_n": SECTOR_LEADER_TOP_N,
            "deprecated_phases": ["P3", "P5", "P7", "P8", "P10"],
            "note": "板块池按长期主线指数取Top100；短期主线和P9买入分分别在长期主线Top100内取Top10做标签。",
        },
        "mainline_strength_summary": strength_summary,
        "sector_count": len(records),
        "sectors": records,
    }


def build_sector_tracking_payload(
    *,
    project_root: Path,
    trade_date: date,
    sector_performance: pd.DataFrame,
    sector_leaders: pd.DataFrame,
    phase9_predictions: pd.DataFrame | None = None,
    mainline_scores: pd.DataFrame | None = None,
    source_files: dict[str, str] | None = None,
) -> dict[str, object]:
    performance = _prepare_sector_performance(sector_performance)
    phase9 = _prepare_phase9_predictions(phase9_predictions)
    mainline = _prepare_mainline_scores(mainline_scores)
    leaders = _prepare_sector_leaders(sector_leaders)

    sectors = _build_sector_score_frame(performance=performance, mainline=mainline, phase9=phase9)
    if sectors.empty:
        records: list[dict[str, object]] = []
    else:
        sectors = sectors.copy()
        sectors["selection_score"] = sectors[
            ["long_mainline_score_100", "short_mainline_score_100", "phase9_score_100"]
        ].max(axis=1, skipna=True)
        sectors = sectors.sort_values(
            ["long_mainline_score_100", "short_mainline_score_100", "phase9_score_100", "sector_type", "sector_name"],
            ascending=[False, False, False, True, True],
            na_position="last",
        )
        records = [_sector_record(row=row, leaders=leaders) for _, row in sectors.iterrows()]

    return {
        "trade_date": trade_date.isoformat(),
        "source_files": source_files or {},
        "selection_policy": {
            "long_mainline_top_n": LONG_MAINLINE_TOP_N,
            "short_mainline_top_n": SHORT_MAINLINE_TOP_N,
            "phase9_buy_top_n": PHASE9_BUY_TOP_N,
            "leader_top_n": SECTOR_LEADER_TOP_N,
            "scope": "all_sectors",
            "note": "主线跟踪表显示所有板块；三张sheet分别按长期主线指数、短期主线指数、P9买入分排序。",
        },
        "sector_count": len(records),
        "sectors": records,
    }


def write_sector_watchlist(
    *,
    project_root: Path,
    trade_date: date,
    payload: dict[str, object],
) -> Path:
    target = watchlist_sectors_path(project_root, trade_date)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_sector_watchlist_csv(target.with_suffix(".csv"), payload)
    return target


def load_sector_watchlist(*, project_root: Path, trade_date: date) -> dict[str, object]:
    target = watchlist_sectors_path(project_root, trade_date)
    if not target.exists():
        raise FileNotFoundError(f"Sector watchlist not found for {trade_date.isoformat()}: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def build_sector_watchlist_from_files(*, project_root: Path, trade_date: date) -> dict[str, object]:
    performance_path = sector_performance_path(project_root, trade_date)
    leaders_path = sector_leaders_path(project_root, trade_date)
    phase9_path = sector_phase9_predictions_path(project_root, trade_date)
    mainline_frame = _read_or_build_mainline_scores(project_root=project_root, trade_date=trade_date)
    return build_sector_watchlist(
        project_root=project_root,
        trade_date=trade_date,
        sector_performance=_read_optional_csv(performance_path),
        sector_leaders=_read_optional_csv(leaders_path),
        phase9_predictions=_read_optional_csv(phase9_path),
        mainline_scores=mainline_frame,
        source_files={
            "sector_performance": str(performance_path),
            "sector_leaders": str(leaders_path),
            "phase9": str(phase9_path) if phase9_path.exists() else "",
            "mainline_scores": str(_mainline_scores_path(project_root, trade_date)),
        },
    )


def build_sector_tracking_payload_from_files(*, project_root: Path, trade_date: date) -> dict[str, object]:
    performance_path = sector_performance_path(project_root, trade_date)
    leaders_path = sector_leaders_path(project_root, trade_date)
    phase9_path = sector_phase9_predictions_path(project_root, trade_date)
    mainline_frame = _read_or_build_mainline_scores(project_root=project_root, trade_date=trade_date)
    return build_sector_tracking_payload(
        project_root=project_root,
        trade_date=trade_date,
        sector_performance=_read_optional_csv(performance_path),
        sector_leaders=_read_optional_csv(leaders_path),
        phase9_predictions=_read_optional_csv(phase9_path),
        mainline_scores=mainline_frame,
        source_files={
            "sector_performance": str(performance_path),
            "sector_leaders": str(leaders_path),
            "phase9": str(phase9_path) if phase9_path.exists() else "",
            "mainline_scores": str(_mainline_scores_path(project_root, trade_date)),
        },
    )


def _build_sector_score_frame(
    *,
    performance: pd.DataFrame,
    mainline: pd.DataFrame,
    phase9: pd.DataFrame,
) -> pd.DataFrame:
    sector_frames = [frame for frame in (performance, mainline, phase9) if not frame.empty]
    if not sector_frames:
        return pd.DataFrame()

    sectors = sector_frames[0].copy()
    for frame in sector_frames[1:]:
        sectors = sectors.merge(frame, on=["sector_type", "sector_label"], how="outer", suffixes=("", "_extra"))
        for column in ("trade_date", "sector_name", "member_count", "valid_count"):
            extra_column = f"{column}_extra"
            if extra_column in sectors.columns:
                if column in sectors.columns:
                    sectors[column] = sectors[column].where(sectors[column].notna(), sectors[extra_column])
                else:
                    sectors[column] = sectors[extra_column]
                sectors = sectors.drop(columns=[extra_column])
        sectors = sectors.drop(columns=[column for column in sectors.columns if column.endswith("_extra")], errors="ignore")

    sectors["long_mainline_score_100"] = pd.to_numeric(sectors.get("long_mainline_score"), errors="coerce")
    sectors["phase9_score_100"] = pd.to_numeric(sectors.get("phase9_score_100"), errors="coerce")
    sectors["short_mainline_score_100"] = sectors.apply(_short_mainline_score, axis=1)
    sectors = _apply_sector_selection_policy(sectors)
    return sectors


def _apply_sector_selection_policy(sectors: pd.DataFrame) -> pd.DataFrame:
    result = sectors.copy()
    result["selected_as_long_mainline"] = False
    result["selected_as_short_mainline"] = False
    result["selected_as_phase9_buy"] = False
    result["long_mainline_rank"] = pd.NA
    result["short_mainline_rank_in_long_pool"] = pd.NA
    result["phase9_rank_in_long_pool"] = pd.NA
    if result.empty:
        return result

    long_sorted = result.dropna(subset=["long_mainline_score_100"]).sort_values(
        ["long_mainline_score_100", "sector_type", "sector_name"],
        ascending=[False, True, True],
        na_position="last",
    )
    long_pool = long_sorted.head(LONG_MAINLINE_TOP_N).copy()
    for rank, index in enumerate(long_pool.index, start=1):
        result.at[index, "selected_as_long_mainline"] = True
        result.at[index, "long_mainline_rank"] = rank
    if long_pool.empty:
        return result

    short_sorted = result.loc[long_pool.index].dropna(subset=["short_mainline_score_100"]).sort_values(
        ["short_mainline_score_100", "long_mainline_score_100", "sector_type", "sector_name"],
        ascending=[False, False, True, True],
        na_position="last",
    )
    for rank, index in enumerate(short_sorted.head(SHORT_MAINLINE_TOP_N).index, start=1):
        result.at[index, "selected_as_short_mainline"] = True
        result.at[index, "short_mainline_rank_in_long_pool"] = rank

    phase9_sorted = result.loc[long_pool.index].dropna(subset=["phase9_score_100"]).sort_values(
        ["phase9_score_100", "long_mainline_score_100", "sector_type", "sector_name"],
        ascending=[False, False, True, True],
        na_position="last",
    )
    for rank, index in enumerate(phase9_sorted.head(PHASE9_BUY_TOP_N).index, start=1):
        result.at[index, "selected_as_phase9_buy"] = True
        result.at[index, "phase9_rank_in_long_pool"] = rank
    return result


def _mainline_strength_summary(sectors: pd.DataFrame) -> dict[str, object]:
    if sectors.empty or "long_mainline_score_100" not in sectors.columns:
        return {
            "state": "unknown",
            "note": "没有可用长期主线指数。",
        }
    selected = sectors[sectors.get("selected_as_long_mainline", pd.Series(False, index=sectors.index)).fillna(False)].copy()
    scores = pd.to_numeric(selected.get("long_mainline_score_100"), errors="coerce").dropna().sort_values(ascending=False)
    if scores.empty:
        return {
            "state": "weak",
            "selected_count": 0,
            "note": "长期主线Top100为空，市场主线不清晰。",
        }
    top10_avg = float(scores.head(min(10, len(scores))).mean())
    top30_avg = float(scores.head(min(30, len(scores))).mean())
    pool_min = float(scores.iloc[-1])
    strong_count_90 = int(scores.ge(90.0).sum())
    strong_count_80 = int(scores.ge(80.0).sum())
    if top10_avg >= 90.0 and strong_count_90 >= 10:
        state = "strong"
        note = "头部主线强度较高。"
    elif top10_avg >= 82.0 and strong_count_80 >= 30:
        state = "mixed"
        note = "有主线，但强度分化，需要优先看高分板块。"
    else:
        state = "chaotic"
        note = "入选数量不代表主线清晰，头部强度不足时按市场混沌处理。"
    return {
        "state": state,
        "selected_count": int(len(scores)),
        "top1_score": round(float(scores.iloc[0]), 2),
        "top10_avg_score": round(top10_avg, 2),
        "top30_avg_score": round(top30_avg, 2),
        "pool_min_score": round(pool_min, 2),
        "count_ge_90": strong_count_90,
        "count_ge_80": strong_count_80,
        "note": note,
    }


def _prepare_sector_performance(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    for column in ("sector_type", "sector_label", "sector_name"):
        if column not in result.columns:
            result[column] = ""
        result[column] = result[column].astype(str).str.strip()
    for column in ("avg_pct_change", "amount_weighted_pct_change", "up_ratio", "total_amount", "member_count", "valid_count"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.drop_duplicates(["sector_type", "sector_label"], keep="first")


def _prepare_phase9_predictions(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    for column in ("sector_type", "sector_label", "sector_name"):
        if column not in result.columns:
            result[column] = ""
        result[column] = result[column].astype(str).str.strip()
    for column in (
        "phase9_score_100",
        "phase9_probability",
        "phase9_rank",
        "long_mainline_score",
        "return_5d",
        "return_20d",
        "ma5_slope_pct",
        "ma10_slope_pct",
        "ma20_slope_pct",
        "drawdown_from_peak_120d_pct",
    ):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.drop_duplicates(["sector_type", "sector_label"], keep="first")


def _prepare_mainline_scores(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    for column in ("sector_type", "sector_label", "sector_name"):
        if column not in result.columns:
            result[column] = ""
        result[column] = result[column].astype(str).str.strip()
    for column in (
        "long_mainline_score",
        "return_5d",
        "return_20d",
        "ma5_slope_pct",
        "member_count",
        "valid_count",
    ):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.drop_duplicates(["sector_type", "sector_label"], keep="first")


def _prepare_sector_leaders(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    for column in ("sector_type", "sector_label", "sector_name", "symbol", "name", "leader_type"):
        if column not in result.columns:
            result[column] = ""
        result[column] = result[column].astype(str).str.strip()
    if "symbol" in result.columns:
        result["symbol"] = result["symbol"].str.replace('="', "", regex=False).str.replace('"', "", regex=False).str.zfill(6)
    for column in ("leader_rank", "leader_score", "combined_leader_score", "long_term_leader_score", "swing_leader_score"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _mainline_scores_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "sectors" / f"sector_mainline_scores_{trade_date.isoformat()}.csv"


def _read_or_build_mainline_scores(*, project_root: Path, trade_date: date) -> pd.DataFrame:
    target = _mainline_scores_path(project_root, trade_date)
    if target.exists():
        return _read_optional_csv(target)
    panel = build_sector_phase9_panel(
        project_root=project_root,
        trade_date=trade_date,
        history_days=MAINLINE_HISTORY_DAYS,
        min_members=5,
        include_unlabeled=True,
        min_feature_history_days=60,
    )
    dataset = panel.dataset
    if dataset.empty:
        return pd.DataFrame()
    dated = dataset[dataset["trade_date"].dt.date.le(trade_date)].copy()
    if dated.empty:
        return pd.DataFrame()
    latest_date = dated["trade_date"].max()
    latest = dated[dated["trade_date"].eq(latest_date)].copy()
    columns = [
        "trade_date",
        "sector_type",
        "sector_name",
        "sector_label",
        "member_count",
        "valid_count",
        "long_mainline_score",
        "return_5d",
        "return_20d",
        "ma5_slope_pct",
    ]
    available = [column for column in columns if column in latest.columns]
    latest = latest[available].copy()
    target.parent.mkdir(parents=True, exist_ok=True)
    latest.to_csv(target, index=False, encoding="utf-8-sig")
    return latest


def _short_mainline_score(row: pd.Series) -> float:
    five_day = _linear_score(_number(row.get("return_5d")), low=-3.0, high=8.0)
    twenty_day = _linear_score(_number(row.get("return_20d")), low=-8.0, high=18.0)
    daily_weighted = _linear_score(_number(row.get("amount_weighted_pct_change")), low=-2.0, high=5.0)
    up_ratio = _linear_score(_number(row.get("up_ratio")), low=0.35, high=0.80)
    ma5 = _linear_score(_number(row.get("ma5_slope_pct")), low=-0.8, high=1.2)
    score = 0.25 * five_day + 0.25 * twenty_day + 0.20 * daily_weighted + 0.15 * up_ratio + 0.15 * ma5
    return round(float(score), 2)


def _sector_record(*, row: pd.Series, leaders: pd.DataFrame) -> dict[str, object]:
    sector_type = str(row.get("sector_type", "") or "")
    sector_label = str(row.get("sector_label", "") or "")
    leader_rows = _top_leaders(leaders=leaders, sector_type=sector_type, sector_label=sector_label)
    return {
        "trade_date": _cell(row.get("trade_date")),
        "sector_type": sector_type,
        "sector_name": _cell(row.get("sector_name")),
        "sector_label": sector_label,
        "member_count": _cell(row.get("member_count")),
        "valid_count": _cell(row.get("valid_count")),
        "sector_avg_pct_change": _round(row.get("avg_pct_change"), 4),
        "sector_amount_weighted_pct_change": _round(row.get("amount_weighted_pct_change"), 4),
        "sector_up_count": _cell(row.get("up_count")),
        "sector_up_ratio": _round(row.get("up_ratio"), 4),
        "long_mainline_score_100": _round(row.get("long_mainline_score_100"), 2),
        "short_mainline_score_100": _round(row.get("short_mainline_score_100"), 2),
        "phase9_score_100": _round(row.get("phase9_score_100"), 2),
        "phase9_rank": _cell(row.get("phase9_rank")),
        "long_mainline_rank": _cell(row.get("long_mainline_rank")),
        "short_mainline_rank_in_long_pool": _cell(row.get("short_mainline_rank_in_long_pool")),
        "phase9_rank_in_long_pool": _cell(row.get("phase9_rank_in_long_pool")),
        "selected_as_long_mainline": bool(row.get("selected_as_long_mainline", False)),
        "selected_as_short_mainline": bool(row.get("selected_as_short_mainline", False)),
        "selected_as_phase9_buy": bool(row.get("selected_as_phase9_buy", False)),
        "leader_symbols": [item["symbol"] for item in leader_rows],
        "leader_names": [item["name"] for item in leader_rows],
        "leaders": leader_rows,
    }


def _top_leaders(*, leaders: pd.DataFrame, sector_type: str, sector_label: str) -> list[dict[str, object]]:
    if leaders.empty:
        return []
    group = leaders[
        leaders["sector_type"].astype(str).eq(str(sector_type))
        & leaders["sector_label"].astype(str).eq(str(sector_label))
    ].copy()
    if group.empty:
        return []
    if "is_dual_leader" in group.columns:
        dual = group["is_dual_leader"].astype(str).str.lower().isin({"true", "1", "yes", "是"})
    else:
        dual = pd.Series(False, index=group.index)
    group["_dual_sort"] = dual
    group = group.sort_values(
        ["_dual_sort", "combined_leader_score", "leader_score", "symbol"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    seen: set[str] = set()
    result: list[dict[str, object]] = []
    for _, row in group.iterrows():
        symbol = str(row.get("symbol", "") or "").zfill(6)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        result.append(
            {
                "symbol": symbol,
                "name": _cell(row.get("name")),
                "leader_score": _round(row.get("leader_score"), 2),
                "combined_leader_score": _round(row.get("combined_leader_score"), 2),
                "leader_type": _cell(row.get("leader_type")),
                "leader_tags": _cell(row.get("leader_tags")),
            }
        )
        if len(result) >= SECTOR_LEADER_TOP_N:
            break
    return result


def _write_sector_watchlist_csv(target: Path, payload: dict[str, object]) -> None:
    rows = payload.get("sectors")
    if not isinstance(rows, list):
        rows = []
    output_rows = []
    for index, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            continue
        output_rows.append(
            {
                "交易日期": payload.get("trade_date"),
                "序号": index,
                "板块类型": _sector_type_cn(item.get("sector_type")),
                "板块名称": item.get("sector_name"),
                "板块代码": item.get("sector_label"),
                "当日涨幅%": item.get("sector_avg_pct_change"),
                "成交额加权涨幅%": item.get("sector_amount_weighted_pct_change"),
                "上涨家数占比": item.get("sector_up_ratio"),
                "长期主线指数": item.get("long_mainline_score_100"),
                "长期主线排名": item.get("long_mainline_rank"),
                "短期主线指数": item.get("short_mainline_score_100"),
                "短期主线Top10排名": item.get("short_mainline_rank_in_long_pool"),
                "P9买入分": item.get("phase9_score_100"),
                "P9Top10排名": item.get("phase9_rank_in_long_pool"),
                "P9排名": item.get("phase9_rank"),
                "入选原因": _selection_reason(item),
                "龙头编号": "/".join(item.get("leader_symbols", []) if isinstance(item.get("leader_symbols"), list) else []),
                "龙头名称": "/".join(item.get("leader_names", []) if isinstance(item.get("leader_names"), list) else []),
                "龙头详细": json.dumps(item.get("leaders", []), ensure_ascii=False, default=_json_default),
                "成员数": item.get("member_count"),
                "有效成员数": item.get("valid_count"),
            }
        )
    frame = pd.DataFrame(output_rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "交易日期",
                "序号",
                "板块类型",
                "板块名称",
                "板块代码",
                "当日涨幅%",
                "成交额加权涨幅%",
                "上涨家数占比",
                "长期主线指数",
                "长期主线排名",
                "短期主线指数",
                "短期主线Top10排名",
                "P9买入分",
                "P9Top10排名",
                "P9排名",
                "入选原因",
                "龙头编号",
                "龙头名称",
                "龙头详细",
                "成员数",
                "有效成员数",
            ]
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8-sig")


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


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _linear_score(value: float | None, *, low: float, high: float) -> float:
    if value is None or not math.isfinite(value):
        return 50.0
    if high <= low:
        return 50.0
    return min(100.0, max(0.0, (value - low) / (high - low) * 100.0))


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _round(value: object, digits: int) -> object:
    number = _number(value)
    if number is None:
        return None
    return round(number, digits)


def _cell(value: object) -> object:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)
