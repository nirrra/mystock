from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .daily_returns import read_full_market_daily_returns
from .phase_display import normalize_symbol
from .sector_leaders import sector_leader_scores_all_path


CONCERN_LEADER_SCORE_THRESHOLD = 60.0


@dataclass(frozen=True)
class ConcernSectorResult:
    trade_date: date
    stock_path: Path
    member_path: Path
    stock_count: int
    weak_stock_count: int
    relation_count: int
    sector_count: int


def stock_concern_sectors_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "sectors" / f"stock_concern_sectors_{trade_date.isoformat()}.csv"


def concern_sector_members_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "sectors" / f"concern_sector_members_{trade_date.isoformat()}.csv"


def stock_concern_sectors_json_path(project_root: Path, trade_date: date) -> Path:
    return stock_concern_sectors_path(project_root, trade_date).with_suffix(".json")


def concern_sector_members_json_path(project_root: Path, trade_date: date) -> Path:
    return concern_sector_members_path(project_root, trade_date).with_suffix(".json")


def build_concern_sector_frames_from_files(
    *,
    project_root: Path,
    trade_date: date,
    leader_score_threshold: float = CONCERN_LEADER_SCORE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    leader_scores = _read_csv(sector_leader_scores_all_path(project_root, trade_date), dtype={"symbol": str, "sector_label": str})
    daily_returns = read_full_market_daily_returns(project_root=project_root, trade_date=trade_date)
    universe = _load_universe(project_root)
    return build_concern_sector_frames(
        trade_date=trade_date,
        leader_scores=leader_scores,
        stock_base=_stock_base_frame(daily_returns=daily_returns, universe=universe, leader_scores=leader_scores),
        leader_score_threshold=leader_score_threshold,
    )


def write_concern_sector_frames_from_files(
    *,
    project_root: Path,
    trade_date: date,
    leader_score_threshold: float = CONCERN_LEADER_SCORE_THRESHOLD,
) -> ConcernSectorResult:
    stock_frame, member_frame = build_concern_sector_frames_from_files(
        project_root=project_root,
        trade_date=trade_date,
        leader_score_threshold=leader_score_threshold,
    )
    stock_path = stock_concern_sectors_path(project_root, trade_date)
    member_path = concern_sector_members_path(project_root, trade_date)
    stock_path.parent.mkdir(parents=True, exist_ok=True)
    stock_frame.to_csv(stock_path, index=False, encoding="utf-8-sig")
    member_frame.to_csv(member_path, index=False, encoding="utf-8-sig")
    stock_concern_sectors_json_path(project_root, trade_date).write_text(
        json.dumps(_frame_records(stock_frame), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    concern_sector_members_json_path(project_root, trade_date).write_text(
        json.dumps(_frame_records(member_frame), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ConcernSectorResult(
        trade_date=trade_date,
        stock_path=stock_path,
        member_path=member_path,
        stock_count=int(len(stock_frame)),
        weak_stock_count=int(stock_frame["是否弱势股"].fillna(False).astype(bool).sum()) if "是否弱势股" in stock_frame.columns else 0,
        relation_count=int(len(member_frame)),
        sector_count=int(member_frame[["板块类型", "板块代码"]].drop_duplicates().shape[0]) if not member_frame.empty else 0,
    )


def build_concern_sector_frames(
    *,
    trade_date: date,
    leader_scores: pd.DataFrame,
    stock_base: pd.DataFrame,
    leader_score_threshold: float = CONCERN_LEADER_SCORE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    relations = _prepare_leader_relations(leader_scores, leader_score_threshold=leader_score_threshold)
    stock_base = _prepare_stock_base(stock_base, relations=relations)
    if relations.empty:
        stock_output = _weak_stock_frame(trade_date=trade_date, stock_base=stock_base)
        return stock_output, _empty_member_frame()

    relation_groups = {
        symbol: group.sort_values(["龙头指数", "板块名称"], ascending=[False, True], kind="stable").reset_index(drop=True)
        for symbol, group in relations.groupby("编号", sort=False)
    }
    stock_rows: list[dict[str, object]] = []
    for _, stock in stock_base.sort_values("编号").iterrows():
        symbol = normalize_symbol(stock.get("编号"))
        group = relation_groups.get(symbol)
        if group is None or group.empty:
            stock_rows.append(
                {
                    "交易日期": trade_date.isoformat(),
                    "编号": _excel_symbol(symbol),
                    "名称": _cell(stock.get("名称")),
                    "是否弱势股": True,
                    "关切板块": "",
                    "最高龙头指数": pd.NA,
                    "关切板块数量": 0,
                    "关切板块详细": "[]",
                }
            )
            continue
        details = [
            {
                "板块类型": row["板块类型"],
                "板块名称": row["板块名称"],
                "板块代码": row["板块代码"],
                "龙头指数": _round(row["龙头指数"], 2),
                "长期龙头指数": _round(row["长期龙头指数"], 2),
                "波段龙头指数": _round(row["波段龙头指数"], 2),
                "龙头标签": _cell(row.get("龙头标签")),
            }
            for _, row in group.iterrows()
        ]
        stock_rows.append(
            {
                "交易日期": trade_date.isoformat(),
                "编号": _excel_symbol(symbol),
                "名称": _cell(stock.get("名称")) or _cell(group.iloc[0].get("名称")),
                "是否弱势股": False,
                "关切板块": "/".join(item["板块名称"] for item in details if item["板块名称"]),
                "最高龙头指数": _round(group["龙头指数"].max(), 2),
                "关切板块数量": len(details),
                "关切板块详细": json.dumps(details, ensure_ascii=False),
            }
        )

    stock_output = pd.DataFrame(stock_rows, columns=_stock_columns())
    member_output = relations.copy()
    member_output.insert(0, "交易日期", trade_date.isoformat())
    member_output["编号"] = member_output["编号"].map(_excel_symbol)
    member_output = member_output.loc[:, _member_columns()].sort_values(
        ["板块名称", "龙头指数", "编号"],
        ascending=[True, False, True],
        kind="stable",
    )
    return stock_output, member_output.reset_index(drop=True)


def read_stock_concern_sectors(*, project_root: Path, trade_date: date) -> pd.DataFrame:
    return _read_csv(stock_concern_sectors_path(project_root, trade_date), dtype={"编号": str})


def read_concern_sector_members(*, project_root: Path, trade_date: date) -> pd.DataFrame:
    return _read_csv(concern_sector_members_path(project_root, trade_date), dtype={"编号": str, "板块代码": str})


def _prepare_leader_relations(frame: pd.DataFrame, *, leader_score_threshold: float) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_member_columns()[1:])
    result = frame.copy()
    result["编号"] = _series(result, "symbol", "编号").map(normalize_symbol)
    result["名称"] = _series(result, "name", "名称").fillna("").astype(str)
    result["板块类型"] = _series(result, "sector_type", "板块类型").fillna("").astype(str).str.strip()
    result["板块名称"] = _series(result, "sector_name", "板块名称").fillna("").astype(str).str.strip()
    result["板块代码"] = _series(result, "sector_label", "板块代码").fillna("").astype(str).str.strip()
    result["龙头指数"] = pd.to_numeric(result.get("combined_leader_score", result.get("龙头指数")), errors="coerce")
    result["长期龙头指数"] = pd.to_numeric(result.get("long_term_leader_score", result.get("长期龙头指数")), errors="coerce")
    result["波段龙头指数"] = pd.to_numeric(result.get("swing_leader_score", result.get("波段龙头指数")), errors="coerce")
    result["龙头排名"] = pd.to_numeric(result.get("combined_rank", result.get("龙头排名")), errors="coerce")
    result["龙头标签"] = _series(result, "leader_tags", "龙头标签").fillna("").astype(str)
    result = result[
        result["编号"].astype(str).str.len().eq(6)
        & result["板块名称"].astype(str).str.strip().ne("")
        & result["龙头指数"].ge(float(leader_score_threshold))
    ].copy()
    if result.empty:
        return pd.DataFrame(columns=_member_columns()[1:])
    result = result.drop_duplicates(["编号", "板块类型", "板块代码"], keep="first")
    return result.loc[:, _member_columns()[1:]].sort_values(["编号", "龙头指数"], ascending=[True, False], kind="stable").reset_index(drop=True)


def _stock_base_frame(*, daily_returns: pd.DataFrame, universe: pd.DataFrame, leader_scores: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not daily_returns.empty:
        daily = daily_returns.rename(columns={"symbol": "编号", "name": "名称"}).loc[:, ["编号", "名称"]].copy()
        frames.append(daily)
    if not universe.empty and "symbol" in universe.columns:
        uni = universe.copy()
        uni["编号"] = uni["symbol"].map(normalize_symbol)
        uni["名称"] = _series(uni, "name").fillna("").astype(str)
        frames.append(uni.loc[:, ["编号", "名称"]])
    if not leader_scores.empty and "symbol" in leader_scores.columns:
        leaders = leader_scores.copy()
        leaders["编号"] = leaders["symbol"].map(normalize_symbol)
        leaders["名称"] = _series(leaders, "name").fillna("").astype(str)
        frames.append(leaders.loc[:, ["编号", "名称"]])
    if not frames:
        return pd.DataFrame(columns=["编号", "名称"])
    result = pd.concat(frames, ignore_index=True)
    return _prepare_stock_base(result, relations=pd.DataFrame())


def _prepare_stock_base(stock_base: pd.DataFrame, *, relations: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if stock_base is not None and not stock_base.empty:
        base = stock_base.copy()
        if "编号" not in base.columns and "symbol" in base.columns:
            base["编号"] = base["symbol"]
        if "名称" not in base.columns and "name" in base.columns:
            base["名称"] = base["name"]
        base["编号"] = base["编号"].map(normalize_symbol)
        base["名称"] = _series(base, "名称").fillna("").astype(str)
        frames.append(base.loc[:, ["编号", "名称"]])
    if relations is not None and not relations.empty:
        frames.append(relations.loc[:, ["编号", "名称"]].copy())
    if not frames:
        return pd.DataFrame(columns=["编号", "名称"])
    result = pd.concat(frames, ignore_index=True)
    result = result[result["编号"].astype(str).str.len().eq(6)].copy()
    result["名称"] = result["名称"].fillna("").astype(str)
    result = result.sort_values(["编号", "名称"], ascending=[True, False], kind="stable")
    return result.drop_duplicates("编号", keep="first").reset_index(drop=True)


def _weak_stock_frame(*, trade_date: date, stock_base: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "交易日期": trade_date.isoformat(),
            "编号": _excel_symbol(row["编号"]),
            "名称": row.get("名称", ""),
            "是否弱势股": True,
            "关切板块": "",
            "最高龙头指数": pd.NA,
            "关切板块数量": 0,
            "关切板块详细": "[]",
        }
        for _, row in stock_base.iterrows()
    ]
    return pd.DataFrame(rows, columns=_stock_columns())


def _empty_member_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_member_columns())


def _stock_columns() -> list[str]:
    return ["交易日期", "编号", "名称", "是否弱势股", "关切板块", "最高龙头指数", "关切板块数量", "关切板块详细"]


def _member_columns() -> list[str]:
    return ["交易日期", "编号", "名称", "板块类型", "板块名称", "板块代码", "龙头指数", "长期龙头指数", "波段龙头指数", "龙头排名", "龙头标签"]


def _load_universe(project_root: Path) -> pd.DataFrame:
    for path in (project_root / "data" / "universe.parquet", project_root / "data" / "universe.csv"):
        if not path.exists():
            continue
        try:
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path, dtype={"symbol": str})
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _read_csv(path: Path, *, dtype: dict[str, Any] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=dtype)


def _series(frame: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series([""] * len(frame), index=frame.index, dtype=object)


def _frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {key: _json_value(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _json_value(value: object) -> object:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _round(value: object, digits: int) -> object:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return pd.NA
    if pd.isna(number):
        return pd.NA
    return round(number, digits)


def _cell(value: object) -> object:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return value


def _excel_symbol(symbol: object) -> str:
    return f'="{normalize_symbol(symbol)}"'
