from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PATTERN_PRIORITY = {
    "1": 3.0,
    "3": 2.8,
    "2": 2.5,
    "4": 1.8,
}

LABEL_BONUS = {
    "strong_buy": 1.0,
    "buy": 0.6,
    "neutral": 0.0,
    "sell": -0.6,
    "strong_sell": -1.0,
}

MAINLINE_SECTIONS = (
    "第一梯队主线",
    "第二梯队主线",
    "轮动与预期线",
    "短线事件题材",
)

INDEX_NAME_MARKERS = {
    "指数",
    "上证综指",
    "深证成指",
    "创业板指",
    "沪深300",
    "中证500",
    "中证1000",
    "科创50",
}

PATTERNS_FILE_DATE_RE = re.compile(r"patterns_all_(\d{4}-\d{2}-\d{2})\.csv$")


def _normalize_symbol(value: object) -> str:
    text = str(value).strip()
    if text.startswith('="') and text.endswith('"'):
        text = text[2:-1]
    return text.zfill(6)


def _latest_patterns_file(project_root: Path) -> Path:
    files = sorted(
        (project_root / "reports" / "patterns").glob("patterns_all_*.csv"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError("No reports/patterns/patterns_all_*.csv file found. Run mystock pattern first.")
    return files[0]


def _daily_rating_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith("tradingview_all_rating_20"))


def _stable_score(row: pd.Series, daily_columns: list[str]) -> float:
    avg_score = float(row.get("tradingview_avg_all_rating_5d", 0.0))
    latest_score = float(row.get(daily_columns[-1], avg_score)) if daily_columns else avg_score
    pattern_id = str(row.get("pattern_id", ""))
    label = str(row.get("tradingview_all_rating_label", "")).strip().lower()
    return round(
        avg_score * 4.0
        + latest_score * 1.5
        + PATTERN_PRIORITY.get(pattern_id, 0.0)
        + LABEL_BONUS.get(label, -1.0),
        4,
    )


def _base_tier(row: pd.Series) -> str | None:
    avg_score = float(row.get("tradingview_avg_all_rating_5d", 0.0))
    pattern_id = str(row.get("pattern_id", ""))
    label = str(row.get("tradingview_all_rating_label", "")).strip().lower()

    if label not in {"buy", "strong_buy"}:
        return None
    if avg_score < 0.22:
        return None

    if pattern_id in {"1", "3"} and avg_score >= 0.36:
        return "第一梯队"
    if pattern_id == "2" and avg_score >= 0.35:
        return "第一梯队"
    if pattern_id == "4" and avg_score >= 0.42 and label == "strong_buy":
        return "第一梯队"

    if pattern_id in {"1", "2", "3"} and avg_score >= 0.28:
        return "第二梯队"
    if pattern_id == "4" and avg_score >= 0.32:
        return "第二梯队"

    return "第三梯队"


def _is_true_index_name(name: object) -> bool:
    normalized = str(name).strip().replace(" ", "")
    return any(marker in normalized for marker in INDEX_NAME_MARKERS)


def _parse_pick_sections(markdown_text: str) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if line.startswith("### "):
            current = {"date": line[4:].strip(), "symbols": [], "themes": []}
            sections.append(current)
            continue
        if current is None:
            continue
        if not line.startswith("|") or "股票代码" in line or "--------" in line:
            continue
        columns = [item.strip() for item in line.strip("|").split("|")]
        if len(columns) < 4:
            continue
        symbol = columns[1]
        theme = columns[3]
        if symbol.isdigit():
            current["symbols"].append(symbol)
            current["themes"].append(theme)
    return sections


def _load_symbol_theme_map(picks_path: Path) -> dict[str, str]:
    if not picks_path.exists():
        return {}

    mapping: dict[str, str] = {}
    for section in _parse_pick_sections(picks_path.read_text(encoding="utf-8")):
        symbols = section.get("symbols", [])
        themes = section.get("themes", [])
        if not isinstance(symbols, list) or not isinstance(themes, list):
            continue
        for symbol, theme in zip(symbols, themes):
            if isinstance(symbol, str) and symbol.isdigit() and isinstance(theme, str) and theme:
                mapping.setdefault(symbol, theme)
    return mapping


def _normalize_mainline_theme(title: str) -> str:
    normalized = title.strip().replace(" ", "")
    if "算力硬件" in normalized:
        return "算力硬件"
    if "电池" in normalized:
        return "电池"
    if "创新药" in normalized:
        return "创新药"
    if "机器人" in normalized:
        return "机器人"
    if "稳定币" in normalized or "支付" in normalized:
        return "稳定币/支付"
    return title.strip()


def _load_mainline_document(mainline_path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "exists": mainline_path.exists(),
        "path": str(mainline_path),
        "sections": {section: [] for section in MAINLINE_SECTIONS},
        "one_line_conclusion": "",
    }
    if not mainline_path.exists():
        return result

    text = mainline_path.read_text(encoding="utf-8")
    current_section: str | None = None
    capture_one_line = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            title = line[3:].strip()
            capture_one_line = "一句话结论" in title
            current_section = next((section for section in MAINLINE_SECTIONS if section in title), None)
            continue
        if line.startswith("### ") and current_section:
            theme = _normalize_mainline_theme(line[4:].strip())
            section_items = result["sections"][current_section]
            if theme and theme not in section_items:
                section_items.append(theme)
            continue
        if capture_one_line and line and not line.startswith("#"):
            if line.startswith("`"):
                result["one_line_conclusion"] = line.strip("`")
                capture_one_line = False
                continue
            if not result["one_line_conclusion"]:
                result["one_line_conclusion"] = line.strip("`")
    return result


def _build_market_sentiment(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "level": "偏谨慎观察",
            "summary": "当日没有筛出满足当前阈值的候选，说明形态与强度共振不足，先以观察为主。",
            "candidate_count": 0,
            "first_tier_count": 0,
            "strong_buy_count": 0,
            "buy_count": 0,
            "top_divergence_count": 0,
            "bottom_divergence_count": 0,
        }

    candidate_count = len(rows)
    first_tier_count = sum(1 for row in rows if row["tier"] == "第一梯队")
    strong_buy_count = sum(1 for row in rows if row["tradingview_label"] == "strong_buy")
    buy_count = sum(1 for row in rows if row["tradingview_label"] == "buy")
    top_divergence_count = sum(1 for row in rows if row["macd_top_divergence_15d"])
    bottom_divergence_count = sum(1 for row in rows if row["macd_bottom_divergence_15d"])
    median_avg_5d = pd.Series([float(row["tradingview_avg_5d"]) for row in rows]).median()

    if first_tier_count >= max(1, candidate_count // 2) and strong_buy_count >= max(1, candidate_count // 3) and top_divergence_count <= bottom_divergence_count:
        level = "偏强共振"
    elif top_divergence_count > bottom_divergence_count and first_tier_count <= max(1, candidate_count // 3):
        level = "偏谨慎观察"
    else:
        level = "强中有分化"

    summary = (
        f"当日共筛出 {candidate_count} 只候选，其中第一梯队 {first_tier_count} 只，"
        f"strong_buy {strong_buy_count} 只、buy {buy_count} 只，五日均分中位数 {median_avg_5d:.4f}。"
        f"近15日顶背离 {top_divergence_count} 只、底背离 {bottom_divergence_count} 只，整体情绪{level}。"
    )
    return {
        "level": level,
        "summary": summary,
        "candidate_count": candidate_count,
        "first_tier_count": first_tier_count,
        "strong_buy_count": strong_buy_count,
        "buy_count": buy_count,
        "top_divergence_count": top_divergence_count,
        "bottom_divergence_count": bottom_divergence_count,
    }


def _build_mainline_changes(
    rows: list[dict[str, object]],
    history_sections: list[dict[str, object]],
    mainline_document: dict[str, object],
) -> dict[str, object]:
    sections = mainline_document.get("sections", {})
    current_core = sections.get("第一梯队主线", []) if isinstance(sections, dict) else []
    current_secondary = sections.get("第二梯队主线", []) if isinstance(sections, dict) else []
    current_rotation = sections.get("轮动与预期线", []) if isinstance(sections, dict) else []
    current_event = sections.get("短线事件题材", []) if isinstance(sections, dict) else []

    theme_counts = Counter(
        str(row["theme"])
        for row in rows
        if isinstance(row.get("theme"), str) and row["theme"] and row["theme"] != "未分类"
    )
    latest_history = history_sections[0] if history_sections else None
    previous_theme_counts = Counter(latest_history.get("themes", [])) if isinstance(latest_history, dict) else Counter()

    base_parts: list[str] = []
    if current_core:
        base_parts.append(f"当前主线文件仍以 {', '.join(current_core)} 为核心")
    if current_secondary:
        base_parts.append(f"{', '.join(current_secondary)} 处于次主线")
    if current_rotation:
        base_parts.append(f"{', '.join(current_rotation)} 更偏轮动线")
    if current_event:
        base_parts.append(f"{', '.join(current_event)} 属于短线事件题材")

    if theme_counts:
        lead_theme, lead_count = sorted(theme_counts.items(), key=lambda item: (-item[1], item[0]))[0]
        previous_lead = sorted(previous_theme_counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if previous_theme_counts else None
        if previous_lead and previous_lead == lead_theme:
            base_parts.append(f"今日候选中 {lead_theme} 仍然最集中，共 {lead_count} 只，延续上一期主线分布")
        else:
            base_parts.append(f"今日候选中 {lead_theme} 最集中，共 {lead_count} 只，主线映射较上一期出现变化")
    else:
        base_parts.append("但今日候选大多仍未建立稳定主题映射，先不对个股主线归属做强判断")

    conclusion = mainline_document.get("one_line_conclusion", "")
    if isinstance(conclusion, str) and conclusion:
        base_parts.append(f"主线文件结论为：{conclusion}")

    return {
        "summary": "；".join(base_parts) + "。",
        "current_mainlines": {
            "core": current_core,
            "secondary": current_secondary,
            "rotation": current_rotation,
            "event": current_event,
        },
        "theme_counts": dict(theme_counts),
    }


def _build_pick_changes(rows: list[dict[str, object]], history_sections: list[dict[str, object]]) -> dict[str, object]:
    current_symbols = sorted({str(row["symbol"]) for row in rows})
    if not history_sections:
        return {
            "summary": "首期记录，暂无上一期可比数据。",
            "added": current_symbols,
            "retained": [],
            "removed": [],
        }

    previous_symbols = sorted({str(symbol) for symbol in history_sections[0].get("symbols", [])})
    current_set = set(current_symbols)
    previous_set = set(previous_symbols)

    added = sorted(current_set - previous_set)
    retained = sorted(current_set & previous_set)
    removed = sorted(previous_set - current_set)

    parts = []
    parts.append(f"新增 {len(added)} 只" + (f"：{', '.join(added[:5])}" if added else ""))
    parts.append(f"保留 {len(retained)} 只" + (f"：{', '.join(retained[:5])}" if retained else ""))
    parts.append(f"移除 {len(removed)} 只" + (f"：{', '.join(removed[:5])}" if removed else ""))

    return {
        "summary": "；".join(parts) + "。",
        "added": added,
        "retained": retained,
        "removed": removed,
    }


def _build_notable_stocks(rows: list[dict[str, object]], history_sections: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []

    recent_symbol_counts = Counter()
    for section in history_sections[:5]:
        recent_symbol_counts.update(str(symbol) for symbol in section.get("symbols", []))

    notable: list[dict[str, object]] = []
    for row in rows:
        symbol = str(row["symbol"])
        label = str(row["tradingview_label"])
        repeated = recent_symbol_counts[symbol] > 0
        summary = ""
        if repeated and row["tier"] == "第一梯队" and label == "strong_buy":
            summary = "连续入选且仍在第一梯队，强度延续。"
        elif row["macd_bottom_divergence_15d"]:
            summary = "近15日出现底背离，可结合位置观察修复。"
        elif row["macd_top_divergence_15d"]:
            summary = "近15日出现顶背离，若继续冲高需防追高。"
        elif row["tier"] == "第一梯队" and label == "strong_buy":
            summary = "第一梯队且 strong_buy，属于当日更强共振标的。"
        elif repeated:
            summary = "近几日重复入选，说明资金关注度仍在。"
        if not summary:
            continue
        notable.append(
            {
                "symbol": symbol,
                "name": row["name"],
                "theme": row["theme"],
                "summary": summary,
            }
        )
        if len(notable) >= 5:
            break

    return notable[:5]


def build_candidates(project_root: Path, limit: int) -> dict[str, object]:
    patterns_file = _latest_patterns_file(project_root)
    mainline_path = project_root / "主线.md"
    picks_path = project_root / "选股.md"
    theme_map = _load_symbol_theme_map(picks_path)
    history_sections = _parse_pick_sections(picks_path.read_text(encoding="utf-8")) if picks_path.exists() else []
    mainline_document = _load_mainline_document(mainline_path)

    frame = pd.read_csv(patterns_file)
    frame["symbol"] = frame["symbol"].map(_normalize_symbol)
    frame = frame[~frame["name"].map(_is_true_index_name)].copy()
    daily_columns = _daily_rating_columns(frame)
    required = {"symbol", "name", "pattern_id", "tradingview_avg_all_rating_5d", "tradingview_all_rating_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Latest patterns file is missing required TradingView columns: {missing}")

    frame["stable_score"] = frame.apply(lambda row: _stable_score(row, daily_columns), axis=1)
    frame["base_tier"] = frame.apply(_base_tier, axis=1)
    frame = frame.dropna(subset=["base_tier"]).copy()
    frame = frame.sort_values(["base_tier", "stable_score", "tradingview_avg_all_rating_5d"], ascending=[True, False, False])

    rows: list[dict[str, object]] = []
    for _, row in frame.head(limit).iterrows():
        symbol = row["symbol"]
        rows.append(
            {
                "tier": row["base_tier"],
                "symbol": symbol,
                "name": row["name"],
                "theme": theme_map.get(symbol, "未分类"),
                "pattern_id": str(row["pattern_id"]),
                "macd_top_divergence_15d": bool(row.get("macd_top_divergence_15d", False)),
                "macd_bottom_divergence_15d": bool(row.get("macd_bottom_divergence_15d", False)),
                "tradingview_label": str(row["tradingview_all_rating_label"]).strip().lower(),
                "tradingview_avg_5d": round(float(row["tradingview_avg_all_rating_5d"]), 4),
                "five_day_scores": [round(float(row[column]), 4) for column in daily_columns if pd.notna(row.get(column))],
                "stable_score": round(float(row["stable_score"]), 4),
                "reason": row.get("reason", ""),
                "source_file": str(patterns_file),
            }
        )

    analysis = {
        "market_sentiment": _build_market_sentiment(rows),
        "mainline_changes": _build_mainline_changes(rows, history_sections, mainline_document),
        "pick_changes": _build_pick_changes(rows, history_sections),
        "notable_stocks": _build_notable_stocks(rows, history_sections),
    }

    return {
        "source_file": str(patterns_file),
        "theme_source_file": str(mainline_path),
        "daily_columns": daily_columns,
        "candidates": rows,
        "analysis": analysis,
    }


def infer_trade_date_from_payload(payload: dict[str, object]) -> date:
    source_file = str(payload.get("source_file", "")).strip()
    filename = Path(source_file).name
    match = PATTERNS_FILE_DATE_RE.search(filename)
    if not match:
        raise ValueError(f"Could not infer trade date from source file: {source_file}")
    return datetime.fromisoformat(match.group(1)).date()


def write_candidates_to_picks(
    *,
    project_root: Path,
    payload: dict[str, object],
    picks_filename: str,
    trade_date: date | None = None,
) -> Path:
    from stocks_analyzer.daily_screening import write_picker_payload_to_picks

    resolved_trade_date = trade_date or infer_trade_date_from_payload(payload)
    _, picks_path = write_picker_payload_to_picks(
        project_root=project_root,
        trade_date=resolved_trade_date,
        picker_payload=payload,
        picks_filename=picks_filename,
    )
    return picks_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the latest project pattern report and score stable-profit candidates.")
    parser.add_argument("--project-root", default=r"C:\Users\wdyab\Desktop\wdy\stocks")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--write-picks", action="store_true", help="Write or replace the same-day section in 选股.md")
    parser.add_argument("--picks-file", default="选股.md", help="Markdown picks file to update when --write-picks is enabled")
    parser.add_argument("--trade-date", default=None, help="Optional trade date override in YYYY-MM-DD format")
    args = parser.parse_args()

    payload = build_candidates(Path(args.project_root), limit=args.limit)
    if args.write_picks:
        override_trade_date = datetime.fromisoformat(args.trade_date).date() if args.trade_date else None
        picks_path = write_candidates_to_picks(
            project_root=Path(args.project_root),
            payload=payload,
            picks_filename=args.picks_file,
            trade_date=override_trade_date,
        )
        payload["written_picks_path"] = str(picks_path)
        payload["written_trade_date"] = (override_trade_date or infer_trade_date_from_payload(payload)).isoformat()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
