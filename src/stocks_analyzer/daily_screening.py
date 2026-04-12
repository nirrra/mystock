from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from time import perf_counter

from .config import load_config
from .trading_calendar import is_trading_day


THEME_PRIORITY = ["电池", "算力硬件", "创新药", "机器人", "稳定币/支付"]
PICKS_FILENAME = "选股.md"


@dataclass
class ScreeningResult:
    trade_date: date
    skipped: bool
    message: str
    picks_path: Path | None = None
    report_path: Path | None = None


def run_daily_screening(
    *,
    project_root: Path,
    trade_date: date,
    start_date: str = "20240101",
    picks_filename: str = PICKS_FILENAME,
) -> ScreeningResult:
    config = load_config(project_root / "config" / "default.yaml")
    print(f"[0/4] 检查 {trade_date.isoformat()} 是否为交易日...", flush=True)
    trading_day = is_trading_day(config.provider, trade_date)
    if not trading_day:
        print(f"[0/4] {trade_date.isoformat()} 不是交易日，跳过每日筛选。", flush=True)
        return ScreeningResult(
            trade_date=trade_date,
            skipped=True,
            message=f"{trade_date.isoformat()} 不是交易日，已跳过每日筛选。",
        )
    print(f"[0/4] {trade_date.isoformat()} 是交易日，开始执行每日筛选。", flush=True)

    _run_project_stage(1, 4, "update", project_root, ["update", "--start-date", start_date])
    _run_project_stage(2, 4, "tradingview", project_root, ["tradingview", "--date", trade_date.isoformat()])
    _run_project_stage(3, 4, "divergence", project_root, ["divergence", "--date", trade_date.isoformat()])
    _run_project_stage(4, 4, "pattern", project_root, ["pattern", "--as-of", trade_date.isoformat()])

    picker_payload = _run_project_stock_picker(project_root)
    section, picks_path = write_picker_payload_to_picks(
        project_root=project_root,
        trade_date=trade_date,
        picker_payload=picker_payload,
        picks_filename=picks_filename,
    )

    report_path = _write_run_report(project_root, trade_date, picker_payload, section, picks_path)
    return ScreeningResult(
        trade_date=trade_date,
        skipped=False,
        message=f"已完成 {trade_date.isoformat()} 每日筛选，并写入 {picks_path}",
        picks_path=picks_path,
        report_path=report_path,
    )


def build_daily_section(
    *,
    trade_date: date,
    picker_payload: dict[str, object],
    existing_markdown: str,
    theme_map: dict[str, str],
) -> str:
    rows = list(_normalize_candidates(picker_payload, theme_map))
    grouped = _group_by_tier(rows)
    analysis_blocks = _render_analysis_sections(picker_payload, rows)

    blocks = [f"### {trade_date.year}.{trade_date.month}.{trade_date.day}", ""]
    if not rows:
        blocks.append("当日未筛出满足当前阈值的候选。")
        if analysis_blocks:
            blocks.append("")
            blocks.extend(analysis_blocks)
        return "\n".join(blocks).strip() + "\n"

    tier_order = ["第一梯队", "第二梯队", "第三梯队"]
    for tier in tier_order:
        tier_rows = grouped.get(tier, [])
        if not tier_rows:
            continue
        blocks.extend(render_tier_table(tier_rows))
        blocks.append("")

    blocks.extend(analysis_blocks)
    return "\n".join(blocks).strip() + "\n"


def parse_pick_history(markdown_text: str) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if line.startswith("### "):
            title = line[4:].strip()
            current = {"date": title, "symbols": [], "summary": ""}
            sections.append(current)
            continue
        if current is None:
            continue
        if line.startswith("|") and "股票代码" not in line and "--------" not in line:
            columns = [item.strip() for item in line.strip("|").split("|")]
            if len(columns) >= 3 and columns[1].isdigit():
                current["symbols"].append(columns[1])
        if line.startswith("总结："):
            current["summary"] = line[3:].strip()
    return sections


def build_summary(*, trade_date: date, rows: list[dict[str, str]], history: list[dict[str, object]]) -> str:
    if not rows:
        return "今天没有新增入选标的，说明模式强度或主题共振不足，先以空仓观察为主。"

    theme_counts = Counter(row["theme"] for row in rows)
    known_theme_counts = [(theme, count) for theme, count in theme_counts.items() if theme != "未分类"]
    known_theme_counts.sort(key=lambda item: (-item[1], THEME_PRIORITY.index(item[0]) if item[0] in THEME_PRIORITY else 99, item[0]))

    symbol_counts = Counter()
    for section in history[:5]:
        symbol_counts.update(section.get("symbols", []))
    repeated = [row["symbol"] for row in rows if symbol_counts[row["symbol"]] > 0]

    parts = [f"当日共筛出 {len(rows)} 只股票"]
    if known_theme_counts:
        lead_theme, lead_count = known_theme_counts[0]
        parts.append(f"{lead_theme} 方向占 {lead_count} 只，仍是更集中的主线")
    else:
        parts.append("当前多数标的还没有历史主题映射，后续会随着文件积累逐步稳定")

    first_tier_count = sum(1 for row in rows if row["tier"] == "第一梯队")
    parts.append(f"第一梯队有 {first_tier_count} 只，强度层次{'偏集中' if first_tier_count >= max(1, len(rows) // 2) else '相对分散'}")

    repeated = sorted(set(repeated))
    if repeated:
        parts.append(f"近几日持续重复出现的股票有 {', '.join(repeated[:5])}")
    else:
        parts.append("近几日重复入选的股票不多，说明轮动仍然较快")

    parts.append(f"本次记录日期为 {trade_date.isoformat()}，后续可继续观察重复入选个股是否开始收敛到少数核心票")
    return "；".join(parts) + "。"


def render_tier_table(rows: list[dict[str, str]]) -> list[str]:
    header = [
        "| 梯队 | 股票代码 | 股票名称 | 行业/主线 | 符合模式/背离 | 五日分数 | 五日均分 | TradingView标签 | 推荐理由 |",
        "| ---- | -------- | -------- | --------- | ------------- | -------- | -------: | --------------- | -------- |",
    ]
    lines = []
    for row in rows:
        lines.append(
            "| {tier} | {symbol} | {name} | {theme} | {pattern} | {scores} | {avg_5d} | {label} | {reason} |".format(
                **row
            )
        )
    return header + lines


def prepend_daily_section(*, existing_text: str, section: str) -> str:
    clean_existing = existing_text.strip()
    if not clean_existing:
        return section.strip() + "\n"
    return section.strip() + "\n\n" + clean_existing + "\n"


def upsert_daily_section(*, existing_text: str, section: str, trade_date: date) -> str:
    clean_existing = existing_text.strip()
    if not clean_existing:
        return section.strip() + "\n"

    target_heading = _section_heading(trade_date)
    lines = clean_existing.splitlines()
    headings = [index for index, line in enumerate(lines) if line.strip().startswith("### ")]
    for offset, start in enumerate(headings):
        if lines[start].strip() != target_heading:
            continue
        end = headings[offset + 1] if offset + 1 < len(headings) else len(lines)
        before = lines[:start]
        after = lines[end:]
        merged_parts: list[str] = []
        if before:
            merged_parts.append("\n".join(before).strip())
        merged_parts.append(section.strip())
        if after:
            merged_parts.append("\n".join(after).strip())
        return "\n\n".join(part for part in merged_parts if part) + "\n"
    return prepend_daily_section(existing_text=existing_text, section=section)


def write_picker_payload_to_picks(
    *,
    project_root: Path,
    trade_date: date,
    picker_payload: dict[str, object],
    picks_filename: str = PICKS_FILENAME,
) -> tuple[str, Path]:
    picks_path = project_root / picks_filename
    existing_markdown = picks_path.read_text(encoding="utf-8") if picks_path.exists() else ""
    theme_map = _load_symbol_theme_map(picks_path)
    section = build_daily_section(
        trade_date=trade_date,
        picker_payload=picker_payload,
        existing_markdown=existing_markdown,
        theme_map=theme_map,
    )
    updated_text = upsert_daily_section(existing_text=existing_markdown, section=section, trade_date=trade_date)
    picks_path.write_text(updated_text, encoding="utf-8")
    return section, picks_path


def _normalize_candidates(payload: dict[str, object], theme_map: dict[str, str]) -> list[dict[str, str]]:
    candidates = payload.get("candidates", [])
    rows: list[dict[str, str]] = []
    for item in candidates if isinstance(candidates, list) else []:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).zfill(6)
        tier = str(item.get("tier", "")).strip() or "第三梯队"
        theme = str(item.get("theme", "")).strip() or theme_map.get(symbol, "未分类")
        pattern_id = str(item.get("pattern_id", "")).strip()
        has_top_divergence = bool(item.get("macd_top_divergence_15d", False))
        has_bottom_divergence = bool(item.get("macd_bottom_divergence_15d", False))
        label = str(item.get("tradingview_label", "")).strip().lower()
        avg_5d = float(item.get("tradingview_avg_5d", 0.0) or 0.0)
        score_list = item.get("five_day_scores", [])
        scores = " / ".join(f"{float(score):.4f}" for score in score_list) if isinstance(score_list, list) and score_list else "-"
        rows.append(
            {
                "tier": tier,
                "symbol": symbol,
                "name": str(item.get("name", "")).strip(),
                "theme": theme,
                "pattern": _format_pattern_with_divergence(
                    pattern_id,
                    has_top_divergence=has_top_divergence,
                    has_bottom_divergence=has_bottom_divergence,
                ),
                "scores": scores,
                "avg_5d": f"{avg_5d:.4f}",
                "label": label,
                "reason": _build_reason(
                    theme=theme,
                    pattern_id=pattern_id,
                    label=label,
                    avg_5d=avg_5d,
                    has_top_divergence=has_top_divergence,
                    has_bottom_divergence=has_bottom_divergence,
                ),
            }
        )
    return rows


def _render_analysis_sections(payload: dict[str, object], rows: list[dict[str, str]]) -> list[str]:
    analysis = payload.get("analysis", {})
    if not isinstance(analysis, dict):
        analysis = {}

    market_sentiment = _analysis_summary(analysis.get("market_sentiment"))
    mainline_changes = _analysis_summary(analysis.get("mainline_changes"))
    pick_changes = _analysis_summary(analysis.get("pick_changes"))
    notable_stocks = _render_notable_stocks(analysis.get("notable_stocks"))

    if not market_sentiment:
        market_sentiment = f"当日共筛出 {len(rows)} 只候选，先以技术强度观察为主。" if rows else "当日没有筛出候选，先以观察为主。"
    if not mainline_changes:
        mainline_changes = "当前缺少结构化主线变动数据。"
    if not pick_changes:
        pick_changes = "当前缺少结构化选股变化数据。"
    if not notable_stocks:
        notable_stocks = "暂无特别需要额外提示的个股。"

    return [
        f"当日市场情绪监测：{market_sentiment}",
        f"主线变动：{mainline_changes}",
        f"选股变化：{pick_changes}",
        f"值得注意的股：{notable_stocks}",
    ]


def _analysis_summary(value: object) -> str:
    if isinstance(value, dict):
        summary = value.get("summary")
        if isinstance(summary, str):
            return summary.strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def _render_notable_stocks(value: object) -> str:
    if not isinstance(value, list) or not value:
        return ""

    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).zfill(6)
        name = str(item.get("name", "")).strip()
        summary = str(item.get("summary", "")).strip()
        if not summary:
            continue
        label = f"{symbol}{name}" if name else symbol
        parts.append(f"{label}：{summary}")
    return "；".join(parts) + "。" if parts else ""


def _group_by_tier(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["tier"], []).append(row)
    return grouped


def _load_symbol_theme_map(picks_path: Path) -> dict[str, str]:
    if not picks_path.exists():
        return {}
    mapping: dict[str, str] = {}
    for raw_line in picks_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or "股票代码" in line or "--------" in line:
            continue
        columns = [item.strip() for item in line.strip("|").split("|")]
        if len(columns) < 4:
            continue
        symbol = columns[1]
        theme = columns[3]
        if symbol.isdigit() and theme:
            mapping.setdefault(symbol, theme)
    return mapping


def _section_heading(trade_date: date) -> str:
    return f"### {trade_date.year}.{trade_date.month}.{trade_date.day}"


def _build_reason(
    *,
    theme: str,
    pattern_id: str,
    label: str,
    avg_5d: float,
    has_top_divergence: bool,
    has_bottom_divergence: bool,
) -> str:
    pattern_text = _format_pattern_label(pattern_id)
    divergence_text = _format_divergence_text(has_top_divergence=has_top_divergence, has_bottom_divergence=has_bottom_divergence)
    label_text = label or "unknown"
    if theme == "未分类":
        base = f"{pattern_text} 配合 {label_text}，五日均分 {avg_5d:.4f}"
    else:
        base = f"{theme} 方向里，{pattern_text} 配合 {label_text}，五日均分 {avg_5d:.4f}"
    if divergence_text:
        return f"{base}，近15日出现{divergence_text}，先结合位置判断信号有效性。"
    if theme == "未分类":
        return f"{base}，先按技术强度归入观察池。"
    return f"{base}，优先关注形态和强度同时成立的标的。"


def _format_pattern_label(pattern_id: str) -> str:
    pattern = str(pattern_id).strip()
    return f"pattern {pattern}" if pattern else "-"


def _format_pattern_with_divergence(pattern_id: str, *, has_top_divergence: bool, has_bottom_divergence: bool) -> str:
    pattern_text = _format_pattern_label(pattern_id)
    divergence_text = _format_divergence_text(
        has_top_divergence=has_top_divergence,
        has_bottom_divergence=has_bottom_divergence,
    )
    if not divergence_text:
        return pattern_text
    return f"{pattern_text} + {divergence_text}"


def _format_divergence_text(*, has_top_divergence: bool, has_bottom_divergence: bool) -> str:
    labels: list[str] = []
    if has_top_divergence:
        labels.append("顶背离")
    if has_bottom_divergence:
        labels.append("底背离")
    return " / ".join(labels)


def _run_project_command(project_root: Path, args: list[str]) -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    src_path = str(project_root / "src")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else src_path + os.pathsep + existing_pythonpath
    completed = subprocess.run(
        [sys.executable, "-m", "stocks_analyzer", "--project-root", str(project_root), *args],
        cwd=project_root,
        env=env,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(args)}")


def _run_project_stage(stage_index: int, total_stages: int, stage_name: str, project_root: Path, args: list[str]) -> None:
    started_at = perf_counter()
    print(f"[{stage_index}/{total_stages}] 开始 {stage_name}...", flush=True)
    _run_project_command(project_root, args)
    elapsed = perf_counter() - started_at
    print(f"[{stage_index}/{total_stages}] {stage_name} 完成，用时 {elapsed:.1f}s。", flush=True)


def _run_project_stock_picker(project_root: Path) -> dict[str, object]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    src_path = str(project_root / "src")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else src_path + os.pathsep + existing_pythonpath
    script_path = project_root / "skills" / "project-stock-picker" / "scripts" / "project_stock_picker.py"
    completed = subprocess.run(
        [sys.executable, str(script_path), "--project-root", str(project_root)],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"project-stock-picker failed:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}".strip())
    return json.loads(completed.stdout)


def _write_run_report(
    project_root: Path,
    trade_date: date,
    picker_payload: dict[str, object],
    section: str,
    picks_path: Path,
) -> Path:
    target = project_root / "reports" / "daily_screening"
    target.mkdir(parents=True, exist_ok=True)
    report_path = target / f"daily_screening_{trade_date.isoformat()}.json"
    report = {
        "trade_date": trade_date.isoformat(),
        "source_file": picker_payload.get("source_file"),
        "section_preview": section,
        "picks_path": str(picks_path),
        "candidate_count": len(picker_payload.get("candidates", [])) if isinstance(picker_payload.get("candidates"), list) else 0,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the daily stock screening workflow and prepend picks to 选股.md.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--date", default=None, help="目标日期，格式 YYYY-MM-DD，默认今天")
    parser.add_argument("--start-date", default="20240101", help="更新数据的起始日期，格式 YYYYMMDD")
    parser.add_argument("--picks-file", default=PICKS_FILENAME, help="选股结果 Markdown 文件名")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    trade_date = datetime.fromisoformat(args.date).date() if args.date else date.today()
    result = run_daily_screening(
        project_root=Path(args.project_root).resolve(),
        trade_date=trade_date,
        start_date=args.start_date,
        picks_filename=args.picks_file,
    )
    print(result.message)
    if result.report_path:
        print(f"报告文件：{result.report_path}")


if __name__ == "__main__":
    main()
