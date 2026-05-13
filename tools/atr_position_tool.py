from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from tkinter import Button, Canvas, END, Entry, Frame, Label, StringVar, Text, Tk, messagebox


POSITION_RISK_FRACTION = 0.02
POSITION_STOP_ATR_MULT = 2.0
POSITION_STAGED_EFFECTIVE_RISK_MULT = 0.85
POSITION_MAX_SYMBOL_FRACTION = 0.40

PHASE5_SCORE_COMPONENTS = (
    ("NEGOUTLIER", False),
    ("CRASH", False),
    ("CRASH_count", False),
    ("NCSKEW", False),
    ("DUVOL", False),
    ("SIGMA", False),
    ("MINRET", True),
)

MACD_TEXT = {
    "golden_cross": "金叉",
    "dead_cross": "死叉",
    "above_signal": "信号线上方",
    "below_signal": "信号线下方",
    "bottom_divergence": "底背离",
    "top_divergence": "顶背离",
    "bullish": "量价看多",
    "bearish": "量价看空",
    "none": "无",
}

APP_BG = "#f3f5f7"
CARD_BG = "#ffffff"
TEXT_DARK = "#111827"
TEXT_MUTED = "#667085"
BORDER = "#d8dee6"
ACCENT = "#c24130"
ACCENT_DARK = "#9f2d24"
ACCENT_LIGHT = "#fee2dc"


@dataclass(frozen=True)
class PhaseValue:
    score_100: float | None = None
    raw_score: float | None = None
    rank: int | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PatternValue:
    pattern_ids: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IntradayLookupResult:
    trade_date: str
    quote_datetime: str
    source_label: str
    latest_price: float | None
    pct_change: float | None
    atr14: float | None
    atr_pct: float | None
    max_position_pct: float | None
    macd: dict[str, object]
    phase1: PhaseValue
    phase2: PhaseValue
    phase4: PhaseValue
    phase8: PhaseValue
    source_file: Path


@dataclass(frozen=True)
class PostMarketLookupResult:
    trade_date: str
    close: float | None
    atr14: float | None
    atr_pct: float | None
    max_position_pct: float | None
    macd: dict[str, object]
    phase1: PhaseValue
    phase2: PhaseValue
    phase4: PhaseValue
    phase5: PhaseValue
    phase8: PhaseValue
    pattern: PatternValue
    source_files: dict[str, Path | None]


@dataclass(frozen=True)
class StockLookupResult:
    symbol: str
    name: str
    intraday: IntradayLookupResult | None
    post_market: PostMarketLookupResult


@dataclass(frozen=True)
class StockNameMatch:
    symbol: str
    name: str
    source: str


class StockLookupError(RuntimeError):
    pass


class StockLookupAmbiguousError(StockLookupError):
    def __init__(self, query: str, matches: list[StockNameMatch]) -> None:
        self.query = query
        self.matches = matches
        super().__init__(format_ambiguous_matches(query, matches))


def normalize_symbol(value: object) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    if not digits:
        return ""
    if len(digits) >= 6:
        return digits[-6:]
    return digits.zfill(6)


def find_project_root() -> Path:
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.extend([exe_dir, exe_dir.parent])
    else:
        source_file = Path(__file__).resolve()
        candidates.extend([source_file.parent, source_file.parent.parent])

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "reports" / "atr").is_dir():
            return candidate

    raise StockLookupError("未找到 reports/atr 目录。请把 exe 放在项目主目录下运行。")


def parse_report_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return datetime.min


def report_date_from_path(path: Path, prefix: str) -> str:
    pattern = report_name_pattern(prefix) if prefix else re.compile(r".*?(\d{4}-\d{2}-\d{2})\.csv$")
    match = pattern.match(path.name)
    return match.group(1) if match else ""


def lookup_stock(query: str, project_root: Path | None = None) -> StockLookupResult:
    root = project_root or find_project_root()
    normalized = resolve_stock_query(query, root)
    post_market = lookup_post_market(root, normalized)
    intraday = lookup_intraday(root, normalized)
    name = post_market.phase1.extra.get("name") or ""
    if not name and intraday is not None:
        name = str(intraday.phase1.extra.get("name") or intraday.phase4.extra.get("name") or "")
    if not name:
        atr_row = lookup_row(post_market.source_files.get("ATR"), normalized)
        name = str(first_present(atr_row, ("名称", "name")) or "")
    return StockLookupResult(
        symbol=normalized,
        name=str(name),
        intraday=intraday,
        post_market=post_market,
    )


def resolve_stock_query(query: str, root: Path) -> str:
    text = str(query or "").strip()
    if not text:
        raise StockLookupError("请输入股票代码或名称。")

    symbol = parse_symbol_query(text)
    if symbol:
        return symbol

    matches = find_stock_name_matches(root, text)
    if not matches:
        raise StockLookupError(f"未找到股票名称：{text}")

    exact_matches = [match for match in matches if normalize_name(match.name) == normalize_name(text)]
    if len(exact_matches) == 1:
        return exact_matches[0].symbol
    if len(matches) == 1:
        return matches[0].symbol
    raise StockLookupAmbiguousError(text, exact_matches or matches)


def parse_symbol_query(query: str) -> str:
    text = str(query or "").strip()
    if not text:
        return ""
    if text.startswith('="') and text.endswith('"'):
        text = text[2:-1]
    text = text.strip().strip('"').strip("'")
    if re.fullmatch(r"(?i)(?:sh|sz|bj)?\d{1,6}", text):
        return normalize_symbol(text)
    groups = re.findall(r"(?<!\d)(\d{6})(?!\d)", text)
    if len(groups) == 1:
        return normalize_symbol(groups[0])
    return ""


def find_stock_name_matches(root: Path, query: str) -> list[StockNameMatch]:
    needle = normalize_name(query)
    if not needle:
        return []
    matches: list[StockNameMatch] = []
    seen: set[str] = set()
    for path, source in candidate_name_sources(root):
        if path is None or not path.exists():
            continue
        for row in iter_csv_rows(path):
            symbol = normalize_symbol(first_present(row, ("代码", "symbol", "股票代码")))
            name = str(first_present(row, ("名称", "name", "股票名称")) or "").strip()
            if not symbol or not name or symbol in seen:
                continue
            normalized_name = normalize_name(name)
            if normalized_name == needle or needle in normalized_name:
                seen.add(symbol)
                matches.append(StockNameMatch(symbol=symbol, name=name, source=source))
    return sorted(
        matches,
        key=lambda match: (
            normalize_name(match.name) != needle,
            len(normalize_name(match.name)),
            match.symbol,
        ),
    )


def candidate_name_sources(root: Path) -> list[tuple[Path | None, str]]:
    sources: list[tuple[Path | None, str]] = []
    for directory, prefix, label in (
        (root / "reports" / "atr", "atr", "最新ATR全市场"),
        (root / "reports" / "intraday_screening", "intraday_top20", "最新日中Top20"),
        (root / "reports" / "intraday_screening", "intraday_top20_previous", "上一轮日中Top20"),
        (root / "reports" / "intraday_screening", "intraday_track_stock", "日中跟踪股"),
        (root / "reports" / "intraday_screening", "intraday_screening", "日中全市场"),
        (root / "reports" / "watchlists", "watchlist", "最新watchlist"),
        (root / "reports" / "watchlists", "watchlist_pattern", "最新pattern watchlist"),
        (root / "reports" / "patterns", "patterns_all", "最新pattern结果"),
    ):
        sources.append((find_latest_optional_report_path(directory, prefix), label))
    return sources


def find_latest_optional_report_path(directory: Path, prefix: str) -> Path | None:
    found = find_latest_optional_report(directory, prefix)
    return found[1] if found is not None else None


def iter_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def normalize_name(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip()).lower()


def lookup_post_market(root: Path, symbol: str) -> PostMarketLookupResult:
    trade_date, atr_path = find_latest_report(root / "reports" / "atr", "atr")
    paths = {
        "ATR": atr_path,
        "MACD": root / "reports" / "macd" / f"macd_{trade_date}.csv",
        "Phase1": root / "reports" / "full_market_model" / f"tail_risk_predictions_{trade_date}.csv",
        "Phase2": root / "reports" / "full_market_model" / f"barrier_risk_predictions_{trade_date}.csv",
        "Phase4": root / "reports" / "full_market_model" / f"alpha158_qlib_return_predictions_{trade_date}.csv",
        "Phase5": root / "reports" / "full_market_model" / "mcd_crash_annual_measures.csv",
        "Phase8": root / "reports" / "full_market_model" / f"limit_up_3d_opportunity_predictions_{trade_date}.csv",
        "Pattern": root / "reports" / "patterns" / f"patterns_all_{trade_date}.csv",
    }

    atr = lookup_row(paths["ATR"], symbol)
    if not atr:
        raise StockLookupError(f"最新盘后 ATR 报表中未找到股票代码：{symbol}")

    close = safe_float(first_present(atr, ("收盘价", "close", "price")))
    atr14 = safe_float(first_present(atr, ("ATR14", "atr_14", "atr")))
    atr_pct = parse_atr_percent(
        first_present(atr, ("ATR%", "atr_pct_14", "atr_pct")),
        atr14,
        close,
        value_is_percent="ATR%" in atr,
    )
    max_position_pct = recommended_position_percent(atr_pct) if atr_pct is not None else None

    macd = lookup_row(paths["MACD"], symbol)
    phase1 = lookup_phase_score(paths["Phase1"], symbol, score_column="risk_score", higher_is_better=False)
    phase2 = lookup_phase_score(
        paths["Phase2"],
        symbol,
        score_column="barrier_risk_score",
        higher_is_better=False,
        extra_columns=("is_cusum_event",),
    )
    phase4 = lookup_phase_score(paths["Phase4"], symbol, score_column="return_score", higher_is_better=True)
    phase4_rolling = lookup_phase4_rolling(root, symbol, trade_date)
    if phase4_rolling:
        phase4 = PhaseValue(
            score_100=phase4.score_100,
            raw_score=phase4.raw_score,
            rank=phase4.rank,
            extra={**phase4.extra, **phase4_rolling},
        )
    phase5 = lookup_phase5(paths["Phase5"], symbol, trade_date)
    phase8 = lookup_phase8(paths["Phase8"], symbol)
    pattern = lookup_patterns(paths["Pattern"], symbol)

    return PostMarketLookupResult(
        trade_date=trade_date,
        close=close,
        atr14=atr14,
        atr_pct=atr_pct,
        max_position_pct=max_position_pct,
        macd=macd,
        phase1=phase1,
        phase2=phase2,
        phase4=phase4,
        phase5=phase5,
        phase8=phase8,
        pattern=pattern,
        source_files={key: path if path.exists() else None for key, path in paths.items()},
    )


def lookup_intraday(root: Path, symbol: str) -> IntradayLookupResult | None:
    intraday_dir = root / "reports" / "intraday_screening"
    source_defs = (
        ("intraday_top20", "Top20"),
        ("intraday_top20_previous", "上一轮Top20"),
        ("intraday_track_stock", "跟踪股"),
        ("intraday_screening", "全市场"),
    )
    candidates: list[tuple[datetime, float, int, str, Path, dict[str, str]]] = []
    for priority, (prefix, label) in enumerate(source_defs):
        found = find_latest_optional_report(intraday_dir, prefix)
        if found is None:
            continue
        trade_date, path = found
        row = lookup_row(path, symbol)
        if not row:
            continue
        candidates.append((parse_report_date(trade_date), path.stat().st_mtime, -priority, label, path, row))
    if not candidates:
        return None

    _date_key, _mtime, _priority, label, path, row = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    trade_date = first_present(row, ("intraday_trade_date", "trade_date", "atr_trade_date")) or report_date_from_path(path, "")
    quote_datetime = first_present(row, ("intraday_quote_datetime", "intraday_fetched_at", "intraday_quote_time")) or ""
    latest_price = safe_float(first_present(row, ("atr_close", "latest_price", "close", "price")))
    pct_change_column, pct_change_value = first_present_item(row, ("intraday_pct_change", "涨幅%", "pct_change"))
    pct_change = parse_percent_value(pct_change_value, value_is_percent=pct_change_column in {"intraday_pct_change", "涨幅%"})
    atr14 = safe_float(first_present(row, ("atr_14", "ATR14", "atr")))
    atr_pct = parse_atr_percent(
        first_present(row, ("atr_pct_14", "ATR%", "atr_pct")),
        atr14,
        latest_price,
        value_is_percent="ATR%" in row,
    )
    max_position_pct = safe_float(first_present(row, ("建议总仓位%", "max_position_pct")))
    if max_position_pct is None and atr_pct is not None:
        max_position_pct = recommended_position_percent(atr_pct)

    phase1 = PhaseValue(score_100=safe_float(row.get("phase1_score_100")), rank=safe_int(row.get("phase1_rank")), extra={"name": row.get("name")})
    phase2 = PhaseValue(
        score_100=safe_float(row.get("phase2_score_100")),
        rank=safe_int(row.get("phase2_rank")),
        extra={"is_cusum_event": row.get("phase2_is_cusum_event")},
    )
    phase4 = PhaseValue(
        score_100=safe_float(row.get("phase4_score_100")),
        rank=safe_int(row.get("phase4_rank")),
        extra={
            "name": row.get("name"),
            "phase4_5d_mean": first_present(row, ("phase4_5d_mean",)),
            "phase4_5d_std": first_present(row, ("phase4_5d_std",)),
        },
    )
    phase8 = PhaseValue(
        score_100=safe_float(row.get("phase8_score_100")),
        raw_score=safe_float(row.get("phase8_raw_score")),
        rank=safe_int(row.get("phase8_rank")),
        extra={"today_limit_up_excluded": row.get("today_limit_up_excluded")},
    )
    return IntradayLookupResult(
        trade_date=str(trade_date),
        quote_datetime=str(quote_datetime),
        source_label=label,
        latest_price=latest_price,
        pct_change=pct_change,
        atr14=atr14,
        atr_pct=atr_pct,
        max_position_pct=max_position_pct,
        macd=row,
        phase1=phase1,
        phase2=phase2,
        phase4=phase4,
        phase8=phase8,
        source_file=path,
    )


def find_latest_report(directory: Path, prefix: str) -> tuple[str, Path]:
    pattern = report_name_pattern(prefix)
    files = [path for path in directory.glob(f"{prefix}_*.csv") if pattern.match(path.name)]
    if not files:
        raise StockLookupError(f"未找到 {prefix} 报表：{directory}")

    def sort_key(path: Path) -> tuple[datetime, float]:
        match = pattern.match(path.name)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y-%m-%d"), path.stat().st_mtime
            except ValueError:
                pass
        return datetime.fromtimestamp(path.stat().st_mtime), path.stat().st_mtime

    latest = max(files, key=sort_key)
    match = pattern.match(latest.name)
    if not match:
        raise StockLookupError(f"无法从文件名识别交易日期：{latest.name}")
    return match.group(1), latest


def find_latest_optional_report(directory: Path, prefix: str) -> tuple[str, Path] | None:
    try:
        return find_latest_report(directory, prefix)
    except StockLookupError:
        return None


def report_name_pattern(prefix: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(prefix)}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$")


def lookup_row(path: Path | None, symbol: str) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if normalize_symbol(first_present(row, ("代码", "symbol", "股票代码"))) == symbol:
                return row
    return {}


def lookup_phase_score(
    path: Path | None,
    symbol: str,
    *,
    score_column: str,
    higher_is_better: bool,
    extra_columns: tuple[str, ...] = (),
) -> PhaseValue:
    if path is None or not path.exists():
        return PhaseValue()
    rows: list[tuple[str, float, dict[str, str]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row_symbol = normalize_symbol(row.get("symbol"))
            score = safe_float(row.get(score_column))
            if not row_symbol or score is None:
                continue
            rows.append((row_symbol, score, row))
    if not rows:
        return PhaseValue()

    rows = sorted(rows, key=lambda item: (item[1], item[0]), reverse=higher_is_better)
    deduped: dict[str, tuple[float, dict[str, str], int]] = {}
    for rank, (row_symbol, score, row) in enumerate(rows, start=1):
        if row_symbol not in deduped:
            deduped[row_symbol] = (score, row, rank)
    if symbol not in deduped:
        return PhaseValue()

    raw_score, row, rank = deduped[symbol]
    values = [item[0] for item in deduped.values()]
    score_100 = percentile_score(values, raw_score, higher_is_better=higher_is_better)
    extra = {column: row.get(column) for column in extra_columns if column in row}
    if "name" in row:
        extra["name"] = row.get("name")
    return PhaseValue(score_100=score_100, raw_score=raw_score, rank=rank, extra=extra)


def lookup_phase4_rolling(root: Path, symbol: str, trade_date_text: str, *, window: int = 5) -> dict[str, object]:
    try:
        trade_date_value = date.fromisoformat(trade_date_text)
    except ValueError:
        return {}
    report_dir = root / "reports" / "full_market_model"
    files: list[tuple[date, Path]] = []
    for path in report_dir.glob("alpha158_qlib_return_predictions_*.csv"):
        match = re.match(r"^alpha158_qlib_return_predictions_(\d{4}-\d{2}-\d{2})\.csv$", path.name)
        if not match:
            continue
        try:
            parsed = date.fromisoformat(match.group(1))
        except ValueError:
            continue
        if parsed <= trade_date_value:
            files.append((parsed, path))
    files = sorted(files, key=lambda item: item[0], reverse=True)[: max(int(window), 0)]
    scores: list[float] = []
    for _parsed, path in files:
        score = lookup_phase4_daily_score_100(path, symbol)
        if score is not None:
            scores.append(score)
    if not scores:
        return {}
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    return {
        "phase4_5d_mean": round(mean, 2),
        "phase4_5d_std": round(math.sqrt(variance), 2),
    }


def lookup_phase4_daily_score_100(path: Path, symbol: str) -> float | None:
    rows: list[tuple[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row_symbol = normalize_symbol(row.get("symbol"))
            score = safe_float(row.get("return_score"))
            if not row_symbol or score is None:
                continue
            rows.append((row_symbol, score))
    if not rows:
        return None
    rows = sorted(rows, key=lambda item: (item[1], item[0]), reverse=True)
    deduped: dict[str, float] = {}
    for row_symbol, score in rows:
        deduped.setdefault(row_symbol, score)
    if symbol not in deduped:
        return None
    return percentile_score(list(deduped.values()), deduped[symbol], higher_is_better=True)


def lookup_phase5(path: Path | None, symbol: str, trade_date_text: str) -> PhaseValue:
    if path is None or not path.exists():
        return PhaseValue()
    try:
        trade_year = date.fromisoformat(trade_date_text).year
    except ValueError:
        trade_year = date.today().year

    latest_by_symbol: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row_symbol = normalize_symbol(row.get("symbol"))
            year = safe_float(row.get("year"))
            if not row_symbol or year is None or int(year) > trade_year:
                continue
            current = latest_by_symbol.get(row_symbol)
            current_year = safe_float(current.get("year")) if current else None
            if current is None or current_year is None or int(year) >= int(current_year):
                latest_by_symbol[row_symbol] = row
    if symbol not in latest_by_symbol:
        return PhaseValue()

    target_row = latest_by_symbol[symbol]
    component_scores: list[float] = []
    for column, higher_is_better in PHASE5_SCORE_COMPONENTS:
        values = [safe_float(row.get(column)) for row in latest_by_symbol.values()]
        valid_values = [value for value in values if value is not None]
        target_value = safe_float(target_row.get(column))
        if target_value is None or not valid_values:
            continue
        component_scores.append(percentile_score(valid_values, target_value, higher_is_better=higher_is_better))

    score_100 = round(sum(component_scores) / len(component_scores), 2) if component_scores else None
    extra = {
        "year": target_row.get("year"),
        "NEGOUTLIER": target_row.get("NEGOUTLIER"),
        "CRASH": target_row.get("CRASH"),
        "NCSKEW": target_row.get("NCSKEW"),
        "DUVOL": target_row.get("DUVOL"),
    }
    return PhaseValue(score_100=score_100, extra=extra)


def lookup_phase8(path: Path | None, symbol: str) -> PhaseValue:
    if path is None or not path.exists():
        return PhaseValue()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if normalize_symbol(row.get("symbol")) != symbol:
                continue
            return PhaseValue(
                score_100=safe_float(row.get("phase8_score_100")),
                raw_score=safe_float(row.get("phase8_raw_score")),
                rank=safe_int(row.get("phase8_rank")),
                extra={
                    "today_limit_up_excluded": row.get("today_limit_up_excluded"),
                    "today_high_return_vs_prev_close": row.get("today_high_return_vs_prev_close"),
                    "today_close_return_vs_prev_close": row.get("today_close_return_vs_prev_close"),
                },
            )
    return PhaseValue()


def lookup_patterns(path: Path | None, symbol: str) -> PatternValue:
    if path is None or not path.exists():
        return PatternValue()
    ids: list[str] = []
    reasons: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if normalize_symbol(row.get("symbol")) != symbol:
                continue
            pattern_id = str(row.get("pattern_id") or "").strip()
            if pattern_id and pattern_id not in ids:
                ids.append(pattern_id)
            reason = str(row.get("reason") or "").strip()
            if reason and reason not in reasons:
                reasons.append(reason)
    ids = sorted(ids, key=lambda value: int(value) if value.isdigit() else 99)
    return PatternValue(pattern_ids=ids, reasons=reasons[:3])


def percentile_score(values: list[float], target: float, *, higher_is_better: bool) -> float:
    valid = [value for value in values if math.isfinite(value)]
    if not valid:
        return math.nan
    if len(valid) == 1:
        return 100.0
    if higher_is_better:
        rank = sum(1 for value in valid if value <= target)
    else:
        rank = sum(1 for value in valid if value >= target)
    return round((rank - 1.0) / (len(valid) - 1.0) * 100.0, 2)


def first_present(row: dict[str, str], columns: tuple[str, ...]) -> str | None:
    _column, value = first_present_item(row, columns)
    return value


def first_present_item(row: dict[str, str], columns: tuple[str, ...]) -> tuple[str | None, str | None]:
    for column in columns:
        value = row.get(column)
        if value not in (None, ""):
            return column, value
    return None, None


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("%", "").replace(",", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def safe_int(value: object) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def parse_percent_value(value: object, *, value_is_percent: bool = False) -> float | None:
    number = safe_float(value)
    if number is None:
        return None
    if not value_is_percent and abs(number) <= 1.5:
        return number * 100.0
    return number


def parse_atr_percent(
    value: object,
    atr14: float | None,
    close: float | None,
    *,
    value_is_percent: bool = False,
) -> float | None:
    number = safe_float(value)
    if number is None:
        if atr14 is None or close is None or close <= 0:
            return None
        number = atr14 / close * 100.0
    elif not value_is_percent and 0 < number <= 1.5:
        number *= 100.0
    if number <= 0:
        return None
    return number


def recommended_position_percent(atr_pct: float) -> float:
    atr_ratio = atr_pct / 100.0
    effective_stop_ratio = (
        POSITION_STAGED_EFFECTIVE_RISK_MULT * POSITION_STOP_ATR_MULT * atr_ratio
    )
    position_fraction = POSITION_RISK_FRACTION / effective_stop_ratio
    position_fraction = min(POSITION_MAX_SYMBOL_FRACTION, max(0.0, position_fraction))
    return round(position_fraction * 100.0, 2)


def build_six_dimension_scores(result: StockLookupResult) -> list[tuple[str, float | None]]:
    p1 = latest_phase_score(result, "phase1")
    p2 = latest_phase_score(result, "phase2")
    p4 = latest_phase_score(result, "phase4")
    p5 = result.post_market.phase5.score_100
    p8 = latest_phase_score(result, "phase8")
    phase4_mean = latest_phase4_mean(result)
    position_pct = latest_position_pct(result)
    return [
        ("风险质量", average_scores(p1, p2)),
        ("上涨预期", clamp_score(p4)),
        ("趋势稳定", clamp_score(phase4_mean)),
        ("长期稳健", clamp_score(p5)),
        ("短线爆发", clamp_score(p8)),
        ("仓位空间", position_space_score(position_pct)),
    ]


def latest_phase_score(result: StockLookupResult, phase_name: str) -> float | None:
    if result.intraday is not None:
        value = getattr(result.intraday, phase_name).score_100
        if is_valid_number(value):
            return float(value)
    value = getattr(result.post_market, phase_name).score_100
    return float(value) if is_valid_number(value) else None


def latest_phase4_mean(result: StockLookupResult) -> float | None:
    if result.intraday is not None:
        value = safe_float(result.intraday.phase4.extra.get("phase4_5d_mean"))
        if is_valid_number(value):
            return value
    value = safe_float(result.post_market.phase4.extra.get("phase4_5d_mean"))
    return value if is_valid_number(value) else None


def latest_position_pct(result: StockLookupResult) -> float | None:
    if result.intraday is not None and is_valid_number(result.intraday.max_position_pct):
        return float(result.intraday.max_position_pct)
    if is_valid_number(result.post_market.max_position_pct):
        return float(result.post_market.max_position_pct)
    return None


def average_scores(*values: float | None) -> float | None:
    valid = [float(value) for value in values if is_valid_number(value)]
    if not valid:
        return None
    return clamp_score(sum(valid) / len(valid))


def position_space_score(position_pct: float | None) -> float | None:
    if not is_valid_number(position_pct):
        return None
    return clamp_score(float(position_pct) / (POSITION_MAX_SYMBOL_FRACTION * 100.0) * 100.0)


def clamp_score(value: float | None) -> float | None:
    if not is_valid_number(value):
        return None
    return round(min(100.0, max(0.0, float(value))), 2)


def is_valid_number(value: object) -> bool:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def format_result(result: StockLookupResult) -> str:
    title = f"{result.symbol}"
    if result.name:
        title += f"  {result.name}"

    post = result.post_market
    pattern_text = "未命中"
    if post.pattern.pattern_ids:
        pattern_text = "命中 Pattern " + ",".join(post.pattern.pattern_ids)
    reason_text = ""
    if post.pattern.reasons:
        reason_text = trim_text("；".join(post.pattern.reasons), 120)

    lines = [title]
    if result.intraday is not None:
        intra = result.intraday
        lines.extend(
            [
                "",
                "日中参数",
                f"时间    {format_datetime(intra.trade_date, intra.quote_datetime)}",
                f"来源    {intra.source_label} / {intra.source_file.name}",
                f"价格    {format_number(intra.latest_price, 2)}    涨幅 {format_signed_percent(intra.pct_change)}",
                f"ATR     {format_number(intra.atr14, 4)}    ATR% {format_number(intra.atr_pct, 2)}%    仓位 {format_number(intra.max_position_pct, 2)}%",
                f"Phase   P1 {format_number(intra.phase1.score_100, 1)}   P2 {format_number(intra.phase2.score_100, 1)}   P4 {format_number(intra.phase4.score_100, 1)}   P8 {format_number(intra.phase8.score_100, 1)}",
                f"P4五日   均值 {format_number(safe_float(intra.phase4.extra.get('phase4_5d_mean')), 1)}   波动 {format_number(safe_float(intra.phase4.extra.get('phase4_5d_std')), 1)}",
                f"事件    CUSUM {format_flag(intra.phase2.extra.get('is_cusum_event'))}   P4排名 {format_rank(intra.phase4.rank)}   P8排名 {format_rank(intra.phase8.rank)}",
                f"技术    MACD {translate_macd(intra.macd.get('macd_cross_state'))}   背离 {translate_macd(intra.macd.get('macd_divergence_state'))}   量价 {translate_macd(intra.macd.get('volume_price_divergence_state'))}",
            ]
        )
    else:
        lines.extend(["", "日中参数", "无日中数据"])

    lines.extend(
        [
            "",
            "盘后参数",
            f"日期    {post.trade_date}",
            f"收盘    {format_number(post.close, 2)}",
            f"ATR     {format_number(post.atr14, 4)}    ATR% {format_number(post.atr_pct, 2)}%    仓位 {format_number(post.max_position_pct, 2)}%",
            f"Phase   P1 {format_number(post.phase1.score_100, 1)}   P2 {format_number(post.phase2.score_100, 1)}   P4 {format_number(post.phase4.score_100, 1)}   P5 {format_number(post.phase5.score_100, 1)}   P8 {format_number(post.phase8.score_100, 1)}",
            f"P4五日   均值 {format_number(safe_float(post.phase4.extra.get('phase4_5d_mean')), 1)}   波动 {format_number(safe_float(post.phase4.extra.get('phase4_5d_std')), 1)}",
            f"事件    CUSUM {format_flag(post.phase2.extra.get('is_cusum_event'))}   P4排名 {format_rank(post.phase4.rank)}   P8排名 {format_rank(post.phase8.rank)}",
            f"技术    MACD {translate_macd(post.macd.get('macd_cross_state'))}   背离 {translate_macd(post.macd.get('macd_divergence_state'))}   量价 {translate_macd(post.macd.get('volume_price_divergence_state'))}",
            f"Pattern {pattern_text}",
        ]
    )
    lines.extend(["", "六维评分"])
    for label, score in build_six_dimension_scores(result):
        lines.append(f"{label}：{format_number(score, 1)}")
    if reason_text:
        lines.append(reason_text)
    return "\n".join(lines)


def format_number(value: float | None, digits: int) -> str:
    if value is None or not math.isfinite(value):
        return "无数据"
    return f"{value:.{digits}f}"


def format_datetime(trade_date: str, quote_datetime: str) -> str:
    trade_date = str(trade_date or "").strip()
    quote_datetime = str(quote_datetime or "").strip()
    if not quote_datetime:
        return trade_date or "无数据"
    if trade_date and quote_datetime.startswith(trade_date):
        return quote_datetime
    if trade_date:
        return f"{trade_date} {quote_datetime}"
    return quote_datetime


def format_signed_percent(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "无数据"
    return f"{value:+.2f}%"


def format_rank(value: int | None) -> str:
    return str(value) if value else "无数据"


def format_flag(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"1", "1.0", "true", "yes", "是"}:
        return "是"
    if text in {"0", "0.0", "false", "no", "否"}:
        return "否"
    return "无数据"


def translate_macd(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "无数据"
    return MACD_TEXT.get(text.lower(), text)


def trim_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def format_ambiguous_matches(query: str, matches: list[StockNameMatch]) -> str:
    shown = matches[:20]
    lines = [f"名称“{query}”匹配到多个股票，请输入更完整名称或直接输入代码："]
    for match in shown:
        lines.append(f"- {match.symbol}  {match.name}（{match.source}）")
    if len(matches) > len(shown):
        lines.append(f"... 另有 {len(matches) - len(shown)} 个匹配结果")
    return "\n".join(lines)


def run_cli(symbol: str) -> int:
    try:
        print(format_result(lookup_stock(symbol)))
        return 0
    except StockLookupError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


def run_gui() -> None:
    root = Tk()
    root.title("股票参数查询工具")
    root.resizable(False, False)
    root.configure(bg=APP_BG)

    symbol_value = StringVar()
    shell = Frame(root, bg=APP_BG)
    shell.grid(row=0, column=0, padx=16, pady=14, sticky="nsew")

    Label(
        shell,
        text="股票参数查询工具",
        bg=APP_BG,
        fg=TEXT_DARK,
        font=("Microsoft YaHei UI", 16, "bold"),
    ).grid(row=0, column=0, columnspan=2, sticky="w")
    Label(
        shell,
        text="最新日中结果 + 盘后参数 + 六维评分",
        bg=APP_BG,
        fg=TEXT_MUTED,
        font=("Microsoft YaHei UI", 9),
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 10))

    input_card = Frame(shell, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER)
    input_card.grid(row=2, column=0, columnspan=2, sticky="ew")
    Label(input_card, text="代码/名称", bg=CARD_BG, fg="#334155", font=("Microsoft YaHei UI", 10)).grid(
        row=0,
        column=0,
        padx=(14, 8),
        pady=12,
        sticky="w",
    )
    entry = Entry(
        input_card,
        textvariable=symbol_value,
        width=18,
        relief="flat",
        bg="#f8fafc",
        fg=TEXT_DARK,
        insertbackground=TEXT_DARK,
        font=("Consolas", 12),
    )
    entry.grid(row=0, column=1, padx=(0, 10), pady=12, ipady=5)
    query_button = Button(
        input_card,
        text="查询",
        width=10,
        command=lambda: submit(),
        relief="flat",
        bg=ACCENT,
        fg="#ffffff",
        activebackground=ACCENT_DARK,
        activeforeground="#ffffff",
        font=("Microsoft YaHei UI", 10, "bold"),
    )
    query_button.grid(row=0, column=2, padx=(0, 14), pady=12, ipady=3)

    content = Frame(shell, bg=APP_BG)
    content.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(12, 0))

    result_box = Text(
        content,
        width=58,
        height=27,
        wrap="word",
        relief="flat",
        bg=CARD_BG,
        fg=TEXT_DARK,
        padx=16,
        pady=14,
        font=("Microsoft YaHei UI", 11),
        highlightthickness=1,
        highlightbackground=BORDER,
        spacing1=2,
        spacing3=4,
    )
    result_box.grid(row=0, column=0, sticky="nsew")

    chart = Canvas(
        content,
        width=390,
        height=420,
        bg=CARD_BG,
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    chart.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
    draw_empty_radar(chart)

    result_box.tag_configure("title", foreground=ACCENT_DARK, font=("Microsoft YaHei UI", 15, "bold"), spacing3=8)
    result_box.tag_configure("section", foreground=TEXT_DARK, font=("Microsoft YaHei UI", 12, "bold"), spacing1=8, spacing3=3)
    result_box.tag_configure("muted", foreground=TEXT_MUTED)
    result_box.tag_configure("metric", foreground="#1f2937", font=("Microsoft YaHei UI", 10))
    write_result(result_box, "输入股票代码或名称后点击查询。\n名称支持模糊匹配；如果匹配多个，会列出候选代码。")

    # Keep a reference on shell so nested submit can update both widgets without globals.
    shell._result_box = result_box  # type: ignore[attr-defined]
    shell._chart = chart  # type: ignore[attr-defined]

    def submit() -> None:
        try:
            result = lookup_stock(symbol_value.get())
            draw_radar(chart, result)
            write_result(result_box, format_result(result))
        except StockLookupError as exc:
            draw_empty_radar(chart)
            write_result(result_box, str(exc))
            messagebox.showwarning("查询失败", str(exc), parent=root)

    entry.bind("<Return>", lambda _event: submit())
    entry.focus_set()
    root.mainloop()


def draw_empty_radar(canvas: Canvas) -> None:
    draw_radar_base(canvas, [(label, None) for label, _score in empty_six_dimension_labels()])


def draw_radar(canvas: Canvas, result: StockLookupResult) -> None:
    draw_radar_base(canvas, build_six_dimension_scores(result))


def empty_six_dimension_labels() -> list[tuple[str, None]]:
    return [
        ("风险质量", None),
        ("上涨预期", None),
        ("趋势稳定", None),
        ("长期稳健", None),
        ("短线爆发", None),
        ("仓位空间", None),
    ]


def draw_radar_base(canvas: Canvas, scores: list[tuple[str, float | None]]) -> None:
    canvas.delete("all")
    width = int(canvas["width"])
    height = int(canvas["height"])
    center_x = width // 2
    center_y = height // 2 + 16
    radius = 112
    label_radius = 148
    axis_count = len(scores)
    start_angle = -math.pi / 2

    canvas.create_text(
        20,
        22,
        text="六维参数图（0-100，越高越有利）",
        anchor="w",
        fill=TEXT_DARK,
        font=("Microsoft YaHei UI", 13, "bold"),
    )

    for level in (20, 40, 60, 80, 100):
        points: list[float] = []
        level_radius = radius * level / 100.0
        for index in range(axis_count):
            angle = start_angle + 2 * math.pi * index / axis_count
            points.extend([center_x + level_radius * math.cos(angle), center_y + level_radius * math.sin(angle)])
        canvas.create_polygon(
            points,
            outline="#e2e8f0" if level < 100 else "#cbd5e1",
            fill="",
            width=1,
        )
        if level in {60, 100}:
            canvas.create_text(
                center_x + 6,
                center_y - level_radius,
                text=str(level),
                anchor="w",
                fill="#94a3b8",
                font=("Microsoft YaHei UI", 8),
            )

    polygon_points: list[float] = []
    valid_points = 0
    for index, (label, score) in enumerate(scores):
        angle = start_angle + 2 * math.pi * index / axis_count
        axis_x = center_x + radius * math.cos(angle)
        axis_y = center_y + radius * math.sin(angle)
        canvas.create_line(center_x, center_y, axis_x, axis_y, fill="#d8e0e8", width=1)

        label_x = center_x + label_radius * math.cos(angle)
        label_y = center_y + label_radius * math.sin(angle)
        anchor = "center"
        if math.cos(angle) > 0.35:
            anchor = "w"
        elif math.cos(angle) < -0.35:
            anchor = "e"
        value_text = "--" if score is None else f"{score:.0f}"
        canvas.create_text(
            label_x,
            label_y,
            text=f"{label}\n{value_text}",
            anchor=anchor,
            justify="center",
            fill=TEXT_DARK,
            font=("Microsoft YaHei UI", 9, "bold"),
        )

        if score is None:
            continue
        score_value = min(100.0, max(0.0, float(score)))
        point_radius = radius * score_value / 100.0
        point_x = center_x + point_radius * math.cos(angle)
        point_y = center_y + point_radius * math.sin(angle)
        polygon_points.extend([point_x, point_y])
        valid_points += 1
        canvas.create_oval(
            point_x - 4,
            point_y - 4,
            point_x + 4,
            point_y + 4,
            fill=ACCENT,
            outline=ACCENT_DARK,
            tags=("score_point",),
        )

    if valid_points == axis_count:
        canvas.create_polygon(polygon_points, fill=ACCENT_LIGHT, outline=ACCENT_DARK, width=2)
        canvas.tag_raise("score_point")


def write_result(widget: Text, content: str) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", END)
    for index, line in enumerate(content.splitlines()):
        tag = None
        if index == 0:
            tag = "title"
        elif line in {"日中参数", "盘后参数", "六维评分"}:
            tag = "section"
        elif line.startswith("来源") or line == "无日中数据":
            tag = "muted"
        elif index > 0 and line.strip():
            tag = "metric"
        widget.insert(END, line + "\n", tag)
    widget.configure(state="disabled")


def main() -> int:
    parser = argparse.ArgumentParser(description="查询最新日中与盘后股票参数、Phase分、Pattern、ATR仓位和六维评分。")
    parser.add_argument("--symbol", help="股票代码或名称；不传则启动 GUI。")
    args = parser.parse_args()
    if args.symbol:
        return run_cli(args.symbol)
    run_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
