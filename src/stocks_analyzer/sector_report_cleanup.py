from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re

from .sector_membership import sector_performance_dir


KEEP_LATEST_PATTERNS = (
    re.compile(r"^sector_performance_(\d{4}-\d{2}-\d{2})\.csv$"),
    re.compile(r"^sector_leader_scores_all_(\d{4}-\d{2}-\d{2})\.csv$"),
    re.compile(r"^sector_phase9_buy_score_predictions_(\d{4}-\d{2}-\d{2})\.csv$"),
    re.compile(r"^stock_concern_sectors_(\d{4}-\d{2}-\d{2})\.csv$"),
    re.compile(r"^concern_sector_members_(\d{4}-\d{2}-\d{2})\.csv$"),
)

DROP_DAILY_PATTERNS = (
    re.compile(r"^sector_leaders_(\d{4}-\d{2}-\d{2})\.csv$"),
    re.compile(r"^sector_leaders_summary_(\d{4}-\d{2}-\d{2})\.csv$"),
    re.compile(r"^skipped_sector_leaders_(\d{4}-\d{2}-\d{2})\.csv$"),
    re.compile(r"^sector_pullback_metrics_(\d{4}-\d{2}-\d{2})\.csv$"),
    re.compile(r"^sector_mainline_scores_(\d{4}-\d{2}-\d{2})\.csv$"),
)

ALWAYS_KEEP_FILENAMES = {
    "sector_mainline_daily_tracking.xlsx",
    "sector_mainline_intraday_tracking.xlsx",
}

OBSOLETE_REPORT_PATTERNS = (
    re.compile(r"^watchlist_stocks_\d{4}-\d{2}-\d{2}\.(?:json|csv)$"),
    re.compile(r"^watchlist_mainline_stocks_\d{4}-\d{2}-\d{2}\.(?:json|csv)$"),
    re.compile(r"^intraday_pool_\d{4}-\d{2}-\d{2}\.(?:json|csv)$"),
    re.compile(r"^watchlist_\d{4}-\d{2}-\d{2}\.(?:json|csv)$"),
    re.compile(r"^watchlist_pattern_\d{4}-\d{2}-\d{2}\.(?:json|csv)$"),
    re.compile(r"^intraday_pool_screening_\d{4}-\d{2}-\d{2}\.csv$"),
    re.compile(r"^intraday_track_stock_\d{4}-\d{2}-\d{2}\.csv$"),
    re.compile(r"^intraday_top20(?:_previous)?_\d{4}-\d{2}-\d{2}\.csv$"),
)


@dataclass(frozen=True)
class SectorReportCleanupResult:
    sector_dir: Path
    kept_files: tuple[str, ...]
    deleted_files: tuple[str, ...]


def cleanup_sector_reports(*, project_root: Path, trade_date: date | None = None) -> SectorReportCleanupResult:
    sector_dir = sector_performance_dir(project_root)
    if not sector_dir.exists():
        return SectorReportCleanupResult(sector_dir=sector_dir, kept_files=tuple(), deleted_files=tuple())

    files = [path for path in sector_dir.iterdir() if path.is_file()]
    keep_names = set(ALWAYS_KEEP_FILENAMES)
    keep_names.update(_latest_names_by_pattern(files, trade_date=trade_date))

    deleted: list[str] = []
    kept: list[str] = []
    for path in sorted(files, key=lambda item: item.name):
        if path.name in keep_names:
            kept.append(path.name)
            continue
        if _matches_any(path.name, DROP_DAILY_PATTERNS) or _matches_any(path.name, KEEP_LATEST_PATTERNS):
            path.unlink(missing_ok=True)
            deleted.append(path.name)
    return SectorReportCleanupResult(
        sector_dir=sector_dir,
        kept_files=tuple(sorted(kept)),
        deleted_files=tuple(deleted),
    )


def cleanup_obsolete_daily_reports(*, project_root: Path) -> tuple[Path, ...]:
    roots = (
        project_root / "reports" / "watchlists",
        project_root / "reports" / "intraday_screening",
    )
    deleted: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.iterdir():
            if not path.is_file():
                continue
            if _matches_any(path.name, OBSOLETE_REPORT_PATTERNS):
                path.unlink(missing_ok=True)
                deleted.append(path)
    return tuple(deleted)


def _latest_names_by_pattern(files: list[Path], *, trade_date: date | None) -> set[str]:
    keep_names: set[str] = set()
    preferred_date = trade_date.isoformat() if trade_date is not None else None
    for pattern in KEEP_LATEST_PATTERNS:
        matched: list[tuple[str, str]] = []
        for path in files:
            match = pattern.match(path.name)
            if not match:
                continue
            matched.append((match.group(1), path.name))
        if not matched:
            continue
        if preferred_date:
            preferred = [name for item_date, name in matched if item_date == preferred_date]
            if preferred:
                keep_names.update(preferred)
                continue
        keep_names.add(max(matched, key=lambda item: item[0])[1])
    return keep_names


def _matches_any(name: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.match(name) for pattern in patterns)
