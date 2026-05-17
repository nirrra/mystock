from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from io import StringIO
from pathlib import Path

import logging
import math
import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

from .phase_display import normalize_symbol

try:
    import akshare as ak
except ImportError as exc:  # pragma: no cover
    ak = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    import py_mini_racer
    from akshare.datasets import get_ths_js
except ImportError:  # pragma: no cover
    py_mini_racer = None
    get_ths_js = None


SECTOR_MEMBERSHIP_COLUMNS = [
    "symbol",
    "name",
    "sector_type",
    "sector_name",
    "sector_label",
    "source",
    "updated_at",
]
SECTOR_DISPLAY_COLUMNS = ["industry_names", "concept_names"]
SECTOR_PERFORMANCE_COLUMNS = [
    "trade_date",
    "sector_type",
    "sector_name",
    "sector_label",
    "member_count",
    "valid_count",
    "avg_pct_change",
    "amount_weighted_pct_change",
    "up_count",
    "up_ratio",
    "total_amount",
]
IGNORED_SECTOR_NAMES = {
    "2025年报预增",
    "2026一季报预增",
    "摘帽",
    "新股与次新股",
    "注册制次新股",
}
IGNORED_SECTOR_KEYWORDS = (
    "同花顺",
    "预增",
    "年报",
    "一季报",
    "中报",
    "半年报",
    "三季报",
    "摘帽",
)
DESCRIPTIVE_SECTOR_KEYWORDS = (
    "公司",
    "主营",
    "研发",
    "生产",
    "销售",
    "产品",
    "业务",
    "专注",
    "行业",
    "体系",
    "设立以来",
    "主要包括",
    "发展成为",
    "报告披露",
)
THS_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)
THS_TIMEOUT_SECONDS = 20
THS_RETRY_COUNT = 4
THS_RETRY_BACKOFF_SECONDS = 0.8
THS_REQUEST_DELAY_SECONDS = 0.12
THS_BASIC_TIMEOUT_SECONDS = 15
THS_BASIC_DELAY_SECONDS = 0.03
SECTOR_MEMBERSHIP_REFRESH_DAYS = 7


@dataclass(frozen=True)
class SectorMembershipUpdateResult:
    output_path: Path
    performance_path: Path | None
    performance_trade_date: date | None
    industry_count: int
    concept_count: int
    row_count: int
    performance_row_count: int
    membership_refreshed: bool
    membership_cache_age_days: float | None


def sector_membership_dir(project_root: Path) -> Path:
    return project_root / "data" / "sector_membership"


def sector_membership_path(project_root: Path) -> Path:
    return sector_membership_dir(project_root) / "stock_sector_membership.csv"


def sector_performance_dir(project_root: Path) -> Path:
    return project_root / "reports" / "sectors"


def sector_performance_path(project_root: Path, trade_date: date) -> Path:
    return sector_performance_dir(project_root) / f"sector_performance_{trade_date.isoformat()}.csv"


def update_sector_membership(
    *,
    project_root: Path,
    trade_date: date | None = None,
    daily_dir: Path | None = None,
    max_cache_age_days: int = SECTOR_MEMBERSHIP_REFRESH_DAYS,
    force_refresh: bool = False,
) -> SectorMembershipUpdateResult:
    target = sector_membership_path(project_root)
    previous = load_sector_membership(project_root=project_root)
    cache_age_days = _sector_membership_cache_age_days(target)
    use_cache = (
        not force_refresh
        and target.exists()
        and not previous.empty
        and cache_age_days is not None
        and cache_age_days <= max_cache_age_days
    )
    updated_at = datetime.now().isoformat(timespec="seconds")

    membership_refreshed = False
    if use_cache:
        logging.info(
            "Using cached sector membership: %s, age %.2f days; daily sector performance will still be rebuilt.",
            target,
            cache_age_days,
        )
        frame = previous.copy()
    else:
        if force_refresh:
            logging.info("Refreshing sector membership because --force-refresh was provided.")
        elif not target.exists() or previous.empty:
            logging.info("Refreshing sector membership because local cache is missing or empty.")
        else:
            logging.info(
                "Refreshing sector membership because cache age %.2f days exceeds %s days.",
                cache_age_days if cache_age_days is not None else math.nan,
                max_cache_age_days,
            )
        profile_rows = _fetch_ths_stock_profile_rows(project_root=project_root, updated_at=updated_at)
        industry_rows = [row for row in profile_rows if row.get("sector_type") == "industry"]
        concept_rows = [row for row in profile_rows if row.get("sector_type") == "concept"]
        if not industry_rows and not concept_rows:
            logging.warning("THS stock profile pages returned no rows; fallback to THS board detail pages.")
            industry_rows = _fetch_ths_sector_rows(
                sector_type="industry",
                sector_frame=_fetch_ths_sector_list(url_root="thshy", seed_code="881272"),
                url_root="thshy",
                source="ths_industry",
                updated_at=updated_at,
            )
            concept_rows = _fetch_ths_sector_rows(
                sector_type="concept",
                sector_frame=_fetch_ths_sector_list(url_root="gn", seed_code="309130", include_summary=True),
                url_root="gn",
                source="ths_concept",
                updated_at=updated_at,
            )
        rows = [*industry_rows, *concept_rows]
        membership_refreshed = bool(industry_rows or concept_rows)
        if not industry_rows and not previous.empty:
            logging.warning("No industry membership rows fetched; preserving existing industry cache.")
            rows.extend(previous.loc[previous["sector_type"].eq("industry")].to_dict("records"))
        if not concept_rows and not previous.empty:
            logging.warning("No concept membership rows fetched; preserving existing concept cache.")
            rows.extend(previous.loc[previous["sector_type"].eq("concept")].to_dict("records"))
        frame = pd.DataFrame(rows, columns=SECTOR_MEMBERSHIP_COLUMNS)
    if not frame.empty:
        frame = frame.drop_duplicates(["symbol", "sector_type", "sector_label"], keep="first")
        frame = frame.sort_values(["symbol", "sector_type", "sector_name"]).reset_index(drop=True)
    filtered_frame = filter_ignored_sector_membership(frame)

    if membership_refreshed or (not target.exists() and not frame.empty):
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False, encoding="utf-8-sig")
    performance_frame, resolved_trade_date = build_sector_performance(
        project_root=project_root,
        membership=filtered_frame,
        trade_date=trade_date,
        daily_dir=daily_dir,
    )
    performance_target: Path | None = None
    if resolved_trade_date is not None:
        performance_target = sector_performance_path(project_root, resolved_trade_date)
        performance_target.parent.mkdir(parents=True, exist_ok=True)
        performance_frame.to_csv(performance_target, index=False, encoding="utf-8-sig")
    return SectorMembershipUpdateResult(
        output_path=target,
        performance_path=performance_target,
        performance_trade_date=resolved_trade_date,
        industry_count=filtered_frame.loc[filtered_frame["sector_type"].eq("industry"), "sector_label"].nunique() if not filtered_frame.empty else 0,
        concept_count=filtered_frame.loc[filtered_frame["sector_type"].eq("concept"), "sector_label"].nunique() if not filtered_frame.empty else 0,
        row_count=len(filtered_frame),
        performance_row_count=len(performance_frame),
        membership_refreshed=membership_refreshed,
        membership_cache_age_days=cache_age_days,
    )


def _sector_membership_cache_age_days(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        modified_at = datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None
    return max(0.0, (datetime.now() - modified_at).total_seconds() / 86400.0)


def append_sector_display_columns(frame: pd.DataFrame, *, project_root: Path) -> pd.DataFrame:
    result = frame.copy()
    for column in SECTOR_DISPLAY_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    if result.empty or "symbol" not in result.columns:
        return result

    display = build_sector_display_frame(project_root=project_root)
    if display.empty:
        return result

    result["_sector_symbol"] = result["symbol"].map(normalize_symbol)
    result = result.drop(columns=[column for column in SECTOR_DISPLAY_COLUMNS if column in result.columns], errors="ignore")
    result = result.merge(display.rename(columns={"symbol": "_sector_symbol"}), on="_sector_symbol", how="left")
    result = result.drop(columns=["_sector_symbol"], errors="ignore")
    for column in SECTOR_DISPLAY_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result


def build_sector_display_frame(*, project_root: Path) -> pd.DataFrame:
    frame = load_sector_membership(project_root=project_root)
    if frame.empty:
        return pd.DataFrame(columns=["symbol", *SECTOR_DISPLAY_COLUMNS])
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].map(normalize_symbol)
    frame = frame[frame["symbol"].astype(str).str.len().eq(6)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["symbol", *SECTOR_DISPLAY_COLUMNS])

    rows: list[dict[str, object]] = []
    for symbol, group in frame.groupby("symbol", sort=False):
        industries = _join_unique(group.loc[group["sector_type"].eq("industry"), "sector_name"])
        concepts = _join_unique(group.loc[group["sector_type"].eq("concept"), "sector_name"])
        rows.append({"symbol": symbol, "industry_names": industries, "concept_names": concepts})
    return pd.DataFrame(rows, columns=["symbol", *SECTOR_DISPLAY_COLUMNS])


def load_sector_membership(*, project_root: Path) -> pd.DataFrame:
    return load_raw_sector_membership(project_root=project_root, apply_ignore_rules=True)


def load_raw_sector_membership(*, project_root: Path, apply_ignore_rules: bool = False) -> pd.DataFrame:
    path = sector_membership_path(project_root)
    if not path.exists():
        return pd.DataFrame(columns=SECTOR_MEMBERSHIP_COLUMNS)
    frame = pd.read_csv(path, dtype={"symbol": str})
    for column in SECTOR_MEMBERSHIP_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame.loc[:, SECTOR_MEMBERSHIP_COLUMNS].copy()
    if apply_ignore_rules:
        frame = filter_ignored_sector_membership(frame)
    return frame


def filter_ignored_sector_membership(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "sector_name" not in frame.columns:
        return frame.copy()
    result = frame.copy()
    names = result["sector_name"].fillna("").astype(str).str.strip()
    ignored = names.isin(IGNORED_SECTOR_NAMES)
    for keyword in IGNORED_SECTOR_KEYWORDS:
        ignored |= names.str.contains(re.escape(keyword), regex=True, na=False)
    ignored |= names.map(_is_descriptive_sector_name)
    return result.loc[~ignored].reset_index(drop=True)


def _is_descriptive_sector_name(name: str) -> bool:
    text = str(name or "").strip()
    if len(text) < 24:
        return False
    keyword_hits = sum(1 for keyword in DESCRIPTIVE_SECTOR_KEYWORDS if keyword in text)
    if "公司" in text and keyword_hits >= 2:
        return True
    if len(text) >= 36 and keyword_hits >= 3:
        return True
    if re.search(r"\d{4}年.*报告.*披露", text):
        return True
    return False


def build_sector_performance(
    *,
    project_root: Path,
    membership: pd.DataFrame | None = None,
    trade_date: date | None = None,
    daily_dir: Path | None = None,
) -> tuple[pd.DataFrame, date | None]:
    members = membership.copy() if membership is not None else load_sector_membership(project_root=project_root)
    if members.empty:
        return pd.DataFrame(columns=SECTOR_PERFORMANCE_COLUMNS), trade_date
    for column in SECTOR_MEMBERSHIP_COLUMNS:
        if column not in members.columns:
            members[column] = pd.NA
    members = members.loc[:, SECTOR_MEMBERSHIP_COLUMNS].copy()
    members["symbol"] = members["symbol"].map(normalize_symbol)
    members = members[members["symbol"].astype(str).str.len().eq(6)].copy()
    if members.empty:
        return pd.DataFrame(columns=SECTOR_PERFORMANCE_COLUMNS), trade_date

    daily_root = daily_dir if daily_dir is not None else project_root / "data" / "daily"
    resolved_trade_date = trade_date or _latest_daily_trade_date(daily_root=daily_root, symbols=members["symbol"].unique())
    if resolved_trade_date is None:
        return pd.DataFrame(columns=SECTOR_PERFORMANCE_COLUMNS), None

    daily_rows = _load_daily_rows(daily_root=daily_root, symbols=members["symbol"].unique(), trade_date=resolved_trade_date)
    rows: list[dict[str, object]] = []
    for (sector_type, sector_label, sector_name), group in members.groupby(
        ["sector_type", "sector_label", "sector_name"],
        dropna=False,
        sort=True,
    ):
        symbols = sorted({symbol for symbol in group["symbol"].astype(str) if symbol})
        valid = daily_rows[daily_rows["symbol"].isin(symbols)].copy()
        valid_count = len(valid)
        up_count = int((valid["pct_change"] > 0).sum()) if valid_count else 0
        total_amount = float(valid["amount"].sum()) if valid_count else 0.0
        avg_pct_change = float(valid["pct_change"].mean()) if valid_count else math.nan
        weighted = math.nan
        if valid_count and total_amount > 0:
            weighted = float((valid["pct_change"] * valid["amount"]).sum() / total_amount)
        rows.append(
            {
                "trade_date": resolved_trade_date.isoformat(),
                "sector_type": sector_type,
                "sector_name": sector_name,
                "sector_label": sector_label,
                "member_count": len(symbols),
                "valid_count": valid_count,
                "avg_pct_change": round(avg_pct_change, 6) if math.isfinite(avg_pct_change) else pd.NA,
                "amount_weighted_pct_change": round(weighted, 6) if math.isfinite(weighted) else pd.NA,
                "up_count": up_count,
                "up_ratio": round(up_count / valid_count, 6) if valid_count else pd.NA,
                "total_amount": round(total_amount, 2) if valid_count else pd.NA,
            }
        )
    frame = pd.DataFrame(rows, columns=SECTOR_PERFORMANCE_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["sector_type", "amount_weighted_pct_change"], ascending=[True, False], na_position="last")
        frame = frame.reset_index(drop=True)
    return frame, resolved_trade_date


def _fetch_ths_sector_rows(
    *,
    sector_type: str,
    sector_frame: pd.DataFrame,
    url_root: str,
    source: str,
    updated_at: str,
) -> list[dict[str, object]]:
    if sector_frame.empty or "code" not in sector_frame.columns or "name" not in sector_frame.columns:
        return []
    rows: list[dict[str, object]] = []
    sectors = sector_frame.to_dict("records")
    for index, sector in enumerate(sectors, start=1):
        label = str(sector.get("code", "") or "").strip()
        name = str(sector.get("name", "") or "").strip()
        if not label or not name:
            continue
        try:
            members = _fetch_ths_sector_members(url_root=url_root, sector_label=label)
        except Exception as exc:
            logging.warning("Failed to fetch sector detail %s %s: %s", source, label, exc)
            continue
        if members.empty or "代码" not in members.columns:
            continue
        member_name_column = "名称" if "名称" in members.columns else None
        for member in members.to_dict("records"):
            symbol = normalize_symbol(member.get("代码", ""))
            if not symbol:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "name": str(member.get(member_name_column, "") or "") if member_name_column else "",
                    "sector_type": sector_type,
                    "sector_name": name,
                    "sector_label": label,
                    "source": source,
                    "updated_at": updated_at,
                }
            )
        if index == 1 or index % 50 == 0 or index == len(sectors):
            logging.info("THS %s membership progress: %s/%s", sector_type, index, len(sectors))
    return rows


def _fetch_ths_stock_profile_rows(*, project_root: Path, updated_at: str) -> list[dict[str, object]]:
    universe = _load_stock_universe(project_root=project_root)
    if universe.empty:
        logging.warning("No local universe found for THS stock profile membership fallback.")
        return []
    rows: list[dict[str, object]] = []
    records = universe.to_dict("records")
    for index, stock in enumerate(records, start=1):
        symbol = normalize_symbol(stock.get("symbol", ""))
        if not symbol:
            continue
        name = str(stock.get("name", "") or "")
        try:
            company_html = _ths_basic_get(symbol=symbol, page="company.html")
            industry = _parse_ths_stock_industry(company_html)
            if industry:
                rows.append(
                    {
                        "symbol": symbol,
                        "name": name,
                        "sector_type": "industry",
                        "sector_name": industry,
                        "sector_label": industry,
                        "source": "ths_stock_profile",
                        "updated_at": updated_at,
                    }
                )
        except Exception as exc:
            logging.debug("Failed to fetch THS company profile %s: %s", symbol, exc)
        try:
            concept_html = _ths_basic_get(symbol=symbol, page="concept.html")
            for concept_name, concept_label in _parse_ths_stock_concepts(concept_html):
                rows.append(
                    {
                        "symbol": symbol,
                        "name": name,
                        "sector_type": "concept",
                        "sector_name": concept_name,
                        "sector_label": concept_label,
                        "source": "ths_stock_profile",
                        "updated_at": updated_at,
                    }
                )
        except Exception as exc:
            logging.debug("Failed to fetch THS concept profile %s: %s", symbol, exc)
        if index == 1 or index % 100 == 0 or index == len(records):
            logging.info("THS stock profile membership progress: %s/%s", index, len(records))
    return rows


def _load_stock_universe(*, project_root: Path) -> pd.DataFrame:
    universe_path = project_root / "data" / "universe" / "main_board.parquet"
    if universe_path.exists():
        try:
            frame = pd.read_parquet(universe_path)
        except Exception as exc:
            logging.warning("Failed to read universe file %s: %s", universe_path, exc)
            frame = pd.DataFrame()
    else:
        frame = pd.DataFrame()
    if frame.empty:
        daily_root = project_root / "data" / "daily"
        rows = [{"symbol": path.stem, "name": ""} for path in sorted(daily_root.glob("*.parquet"))]
        frame = pd.DataFrame(rows)
    if frame.empty or "symbol" not in frame.columns:
        return pd.DataFrame(columns=["symbol", "name"])
    result = frame.copy()
    result["symbol"] = result["symbol"].map(normalize_symbol)
    result = result[result["symbol"].astype(str).str.len().eq(6)].copy()
    if "name" not in result.columns:
        result["name"] = ""
    return result.loc[:, ["symbol", "name"]].drop_duplicates("symbol").reset_index(drop=True)


def _ths_basic_get(*, symbol: str, page: str) -> str:
    time.sleep(THS_BASIC_DELAY_SECONDS)
    url = f"https://basic.10jqka.com.cn/{symbol}/{page}"
    headers = {
        "User-Agent": THS_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": f"https://basic.10jqka.com.cn/{symbol}/",
    }
    last_error: Exception | None = None
    for trust_env in (False, True):
        try:
            session = _ths_basic_direct_session() if not trust_env else requests
            response = session.get(url, headers=headers, timeout=THS_BASIC_TIMEOUT_SECONDS)
            response.raise_for_status()
            response.encoding = "gbk"
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch THS basic page: {url}")


@lru_cache(maxsize=1)
def _ths_basic_direct_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def _parse_ths_stock_industry(html: str) -> str:
    match = re.search(r"所属申万行业：</strong><span>(.*?)</span>", html, flags=re.S)
    if match is None:
        match = re.search(r"所属行业：</span>\s*<span[^>]*>(.*?)</span>", html, flags=re.S)
    if match is None:
        return ""
    text = BeautifulSoup(match.group(1), features="lxml").get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" — ", "-").replace("—", "-")
    return text


def _parse_ths_stock_concepts(html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, features="lxml")
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for cell in soup.find_all(name="td", attrs={"class": re.compile(r"\bgnName\b")}):
        name = cell.get_text(" ", strip=True)
        if not name or name in seen:
            continue
        label = str(cell.get("clid") or name).strip() or name
        result.append((name, label))
        seen.add(name)
    if result:
        return result
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return result
    for table in tables:
        if "概念名称" not in table.columns:
            continue
        for value in table["概念名称"].dropna().astype(str):
            name = value.strip()
            if not name or name in seen:
                continue
            result.append((name, name))
            seen.add(name)
    return result


def _fetch_ths_sector_list(*, url_root: str, seed_code: str, include_summary: bool = False) -> pd.DataFrame:
    url = f"https://q.10jqka.com.cn/{url_root}/detail/code/{seed_code}/"
    try:
        html = _ths_get(url)
    except Exception as exc:
        logging.warning("Failed to fetch THS %s sector category page: %s", url_root, exc)
        return pd.DataFrame(columns=["name", "code"])
    sectors = _parse_ths_sector_links(html, url_root=url_root)
    if include_summary:
        sectors.update(_fetch_ths_concept_summary_links())
    rows = [{"name": name, "code": code} for name, code in sorted(sectors.items(), key=lambda item: item[0])]
    return pd.DataFrame(rows, columns=["name", "code"])


def _parse_ths_sector_links(html: str, *, url_root: str) -> dict[str, str]:
    soup = BeautifulSoup(html, features="lxml")
    result: dict[str, str] = {}
    for link in soup.find_all(name="a"):
        href = str(link.get("href") or "")
        if f"/{url_root}/detail/code/" not in href:
            continue
        code = href.rstrip("/").split("/")[-1].strip()
        name = link.get_text(strip=True)
        if name and code:
            result[name] = code
    return result


def _fetch_ths_concept_summary_links() -> dict[str, str]:
    result: dict[str, str] = {}
    first_url = "https://q.10jqka.com.cn/gn/index/field/addtime/order/desc/page/1/ajax/1/"
    try:
        first_html = _ths_get(first_url)
    except Exception as exc:
        logging.warning("Failed to fetch THS concept summary page: %s", exc)
        return result
    page_count = _extract_ths_page_count(first_html)
    result.update(_parse_ths_sector_links(first_html, url_root="gn"))
    for page in range(2, page_count + 1):
        page_url = f"https://q.10jqka.com.cn/gn/index/field/addtime/order/desc/page/{page}/ajax/1/"
        try:
            html = _ths_get(page_url, referer=first_url)
        except Exception as exc:
            logging.warning("Failed to fetch THS concept summary page %s: %s", page, exc)
            break
        page_links = _parse_ths_sector_links(html, url_root="gn")
        if not page_links:
            break
        result.update(page_links)
    return result


def _fetch_ths_sector_members(*, url_root: str, sector_label: str) -> pd.DataFrame:
    base_url = f"https://q.10jqka.com.cn/{url_root}/detail/code/{sector_label}/"
    base_text = _ths_get(base_url)
    frames = _parse_ths_member_tables(base_text)
    page_count = _extract_ths_page_count(base_text)
    for page in range(2, page_count + 1):
        page_url = f"https://q.10jqka.com.cn/{url_root}/detail/page/{page}/ajax/1/code/{sector_label}/"
        text = _ths_get(page_url, referer=base_url)
        page_frames = _parse_ths_member_tables(text)
        if not page_frames:
            break
        frames.extend(page_frames)
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    if "代码" not in frame.columns:
        return pd.DataFrame()
    frame["代码"] = frame["代码"].map(normalize_symbol)
    frame = frame[frame["代码"].astype(str).str.len().eq(6)].copy()
    if "名称" not in frame.columns:
        frame["名称"] = ""
    return frame.drop_duplicates("代码", keep="first").reset_index(drop=True)


def _parse_ths_member_tables(html: str) -> list[pd.DataFrame]:
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return []
    result: list[pd.DataFrame] = []
    for table in tables:
        if "代码" in table.columns and "名称" in table.columns:
            result.append(table)
    return result


def _extract_ths_page_count(html: str) -> int:
    soup = BeautifulSoup(html, features="lxml")
    page_info = soup.find(name="span", attrs={"class": "page_info"})
    if page_info is None:
        return 1
    text = page_info.get_text(strip=True)
    if "/" not in text:
        return 1
    try:
        return max(1, int(text.split("/")[-1]))
    except ValueError:
        return 1


def _ths_get(url: str, *, referer: str | None = None) -> str:
    last_error: Exception | None = None
    for attempt in range(THS_RETRY_COUNT):
        if attempt > 0:
            _ths_v_code.cache_clear()
            time.sleep(THS_RETRY_BACKOFF_SECONDS * attempt)
        else:
            time.sleep(THS_REQUEST_DELAY_SECONDS)
        for candidate_url in _ths_url_variants(url):
            headers = _ths_headers(referer=referer)
            for trust_env in (False, True):
                try:
                    response = _ths_request(candidate_url, headers=headers, trust_env=trust_env)
                except requests.RequestException as exc:
                    last_error = exc
                    continue
                if response.status_code in {401, 403}:
                    last_error = requests.HTTPError(f"{response.status_code} Client Error for url: {candidate_url}", response=response)
                    continue
                try:
                    response.raise_for_status()
                except requests.RequestException as exc:
                    last_error = exc
                    continue
                return response.text
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch THS URL: {url}")


def _ths_request(url: str, *, headers: dict[str, str], trust_env: bool) -> requests.Response:
    if trust_env:
        return requests.get(url, headers=headers, timeout=THS_TIMEOUT_SECONDS)
    session = _ths_direct_session()
    return session.get(url, headers=headers, timeout=THS_TIMEOUT_SECONDS)


@lru_cache(maxsize=1)
def _ths_direct_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def _ths_url_variants(url: str) -> list[str]:
    if url.startswith("https://"):
        return [url, "http://" + url[len("https://") :]]
    if url.startswith("http://"):
        return [url, "https://" + url[len("http://") :]]
    return [url]


def _ths_headers(*, referer: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": THS_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    if referer:
        headers["Referer"] = referer
    try:
        v_code = _ths_v_code()
    except Exception as exc:
        logging.warning("Failed to build THS anti-scraping cookie: %s", exc)
        v_code = ""
    if v_code:
        headers["Cookie"] = f"v={v_code}"
        headers["hexin-v"] = v_code
    return headers


@lru_cache(maxsize=1)
def _ths_v_code() -> str:
    if py_mini_racer is None or get_ths_js is None:
        return ""
    js_code = py_mini_racer.MiniRacer()
    js_path = get_ths_js("ths.js")
    with open(js_path, encoding="utf-8") as handle:
        js_code.eval(handle.read())
    return str(js_code.call("v") or "")


def _latest_daily_trade_date(*, daily_root: Path, symbols) -> date | None:
    latest: pd.Timestamp | None = None
    for symbol in symbols:
        frame = _read_daily_bars_for_symbol(daily_root=daily_root, symbol=str(symbol))
        if frame.empty or "trade_date" not in frame.columns:
            continue
        dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna()
        if dates.empty:
            continue
        symbol_latest = dates.max()
        if latest is None or symbol_latest > latest:
            latest = symbol_latest
    return latest.date() if latest is not None else None


def _load_daily_rows(*, daily_root: Path, symbols, trade_date: date) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        normalized = normalize_symbol(symbol)
        if not normalized:
            continue
        frame = _read_daily_bars_for_symbol(daily_root=daily_root, symbol=normalized)
        if frame.empty or "trade_date" not in frame.columns:
            continue
        data = frame.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.date
        data = data[data["trade_date"].eq(trade_date)].copy()
        if data.empty:
            continue
        row = data.iloc[-1]
        pct_change = _row_pct_change(frame, row)
        amount = pd.to_numeric(pd.Series([row.get("amount", pd.NA)]), errors="coerce").iloc[0]
        if pd.isna(pct_change) or pd.isna(amount):
            continue
        rows.append(
            {
                "symbol": normalized,
                "pct_change": float(pct_change),
                "amount": max(float(amount), 0.0),
            }
        )
    return pd.DataFrame(rows, columns=["symbol", "pct_change", "amount"])


def _read_daily_bars_for_symbol(*, daily_root: Path, symbol: str) -> pd.DataFrame:
    target = daily_root / f"{normalize_symbol(symbol)}.parquet"
    if not target.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(target)
    except Exception as exc:
        logging.warning("Failed to read daily bars for %s: %s", symbol, exc)
        return pd.DataFrame()


def _row_pct_change(frame: pd.DataFrame, row: pd.Series) -> object:
    if "pct_change" in row.index:
        value = pd.to_numeric(pd.Series([row.get("pct_change")]), errors="coerce").iloc[0]
        if pd.notna(value):
            return float(value)
    if "close" not in frame.columns:
        return pd.NA
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    current_date = pd.to_datetime(row.get("trade_date"), errors="coerce")
    if pd.isna(current_date):
        return pd.NA
    current_index = data.index[data["trade_date"].dt.date.eq(current_date.date())]
    if len(current_index) == 0 or int(current_index[-1]) == 0:
        return pd.NA
    index = int(current_index[-1])
    close = pd.to_numeric(pd.Series([data.loc[index, "close"]]), errors="coerce").iloc[0]
    prev_close = pd.to_numeric(pd.Series([data.loc[index - 1, "close"]]), errors="coerce").iloc[0]
    if pd.isna(close) or pd.isna(prev_close) or float(prev_close) == 0:
        return pd.NA
    return float((float(close) / float(prev_close) - 1.0) * 100.0)


def _join_unique(values: pd.Series) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for value in values.dropna().astype(str):
        text = value.strip()
        if not text or text in seen:
            continue
        result.append(text)
        seen.add(text)
    return "/".join(result)
