from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .storage import Storage
from .watchlist import WATCHLIST_FILENAME_RE, extract_watchlist_symbols, watchlists_dir


INTRADAY_DATA_INTERFACES = ("eastmoney_direct", "sina_raw")
INTRADAY_REQUIRED_COLUMNS = ("symbol", "open", "close", "high", "low", "volume", "amount")


@dataclass(slots=True)
class IntradayUpdateResult:
    source: str
    requested_symbols: list[str]
    updated_symbols: list[str]
    failed_symbols: list[str]
    target_paths: list[Path]
    source_watchlist_path: Path | None


def run_intraday_update(
    *,
    storage: Storage,
    project_root: Path,
    source: str = "eastmoney_direct",
    symbols: list[str] | None = None,
    limit: int | None = None,
    watchlist_only: bool = False,
    timeout_seconds: float = 15.0,
    chunk_size: int = 50,
) -> IntradayUpdateResult:
    resolved_source = _normalize_source(source)
    source_watchlist_path: Path | None = None

    if symbols:
        requested_symbols = _normalize_symbols(symbols)
    elif watchlist_only:
        source_watchlist_path, requested_symbols = load_latest_watchlist_symbols(project_root)
    else:
        requested_symbols = load_universe_symbols(storage)

    if limit is not None:
        requested_symbols = requested_symbols[:limit]
    if not requested_symbols:
        raise RuntimeError("No symbols selected for intraday update.")

    quote_frame = fetch_intraday_quotes(
        requested_symbols,
        source=resolved_source,
        timeout_seconds=timeout_seconds,
        chunk_size=chunk_size,
    )
    normalized = normalize_intraday_quotes(quote_frame, source=resolved_source)
    if normalized.empty:
        raise RuntimeError(f"No intraday rows returned by {resolved_source}.")

    target_paths: list[Path] = []
    updated_symbols: list[str] = []
    requested_set = set(requested_symbols)

    for symbol, group in normalized.groupby("symbol", sort=False):
        if symbol not in requested_set:
            continue
        latest = group.sort_values(["quote_datetime", "fetched_at"], na_position="first").tail(1).reset_index(drop=True)
        target_paths.append(storage.save_intraday_bars(symbol, latest))
        updated_symbols.append(symbol)

    failed_symbols = [symbol for symbol in requested_symbols if symbol not in set(updated_symbols)]
    return IntradayUpdateResult(
        source=resolved_source,
        requested_symbols=requested_symbols,
        updated_symbols=updated_symbols,
        failed_symbols=failed_symbols,
        target_paths=target_paths,
        source_watchlist_path=source_watchlist_path,
    )


def load_universe_symbols(storage: Storage) -> list[str]:
    universe = storage.load_universe().copy()
    if "symbol" not in universe.columns:
        raise RuntimeError("Universe file lacks symbol column.")
    return _normalize_symbols(universe["symbol"].astype(str).tolist())


def load_latest_watchlist_symbols(project_root: Path) -> tuple[Path, list[str]]:
    candidates: list[tuple[datetime, Path]] = []
    for path in watchlists_dir(project_root).glob("watchlist_*.json"):
        match = WATCHLIST_FILENAME_RE.fullmatch(path.name)
        if not match:
            continue
        parsed = datetime.fromisoformat(match.group(1))
        candidates.append((parsed, path))

    if not candidates:
        raise FileNotFoundError(f"No main watchlist JSON found in {watchlists_dir(project_root)}")

    _, latest_path = max(candidates, key=lambda item: item[0])
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    symbols = extract_watchlist_symbols(payload)
    if not symbols:
        raise RuntimeError(f"Latest watchlist has no candidate symbols: {latest_path}")
    return latest_path, symbols


def fetch_intraday_quotes(
    symbols: list[str],
    *,
    source: str,
    timeout_seconds: float = 15.0,
    chunk_size: int = 50,
) -> pd.DataFrame:
    resolved_source = _normalize_source(source)
    normalized_symbols = _normalize_symbols(symbols)
    if resolved_source == "eastmoney_direct":
        return _fetch_eastmoney_quotes(normalized_symbols, timeout_seconds=timeout_seconds, chunk_size=chunk_size)
    if resolved_source == "sina_raw":
        return _fetch_sina_quotes(normalized_symbols, timeout_seconds=timeout_seconds, chunk_size=chunk_size)
    raise ValueError(f"Unsupported intraday data interface: {source}")


def normalize_intraday_quotes(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = frame.copy()
    missing = [column for column in INTRADAY_REQUIRED_COLUMNS if column not in normalized.columns]
    if missing:
        raise RuntimeError(f"Intraday frame from {source} is missing required columns: {missing}")

    normalized["symbol"] = _normalize_symbol_series(normalized["symbol"])
    for column in ("open", "close", "high", "low", "pre_close", "volume", "amount", "pct_change"):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        else:
            normalized[column] = pd.NA

    if "quote_datetime" not in normalized.columns:
        normalized["quote_datetime"] = pd.NaT
    normalized["quote_datetime"] = pd.to_datetime(normalized["quote_datetime"], errors="coerce")

    quote_dates = normalized["quote_datetime"].dt.date.astype("string")
    if "quote_date" in normalized.columns:
        explicit_dates = pd.to_datetime(normalized["quote_date"], errors="coerce").dt.date.astype("string")
        quote_dates = quote_dates.fillna(explicit_dates)
    normalized["trade_date"] = pd.to_datetime(quote_dates, errors="coerce")

    fallback_date = pd.Timestamp.today().normalize()
    normalized["trade_date"] = normalized["trade_date"].fillna(fallback_date)
    normalized["fetched_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    normalized["source"] = source
    normalized["provisional"] = True

    keep_columns = [
        "trade_date",
        "symbol",
        "name",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "volume",
        "amount",
        "pct_change",
        "quote_datetime",
        "quote_date",
        "quote_time",
        "source",
        "fetched_at",
        "provisional",
    ]
    for column in keep_columns:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    return normalized.loc[:, keep_columns].sort_values("symbol").reset_index(drop=True)


def _fetch_eastmoney_quotes(symbols: list[str], *, timeout_seconds: float, chunk_size: int) -> pd.DataFrame:
    frames = []
    for chunk in _chunks(symbols, chunk_size):
        secids = ",".join(_eastmoney_secid(symbol) for symbol in chunk)
        params = {
            "fltt": "2",
            "invt": "2",
            "fields": "f12,f14,f2,f17,f15,f16,f18,f5,f6,f3,f124",
            "secids": secids,
        }
        url = "https://push2.eastmoney.com/api/qt/ulist.np/get?" + urllib.parse.urlencode(params)
        payload = json.loads(_fetch_text(url, timeout_seconds=timeout_seconds))
        rows = (payload.get("data") or {}).get("diff") or []
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    renamed = raw.rename(
        columns={
            "f12": "symbol",
            "f14": "name",
            "f17": "open",
            "f2": "close",
            "f15": "high",
            "f16": "low",
            "f18": "pre_close",
            "f5": "volume",
            "f6": "amount",
            "f3": "pct_change",
            "f124": "quote_timestamp",
        }
    )
    quote_datetime = (
        pd.to_datetime(pd.to_numeric(renamed.get("quote_timestamp"), errors="coerce"), unit="s", errors="coerce", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    renamed["quote_datetime"] = quote_datetime
    renamed["quote_date"] = quote_datetime.dt.date.astype("string")
    renamed["quote_time"] = quote_datetime.dt.strftime("%H:%M:%S")
    return renamed


def _fetch_sina_quotes(symbols: list[str], *, timeout_seconds: float, chunk_size: int) -> pd.DataFrame:
    rows = []
    for chunk in _chunks(symbols, chunk_size):
        codes = ",".join(_sina_code(symbol) for symbol in chunk)
        url = "https://hq.sinajs.cn/list=" + codes
        text = _fetch_text(
            url,
            timeout_seconds=timeout_seconds,
            encoding="gbk",
            headers={"Referer": "https://finance.sina.com.cn/"},
        )
        rows.extend(_parse_sina_response(text))
    return pd.DataFrame(rows)


def _parse_sina_response(text: str) -> list[dict[str, object]]:
    rows = []
    pattern = re.compile(r"var hq_str_(?P<code>[a-z]{2}\d{6})=\"(?P<body>[^\"]*)\";")
    for match in pattern.finditer(text):
        body = match.group("body")
        if not body:
            continue
        fields = body.split(",")
        if len(fields) < 32:
            continue
        quote_date = fields[30]
        quote_time = fields[31]
        rows.append(
            {
                "symbol": match.group("code")[-6:],
                "name": fields[0],
                "open": _to_float(fields[1]),
                "pre_close": _to_float(fields[2]),
                "close": _to_float(fields[3]),
                "high": _to_float(fields[4]),
                "low": _to_float(fields[5]),
                "volume": _to_float(fields[8]),
                "amount": _to_float(fields[9]),
                "quote_date": quote_date,
                "quote_time": quote_time,
                "quote_datetime": f"{quote_date} {quote_time}",
            }
        )
    return rows


def _fetch_text(
    url: str,
    *,
    timeout_seconds: float,
    encoding: str = "utf-8",
    headers: dict[str, str] | None = None,
) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 stocks-analyzer-intraday/1.0",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return response.read().decode(encoding, errors="replace")


def _normalize_source(source: str) -> str:
    normalized = str(source or "eastmoney_direct").strip().lower()
    aliases = {
        "eastmoney": "eastmoney_direct",
        "em": "eastmoney_direct",
        "sina": "sina_raw",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in INTRADAY_DATA_INTERFACES:
        raise ValueError(f"Unsupported intraday data interface: {source}")
    return normalized


def _normalize_symbols(symbols: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in symbols:
        symbol = re.sub(r"\D", "", str(value))[-6:].zfill(6)
        if symbol == "000000" or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    return normalized


def _normalize_symbol_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\D", "", regex=True).str[-6:].str.zfill(6)


def _chunks(items: list[str], chunk_size: int) -> list[list[str]]:
    size = max(1, int(chunk_size))
    return [items[index : index + size] for index in range(0, len(items), size)]


def _eastmoney_secid(symbol: str) -> str:
    market = "1" if _market_prefix(symbol) == "sh" else "0"
    return f"{market}.{symbol}"


def _sina_code(symbol: str) -> str:
    return f"{_market_prefix(symbol)}{symbol}"


def _market_prefix(symbol: str) -> str:
    code = str(symbol).zfill(6)
    if code.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return "sh"
    return "sz"


def _to_float(value: object) -> float | None:
    text = str(value).strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None
