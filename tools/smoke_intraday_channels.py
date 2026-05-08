from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd


DEFAULT_CHANNELS = [
    "akshare_spot_em",
    "eastmoney_direct",
    "sina_raw",
    "tencent_raw",
    "tushare_realtime_quote",
    "tushare_rt_k",
    "akshare_minute_em_loop",
]

REQUIRED_COLUMNS = ["symbol", "open", "close", "high", "low", "volume", "amount"]


@dataclass(slots=True)
class ChannelResult:
    channel: str
    status: str
    requested_symbols: int
    returned_rows: int = 0
    matched_symbols: int = 0
    valid_ohlcv_rows: int = 0
    meets_target: bool = False
    duration_seconds: float = 0.0
    columns: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    sample_rows: list[dict[str, object]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]

    if args.use_project_proxy:
        apply_project_proxy(project_root / args.config)

    symbols = load_symbols(project_root / args.universe, sample_size=args.sample_size)
    if not symbols:
        raise SystemExit(f"No symbols loaded from {project_root / args.universe}")

    channels = resolve_channels(args.channels)
    results: list[ChannelResult] = []

    print(f"Project root: {project_root}")
    print(f"Universe: {project_root / args.universe}")
    print(f"Sample symbols: {len(symbols)}")
    print(f"Channels: {', '.join(channels)}")
    print()

    runners: dict[str, Callable[[list[str], argparse.Namespace], pd.DataFrame]] = {
        "akshare_spot_em": run_akshare_spot_em,
        "eastmoney_direct": run_eastmoney_direct,
        "sina_raw": run_sina_raw,
        "tencent_raw": run_tencent_raw,
        "tushare_realtime_quote": run_tushare_realtime_quote,
        "tushare_rt_k": run_tushare_rt_k,
        "akshare_minute_em_loop": run_akshare_minute_em_loop,
    }

    for channel in channels:
        runner = runners[channel]
        started = time.perf_counter()
        try:
            frame = runner(symbols, args)
            elapsed = time.perf_counter() - started
            result = summarize_frame(
                channel=channel,
                symbols=symbols,
                frame=frame,
                elapsed=elapsed,
                target_rows=args.target_rows,
            )
        except SkipChannel as exc:
            elapsed = time.perf_counter() - started
            result = ChannelResult(
                channel=channel,
                status="skipped",
                requested_symbols=len(symbols),
                duration_seconds=round(elapsed, 3),
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - smoke script should isolate channel failures.
            elapsed = time.perf_counter() - started
            result = ChannelResult(
                channel=channel,
                status="failed",
                requested_symbols=len(symbols),
                duration_seconds=round(elapsed, 3),
                error=f"{type(exc).__name__}: {exc}",
            )

        results.append(result)
        print_result_line(result)

    report_path = write_report(project_root, args.report, results)
    print()
    print(f"Report written: {report_path}")

    if args.strict and not any(result.meets_target for result in results):
        return 2
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test whether free or optional A-share intraday channels can return "
            "at least 100 provisional daily OHLCV rows during market hours."
        )
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--universe", default="data/universe/main_board.parquet")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--target-rows", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--channels", default="all", help="Comma-separated channel names or 'all'.")
    parser.add_argument("--minute-period", default="5", choices=["1", "5", "15", "30", "60"])
    parser.add_argument("--minute-time-budget", type=float, default=90.0)
    parser.add_argument("--report", default=None, help="JSON report path. Defaults to reports/intraday_channel_smoke_*.json.")
    parser.add_argument("--use-project-proxy", action="store_true", help="Apply network proxy values from config/default.yaml.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if no channel reaches target rows.")
    return parser.parse_args()


def resolve_channels(raw_channels: str) -> list[str]:
    if raw_channels.strip().lower() == "all":
        return DEFAULT_CHANNELS.copy()
    requested = [item.strip() for item in raw_channels.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(DEFAULT_CHANNELS))
    if unknown:
        raise SystemExit(f"Unknown channel(s): {', '.join(unknown)}")
    return requested


def apply_project_proxy(config_path: Path) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise SkipChannel("PyYAML is not installed, cannot read project proxy config") from exc

    if not config_path.exists():
        raise SkipChannel(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    network = raw.get("network") or {}
    mapping = {
        "http_proxy": "HTTP_PROXY",
        "https_proxy": "HTTPS_PROXY",
        "no_proxy": "NO_PROXY",
    }
    for source_key, env_key in mapping.items():
        value = str(network.get(source_key) or "").strip()
        if value and not os.environ.get(env_key):
            os.environ[env_key] = value
            os.environ[env_key.lower()] = value


def load_symbols(universe_path: Path, *, sample_size: int) -> list[str]:
    frame = pd.read_parquet(universe_path)
    if "symbol" not in frame.columns:
        raise RuntimeError(f"Universe file lacks symbol column: {universe_path}")
    symbols = (
        frame["symbol"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return [symbol for symbol in symbols if len(symbol) == 6][:sample_size]


def summarize_frame(
    *,
    channel: str,
    symbols: list[str],
    frame: pd.DataFrame,
    elapsed: float,
    target_rows: int,
) -> ChannelResult:
    normalized = frame.copy()
    if "symbol" in normalized.columns:
        normalized["symbol"] = normalize_symbol_series(normalized["symbol"])

    missing = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    valid_rows = 0
    if not missing:
        numeric = normalized[["open", "close", "high", "low", "volume", "amount"]].apply(pd.to_numeric, errors="coerce")
        valid_rows = int(numeric[["open", "close", "high", "low"]].gt(0).all(axis=1).sum())

    matched = 0
    if "symbol" in normalized.columns:
        matched = int(normalized["symbol"].isin(set(symbols)).sum())

    returned_rows = int(len(normalized))
    meets_target = valid_rows >= target_rows
    result = ChannelResult(
        channel=channel,
        status="ok" if returned_rows else "empty",
        requested_symbols=len(symbols),
        returned_rows=returned_rows,
        matched_symbols=matched,
        valid_ohlcv_rows=valid_rows,
        meets_target=meets_target,
        duration_seconds=round(elapsed, 3),
        columns=[str(column) for column in normalized.columns],
        missing_columns=missing,
        sample_rows=json_safe_records(normalized.head(3)),
    )
    if channel.endswith("_loop"):
        result.notes.append("Loop channel: requests symbols one by one, not a true batch endpoint.")
    return result


def print_result_line(result: ChannelResult) -> None:
    mark = "PASS" if result.meets_target else "FAIL"
    if result.status == "skipped":
        mark = "SKIP"
    elif result.status == "failed":
        mark = "ERR"
    error = f" | {result.error}" if result.error else ""
    print(
        f"{result.channel:<26} {mark:<4} "
        f"rows={result.returned_rows:<5} valid={result.valid_ohlcv_rows:<5} "
        f"matched={result.matched_symbols:<5} time={result.duration_seconds:>7.3f}s{error}"
    )


def write_report(project_root: Path, raw_report: str | None, results: list[ChannelResult]) -> Path:
    if raw_report:
        report_path = Path(raw_report)
        if not report_path.is_absolute():
            report_path = project_root / report_path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = project_root / "reports" / f"intraday_channel_smoke_{stamp}.json"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results": [asdict(result) for result in results],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def run_akshare_spot_em(symbols: list[str], args: argparse.Namespace) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:
        raise SkipChannel("akshare is not installed") from exc

    raw = ak.stock_zh_a_spot_em()
    frame = raw.rename(
        columns={
            "代码": "symbol",
            "名称": "name",
            "今开": "open",
            "最新价": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "昨收": "pre_close",
            "涨跌幅": "pct_change",
        }
    )
    if "symbol" in frame.columns:
        sample_set = set(symbols)
        matched = frame[normalize_symbol_series(frame["symbol"]).isin(sample_set)].copy()
        if len(matched) >= min(len(symbols), args.target_rows):
            return matched
    return frame


def run_eastmoney_direct(symbols: list[str], args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for chunk in chunks(symbols, args.chunk_size):
        secids = ",".join(eastmoney_secid(symbol) for symbol in chunk)
        params = {
            "fltt": "2",
            "invt": "2",
            "fields": "f12,f14,f2,f17,f15,f16,f18,f5,f6,f3,f124",
            "secids": secids,
        }
        url = "https://push2.eastmoney.com/api/qt/ulist.np/get?" + urllib.parse.urlencode(params)
        data = json.loads(fetch_text(url, timeout=args.timeout))
        rows = (data.get("data") or {}).get("diff") or []
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    return raw.rename(
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


def run_sina_raw(symbols: list[str], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for chunk in chunks(symbols, args.chunk_size):
        codes = ",".join(sina_code(symbol) for symbol in chunk)
        url = "https://hq.sinajs.cn/list=" + codes
        text = fetch_text(
            url,
            timeout=args.timeout,
            encoding="gbk",
            headers={"Referer": "https://finance.sina.com.cn/"},
        )
        rows.extend(parse_sina_response(text))
    return pd.DataFrame(rows)


def run_tencent_raw(symbols: list[str], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for chunk in chunks(symbols, args.chunk_size):
        codes = ",".join(tencent_code(symbol) for symbol in chunk)
        url = "https://qt.gtimg.cn/q=" + codes
        text = fetch_text(
            url,
            timeout=args.timeout,
            encoding="gbk",
            headers={"Referer": "https://gu.qq.com/"},
        )
        rows.extend(parse_tencent_response(text))
    return pd.DataFrame(rows)


def run_tushare_realtime_quote(symbols: list[str], args: argparse.Namespace) -> pd.DataFrame:
    try:
        import tushare as ts
    except ImportError as exc:
        raise SkipChannel("tushare is not installed") from exc

    token = os.environ.get("TUSHARE_TOKEN") or os.environ.get("TUSHARE_PRO_TOKEN")
    if token:
        ts.set_token(token)

    frames = []
    for chunk in chunks(symbols, min(args.chunk_size, 50)):
        ts_codes = ",".join(tushare_code(symbol) for symbol in chunk)
        frame = ts.realtime_quote(ts_code=ts_codes)
        if frame is not None and not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    raw.columns = [str(column).lower() for column in raw.columns]
    return raw.rename(
        columns={
            "ts_code": "symbol",
            "price": "close",
            "pre_close": "pre_close",
        }
    )


def run_tushare_rt_k(symbols: list[str], args: argparse.Namespace) -> pd.DataFrame:
    try:
        import tushare as ts
    except ImportError as exc:
        raise SkipChannel("tushare is not installed") from exc

    token = os.environ.get("TUSHARE_TOKEN") or os.environ.get("TUSHARE_PRO_TOKEN")
    if token:
        ts.set_token(token)
    pro = ts.pro_api(token) if token else ts.pro_api()
    ts_codes = ",".join(tushare_code(symbol) for symbol in symbols)
    frame = pro.rt_k(ts_code=ts_codes)
    if frame is None:
        return pd.DataFrame()
    return frame.rename(columns={"ts_code": "symbol"})


def run_akshare_minute_em_loop(symbols: list[str], args: argparse.Namespace) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:
        raise SkipChannel("akshare is not installed") from exc

    today = datetime.now().strftime("%Y-%m-%d")
    start_date = f"{today} 09:30:00"
    end_date = f"{today} 15:00:00"
    rows = []
    started = time.perf_counter()

    for symbol in symbols:
        if time.perf_counter() - started > args.minute_time_budget:
            break
        try:
            raw = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                period=args.minute_period,
                adjust="",
            )
        except Exception as exc:  # noqa: BLE001 - one bad symbol should not stop the loop.
            rows.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
            continue

        if raw is None or raw.empty:
            rows.append({"symbol": symbol, "error": "empty"})
            continue

        normalized = raw.rename(
            columns={
                "时间": "time",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
            }
        ).copy()
        normalized["time"] = pd.to_datetime(normalized["time"], errors="coerce")
        same_day = normalized[normalized["time"].dt.strftime("%Y-%m-%d").eq(today)]
        if same_day.empty:
            rows.append({"symbol": symbol, "error": "no current-day minute rows"})
            continue
        latest = same_day.sort_values("time").iloc[-1].to_dict()
        latest["symbol"] = symbol
        latest["minute_rows"] = int(len(same_day))
        rows.append(latest)

    return pd.DataFrame(rows)


def fetch_text(
    url: str,
    *,
    timeout: float,
    encoding: str = "utf-8",
    headers: dict[str, str] | None = None,
) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 intraday-channel-smoke/1.0",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode(encoding, errors="replace")


def parse_sina_response(text: str) -> list[dict[str, object]]:
    rows = []
    pattern = re.compile(r"var hq_str_(?P<code>[a-z]{2}\d{6})=\"(?P<body>[^\"]*)\";")
    for match in pattern.finditer(text):
        body = match.group("body")
        if not body:
            continue
        fields = body.split(",")
        if len(fields) < 32:
            continue
        rows.append(
            {
                "symbol": match.group("code")[-6:],
                "name": fields[0],
                "open": to_float(fields[1]),
                "pre_close": to_float(fields[2]),
                "close": to_float(fields[3]),
                "high": to_float(fields[4]),
                "low": to_float(fields[5]),
                "bid": to_float(fields[6]),
                "ask": to_float(fields[7]),
                "volume": to_float(fields[8]),
                "amount": to_float(fields[9]),
                "quote_date": fields[30],
                "quote_time": fields[31],
            }
        )
    return rows


def parse_tencent_response(text: str) -> list[dict[str, object]]:
    rows = []
    pattern = re.compile(r'v_(?P<code>[a-z]{2}\d{6})="(?P<body>[^"]*)";')
    for match in pattern.finditer(text):
        fields = match.group("body").split("~")
        if len(fields) < 45:
            continue
        rows.append(
            {
                "symbol": match.group("code")[-6:],
                "name": fields[1],
                "close": to_float(fields[3]),
                "pre_close": to_float(fields[4]),
                "open": to_float(fields[5]),
                "quote_time": fields[30],
                "change": to_float(fields[31]),
                "pct_change": to_float(fields[32]),
                "high": to_float(fields[33]),
                "low": to_float(fields[34]),
                "volume": to_float(fields[36]),
                "amount": to_float(fields[37]),
                "turnover": to_float(fields[38]),
            }
        )
    return rows


def chunks(items: list[str], chunk_size: int) -> list[list[str]]:
    size = max(1, int(chunk_size))
    return [items[index : index + size] for index in range(0, len(items), size)]


def normalize_symbol_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\D", "", regex=True).str[-6:].str.zfill(6)


def sina_code(symbol: str) -> str:
    return f"{market_prefix(symbol)}{symbol}"


def tencent_code(symbol: str) -> str:
    return f"{market_prefix(symbol)}{symbol}"


def eastmoney_secid(symbol: str) -> str:
    market = "1" if market_prefix(symbol) == "sh" else "0"
    return f"{market}.{symbol}"


def tushare_code(symbol: str) -> str:
    suffix = "SH" if market_prefix(symbol) == "sh" else "SZ"
    return f"{symbol}.{suffix}"


def market_prefix(symbol: str) -> str:
    code = str(symbol).zfill(6)
    if code.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return "sh"
    return "sz"


def to_float(value: object) -> float | None:
    text = str(value).strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def json_safe_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records = frame.replace({pd.NA: None}).to_dict(orient="records")
    safe_records: list[dict[str, object]] = []
    for record in records:
        safe_records.append({str(key): json_safe_value(value) for key, value in record.items()})
    return safe_records


def json_safe_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return value


class SkipChannel(RuntimeError):
    pass


if __name__ == "__main__":
    sys.exit(main())
