from __future__ import annotations

import pytest

import stocks_analyzer.data_sources.baostock_provider as baostock_provider


class FakeLoginResult:
    def __init__(self, error_code: str, error_msg: str) -> None:
        self.error_code = error_code
        self.error_msg = error_msg


class FakeQueryResult:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.error_code = "0"
        self.error_msg = "success"
        self.fields = ["calendar_date", "is_trading_day"]
        self._rows = rows
        self._index = -1

    def next(self) -> bool:
        self._index += 1
        return self._index < len(self._rows)

    def get_row_data(self) -> list[str]:
        row = self._rows[self._index]
        return [row["calendar_date"], row["is_trading_day"]]


class FakeBaoStock:
    def __init__(self, login_results: list[FakeLoginResult], trade_day_flag: str = "1") -> None:
        self._login_results = login_results
        self._trade_day_flag = trade_day_flag
        self.login_calls = 0
        self.logout_calls = 0
        self.query_calls = 0

    def login(self) -> FakeLoginResult:
        result = self._login_results[min(self.login_calls, len(self._login_results) - 1)]
        self.login_calls += 1
        return result

    def logout(self) -> None:
        self.logout_calls += 1

    def query_trade_dates(self, *, start_date: str, end_date: str) -> FakeQueryResult:
        self.query_calls += 1
        return FakeQueryResult(
            [
                {
                    "calendar_date": start_date,
                    "is_trading_day": self._trade_day_flag,
                }
            ]
        )


class ExceptionalBaoStock:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.login_calls = 0

    def login(self) -> FakeLoginResult:
        self.login_calls += 1
        if self.login_calls <= self.failures:
            raise OSError("socket reset")
        return FakeLoginResult("0", "success")


def test_login_baostock_retries_until_success(monkeypatch) -> None:
    fake_bs = FakeBaoStock(
        [
            FakeLoginResult("10002007", "网络接收错误"),
            FakeLoginResult("10002007", "网络接收错误"),
            FakeLoginResult("0", "success"),
        ]
    )
    sleep_calls: list[float] = []

    monkeypatch.setattr(baostock_provider, "bs", fake_bs)
    monkeypatch.setattr(baostock_provider, "sleep", lambda seconds: sleep_calls.append(seconds))

    result = baostock_provider.login_baostock(retry_delay_seconds=0.25)

    assert result.error_code == "0"
    assert fake_bs.login_calls == 3
    assert sleep_calls == [0.25, 0.25]


def test_login_baostock_raises_after_retry_budget(monkeypatch) -> None:
    fake_bs = FakeBaoStock(
        [
            FakeLoginResult("10002007", "网络接收错误"),
            FakeLoginResult("10002007", "网络接收错误"),
            FakeLoginResult("10002007", "网络接收错误"),
        ]
    )
    sleep_calls: list[float] = []

    monkeypatch.setattr(baostock_provider, "bs", fake_bs)
    monkeypatch.setattr(baostock_provider, "sleep", lambda seconds: sleep_calls.append(seconds))

    with pytest.raises(RuntimeError, match="baostock login failed after 3 attempt\\(s\\): 10002007 网络接收错误"):
        baostock_provider.login_baostock(retry_delay_seconds=0.1)

    assert fake_bs.login_calls == 3
    assert sleep_calls == [0.1, 0.1]


def test_login_baostock_retries_when_login_raises(monkeypatch) -> None:
    fake_bs = ExceptionalBaoStock(failures=2)
    sleep_calls: list[float] = []

    monkeypatch.setattr(baostock_provider, "bs", fake_bs)
    monkeypatch.setattr(baostock_provider, "sleep", lambda seconds: sleep_calls.append(seconds))

    result = baostock_provider.login_baostock(retry_delay_seconds=0.2)

    assert result.error_code == "0"
    assert fake_bs.login_calls == 3
    assert sleep_calls == [0.2, 0.2]
