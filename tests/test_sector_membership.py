from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

import stocks_analyzer.sector_membership as sector_membership
from stocks_analyzer.sector_membership import append_sector_display_columns, build_sector_display_frame, sector_membership_path


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_build_sector_display_frame_joins_multiple_industries_and_concepts() -> None:
    tmp_path = _make_workspace_tmp_dir("sector_membership")
    target = sector_membership_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "symbol": "600000",
                "name": "测试股份",
                "sector_type": "industry",
                "sector_name": "银行",
                "sector_label": "new_yh",
                "source": "ths_industry",
                "updated_at": "2026-05-15T10:00:00",
            },
            {
                "symbol": "600000",
                "name": "测试股份",
                "sector_type": "industry",
                "sector_name": "沪深300",
                "sector_label": "new_hs300",
                "source": "ths_industry",
                "updated_at": "2026-05-15T10:00:00",
            },
            {
                "symbol": "600000",
                "name": "测试股份",
                "sector_type": "concept",
                "sector_name": "央企改革",
                "sector_label": "gn_yqgg",
                "source": "ths_concept",
                "updated_at": "2026-05-15T10:00:00",
            },
            {
                "symbol": "600000",
                "name": "测试股份",
                "sector_type": "concept",
                "sector_name": "破净股",
                "sector_label": "gn_pjg",
                "source": "ths_concept",
                "updated_at": "2026-05-15T10:00:00",
            },
        ]
    ).to_csv(target, index=False, encoding="utf-8-sig")

    display = build_sector_display_frame(project_root=tmp_path)

    assert display.loc[0, "symbol"] == "600000"
    assert display.loc[0, "industry_names"] == "银行/沪深300"
    assert display.loc[0, "concept_names"] == "央企改革/破净股"


def test_sector_membership_ignores_non_thematic_boards_by_default() -> None:
    tmp_path = _make_workspace_tmp_dir("sector_membership_ignore")
    target = sector_membership_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "symbol": "600000",
                "name": "测试股份",
                "sector_type": "concept",
                "sector_name": "2025年报预增",
                "sector_label": "886107",
                "source": "ths_concept",
                "updated_at": "2026-05-15T10:00:00",
            },
            {
                "symbol": "600000",
                "name": "测试股份",
                "sector_type": "concept",
                "sector_name": "同花顺漂亮50",
                "sector_label": "885931",
                "source": "ths_concept",
                "updated_at": "2026-05-15T10:00:00",
            },
            {
                "symbol": "600000",
                "name": "测试股份",
                "sector_type": "concept",
                "sector_name": "摘帽",
                "sector_label": "885999",
                "source": "ths_concept",
                "updated_at": "2026-05-15T10:00:00",
            },
            {
                "symbol": "600000",
                "name": "测试股份",
                "sector_type": "concept",
                "sector_name": "新股与次新股",
                "sector_label": "885598",
                "source": "ths_concept",
                "updated_at": "2026-05-15T10:00:00",
            },
            {
                "symbol": "600000",
                "name": "测试股份",
                "sector_type": "concept",
                "sector_name": "注册制次新股",
                "sector_label": "885948",
                "source": "ths_concept",
                "updated_at": "2026-05-15T10:00:00",
            },
            {
                "symbol": "600000",
                "name": "测试股份",
                "sector_type": "concept",
                "sector_name": "2022年半年度报告披露显示，公司自设立以来一直专注于休闲食品的研发、生产和销售",
                "sector_label": "desc_001",
                "source": "ths_concept",
                "updated_at": "2026-05-15T10:00:00",
            },
            {
                "symbol": "600000",
                "name": "测试股份",
                "sector_type": "concept",
                "sector_name": "商业航天",
                "sector_label": "886091",
                "source": "ths_concept",
                "updated_at": "2026-05-15T10:00:00",
            },
        ]
    ).to_csv(target, index=False, encoding="utf-8-sig")

    membership = sector_membership.load_sector_membership(project_root=tmp_path)
    raw = sector_membership.load_raw_sector_membership(project_root=tmp_path)
    display = build_sector_display_frame(project_root=tmp_path)

    assert raw["sector_name"].tolist() == [
        "2025年报预增",
        "同花顺漂亮50",
        "摘帽",
        "新股与次新股",
        "注册制次新股",
        "2022年半年度报告披露显示，公司自设立以来一直专注于休闲食品的研发、生产和销售",
        "商业航天",
    ]
    assert membership["sector_name"].tolist() == ["商业航天"]
    assert display.loc[0, "concept_names"] == "商业航天"


def test_append_sector_display_columns_preserves_input_rows_when_cache_missing() -> None:
    tmp_path = _make_workspace_tmp_dir("sector_membership_missing")
    frame = pd.DataFrame([{"symbol": "1", "name": "平安银行"}])

    result = append_sector_display_columns(frame, project_root=tmp_path)

    assert result.loc[0, "symbol"] == "1"
    assert "industry_names" in result.columns
    assert "concept_names" in result.columns


def test_build_sector_performance_uses_amount_weighted_returns() -> None:
    tmp_path = _make_workspace_tmp_dir("sector_performance")
    daily_dir = tmp_path / "data" / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-05-14", "symbol": "600000", "close": 10.0, "pct_change": 0.0, "amount": 80.0},
            {"trade_date": "2026-05-15", "symbol": "600000", "close": 10.2, "pct_change": 2.0, "amount": 100.0},
        ]
    ).to_parquet(daily_dir / "600000.parquet", index=False)
    pd.DataFrame(
        [
            {"trade_date": "2026-05-14", "symbol": "600001", "close": 10.0, "pct_change": 0.0, "amount": 200.0},
            {"trade_date": "2026-05-15", "symbol": "600001", "close": 9.9, "pct_change": -1.0, "amount": 300.0},
        ]
    ).to_parquet(daily_dir / "600001.parquet", index=False)
    membership = pd.DataFrame(
        [
            {
                "symbol": "600000",
                "name": "测试甲",
                "sector_type": "industry",
                "sector_name": "电子元件",
                "sector_label": "new_dzqj",
                "source": "ths_industry",
                "updated_at": "2026-05-15T10:00:00",
            },
            {
                "symbol": "600001",
                "name": "测试乙",
                "sector_type": "industry",
                "sector_name": "电子元件",
                "sector_label": "new_dzqj",
                "source": "ths_industry",
                "updated_at": "2026-05-15T10:00:00",
            },
            {
                "symbol": "600000",
                "name": "测试甲",
                "sector_type": "concept",
                "sector_name": "机器人",
                "sector_label": "gn_jqr",
                "source": "ths_concept",
                "updated_at": "2026-05-15T10:00:00",
            },
        ]
    )

    frame, resolved = sector_membership.build_sector_performance(
        project_root=tmp_path,
        membership=membership,
        trade_date=date(2026, 5, 15),
        daily_dir=daily_dir,
    )

    assert resolved == date(2026, 5, 15)
    industry = frame[frame["sector_type"].eq("industry")].iloc[0]
    assert industry["member_count"] == 2
    assert industry["valid_count"] == 2
    assert industry["avg_pct_change"] == 0.5
    assert industry["amount_weighted_pct_change"] == -0.25
    assert industry["up_count"] == 1
    assert industry["up_ratio"] == 0.5
    assert industry["total_amount"] == 400.0


def test_update_sector_membership_preserves_existing_type_when_fetch_fails(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("sector_membership_preserve")
    target = sector_membership_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "symbol": "600000",
                "name": "测试股份",
                "sector_type": "concept",
                "sector_name": "央企改革",
                "sector_label": "gn_yqgg",
                "source": "ths_concept",
                "updated_at": "2026-05-14T10:00:00",
            }
        ]
    ).to_csv(target, index=False, encoding="utf-8-sig")

    def fake_list(*, url_root: str, seed_code: str, include_summary: bool = False) -> pd.DataFrame:
        if url_root == "thshy":
            return pd.DataFrame([{"code": "881270", "name": "电子元件"}])
        if url_root == "gn":
            return pd.DataFrame(columns=["name", "code"])
        raise AssertionError(f"unexpected THS root: {url_root}")

    def fake_members(*, url_root: str, sector_label: str) -> pd.DataFrame:
        assert url_root == "thshy"
        assert sector_label == "881270"
        return pd.DataFrame([{"代码": "600000", "名称": "测试股份"}])

    monkeypatch.setattr(sector_membership, "_fetch_ths_sector_list", fake_list)
    monkeypatch.setattr(sector_membership, "_fetch_ths_sector_members", fake_members)

    result = sector_membership.update_sector_membership(project_root=tmp_path, force_refresh=True)
    frame = pd.read_csv(result.output_path, dtype={"symbol": str})

    assert result.industry_count == 1
    assert result.concept_count == 1
    assert set(frame["sector_type"]) == {"industry", "concept"}
    assert frame.loc[frame["sector_type"].eq("concept"), "sector_name"].iloc[0] == "央企改革"


def test_update_sector_membership_fetches_ths_commercial_space(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("sector_membership_commercial_space")

    def fake_list(*, url_root: str, seed_code: str, include_summary: bool = False) -> pd.DataFrame:
        if url_root == "thshy":
            return pd.DataFrame([{"code": "881272", "name": "航天航空"}])
        if url_root == "gn":
            return pd.DataFrame([{"code": "309130", "name": "商业航天"}])
        raise AssertionError(f"unexpected THS root: {url_root}")

    def fake_members(*, url_root: str, sector_label: str) -> pd.DataFrame:
        if url_root == "thshy":
            assert sector_label == "881272"
            return pd.DataFrame([{"代码": "600118", "名称": "中国卫星"}])
        if url_root == "gn":
            assert sector_label == "309130"
            return pd.DataFrame(
                [
                    {"代码": "600118", "名称": "中国卫星"},
                    {"代码": "600879", "名称": "航天电子"},
                ]
            )
        raise AssertionError(f"unexpected THS root: {url_root}")

    monkeypatch.setattr(sector_membership, "_fetch_ths_sector_list", fake_list)
    monkeypatch.setattr(sector_membership, "_fetch_ths_sector_members", fake_members)

    result = sector_membership.update_sector_membership(project_root=tmp_path)
    frame = pd.read_csv(result.output_path, dtype={"symbol": str})
    display = sector_membership.build_sector_display_frame(project_root=tmp_path)

    assert result.concept_count == 1
    commercial = frame[frame["sector_name"].eq("商业航天")]
    assert commercial["sector_label"].astype(str).unique().tolist() == ["309130"]
    assert sorted(commercial["symbol"].tolist()) == ["600118", "600879"]
    assert display.loc[display["symbol"].eq("600118"), "concept_names"].iloc[0] == "商业航天"


def test_update_sector_membership_reuses_fresh_cache_but_rebuilds_daily_performance(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("sector_membership_fresh_cache")
    target = sector_membership_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "symbol": "600000",
                "name": "测试股份",
                "sector_type": "industry",
                "sector_name": "银行",
                "sector_label": "银行",
                "source": "ths_stock_profile",
                "updated_at": "2026-05-15T10:00:00",
            }
        ]
    ).to_csv(target, index=False, encoding="utf-8-sig")
    daily_dir = tmp_path / "data" / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-05-14", "symbol": "600000", "close": 10.0, "amount": 100.0},
            {"trade_date": "2026-05-15", "symbol": "600000", "close": 10.2, "amount": 200.0},
        ]
    ).to_parquet(daily_dir / "600000.parquet", index=False)

    def fail_fetch(*args, **kwargs):
        raise AssertionError("fresh membership cache should skip network refresh")

    monkeypatch.setattr(sector_membership, "_fetch_ths_stock_profile_rows", fail_fetch)

    result = sector_membership.update_sector_membership(project_root=tmp_path, trade_date=date(2026, 5, 15), daily_dir=daily_dir)

    assert result.membership_refreshed is False
    assert result.performance_path is not None
    assert result.performance_path.exists()
    performance = pd.read_csv(result.performance_path)
    assert performance.loc[0, "avg_pct_change"] == 2.0


def test_update_sector_membership_refreshes_stale_cache(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("sector_membership_stale_cache")
    target = sector_membership_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "symbol": "600000",
                "name": "旧股份",
                "sector_type": "industry",
                "sector_name": "旧行业",
                "sector_label": "旧行业",
                "source": "ths_stock_profile",
                "updated_at": "2026-05-01T10:00:00",
            }
        ]
    ).to_csv(target, index=False, encoding="utf-8-sig")
    stale_time = time.time() - 8 * 86400
    target.touch()
    import os

    os.utime(target, (stale_time, stale_time))

    monkeypatch.setattr(
        sector_membership,
        "_fetch_ths_stock_profile_rows",
        lambda *, project_root, updated_at: [
            {
                "symbol": "600001",
                "name": "新股份",
                "sector_type": "industry",
                "sector_name": "新行业",
                "sector_label": "新行业",
                "source": "ths_stock_profile",
                "updated_at": updated_at,
            }
        ],
    )

    result = sector_membership.update_sector_membership(project_root=tmp_path)
    frame = pd.read_csv(target, dtype={"symbol": str})

    assert result.membership_refreshed is True
    assert frame["symbol"].tolist() == ["600001"]
    assert frame["sector_name"].tolist() == ["新行业"]


def test_fetch_ths_sector_members_parses_paginated_tables(monkeypatch) -> None:
    first_page = """
    <table><thead><tr><th>代码</th><th>名称</th></tr></thead>
    <tbody><tr><td>600118</td><td>中国卫星</td></tr></tbody></table>
    <span class="page_info">1/2</span>
    """
    second_page = """
    <table><thead><tr><th>代码</th><th>名称</th></tr></thead>
    <tbody><tr><td>600879</td><td>航天电子</td></tr></tbody></table>
    """
    seen_urls: list[str] = []

    def fake_get(url: str, *, referer: str | None = None) -> str:
        seen_urls.append(url)
        if "/page/2/" in url:
            assert referer == "https://q.10jqka.com.cn/gn/detail/code/309130/"
            return second_page
        return first_page

    monkeypatch.setattr(sector_membership, "_ths_get", fake_get)

    frame = sector_membership._fetch_ths_sector_members(url_root="gn", sector_label="309130")

    assert seen_urls == [
        "https://q.10jqka.com.cn/gn/detail/code/309130/",
        "https://q.10jqka.com.cn/gn/detail/page/2/ajax/1/code/309130/",
    ]
    assert frame["代码"].tolist() == ["600118", "600879"]


def test_fetch_ths_sector_list_merges_category_and_summary_links(monkeypatch) -> None:
    category = """
    <div class="cate_inner">
      <a href="http://q.10jqka.com.cn/gn/detail/code/309130/">商业航天</a>
    </div>
    """
    summary = """
    <table><tbody><tr><td><a href="http://q.10jqka.com.cn/gn/detail/code/309131/">机器人</a></td></tr></tbody></table>
    <span class="page_info">1/1</span>
    """

    def fake_get(url: str, *, referer: str | None = None) -> str:
        if "/gn/index/" in url:
            return summary
        return category

    monkeypatch.setattr(sector_membership, "_ths_get", fake_get)

    frame = sector_membership._fetch_ths_sector_list(url_root="gn", seed_code="309130", include_summary=True)

    assert set(frame["name"]) == {"商业航天", "机器人"}
    assert set(frame["code"].astype(str)) == {"309130", "309131"}


def test_fetch_ths_stock_profile_rows_parses_industry_and_concepts(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("sector_membership_stock_profile")
    company_html = "<strong class='hltip fl'>所属申万行业：</strong><span>国防军工 — 航天装备Ⅱ</span>"
    concept_html = """
    <table><tbody>
      <tr><td class="gnName" clid="886078">商业航天</td></tr>
      <tr><td class="gnName" clid="300800">安防</td></tr>
    </tbody></table>
    """

    monkeypatch.setattr(
        sector_membership,
        "_load_stock_universe",
        lambda *, project_root: pd.DataFrame([{"symbol": "600118", "name": "中国卫星"}]),
    )

    def fake_basic_get(*, symbol: str, page: str) -> str:
        assert symbol == "600118"
        if page == "company.html":
            return company_html
        if page == "concept.html":
            return concept_html
        raise AssertionError(page)

    monkeypatch.setattr(sector_membership, "_ths_basic_get", fake_basic_get)

    rows = sector_membership._fetch_ths_stock_profile_rows(project_root=tmp_path, updated_at="2026-05-15T10:00:00")

    industry_rows = [row for row in rows if row["sector_type"] == "industry"]
    concept_rows = [row for row in rows if row["sector_type"] == "concept"]
    assert industry_rows[0]["sector_name"] == "国防军工-航天装备Ⅱ"
    assert {row["sector_name"] for row in concept_rows} == {"商业航天", "安防"}
    assert {row["sector_label"] for row in concept_rows} == {"886078", "300800"}
