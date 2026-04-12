from __future__ import annotations

from datetime import date

from pathlib import Path

from stocks_analyzer.daily_screening import build_daily_section, parse_pick_history, prepend_daily_section, run_daily_screening, upsert_daily_section


def test_prepend_daily_section_puts_latest_on_top() -> None:
    existing = "### 2026.4.10\n\n旧内容\n"
    section = "### 2026.4.11\n\n新内容\n"

    merged = prepend_daily_section(existing_text=existing, section=section)

    assert merged.startswith("### 2026.4.11")
    assert "### 2026.4.10" in merged


def test_upsert_daily_section_replaces_same_day_section() -> None:
    existing = """### 2026.4.11

旧内容

### 2026.4.10

更旧内容
"""
    section = "### 2026.4.11\n\n新内容\n"

    merged = upsert_daily_section(existing_text=existing, section=section, trade_date=date(2026, 4, 11))

    assert merged.count("### 2026.4.11") == 1
    assert "新内容" in merged
    assert "\n旧内容\n" not in merged
    assert "### 2026.4.10" in merged


def test_parse_pick_history_extracts_symbols() -> None:
    markdown = """### 2026.4.11

| 梯队 | 股票代码 | 股票名称 | 行业/主线 | 符合模式 | 五日分数 | 五日均分 | TradingView标签 | 推荐理由 |
| ---- | -------- | -------- | --------- | -------- | -------- | -------: | --------------- | -------- |
| 第一梯队 | 002579 | 中京电子 | 算力硬件 | pattern 1 | 0.1 | 0.1 | buy | demo |

总结：示例总结。
"""

    history = parse_pick_history(markdown)

    assert history[0]["symbols"] == ["002579"]
    assert history[0]["summary"] == "示例总结。"


def test_build_daily_section_includes_summary_and_tables() -> None:
    payload = {
        "candidates": [
            {
                "tier": "第一梯队",
                "symbol": "2579",
                "name": "中京电子",
                "theme": "算力硬件",
                "pattern_id": "1",
                "macd_top_divergence_15d": True,
                "tradingview_label": "buy",
                "tradingview_avg_5d": 0.44,
                "five_day_scores": [0.46, 0.60, 0.46, 0.26, 0.40],
            },
            {
                "tier": "第二梯队",
                "symbol": "603803",
                "name": "瑞斯康达",
                "theme": "未分类",
                "pattern_id": "3",
                "macd_bottom_divergence_15d": True,
                "tradingview_label": "buy",
                "tradingview_avg_5d": 0.32,
                "five_day_scores": [0.20, 0.10, 0.42, 0.46, 0.44],
            },
        ],
        "analysis": {
            "market_sentiment": {"summary": "当日共筛出 2 只候选，整体强中有分化。"},
            "mainline_changes": {"summary": "当前主线仍以算力硬件、电池为核心。"},
            "pick_changes": {"summary": "新增 1 只：603803；保留 1 只：002579；移除 0 只。"},
            "notable_stocks": [
                {"symbol": "002579", "name": "中京电子", "summary": "近15日出现顶背离，若继续冲高需防追高。"},
                {"symbol": "603803", "name": "瑞斯康达", "summary": "近15日出现底背离，可结合位置观察修复。"},
            ],
        },
    }
    existing = """### 2026.4.10

| 梯队 | 股票代码 | 股票名称 | 行业/主线 | 符合模式 | 五日分数 | 五日均分 | TradingView标签 | 推荐理由 |
| ---- | -------- | -------- | --------- | -------- | -------- | -------: | --------------- | -------- |
| 第一梯队 | 002579 | 中京电子 | 算力硬件 | pattern 1 | 0.1 | 0.1 | buy | demo |

总结：旧总结。
"""

    section = build_daily_section(
        trade_date=date(2026, 4, 11),
        picker_payload=payload,
        existing_markdown=existing,
        theme_map={"002579": "算力硬件"},
    )

    assert section.startswith("### 2026.4.11")
    assert "| 第一梯队 | 002579 | 中京电子 | 算力硬件 |" in section
    assert "| 第二梯队 | 603803 | 瑞斯康达 | 未分类 |" in section
    assert "pattern 1 + 顶背离" in section
    assert "pattern 3 + 底背离" in section
    assert "当日市场情绪监测：" in section
    assert "主线变动：" in section
    assert "选股变化：" in section
    assert "值得注意的股：" in section
    assert "002579中京电子" in section
    assert "603803瑞斯康达" in section


def test_build_daily_section_handles_empty_candidates_with_analysis() -> None:
    payload = {
        "candidates": [],
        "analysis": {
            "market_sentiment": {"summary": "当日没有筛出候选，先以观察为主。"},
            "mainline_changes": {"summary": "当前主线文件仍以算力硬件、电池为核心。"},
            "pick_changes": {"summary": "首期记录，暂无上一期可比数据。"},
            "notable_stocks": [],
        },
    }

    section = build_daily_section(
        trade_date=date(2026, 4, 11),
        picker_payload=payload,
        existing_markdown="",
        theme_map={},
    )

    assert "当日未筛出满足当前阈值的候选。" in section
    assert "当日市场情绪监测：" in section
    assert "主线变动：" in section
    assert "选股变化：" in section
    assert "值得注意的股：暂无特别需要额外提示的个股。" in section


def test_run_daily_screening_runs_divergence_between_tradingview_and_pattern(monkeypatch, tmp_path: Path, capsys) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr("stocks_analyzer.daily_screening.is_trading_day", lambda provider, trade_date: True)
    monkeypatch.setattr(
        "stocks_analyzer.daily_screening.load_config",
        lambda path: type("Config", (), {"provider": "mock"})(),
    )

    def fake_run_project_command(project_root: Path, args: list[str]) -> None:
        commands.append(args)

    monkeypatch.setattr("stocks_analyzer.daily_screening._run_project_command", fake_run_project_command)
    monkeypatch.setattr(
        "stocks_analyzer.daily_screening._run_project_stock_picker",
        lambda project_root: {
            "candidates": [],
            "analysis": {
                "market_sentiment": {"summary": "当日没有筛出候选，先以观察为主。"},
                "mainline_changes": {"summary": "当前缺少结构化主线变动数据。"},
                "pick_changes": {"summary": "首期记录，暂无上一期可比数据。"},
                "notable_stocks": [],
            },
        },
    )

    result = run_daily_screening(project_root=tmp_path, trade_date=date(2026, 4, 11))
    output = capsys.readouterr().out

    assert result.skipped is False
    assert commands == [
        ["update", "--start-date", "20240101"],
        ["tradingview", "--date", "2026-04-11"],
        ["divergence", "--date", "2026-04-11"],
        ["pattern", "--as-of", "2026-04-11"],
    ]
    assert "[0/4] 检查 2026-04-11 是否为交易日..." in output
    assert "[0/4] 2026-04-11 是交易日，开始执行每日筛选。" in output
    assert "[1/4] 开始 update..." in output
    assert "[4/4] pattern 完成" in output
