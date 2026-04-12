from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_project_stock_picker_module():
    module_path = Path(__file__).resolve().parents[1] / "skills" / "project-stock-picker" / "scripts" / "project_stock_picker.py"
    spec = importlib.util.spec_from_file_location("project_stock_picker_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load project_stock_picker module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_candidates_reads_new_patterns_path_and_filters_true_indexes(tmp_path: Path) -> None:
    module = _load_project_stock_picker_module()

    patterns_dir = tmp_path / "reports" / "patterns"
    patterns_dir.mkdir(parents=True)
    (tmp_path / "reports").mkdir(exist_ok=True)

    pd.DataFrame(
        [
            {
                "symbol": '="002579"',
                "name": "中京电子",
                "pattern_id": "1",
                "tradingview_avg_all_rating_5d": 0.44,
                "tradingview_all_rating_label": "buy",
                "tradingview_all_rating_2026-04-08": 0.46,
                "tradingview_all_rating_2026-04-09": 0.60,
                "tradingview_all_rating_2026-04-10": 0.46,
                "tradingview_all_rating_2026-04-11": 0.26,
                "tradingview_all_rating_2026-04-12": 0.40,
                "macd_top_divergence_15d": True,
                "macd_bottom_divergence_15d": False,
                "reason": "demo-1",
            },
            {
                "symbol": "603803",
                "name": "瑞斯康达",
                "pattern_id": "1",
                "tradingview_avg_all_rating_5d": 0.44,
                "tradingview_all_rating_label": "buy",
                "tradingview_all_rating_2026-04-08": 0.46,
                "tradingview_all_rating_2026-04-09": 0.60,
                "tradingview_all_rating_2026-04-10": 0.46,
                "tradingview_all_rating_2026-04-11": 0.26,
                "tradingview_all_rating_2026-04-12": 0.40,
                "macd_top_divergence_15d": False,
                "macd_bottom_divergence_15d": False,
                "reason": "demo-2",
            },
            {
                "symbol": "000001",
                "name": "上证综指",
                "pattern_id": "1",
                "tradingview_avg_all_rating_5d": 0.52,
                "tradingview_all_rating_label": "strong_buy",
                "tradingview_all_rating_2026-04-08": 0.52,
                "tradingview_all_rating_2026-04-09": 0.51,
                "tradingview_all_rating_2026-04-10": 0.52,
                "tradingview_all_rating_2026-04-11": 0.53,
                "tradingview_all_rating_2026-04-12": 0.52,
                "macd_top_divergence_15d": False,
                "macd_bottom_divergence_15d": False,
                "reason": "index",
            },
        ]
    ).to_csv(patterns_dir / "patterns_all_2026-04-12.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "symbol": "999999",
                "name": "旧路径文件",
                "pattern_id": "4",
                "tradingview_avg_all_rating_5d": 0.99,
                "tradingview_all_rating_label": "strong_buy",
                "tradingview_all_rating_2026-04-08": 0.99,
                "tradingview_all_rating_2026-04-09": 0.99,
                "tradingview_all_rating_2026-04-10": 0.99,
                "tradingview_all_rating_2026-04-11": 0.99,
                "tradingview_all_rating_2026-04-12": 0.99,
            }
        ]
    ).to_csv(tmp_path / "reports" / "patterns_all_2026-04-12.csv", index=False, encoding="utf-8-sig")

    (tmp_path / "主线.md").write_text(
        "\n".join(
            [
                "# A股市场主线",
                "",
                "## 1. 第一梯队主线",
                "### AI算力硬件",
                "### 电池新能源",
                "",
                "## 2. 第二梯队主线",
                "### 创新药",
                "",
                "## 3. 轮动与预期线",
                "### 机器人",
                "",
                "## 4. 短线事件题材",
                "### 稳定币 / 支付",
                "",
                "## 一句话结论",
                "`A股主线是 AI算力硬件 与 电池新能源。`",
            ]
        ),
        encoding="utf-8",
    )

    (tmp_path / "选股.md").write_text(
        "\n".join(
            [
                "### 2026.4.11",
                "",
                "| 梯队 | 股票代码 | 股票名称 | 行业/主线 | 符合模式/背离 | 五日分数 | 五日均分 | TradingView标签 | 推荐理由 |",
                "| ---- | -------- | -------- | --------- | ------------- | -------- | -------: | --------------- | -------- |",
                "| 第一梯队 | 002579 | 中京电子 | 算力硬件 | pattern 1 | 0.1 | 0.1 | buy | demo |",
                "| 第二梯队 | 600000 | 测试旧票 | 电池 | pattern 2 | 0.1 | 0.1 | buy | demo |",
            ]
        ),
        encoding="utf-8",
    )

    payload = module.build_candidates(tmp_path, limit=10)

    assert Path(payload["source_file"]).parent.name == "patterns"
    assert Path(payload["theme_source_file"]).name == "主线.md"

    symbols = [item["symbol"] for item in payload["candidates"]]
    assert symbols == ["002579", "603803"]

    stable_scores = {item["symbol"]: item["stable_score"] for item in payload["candidates"]}
    assert stable_scores["002579"] == stable_scores["603803"]

    analysis = payload["analysis"]
    assert analysis["pick_changes"]["retained"] == ["002579"]
    assert analysis["pick_changes"]["removed"] == ["600000"]
    assert "算力硬件" in analysis["mainline_changes"]["current_mainlines"]["core"]
    assert "电池" in analysis["mainline_changes"]["current_mainlines"]["core"]
    assert analysis["market_sentiment"]["summary"]
    assert any(item["symbol"] == "002579" for item in analysis["notable_stocks"])


def test_write_candidates_to_picks_replaces_same_trade_date_section(tmp_path: Path) -> None:
    module = _load_project_stock_picker_module()

    patterns_dir = tmp_path / "reports" / "patterns"
    patterns_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "symbol": "002579",
                "name": "中京电子",
                "pattern_id": "1",
                "tradingview_avg_all_rating_5d": 0.44,
                "tradingview_all_rating_label": "buy",
                "tradingview_all_rating_2026-04-08": 0.46,
                "tradingview_all_rating_2026-04-09": 0.60,
                "tradingview_all_rating_2026-04-10": 0.46,
                "tradingview_all_rating_2026-04-11": 0.26,
                "tradingview_all_rating_2026-04-12": 0.40,
                "macd_top_divergence_15d": False,
                "macd_bottom_divergence_15d": True,
                "reason": "demo",
            }
        ]
    ).to_csv(patterns_dir / "patterns_all_2026-04-12.csv", index=False, encoding="utf-8-sig")

    (tmp_path / "主线.md").write_text(
        "\n".join(
            [
                "# A股市场主线",
                "",
                "## 1. 第一梯队主线",
                "### AI算力硬件",
                "",
                "## 一句话结论",
                "`A股主线是 AI算力硬件。`",
            ]
        ),
        encoding="utf-8",
    )

    picks_path = tmp_path / "选股.md"
    picks_path.write_text(
        "\n".join(
            [
                "### 2026.4.12",
                "",
                "旧内容",
                "",
                "### 2026.4.11",
                "",
                "更旧内容",
            ]
        ),
        encoding="utf-8",
    )

    payload = module.build_candidates(tmp_path, limit=10)
    written_path = module.write_candidates_to_picks(project_root=tmp_path, payload=payload, picks_filename="选股.md")

    written = written_path.read_text(encoding="utf-8")
    assert written_path == picks_path
    assert written.count("### 2026.4.12") == 1
    assert "\n旧内容\n" not in written
    assert "| 第一梯队 | 002579 | 中京电子 |" in written
    assert "当日市场情绪监测：" in written
    assert "### 2026.4.11" in written
