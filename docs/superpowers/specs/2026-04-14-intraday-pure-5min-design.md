# Intraday 纯 5 分钟复筛设计

## 1. 目标

将 `intraday-screening` 从“盘中再跑一遍日线结果 + 5 分钟排序”改成“纯 5 分钟复筛”。

新的盘中流程：

- 读取最新或指定日期的 `watchlist`
- 不再调用 `update`
- 不再生成盘中阶段的 `tradingview / divergence / pattern`
- 直接抓取 `watchlist` 股票的当日 5 分钟线
- 只输出一份盘中排序 CSV 和一份 JSON 摘要

## 2. 输入与输出

输入继续是 `watchlist.json`。

CSV 仍保留一部分日线辅助字段，但这些字段只从 `watchlist.json` 中读取，不再回读 `patterns_all_*.csv` 或盘中重新计算：

- `代码`
- `名称`
- `形态`
- `TradingView五日均分`
- `TradingView评级`
- `日线顶背离`
- `日线底背离`

盘中字段继续来自 5 分钟事件打分：

- `5分钟分数`
- `量价背离命中/类型/分`
- `5分钟MACD背离命中/类型/分`
- `5分钟金叉死叉命中/类型/分`
- `均线事件命中/类型/分`

输出文件收缩为：

- `reports/intraday_screening/<date>/intraday_rank_<date>.csv`
- `reports/intraday_screening/<date>/intraday_screening_<date>.json`

JSON 只保留：

- `trade_date`
- `watchlist_date`
- `watchlist_path`
- `symbol_count`
- `symbols`
- `intraday_rank_path`

## 3. 代码范围

### 3.1 `cli.py`

- 删掉 `_run_intraday_screening(...)` 中对 `_run_update(...)` 的调用
- 删掉 `_run_tradingview(...)`、`_run_divergence(...)`、`_run_pattern(...)` 的盘中调用
- 只保留读 `watchlist`、生成 `intraday_rank.csv`、写 JSON

### 3.2 `intraday_ranking.py`

- 新增直接从 `watchlist` 候选项构造基础表的入口
- 盘中排序只依赖 `watchlist` 候选项和 5 分钟数据
- 仍保留既有 5 分钟事件与分数逻辑

## 4. 测试

- `intraday-screening` 不再调用 `_run_update / _run_tradingview / _run_divergence / _run_pattern`
- JSON 不再包含 `tradingview_path / divergence_path / pattern_path`
- 排名 CSV 仍可从 `watchlist.json` 带出日线辅助字段
- 5 分钟排序逻辑继续可测
