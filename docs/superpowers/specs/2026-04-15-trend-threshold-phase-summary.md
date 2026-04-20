# Trend Threshold Phase Summary

## 1. 当前阶段目标

本阶段的目标是把“收盘评分 -> 次日开盘买入”的趋势交易链路，从固定默认阈值推进到“可研究、可比较、可按信号类型分开配置”的状态。

这一阶段重点不是继续增加技术指标，而是回答三个问题：

- 现有评分体系里，哪些门槛真的有效
- `breakout` 和 `pullback` 是否应该共用同一套阈值
- 当前默认阈值是否应该被研究结果替换

## 2. 已完成能力

### 2.1 次日开盘评分与回测链路

已完成一条独立于 `pattern/watchlist` 的趋势研究链路：

- `trend-universe`
- `trend-signals`
- `trend-score`
- `trend-entries`
- `backtest-signals`
- `backtest-portfolio`
- `backtest-entries`
- `backtest-entries-portfolio`

其中：

- `trend-score` 负责在 `breakout/pullback` setup 上叠加多指标评分
- `trend-entries` 负责生成“收盘评分后，次日开盘买入”的候选
- `backtest-entries` 和 `backtest-entries-portfolio` 负责 `next_open` 口径回测

### 2.2 阈值研究命令

已新增独立命令：

```bash
python -m stocks_analyzer --project-root . research-thresholds --date 2026-04-10 --start-date 2025-01-01 --sample-mode monthly
```

该命令当前可以完成：

- 从历史区间构建评分样本
- 按 `next_open` 口径回测 `5/10/20/40` 日表现
- 比较强势组、弱势组、底部组的指标分布
- 生成单指标候选阈值
- 生成组合阈值候选
- 评估 `current_default_rules` 与候选阈值的差异
- 输出 `breakout/pullback` 分开的候选默认阈值

### 2.3 进度日志

`research-thresholds` 已接入进度显示，包括：

- 股票扫描进度
- 样本构建阶段日志
- 分布统计阶段日志
- 候选阈值生成阶段日志
- 组合阈值评估阶段日志
- 报表写出阶段日志

## 3. 当前评分与阈值研究结论

### 3.1 总体结论

从当前历史研究结果看：

- `buy_score` 是当前最有效的主门槛
- `10日持有` 比 `5日持有` 更适合作为主要观察周期
- `breakout` 和 `pullback` 不适合共用完全相同的默认阈值
- `MACD`、`trend_base_score` 等分项更适合继续保留在评分体系中，不一定都适合做第二层硬门槛

### 3.2 `breakout` 结论

按当前研究结果，`breakout` 在 `10日持有` 下的较优候选组合是：

- `buy_score >= 81.3308`
- `price_action_score >= 75.0373`

相对旧的全局默认阈值，这组规则在样本内表现更好，因此已经被采纳为当前 `breakout` 的默认覆盖规则。

### 3.3 `pullback` 结论

当前研究结果显示，`pullback` 的候选阈值尚未稳定优于旧默认规则。

因此当前阶段的处理方式是：

- `pullback` 暂时继续沿用原有全局默认阈值
- 后续单独继续做 `pullback` 阈值研究，不和 `breakout` 混在一起直接替换

## 4. 当前生效的默认规则

### 4.1 全局默认阈值

当前全局默认阈值仍然保留为：

- `buy_score >= 80`
- `trend_base_score >= 65`
- `price_action_score >= 60`
- `macd_score >= 35`
- `positive_indicator_count >= 3`

### 4.2 分信号覆盖规则

当前代码已支持按 `signal_type` 分开阈值。

当前生效逻辑是：

- `breakout`
  - `buy_score >= 81.3308`
  - `price_action_score >= 75.0373`
- `pullback`
  - 继续沿用全局默认阈值

也就是说：

- `breakout` 已切换到研究阈值
- `pullback` 仍在观察阶段

## 5. 输出文件

本阶段新增或重点依赖的输出目录包括：

- `reports/trend_scores/`
- `reports/trend_entries/`
- `reports/backtests/entries/`
- `reports/backtests/entries_portfolio/`
- `reports/threshold_research/`

其中 `reports/threshold_research/` 当前会输出：

- `threshold_samples_*.csv`
- `threshold_distributions_*.csv`
- `threshold_candidates_*.csv`
- `threshold_candidate_eval_*.csv`
- `threshold_combo_candidates_*.csv`
- `threshold_combo_eval_*.csv`
- `threshold_default_candidates_*.csv`

最后这一张 `threshold_default_candidates_*.csv` 是当前最重要的阶段性结论表，用来直接看：

- `breakout` 当前建议阈值
- `pullback` 当前建议阈值
- 与 `current_default_rules` 相比是否变好

## 6. 本阶段验证结果

当前阶段已完成的验证包括：

- `pytest tests/test_trend_trading.py tests/test_trend_threshold_research.py -q`
- `pytest tests/test_cli.py -q`
- `pytest tests/test_trend_trading.py tests/test_trend_threshold_research.py tests/test_intraday_screening.py tests/test_update_fallback.py tests/test_watchlist.py tests/test_pattern_tradingview_scores.py -q`
- 最小真实数据工作区下的 `trend-entries` 冒烟验证
- 最小真实数据工作区下的 `research-thresholds` 冒烟验证

## 7. 当前阶段结论

可以认为本阶段已经完成以下转变：

- 从“固定一套全局默认阈值”升级为“可研究、可比较、可输出候选阈值”
- 从“所有 setup 共用一套默认门槛”升级为“支持按 `breakout/pullback` 分开阈值”
- 从“凭主观调整阈值”升级为“先做样本研究，再按结果更新规则”

## 8. 下一阶段建议

下一阶段最值得做的事情只有一件：

- 单独继续研究 `pullback` 阈值

理由很简单：

- `breakout` 当前已经找到一版比旧默认更强的规则
- `pullback` 目前还没有稳定优于旧默认

因此下一阶段应重点回答：

- `pullback` 是否应该继续用 `price_action_score` 做第二门槛
- `pullback` 是否应该改用 `trend_base_score` 或 `positive_indicator_count` 作为辅助门槛
- `pullback` 在 `10日` 与 `20日` 下是否需要不同阈值
