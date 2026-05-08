# 日中选股名单写入指导

本文用于 14:45 生成“日中选股名单”。它是日中参考清单，不替代盘后 [选股.md](/C:/Users/wdyab/Desktop/wdy/stocks/选股.md) 的最终写入规则。盘后正式更新仍以 [picks-writing-guide.md](/C:/Users/wdyab/Desktop/wdy/stocks/docs/picks-writing-guide.md) 为准。

日中名单的目标是：在当日尚未收盘、日 K 仍是临时数据的情况下，结合 [主线.md](/C:/Users/wdyab/Desktop/wdy/stocks/主线.md)、盘中 `intraday-screening` 结果和前一日 pattern 背景，整理出少量可以在尾盘或次日重点观察的股票。

## 使用范围

本指导适用于每个交易日 14:45 调用 GPT 生成日中选股名单。

必须读取或参考：

```text
主线.md
reports/intraday_screening/intraday_top20_YYYY-MM-DD.csv
reports/intraday_screening/intraday_top20_previous_YYYY-MM-DD.csv
reports/intraday_screening/intraday_screening_YYYY-MM-DD.csv
reports/intraday_screening/intraday_track_stock_YYYY-MM-DD.csv
track_stock.xlsx
```

如果 `intraday_top20_previous_YYYY-MM-DD.csv` 不存在，说明没有上一轮 focus 缓存，直接使用全市场结果和 `intraday_top20`。

## 数据口径

`intraday-screening` 使用 `data/intraday` 下的盘中临时日 K，不写入 `data/daily`。它只跑 MACD、ATR、Phase1、Phase2、Phase4，不跑 Phase5、Phase7，也不重新识别 pattern。

因此日中名单必须遵守：

- 盘中数据是临时参考，不写成“确认收盘形态”。
- `prev_pattern_*` 只能作为前一日形态背景，不写成“今日新命中 pattern”。
- 没有 Phase7，当天交易环境只能结合主线和盘中强弱描述，不补写 Phase7 许可。
- 没有 Phase5，不做长周期极端风险判断；盘后正式名单再补。
- `intraday_pct_change` 是盘中涨幅。涨幅超过 8% 的股票不进入日中主表。

## 选股原则

### 1. 主线优先

选股时必须先看 [主线.md](/C:/Users/wdyab/Desktop/wdy/stocks/主线.md)。

优先级：

1. 当前第一梯队主线中的股票
2. 当前第二梯队主线中的股票
3. 主线延伸、轮动和预期差方向
4. 仅技术面强、但无法归入主线的股票

如果股票技术分很高，但与当前主线无关，默认降级为“观察”，不要放在日中主表前列。

### 2. 从 Top20 开始，不从全市场随意重挑

日中主表优先从 `intraday_top20_YYYY-MM-DD.csv` 里选。这个文件已经做过基础过滤：

- `phase1_score_100 > 40`
- `phase2_score_100 > 40`
- `intraday_pct_change <= 8`
- 按 `phase4_score_100` 从高到低取前 20

另外，`track_stock.xlsx` 的手动跟踪股票会额外并入 `intraday_top20_YYYY-MM-DD.csv`。这类行用 `track_stock = true` 和 `intraday_selection_source = track_stock` 或 `top20+track_stock` 标记。手动跟踪股是“必须点评的监测对象”，但不是自动买入名单。

全市场 `intraday_screening_YYYY-MM-DD.csv` 只用于补充解释、查看同主线其他股票，不能绕过 Top20 大量重挑。

### 3. P1/P2 是盘中风险底线

`phase1_score_100` 和 `phase2_score_100` 都是 0-100 买入友好分，越高越安全。

写日中名单时：

- 两项都高于 60：优先。
- 40 到 60：可以进入观察，但推荐理由必须更依赖主线和技术触发。
- 任一不高于 40：不进入日中主表。
- 任一低于 30：默认不写入名单，只能在风险备注里点到。

### 4. Phase4 只在同主线内排序

`phase4_score_100` 是横截面收益排序分，不能单独替代主线判断。

正确用法：

- 先按主线归类。
- 同一主线内，优先选择 `phase4_score_100` 更高、P1/P2 更稳、涨幅不过热的股票。
- 主线不匹配时，即使 Phase4 很高，也不要压过主线内质量合格的股票。

### 5. 涨幅约束

盘中涨幅是追高风险的核心约束。

- `intraday_pct_change <= 3%`：更适合观察低吸或确认转强。
- `3% < intraday_pct_change <= 5%`：可观察，但要说明不宜追高。
- `5% < intraday_pct_change <= 8%`：只在主线很强、技术触发很明确时列入，必须提示尾盘回落风险。
- `intraday_pct_change > 8%`：不进入日中主表。

### 6. 技术状态只做辅助确认

优先看这些字段：

- `macd_cross_state`
- `macd_divergence_state`
- `volume_price_divergence_state`
- `macd_bottom_divergence_15d`
- `macd_top_divergence_15d`
- `atr_14`
- `atr_pct_14`
- `atr_stop_loss_2x`
- `atr_take_profit_2x`
- `prev_pattern_id`
- `prev_reason`

底背离、金叉、量价看多是加分项。顶背离、死叉、量价看空是降级项。ATR 止损空间过大时，即使分数高，也只能写成观察。

### 7. 手动跟踪股票必须单独总结

日中名单必须包含“手动跟踪股票总结”。数据优先来自 `intraday_track_stock_YYYY-MM-DD.csv`；如果该文件不存在，就在 `intraday_top20_YYYY-MM-DD.csv` 中筛选 `track_stock = true` 的行。

写法要求：

- 每只 `track_stock` 都要给出“适合日中关注 / 观察 / 不适合”的结论。
- 如果跟踪股不属于当前主线，默认降级为观察或不选。
- 如果 P1/P2 任一不高于 40，默认不进入日中主表，但仍要在跟踪总结里说明风险。
- 如果盘中涨幅高于 8%，不进入日中主表，只写“涨幅过高，不追”。
- 如果跟踪股同时也是 Top20，说明它既在技术筛选内，也在手动跟踪列表内。

## 输出格式

日中名单建议单独输出，不直接覆盖 [选股.md](/C:/Users/wdyab/Desktop/wdy/stocks/选股.md)。若需要写入文件，应写成独立小节，等盘后再按 [picks-writing-guide.md](/C:/Users/wdyab/Desktop/wdy/stocks/docs/picks-writing-guide.md) 更新正式名单。

推荐格式：

```md
### YYYY.M.D 日中观察

主线判断：一句话说明当前主线是否延续、是否有切换迹象。
盘中状态：一句话说明候选强弱、涨幅分布和追高风险。

日中选股名单：

| 股票代码 | 股票名称 | 主线 | P1/P2 | P4 | 盘中涨幅 | 技术/前日形态 | 日中判断 |
| -------- | -------- | ---- | ----- | --: | -------: | ------------- | -------- |
| 000000 | 示例股票 | 算力硬件 | 72 / 66 | 86 | 2.8% | 前日模式2 + 金叉 | 属于当前主线，P1/P2 合格，P4 排序靠前，涨幅未过热，尾盘重点看能否站稳关键位。 |

手动跟踪股票总结：

| 股票代码 | 股票名称 | 是否日中关注 | P1/P2 | P4 | 盘中涨幅 | 判断 |
| -------- | -------- | ------------ | ----- | --: | -------: | ---- |
| 000000 | 示例股票 | 观察 | 55 / 48 | 62 | 1.5% | 在跟踪列表内但不属于当前第一主线，P1/P2 仅中等，尾盘只观察是否放量转强。 |

不选或降级：

- 000000示例股票：P4 很高但不属于当前主线，且盘中涨幅接近 8%，只观察不追。
```

## 写作要求

- 结果先于分析，先给表，再给简短说明。
- 每只股票的判断必须同时回答：属于什么主线、风险分是否合格、是否过热、技术触发是什么。
- 必须写“手动跟踪股票总结”；不要因为跟踪股没有进入日中主表就省略。
- 不要写“确定买入”“必涨”“下午拉升”这类确定性措辞。
- 不要把全市场高 P4 股票全部放进主表。
- 不要把前一日 pattern 写成今日 pattern。
- 如果主线内没有合格股票，可以少选或不选，不用用非主线股票补满数量。
- 最终盘后写入 [选股.md](/C:/Users/wdyab/Desktop/wdy/stocks/选股.md) 时，必须重新参考 [picks-writing-guide.md](/C:/Users/wdyab/Desktop/wdy/stocks/docs/picks-writing-guide.md)。
