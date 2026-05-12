# 日中选股名单写入指导

本文用于 14:45 生成“日中选股名单”。它是日中参考清单，不替代盘后 [选股.md](/C:/Users/wdyab/Desktop/wdy/stocks/选股.md) 的最终写入规则。盘后正式更新仍以 [picks-writing-guide.md](/C:/Users/wdyab/Desktop/wdy/stocks/docs/picks-writing-guide.md) 为准。

日中名单的目标是：在当日尚未收盘、日 K 仍是临时数据的情况下，结合 [主线.md](/C:/Users/wdyab/Desktop/wdy/stocks/主线.md)、盘中 `intraday-screening` 结果和前一日 pattern 背景，整理出少量可以在尾盘或次日重点观察的股票。

## 使用范围

本指导适用于每个交易日 14:45 调用 GPT 生成日中选股名单。

如果 [选股-日中.md](/C:/Users/wdyab/Desktop/wdy/stocks/选股-日中.md) 中已经有对应日期的日中选股列表，必须先删除该日期已有列表并完全忽略旧内容，只根据最新盘中筛选结果、最新主线和本指导重新列出该日期日中选股列表。

必须读取或参考：

```text
主线.md
reports/intraday_screening/intraday_top20_YYYY-MM-DD.csv
reports/intraday_screening/intraday_top20_previous_YYYY-MM-DD.csv
reports/intraday_screening/intraday_screening_YYYY-MM-DD.csv
reports/intraday_screening/intraday_track_stock_YYYY-MM-DD.csv
track_stock.xlsx
```

`intraday_top20_previous_YYYY-MM-DD.csv` 是“上一轮 Top20/focus 股票池，在本次盘中数据下重新计算后的结果”。它不是旧文件原样复制，而是用于快速查看上一轮表现好的股票在本次盘中是否继续保持强度。

日中名单必须先确定本次参考源：

- 如果 `intraday_top20_previous_YYYY-MM-DD.csv` 不存在，说明没有上一轮 focus 缓存，直接使用 `intraday_top20_YYYY-MM-DD.csv`。
- 如果 `intraday_top20_YYYY-MM-DD.csv` 的文件修改时间晚于 `intraday_top20_previous_YYYY-MM-DD.csv`，说明新的全市场 Top20 已经重新生成，优先参考 `intraday_top20_YYYY-MM-DD.csv`。
- 否则优先参考 `intraday_top20_previous_YYYY-MM-DD.csv`，因为完整全市场名单可能尚未跑完，此时应先看上一轮 Top20 在本次盘中的最新表现。

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

### 2. 优先看可用的 Top20 参考源，不从全市场随意重挑

日中主表不直接从全市场随意重挑，而是先按文件时间确定一个“当前参考源”：

```text
如果 intraday_top20_YYYY-MM-DD.csv 比 intraday_top20_previous_YYYY-MM-DD.csv 更新：
    当前参考源 = intraday_top20_YYYY-MM-DD.csv
否则：
    当前参考源 = intraday_top20_previous_YYYY-MM-DD.csv
```

这样做是为了提高日中效率：完整全市场 Top20 重新生成可能较慢；在新 Top20 尚未更新完成时，优先查看上一轮 Top20/focus 股票在本次盘中数据下的表现。

当前参考源里的股票已经做过基础过滤，但过滤口径按来源分层：

- 非前日 pattern 背景票：`phase1_score_100 >= 40`、`phase2_score_100 >= 50`、`phase4_score_100 >= 70`
- 前日 pattern 背景票：取消 P1/P2 分数底线，只要求 `phase4_score_100 > 70`
- 所有来源都要求 `intraday_pct_change <= 8`
- 计算 `intraday_focus_score = phase4_score_100 + 0.08 * P1_center + 0.12 * P2_center`
- 其中 `P1_center = max(0, 100 - 2 * abs(P1 - 80))`，`P2_center = max(0, 100 - 2 * abs(P2 - 80))`
- 符合 P4 `> 70` 的前日 pattern 背景票优先保留，再用非 pattern 票按 `intraday_focus_score` 从高到低补足 20；同分时再看 `phase4_score_100 / phase1_center_score / phase2_center_score`

另外，`track_stock.xlsx` 的手动跟踪股票会额外并入 `intraday_top20_YYYY-MM-DD.csv`。这类行用 `track_stock = true` 和 `intraday_selection_source = track_stock` 或 `top20+track_stock` 标记。手动跟踪股是“必须点评的监测对象”，但不是自动买入名单。

全市场 `intraday_screening_YYYY-MM-DD.csv` 只用于补充解释、查看同主线其他股票，不能绕过当前参考源大量重挑。只有当当前参考源明显缺少主线内候选时，才可从全市场结果中少量补充观察对象，并必须说明这是补充观察，不是 Top20 主筛结果。

### 3. P1/P2 是非 pattern 票的盘中风险底线

`phase1_score_100` 和 `phase2_score_100` 都是 0-100 买入友好分，低分是风险，但不是越高越好。当前回测更偏好 P1/P2 在 `70-90`，尤其接近 `80` 的状态。

写日中名单时：

- 对非 pattern 背景票，P1 `>= 40`、P2 `>= 50` 是自动进入 Top20 的风险底线。
- 对前一日 pattern 背景票，P1/P2 不作为剔除条件；只要 P4 `> 70` 且涨幅不过热，可以进入日中主表候选，但必须写清风险分偏低意味着波动和止损压力更大。
- P1/P2 接近 `80`：优先。
- P1/P2 过高，例如都在 `90+`：不自动升为最高优先级，需要结合 Phase4 和主线判断。
- 非 pattern 背景票任一低于自动底线：不进入日中主表。
- 前一日 pattern 背景票如果 P1/P2 低于 30，不直接剔除，但要降低优先级，并在判断中强调只适合等回踩/确认。

### 4. 内部排序分只做参考，不写入表格

`phase4_score_100` 是横截面收益排序分，但日中 Phase4 Top 之间差距常常很小，不能单独替代风险质量判断。

日中自动排序内部使用：

```text
P1_center = max(0, 100 - 2 * abs(phase1_score_100 - 80))
P2_center = max(0, 100 - 2 * abs(phase2_score_100 - 80))
intraday_focus_score = phase4_score_100 + 0.08 * P1_center + 0.12 * P2_center
```

这个分数只用于生成和排序日中 Top20/focus 参考源，不是新的模型预测目标，也不写入最终日中选股表。

正确用法：

- 先按主线归类。
- 同一主线内，优先选择 P1/P2 更接近 `80`、Phase4 更高、涨幅不过热的股票。
- 同一候选池内，优先看 centered Top20 中同时进入五日 P4 均分 Top5 的股票；这类股票说明当日风险位置和 Phase4 连续强度同时较好。
- 如果两只股票 Phase4 接近，优先选 P1/P2 更接近 `80` 的一只。
- 主线不匹配时，即使 Phase4 或内部排序分很高，也不要压过主线内质量合格的股票。

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
- `建议总仓位%`
- `atr_stop_loss_2x`
- `atr_take_profit_2x`
- `prev_pattern_id`
- `prev_reason`

底背离、金叉、量价看多是加分项。顶背离、死叉、量价看空是降级项。ATR 止损空间过大、`建议总仓位%` 明显偏低时，即使分数高，也只能写成观察。

`建议总仓位%` 是按“单笔总最大亏损 2%、`D = 2ATR14`、第二批回撤 `D/2`、第三次加仓前有效平均止损距离 `0.85D`、单一标的最高 40%”推导出的计划满仓比例，不是第一批买入比例。公式为 `min(40, 100 * 0.02 * 当前价 / (0.85 * 2 * ATR14))`；如果 `ATR%` 用百分数数值表示，例如 `4` 表示 `4%`，等价于 `min(40, 117.65 / ATR%)`。

### 7. 手动跟踪股票必须单独总结

日中名单必须包含“手动跟踪股票总结”。数据优先来自 `intraday_track_stock_YYYY-MM-DD.csv`；如果该文件不存在，就在 `intraday_top20_YYYY-MM-DD.csv` 中筛选 `track_stock = true` 的行。

写法要求：

- 每只 `track_stock` 都要给出“适合日中关注 / 观察 / 不适合”的结论。
- 如果跟踪股不属于当前主线，默认降级为观察或不选。
- 如果不是前日 pattern 背景票，P1 `< 40` 或 P2 `< 50` 默认不进入日中主表，但仍要在跟踪总结里说明风险。
- 如果是前日 pattern 背景票，P1/P2 低分不直接剔除，但 P4 必须 `> 70`，且要说明“形态弹性较强但风险分偏低”。
- 如果盘中涨幅高于 8%，不进入日中主表，只写“涨幅过高，不追”。
- 如果跟踪股同时也是 Top20，说明它既在技术筛选内，也在手动跟踪列表内。

## 输出格式

日中名单建议单独输出，不直接覆盖 [选股.md](/C:/Users/wdyab/Desktop/wdy/stocks/选股.md)。若需要写入文件，应写成独立小节，等盘后再按 [picks-writing-guide.md](/C:/Users/wdyab/Desktop/wdy/stocks/docs/picks-writing-guide.md) 更新正式名单。

推荐格式：

```md
### YYYY.M.D 日中观察

日中选股名单：

| 股票代码 | 股票名称 | 主线 | P1/P2/P4 | P4五日均/std | 盘中涨幅 | 建议总仓位 | 技术/前日形态 |
| -------- | -------- | ---- | -------- | ------------- | -------: | -----------: | ------------- |
| 000000 | 示例股票 | 算力硬件 | 80 / 78 / 86 | 88 / 5 | 2.8% | 28.7% | 前日模式2 + 金叉 |

日中判断：

- 000000示例股票：可观察。属于当前主线，P1/P2 接近 80，涨幅未过热，尾盘重点看能否站稳关键位。

手动跟踪股票总结：

| 股票代码 | 股票名称 | 是否日中关注 | P1/P2/P4 | P4五日均/std | 盘中涨幅 | 建议总仓位 | 判断 |
| -------- | -------- | ------------ | -------- | ------------- | -------: | -----------: | ---- |
| 000000 | 示例股票 | 观察 | 55 / 48 / 62 | 62 / 18 | 1.5% | 21.2% | P2 低于自动入表底线，只观察是否放量转强。 |
```

## 写作要求

- 结果先于分析，先给表，再给简短说明。
- 日中选股表中不显示 `Focus` 或 `intraday_focus_score`。
- 日中选股表中把 P1、P2、P4 合并成一列，写成 `P1 / P2 / P4`。
- 日中选股表和手动跟踪股票总结中必须显示 `P4五日均/std`，紧跟在 `P1/P2/P4` 后面，写成 `均值 / 标准差`。
- 日中选股表中不放长判断，判断统一放在表格下方的 `日中判断` 短文本里。
- 每只股票的判断必须尽量短，直接回答：可买、观察、不追或不适合。
- 必须写“手动跟踪股票总结”；不要因为跟踪股没有进入日中主表就省略。
- 不需要单独写“主线判断”和“盘中状态”。
- 不需要写“上一轮重点股延续”。
- 不需要写“不选或降级”清单。
- 不要写“确定买入”“必涨”“下午拉升”这类确定性措辞。
- 不要把全市场高 P4 股票全部放进主表；非 pattern 背景票 P4 高但 P1/P2 一般，默认只能观察。
- 前一日 pattern 背景票可以绕过 P1/P2 底线，但必须有 P4 `> 70`，且不能忽略止损和仓位约束。
- 不要把前一日 pattern 写成今日 pattern。
- 如果主线内没有合格股票，可以少选或不选，不用用非主线股票补满数量。
- 最终盘后写入 [选股.md](/C:/Users/wdyab/Desktop/wdy/stocks/选股.md) 时，必须重新参考 [picks-writing-guide.md](/C:/Users/wdyab/Desktop/wdy/stocks/docs/picks-writing-guide.md)。
