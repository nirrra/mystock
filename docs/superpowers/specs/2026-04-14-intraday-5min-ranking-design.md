# Intraday 5 分钟排序增强设计

## 1. 背景与目标

当前项目的 `intraday-screening` 已经能基于上一交易日 `watchlist` 读取候选股，只对候选股刷新数据，并生成三份日线侧结果：

- `tradingview`
- `divergence`
- `pattern`

但它目前仍然缺少一个真正面向盘中执行的排序层。现有产物更像“把候选股再跑一遍日线技术分析”，而不是“根据当天盘中的实际强弱给 watchlist 重新排优先级”。这会导致一个实际问题：同一批候选股在日线层面都不错，但到了盘中，强弱分化往往已经出现，系统却没有给出统一、可比较的盘中排序结果。

本次设计的目标是，在不推翻现有 `intraday-screening` 主流程的前提下，为 `watchlist` 候选股补一层基于 5 分钟线的盘中评分，并与现有日线结果一起汇总成一份新的排序 CSV。最终结果用于回答这个问题：

在当前交易日内，这批候选股里，哪些股票的盘中状态更值得优先关注。

## 2. 问题定义

当前 `intraday-screening` 的行为是：

1. 读取最近一期 `watchlist`
2. 对候选股逐只执行 `update`
3. 生成 `tradingview`、`divergence`、`pattern` 三份结果
4. 写一个只包含路径和基础元数据的 JSON 报告

也就是说，它的输出中没有：

- 单独的 `daily_score`
- 单独的 `intraday_5m_score`
- 基于盘中 5 分钟线的事件命中明细
- 一个可以直接用于盘中排序和比对的汇总 CSV

用户希望补齐这层能力，但要求边界明确：

- `daily_score` 不重新设计，直接复用项目现有的日线打分公式
- `intraday_5m_score` 单独计算，只基于本交易日 5 分钟线
- 最终输出一个新的 CSV
- CSV 中既要有现有日线结果，也要列出 5 分钟事件是否命中
- 排序按 `intraday_5m_score` 而不是按综合混合分

## 3. 目标边界

### 3.1 目标

- 在 `intraday-screening` 末尾新增一份盘中排序 CSV
- `daily_score` 直接复用现有 `watchlist` 打分逻辑
- 新增 `intraday_5m_score`，只看本交易日 5 分钟线
- 盘中分数由以下事件组成：
  - 成交量背离
  - MACD 背离
  - 金叉死叉
  - 是否重新站上均线或回踩均线
- CSV 中保留每类事件的命中情况和类型
- 最终按 `intraday_5m_score` 降序排序
- JSON 报告中新增该 CSV 路径

### 3.2 非目标

- 不改写现有 `daily-screening` 主流程
- 不调整 `watchlist` 的生成逻辑或分层逻辑
- 不把日线和 5 分钟线揉成单一黑盒总分
- 不在第一版引入分钟级实时流或持续刷新机制
- 不扩展新的日线特征公式
- 不把 5 分钟评分接入自动下单或自动决策链路

## 4. 方案比较

### 4.1 在 `intraday-screening` 末尾新增汇总排序步骤

做法是在现有 `intraday-screening` 跑完三份日线结果后，新增一步：

- 读取 `pattern/tradingview/divergence` 结果
- 拉取本交易日 5 分钟线
- 计算 `intraday_5m_score`
- 生成排序 CSV
- 回写 JSON 报告路径

优点：

- 对现有命令行为影响最小
- 与当前项目结构最贴合
- 容易测试
- 风险集中在新模块，不会破坏原有日线逻辑

缺点：

- 盘中评分逻辑是新增层，不属于现有技术结果模块的一部分

结论：

这是推荐方案。

### 4.2 为 5 分钟线新增独立报告子命令

做法是新增类似 `tradingview` / `divergence` 的 5 分钟报告命令，再由 `intraday-screening` 统一汇总。

优点：

- 结构更“对称”
- 后续继续扩展盘中功能时更清晰

缺点：

- 第一版成本更高
- 命令接口和输出目录会更复杂
- 当前需求只是盘中排序，不需要提前抽象到这个程度

结论：

暂不采用。

### 4.3 建立完全配置化的评分引擎

做法是把日线与 5 分钟线的所有事件、阈值和权重都抽象成通用配置。

优点：

- 后续调权灵活

缺点：

- 当前需求过小，不值得引入更高复杂度
- 配置层和调试成本会明显上升

结论：

第一版不采用。

## 5. 总体设计

本次采用“保留原有盘中复筛主流程，在末尾增加盘中排序汇总”的设计。

整体数据流如下：

`watchlist -> update(日线刷新) -> tradingview/divergence/pattern -> 5分钟线抓取 -> intraday评分汇总 -> intraday排序CSV + JSON报告补充`

这里有两个分数：

- `daily_score`
- `intraday_5m_score`

其中：

- `daily_score` 只负责表达这只股票现有日线技术面本身的稳定度
- `intraday_5m_score` 只负责表达这只股票在本交易日盘中的即时状态

最终排序规则固定为：

1. `intraday_5m_score` 降序
2. `daily_score` 降序
3. `symbol` 升序

这样做的含义很明确：盘中强弱优先，日线质量作为次级参考，而不是主导当天排序。

## 6. 分数设计

### 6.1 `daily_score`

`daily_score` 不重新设计，也不再额外引入新公式。它直接复用当前 `watchlist` 使用的稳定度打分逻辑，也就是 `watchlist.py` 中的 `_stable_score(...)`。

该分数当前由以下信息派生：

- `tradingview_avg_all_rating_5d`
- 最近一个 TradingView 日评分
- `pattern_id`
- `tradingview_all_rating_label`

这意味着本次改动不会重新解释“日线分数是什么”，只是在 `intraday-screening` 汇总时复用这套既有口径，把它显式写进新的排序 CSV。

### 6.2 `intraday_5m_score`

`intraday_5m_score` 是一个独立的 5 分钟线事件加权分，以 `50` 作为中性基准，最后截断到 `[0, 100]`。

公式为：

```text
intraday_5m_score =
clip(
  50
  + intraday_volume_score
  + intraday_macd_divergence_score
  + intraday_macd_cross_score
  + intraday_ma_score,
  0,
  100
)
```

这个设计意味着：

- 没有明显盘中优势或劣势时，分数会靠近 50
- 多个偏强信号叠加时，分数上升
- 多个偏弱信号叠加时，分数下降
- 最终值足够直观，便于排序和人工判断

## 7. 5 分钟事件定义与权重

### 7.1 数据范围

第一版只使用“本交易日内”的 5 分钟线数据，不跨日拼接历史分钟线。

原因有两个：

- 用户明确希望盘中评分反映当日状态
- 跨日拼接会引入隔夜跳空和交易节奏变化，使第一版事件口径更难解释

若当日 5 分钟线数据不足以完成某项事件判断，则该事件视为未命中，对应分项分数记为 `0`。

### 7.2 成交量背离

成交量背离采用最近两个价格枢轴点比较的方式，而不是只比较单根 K 线。这样可以降低盘中噪声。

定义：

- `bullish`
  - 最近两个低点中，价格创新低
  - 但对应枢轴附近平均成交量下降，说明下跌过程中的抛压减弱
- `bearish`
  - 最近两个高点中，价格创新高
  - 但对应枢轴附近平均成交量下降，说明上冲过程中的跟随量不足

权重：

- `bullish`：`+12`
- `bearish`：`-15`

输出字段：

- `intraday_volume_divergence_hit`
- `intraday_volume_divergence_type`
- `intraday_volume_score`

### 7.3 MACD 背离

MACD 背离沿用现有日线背离模块的判定思路，但把输入改成 5 分钟线，并使用分钟时间戳作为时间字段。

定义：

- `bullish`
  - 最近两个低点中，价格更低
  - 但 `macd_dif` 更高
- `bearish`
  - 最近两个高点中，价格更高
  - 但 `macd_dif` 更低

权重：

- `bullish`：`+20`
- `bearish`：`-20`

输出字段：

- `intraday_macd_divergence_hit`
- `intraday_macd_divergence_type`
- `intraday_macd_divergence_score`

### 7.4 金叉死叉

金叉死叉只统计最近 `3` 根 5 分钟 K 线内的新发生事件，避免较早信号对当前盘中状态产生过强影响。

定义：

- `golden_cross`
  - `macd_dif` 从下向上穿越 `macd_dea`
- `death_cross`
  - `macd_dif` 从上向下穿越 `macd_dea`
- 若交叉之后最近几根 `macd_hist` 沿同方向继续增强，则视为“交叉后延续”

权重：

- `golden_cross`：`+12`
- `golden_cross_continuation`：额外 `+6`
- `death_cross`：`-12`
- `death_cross_continuation`：额外 `-6`

第一版输出里，类型字段只保留一个最终结果值：

- `golden_cross`
- `golden_cross_continuation`
- `death_cross`
- `death_cross_continuation`
- `none`

输出字段：

- `intraday_macd_cross_hit`
- `intraday_macd_cross_type`
- `intraday_macd_cross_score`

### 7.5 重新站上均线 / 回踩均线

第一版只使用短周期均线：

- `ma_5`
- `ma_10`

定义：

- `reclaim_ma`
  - 前一到两根收盘价还位于短均线组下方
  - 最新一根重新站上 `ma_5` 与 `ma_10`
- `pullback_hold_ma`
  - 股票先运行在短均线组上方
  - 近几根回踩到 `ma_5` 或 `ma_10` 附近
  - 最新一根未有效跌破，并重新转强
- `break_ma`
  - 最新一根有效跌破 `ma_5` 与 `ma_10`

权重：

- `reclaim_ma`：`+12`
- `pullback_hold_ma`：`+15`
- `break_ma`：`-12`

输出字段：

- `intraday_ma_event_hit`
- `intraday_ma_event_type`
- `intraday_ma_score`

## 8. 输出设计

### 8.1 新增 CSV

新增输出文件：

`reports/intraday_screening/<trade_date>/intraday_rank_<trade_date>.csv`

该 CSV 仅包含本次 `intraday-screening` 处理的候选股。

第一版固定输出字段如下：

- `rank`
- `symbol`
- `name`
- `pattern_ids`
- `tradingview_avg_all_rating_5d`
- `tradingview_all_rating_label`
- `daily_macd_top_divergence_15d`
- `daily_macd_bottom_divergence_15d`
- `daily_score`
- `intraday_5m_score`
- `intraday_volume_divergence_hit`
- `intraday_volume_divergence_type`
- `intraday_volume_score`
- `intraday_macd_divergence_hit`
- `intraday_macd_divergence_type`
- `intraday_macd_divergence_score`
- `intraday_macd_cross_hit`
- `intraday_macd_cross_type`
- `intraday_macd_cross_score`
- `intraday_ma_event_hit`
- `intraday_ma_event_type`
- `intraday_ma_score`

排序顺序：

- 按 `intraday_5m_score` 降序
- 再按 `daily_score` 降序
- 再按 `symbol` 升序

### 8.2 JSON 报告

现有 `intraday_screening_<trade_date>.json` 结构保留，在其基础上新增：

- `intraday_rank_path`

## 9. 模块变更范围

### 9.1 `src/stocks_analyzer/data_sources/baostock_provider.py`

- 复用现有 `get_intraday_bars(...)`
- 确保 5 分钟线查询参数由新汇总模块正确传入

### 9.2 新增盘中评分模块

新增独立模块：

- `src/stocks_analyzer/intraday_ranking.py`

职责：

- 拉取单只股票本交易日 5 分钟线
- 计算分钟级指标
- 判定 5 分钟事件
- 计算 `intraday_5m_score`
- 汇总日线结果与盘中结果
- 生成最终排序 DataFrame

之所以单独建模块，而不是把逻辑直接塞进 `cli.py`，是因为盘中评分已经形成独立职责，继续堆在 CLI 文件里会让入口代码进一步膨胀。

### 9.3 `src/stocks_analyzer/cli.py`

- 在 `_run_intraday_screening(...)` 中，在现有 `tradingview/divergence/pattern` 之后追加调用新的盘中汇总步骤
- 生成 `intraday_rank_<date>.csv`
- 在 JSON 报告中写入新路径

### 9.4 复用现有模块

- 复用 `watchlist.py` 中的 `_stable_score(...)` 作为 `daily_score`
- 复用 `macd_divergence.py` 的整体背离检测思路
- 复用 `indicators.py` 中已有的 MACD 计算方式，但时间列需适配 5 分钟线

## 10. 错误处理与降级行为

需要明确以下降级规则：

- 若单只股票 5 分钟线拉取失败：
  - 不中断整个 `intraday-screening`
  - 该股票的分钟事件全部记为未命中
  - `intraday_5m_score` 记为中性值 `50`
  - 在日志中输出 warning
- 若单只股票 5 分钟线条数不足：
  - 无法判定的事件记为未命中，分项记为 `0`
- 若日线侧某份报告缺字段或缺行：
  - 继续使用可获得字段
  - 缺失的日线分数项按现有逻辑保守处理

这里的原则是：盘中排序增强属于附加层，不应因为部分分钟数据异常让整轮复筛失败。

## 11. 测试设计

至少补充以下测试：

### 11.1 汇总路径测试

验证 `intraday-screening` 在原有三份 CSV 之外，会新增生成 `intraday_rank_<date>.csv`，并把路径写入 JSON 报告。

### 11.2 `daily_score` 复用测试

构造样例行，验证 `intraday-screening` 中写出的 `daily_score` 与 `watchlist.py` 现有 `_stable_score(...)` 结果一致，避免口径漂移。

### 11.3 5 分钟事件判定测试

分别为以下事件构造最小数据样例：

- 成交量背离 `bullish`
- 成交量背离 `bearish`
- MACD 底背离
- MACD 顶背离
- 金叉 / 死叉
- 重新站上均线
- 回踩均线后企稳
- 跌破均线

验证类型字段与分项分数都正确。

### 11.4 排序测试

构造多只股票的模拟结果，验证最终 CSV 的排序规则为：

1. `intraday_5m_score` 降序
2. `daily_score` 降序
3. `symbol` 升序

### 11.5 异常降级测试

模拟 5 分钟线拉取失败，验证：

- 流程不中断
- 该股票 `intraday_5m_score = 50`
- 事件字段为未命中

## 12. 风险与取舍

第一，5 分钟线本身噪声更大，尤其是在盘中前半小时和午后尾盘。即使使用枢轴点和最近若干根 K 线确认，事件仍然比日线更容易反复。这是分钟级分析天然存在的代价，因此第一版不追求“绝对精准”，而追求“同一批候选股之间有一致的盘中比较口径”。

第二，`intraday_5m_score` 当前是事件加权分，不是回测校准后的统计分数。因此它更适合做排序参考，而不是做阈值化的机械买卖决策。用户在使用时应把它理解为“盘中强弱相对分”，而不是“收益概率”。

第三，`daily_score` 继续沿用现有 `watchlist` 公式，优点是口径稳定，缺点是它本身并不是为了盘中排序而设计的。但这正符合本次范围：先复用既有日线质量分，不在本次改动中重构日线评分体系。

## 13. 验收标准

本次设计完成后的验收标准如下：

- `intraday-screening` 在原有产物基础上新增一份盘中排序 CSV
- CSV 中包含用户要求的日线字段、两类分数与 5 分钟事件命中情况
- `daily_score` 复用现有 `watchlist` 打分公式，不另起口径
- `intraday_5m_score` 基于本交易日 5 分钟线，按确认的事件与权重计算
- 最终排序按 `intraday_5m_score` 降序
- JSON 报告中新增新 CSV 路径
- 新增测试覆盖评分、事件命中、排序和异常降级行为
