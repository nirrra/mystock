# Daily Screening Trend Filter Design

## 1. 目标

本设计的目标是在不推翻现有 `daily-screening -> pattern -> watchlist` 体系的前提下，引入一条独立的全市场趋势评分链路，并把它作为 `watchlist` 生成前的一道新增初筛。

本次改动要解决的问题是：

- 现有 `watchlist` 主要基于旧的 `pattern + TradingView + stable_score` 体系筛出候选
- 项目已经有一套更面向趋势交易的 `buy_score / price_action_score` 评分体系
- 需要把这套新评分并入 `daily-screening`，但不直接推翻旧规则

本次设计的最终口径是：

- 保留原有所有筛选条件
- 新增 `trend` 指令，按指定交易日对全市场输出趋势评分结果
- `daily-screening` 在生成 `watchlist` 前读取 `trend` 结果
- 只有同时满足“旧体系通过 + 新趋势宽松阈值通过”的股票，才进入最终 `watchlist`

## 2. 非目标

本次设计明确不做以下事情：

- 不重写现有 `pattern` 识别逻辑
- 不移除旧的 `history_momentum_filter`
- 不让 `MACD`、量价背离、金叉死叉参与第一版硬筛选
- 不重写 `watchlist` 的主排序逻辑
- 不把 `daily-screening` 直接替换为 `trend-entries`
- 不修改 `选股.md` 的人工整理流程

## 3. 已确认的设计决策

本次需求的已确认决策如下：

- 新增一条独立的 `trend` 指令
- `trend` 对全市场打分，不只处理 `pattern` 候选
- `daily-screening` 读取 `trend` 输出，并与旧 `pattern/watchlist` 候选做严格交集
- 若某只股票没有对应的 `trend` 评分记录，则直接剔除
- 第一版硬筛选只看：
  - `buy_score`
  - `price_action_score`
- `MACD`、量价背离、金叉死叉只作为展示字段输出
- 第一版阈值使用统一全局配置，后续再扩展为按 `breakout/pullback` 分开
- `watchlist` 的主排序仍保持：
  - `tier -> stable_score -> tradingview_avg_5d`

## 4. 现有体系回顾

当前 `daily-screening` 的主流程为：

`update -> tradingview -> divergence -> pattern -> watchlist`

其中现有的“旧体系”实际上已经包含多层初筛：

- 全局动量初筛：`history_momentum_filter`
  - 默认规则是最近 `200` 个交易日中至少出现过一个 `5` 日窗口涨幅 `>= 10%`
- 各 `type1~type4` 自身的趋势、涨幅、位置、量能、整理形态条件
- `watchlist` 生成规则
  - `TradingView` 标签过滤
  - 不同 `pattern_id` 的 5 日均分阈值
  - `stable_score`
  - `tier`

因此，本次设计不是“新增第一层初筛”，而是在旧体系之后新增一层“趋势质量复核”。

## 5. 总体架构

建议将新流程改为：

`update -> tradingview -> divergence -> pattern -> trend -> merge -> watchlist`

各模块职责如下：

- `daily-screening`
  - 仍负责串行调度
  - 不直接计算趋势分数
- `pattern`
  - 继续按旧逻辑输出当日技术候选
- `trend`
  - 对全市场输出指定日期的趋势评分表
- `watchlist` 构建阶段
  - 读取 `pattern` 候选
  - 读取 `trend` 评分表
  - 按 `symbol` 做严格交集
  - 应用新增宽松阈值
  - 生成最终 `watchlist`

这种拆分的核心原则是：

- 旧体系负责“形态入围”
- 新体系负责“趋势质量复核”
- `watchlist` 只保留两套体系都认可的候选

## 6. `trend` 指令设计

### 6.1 职责

新增 `trend` 指令，作用是为指定交易日输出全市场趋势评分结果，而不是直接生成买入名单。

推荐语义：

```bash
python -m stocks_analyzer --project-root . trend --date 2026-04-18
```

### 6.2 输入

- 指定交易日
- 本地主板股票 universe
- 本地日线数据
- 现有趋势评分配置

### 6.3 输出

建议输出：

- `reports/trend/trend_YYYY-MM-DD.csv`
- 可选：`reports/trend/trend_YYYY-MM-DD.json`

CSV 至少包含以下字段：

- `trade_date`
- `symbol`
- `name`
- `signal_type`
- `trend_base_score`
- `price_action_score`
- `macd_score`
- `buy_score`
- `positive_indicator_count`
- `trigger_reason`
- `buy_reason`

展示增强字段应包含：

- `macd_cross_state`
- `macd_divergence_state`
- `volume_price_divergence_state`

必要时也可以保留布尔字段，便于程序判断：

- `macd_top_divergence_flag`
- `macd_bottom_divergence_flag`
- `bullish_volume_price_divergence_flag`
- `bearish_volume_price_divergence_flag`

## 7. 新增展示指标定义

第一版里，以下指标只用于展示，不参与硬筛选：

- `macd_score`
- `MACD` 金叉死叉状态
- `MACD` 顶背离/底背离状态
- 量价背离状态

建议将其语义做明确拆分：

- `macd_score`
  - 连续型评分，反映 `MACD` 状态强弱
- `macd_cross_state`
  - 离散标签，例如：
    - `golden_cross`
    - `dead_cross`
    - `above_signal`
    - `below_signal`
- `macd_divergence_state`
  - 离散标签，例如：
    - `top_divergence`
    - `bottom_divergence`
    - `none`
- `volume_price_divergence_state`
  - 离散标签，例如：
    - `bullish`
    - `bearish`
    - `none`

这样处理的原因是：

- 连续评分适合后续研究
- 离散标签更适合 `watchlist` 展示和人工阅读
- 可以避免第一版把过多指标直接硬编码进筛选规则

## 8. `daily-screening` 合并逻辑

### 8.1 新执行顺序

`daily-screening` 建议改为以下顺序：

1. 判断交易日
2. `update`
3. `tradingview`
4. `divergence`
5. `pattern`
6. `trend`
7. 读取 `pattern` 候选
8. 读取 `trend` 评分表
9. 做严格交集并应用趋势宽松阈值
10. 输出最终 `watchlist`
11. 写入运行摘要

### 8.2 合并规则

合并键：

- `trade_date`
- `symbol`

实际实现中若文件本身已经按日期隔离，则核心键可以只用 `symbol`，但对外语义仍应视为“同一交易日的同一股票”。

最终进入 `watchlist` 的条件是：

- 股票先通过旧体系
- 股票存在对应的 `trend` 评分记录
- `buy_score >= watchlist_trend_filter.buy_score_min`
- `price_action_score >= watchlist_trend_filter.price_action_score_min`

也就是说：

`final_watchlist = old_watchlist_candidates AND trend_record_exists AND loose_trend_threshold_passed`

### 8.3 无趋势记录时的行为

已确认采用严格交集：

- 没有 `trend` 记录的股票直接剔除
- 不允许旧链路兜底保留

这保证最终 `watchlist` 口径清晰，不会出现“有些票经过趋势复核，有些票没有”的混合状态。

## 9. 阈值配置设计

第一版建议新增独立配置段，例如：

```yaml
watchlist_trend_filter:
  enabled: false
  buy_score_min: 70.0
  price_action_score_min: 55.0
```

第一版明确口径如下：

- 默认 `enabled: false`
- 默认全局宽松阈值使用：
  - `buy_score_min: 70.0`
  - `price_action_score_min: 55.0`

这样处理的原因是：

- 功能上线初期先默认关闭，避免直接改变现有 `watchlist` 结果
- 一旦启用，这组阈值明显宽于现有 `trend_entry_rules`
- 它更适合作为 `watchlist` 的“趋势质量复核”而不是最终买入门槛

设计要求是：

- 第一版使用统一全局阈值
- 代码结构预留二阶段扩展：

```yaml
watchlist_trend_filter:
  enabled: false
  buy_score_min: 70.0
  price_action_score_min: 55.0
  breakout: {}
  pullback: {}
```

第一版不启用 `breakout/pullback` 分开阈值，但需要避免未来扩展时重构配置结构。

## 10. `watchlist` 输出设计

### 10.1 准入逻辑

`watchlist` 准入逻辑新增趋势宽松复核，但主排序逻辑第一版不改。

### 10.2 排序逻辑

第一版排序继续保持：

- `tier`
- `stable_score`
- `tradingview_avg_5d`

原因是：

- 本次核心目标是把趋势评分作为新增准入筛
- 不是立刻重写 `watchlist` 排序体系
- 若同时修改准入和排序，回溯变化来源会变得困难

### 10.3 新增展示字段

写入 `watchlist` 候选项的字段建议增加：

- `buy_score`
- `price_action_score`
- `trend_base_score`
- `macd_score`
- `macd_cross_state`
- `macd_divergence_state`
- `volume_price_divergence_state`
- `positive_indicator_count`
- `signal_type`
- `trigger_reason`
- `buy_reason`

这样 `watchlist` 既保留现有技术候选语义，也能直接携带趋势复核结果。

## 11. 错误处理

以下情况需要定义明确行为：

### 11.1 `trend` 输出不存在

- `daily-screening` 运行 `trend` 后若未生成对应文件，应直接报错退出
- 不允许静默降级回旧 `watchlist`

### 11.2 `trend` 输出缺少关键列

若缺少以下字段之一，应报错：

- `symbol`
- `buy_score`
- `price_action_score`

### 11.3 合并后候选为空

若旧体系候选不为空，但严格交集后为空：

- 允许生成空 `watchlist`
- 运行摘要中应明确记录为空的候选数
- 后续可在日志中提示是趋势复核导致全部剔除

### 11.4 非交易日

- 保持现有逻辑
- 非交易日直接跳过，不运行后续链路

## 12. 测试要求

至少补以下测试：

- `trend` CLI 参数解析与输出路径测试
- `trend` 结果表字段完整性测试
- `daily-screening` 会调用 `trend` 的流程测试
- `watchlist` 合并阶段的严格交集测试
- 缺失 `trend` 记录时候选被剔除的测试
- `buy_score / price_action_score` 宽松阈值过滤测试
- `MACD` / 量价背离 / 金叉死叉字段透传测试
- 合并后空 `watchlist` 的测试

另外建议补一组最小真实数据冒烟验证：

- 单独运行 `trend --date ...`
- 运行 `daily-screening --date ...`
- 检查最终 `watchlist` 中是否带有新增趋势字段

## 13. 分阶段实施建议

建议按以下顺序落地：

### 阶段 1

- 新增 `trend` 指令
- 产出全市场趋势评分表
- 先确保字段和输出稳定

### 阶段 2

- 在 `daily-screening` 中接入 `trend`
- 对 `watchlist` 生成逻辑加入严格交集和宽松阈值

### 阶段 3

- 将 `MACD`、量价背离、金叉死叉字段完整写入 `watchlist`
- 完成 CLI、单测、冒烟验证

### 阶段 4

- 评估是否需要把 `buy_score / price_action_score` 接入排序层
- 评估是否需要为 `breakout/pullback` 拆分阈值

## 14. 结论

本设计的核心不是用新趋势链路替代旧 `daily-screening`，而是在旧体系后新增一层趋势质量复核。

最终形成的口径是：

- 旧体系负责形态候选入围
- `trend` 负责全市场趋势评分
- `watchlist` 只保留“旧体系通过且趋势宽松初筛通过”的股票

这保证了系统能同时利用：

- 旧 `pattern/watchlist` 体系的稳定性
- 新 `buy_score / price_action_score` 体系的趋势质量识别能力

同时又把 `MACD`、背离、金叉死叉先控制在展示层，避免第一版把规则复杂度推得过高。
