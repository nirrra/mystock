# Daily Trend Trading Design

## 1. 目标

将项目从“日线技术结果 + watchlist”扩展为一条独立的“日 K 趋势交易研究”链路，但不做自动下单。

本设计的第一阶段目标只有两件事：

- 独立定义并产出 `trend universe`，回答“哪些股票属于可交易的日 K 趋势股”
- 在 `trend universe` 之上定义 `breakout` / `pullback` 两类入场信号，并完成基线回测

第一阶段输出的重点是统计结论，不是实盘执行。核心问题是：

- 趋势股能否被稳定定义
- 哪类日 K 买点更有效
- `5/10/20/40` 个交易日中，哪个持有周期的胜率和收益分布更稳定

## 2. 非目标

第一阶段明确不做以下能力：

- 自动交易执行
- 盘中级别择时和成交模拟
- 预设止盈止损规则
- 动态仓位管理
- 行业暴露和风险平价控制
- 替代或重写现有 `pattern/watchlist` 链路

现有 `pattern`、`watchlist`、`intraday-screening` 继续保留原语义。趋势交易链路独立存在，不与其混写。

## 3. 核心原则

### 3.1 趋势股识别独立于 Pattern 识别

`pattern` 用于识别某种形态；`trend universe` 用于识别股票是否已经处于可交易的日 K 上升趋势。两者职责不同，必须拆开。

后续所有趋势交易信号都建立在 `trend universe` 之上：

- 没进入趋势股池，不允许产生趋势交易信号
- 进入趋势股池，不代表一定是买点

### 3.2 先验证统计优势，再制定止盈止损

回测第一阶段不预设止盈止损，而是先做固定持有周期统计。后续再根据胜率、收益分布、回撤分布反推止盈止损和持仓规则。

### 3.3 区分“信号质量统计”和“可执行回测”

第一阶段先支持 `当日收盘确认并按当日收盘入场` 的回测口径，用于快速验证信号质量。报告中必须明确标注这不是严格可执行口径。

后续第二阶段应补充 `次日开盘入场` 对照版本，用于评估前视偏差对收益的影响。

## 4. 总体架构

新增一条独立链路：

`daily data -> trend universe -> trend signals -> signal backtest -> portfolio backtest -> reports`

建议新增模块：

- `trend_universe.py`
  - 趋势股硬过滤
  - `trend_score` 计算
- `trend_signals.py`
  - `breakout` / `pullback` 信号生成
  - `entry_score` 计算
- `trend_backtest.py`
  - 单信号固定持有回测
  - 组合层前 N 等权固定持有回测
- `trend_reporting.py`
  - 明细表、汇总表、净值曲线、JSON 摘要输出

必要时可补一个共享配置模块，例如 `trend_config`，但不修改现有 `type1~type4` 结构来承载趋势交易语义。

## 5. 趋势股定义

趋势股定义采用“双层法”：

- 第一层：硬规则过滤，得到 `trend universe`
- 第二层：对 `trend universe` 内股票计算 `trend_score`

### 5.1 硬规则过滤

一只股票进入 `trend universe`，至少满足以下条件：

- 均线多头：`MA20 > MA60 > MA120`
- 均线向上：`MA20` 和 `MA60` 均高于各自 `10` 个交易日前的值
- 价格位置：收盘价位于 `MA20` 上方，且不低于 `MA60`
- 趋势强度：近 `60` 个交易日收益达到最低门槛
- 趋势质量：近 `60` 个交易日最大回撤不超过上限
- 流动性：近 `20` 个交易日平均成交额达到门槛
- 基础过滤：排除 `ST`、长期停牌、明显异常流动性标的

默认建议参数：

- `min_return_60d = 0.15`
- `max_drawdown_60d = 0.18`
- `min_avg_amount_20d = 100000000`

这些参数应进入配置文件，不写死在代码中。

### 5.2 趋势评分 `trend_score`

对于进入 `trend universe` 的股票，按四个维度计算评分：

- 方向分：均线多头结构、价格相对 `MA20/MA60/MA120` 的位置
- 强度分：`20/60/120` 日收益、距 `60/120` 日新高的距离
- 质量分：`20/60` 日最大回撤、波动率、上涨日占比、趋势效率
- 流动性分：成交额水平和稳定性

`trend_score` 用于：

- 趋势股池排序
- 作为买点评分的基础输入
- 组合回测中的当日选股排序

`trend_score` 必须是可解释的。报表中至少要保留总分和主要分项，避免只输出黑箱结果。

## 6. 入场信号定义

趋势交易信号只在 `trend universe` 内生成。

### 6.1 突破信号 `breakout`

`breakout` 用于识别上升趋势中的再启动。

建议条件：

- 股票已进入 `trend universe`
- 最近 `20~40` 个交易日存在相对清晰的平台整理
- 平台振幅不超过上限，避免高波动震荡股误判为平台
- 当日收盘突破平台高点，或接近 `60/120` 日新高
- 当日成交量相对 `20` 日均量放大
- 突破后离平台高点不能过远，避免追高过度

建议信号字段：

- `signal_type = breakout`
- `platform_high`
- `platform_range_pct`
- `distance_to_breakout_pct`
- `volume_ratio_20`
- `trend_score`
- `entry_score`
- `trigger_reason`

### 6.2 回踩信号 `pullback`

`pullback` 用于识别上升趋势中的回撤再起。

建议条件：

- 股票已进入 `trend universe`
- `MA20/MA60` 继续上行，趋势未破坏
- 从阶段高点出现受控回撤
- 回踩位置接近 `MA20`，必要时可扩展到 `MA60`
- 回踩期间成交量收缩
- 当日出现企稳特征，例如重新站回短均线、阳线、下影线等

建议信号字段：

- `signal_type = pullback`
- `distance_to_ma20`
- `distance_to_ma60`
- `drawdown_from_recent_high`
- `volume_contraction_ratio`
- `trend_score`
- `entry_score`
- `trigger_reason`

### 6.3 信号冲突处理

同一只股票在同一交易日原则上只保留一个主信号，避免重复入场统计。

第一阶段建议处理方式：

- 若同日同时命中 `breakout` 和 `pullback`，按 `entry_score` 高者保留
- 若 `entry_score` 相同，则优先保留 `pullback`

选择该规则的原因是：`pullback` 通常风险预算更清晰，适合先做保守去重。该优先级应写入代码和测试，不依赖隐式排序。

## 7. 回测规则

### 7.1 单信号回测

单信号回测用于验证信号本身的统计优势。

第一阶段统一口径：

- 信号在当日收盘确认
- 入场价使用当日收盘价
- 对每条信号分别统计持有 `5/10/20/40` 个交易日后的结果

每条信号至少输出：

- `entry_date`
- `entry_price`
- `holding_days`
- `exit_date`
- `exit_price`
- `return_pct`
- `max_upside_pct`
- `max_drawdown_pct`
- `min_return_pct`

汇总统计至少包括：

- 样本数
- 胜率
- 平均收益
- 中位数收益
- 收益标准差
- 平均最大回撤
- 平均最大上涨

统计结果至少按以下维度切分：

- `signal_type`
- `holding_days`
- 全部样本 / 高 `trend_score` / 高 `entry_score`

### 7.2 组合回测

组合回测用于回答“如果每天按规则只买前 N 只，会得到怎样的资金曲线”。

第一阶段基线规则：

- 每个交易日生成全部趋势交易信号
- 按综合评分排序
- 只选当天前 `N` 只
- 等权分配资金
- 固定持有 `5/10/20/40` 个交易日
- 到期卖出，不做主动止盈止损

默认排序建议使用：

- `portfolio_rank_score = entry_score * 0.6 + trend_score * 0.4`

默认建议支持 `N = 3/5/10`，写成命令参数。

组合回测输出至少包括：

- 净值曲线
- 回撤曲线
- 年化收益近似值
- 最大回撤
- 胜率
- 平均持仓数
- 模式占比：`breakout` / `pullback`

### 7.3 前视偏差声明

第一阶段允许 `当日收盘入场`，但所有报表必须附带说明：

- 该版本用于信号质量研究
- 不代表严格可执行收益
- 后续应补 `次日开盘入场` 版本作对照

## 8. CLI 与报表

建议新增独立命令：

- `trend-universe`
  - 生成指定日期的趋势股池和趋势评分
- `trend-signals`
  - 识别指定日期的 `breakout` / `pullback` 信号
- `backtest-signals`
  - 运行单信号固定持有回测
- `backtest-portfolio`
  - 运行组合层前 N 等权固定持有回测

建议新增输出目录：

- `reports/trend_universe/`
- `reports/trend_signals/`
- `reports/backtests/signals/`
- `reports/backtests/portfolio/`

建议文件类型：

- 明细 CSV
- 汇总 CSV
- JSON 摘要

其中 JSON 摘要用于自动读取关键结论，CSV 用于人工复盘和二次分析。

## 9. 配置

趋势交易参数需要独立配置，不与现有 `strategies.type1~type4` 混用。

建议新增配置段：

- `trend_universe`
- `trend_signals.breakout`
- `trend_signals.pullback`
- `trend_backtest`

最少应参数化以下内容：

- 趋势股池的均线周期、涨幅门槛、最大回撤门槛、流动性门槛
- 突破平台窗口、平台振幅上限、放量阈值、突破距离阈值
- 回踩靠近均线的阈值、回撤阈值、缩量阈值、企稳规则
- 回测持有周期列表
- 组合回测的前 `N`、排序权重

## 10. 测试要求

第一阶段至少补充以下测试：

- 趋势股硬过滤测试
  - 均线多头、趋势强度、最大回撤、流动性过滤是否正确
- `trend_score` 测试
  - 核心分项和总分是否稳定、可解释
- `breakout` 信号测试
  - 平台突破、量能放大、距离过滤边界是否正确
- `pullback` 信号测试
  - 回踩均线、缩量、企稳边界是否正确
- 信号冲突去重测试
  - 同日双信号只保留一个主信号
- 单信号回测测试
  - `5/10/20/40` 日固定持有收益和区间指标计算正确
- 组合回测测试
  - 当日排序、前 `N` 选股、持有到期卖出逻辑正确
- 无未来函数测试
  - 信号日和评分不能引用未来数据

## 11. 分阶段实施

### Phase 1

- 建立 `trend universe`
- 产出 `trend_score`
- 生成 `breakout` / `pullback` 信号
- 完成单信号固定持有回测
- 完成组合前 `N` 等权固定持有回测
- 输出报表和基础测试

### Phase 2

- 增加 `次日开盘入场` 回测口径
- 比较两种入场口径的偏差
- 基于统计结果推导止盈止损候选规则

### Phase 3

- 将固定持有规则扩展为条件退出
- 评估止盈止损、趋势退出、仓位管理的增量价值

## 12. 验收标准

本设计在第一阶段完成后，应满足以下验收标准：

- 趋势股池与现有 `pattern` 体系完全解耦
- 可以在任意指定交易日输出 `trend universe`
- 可以在 `trend universe` 上稳定输出 `breakout` / `pullback` 信号
- 可以生成 `5/10/20/40` 日固定持有的单信号回测结果
- 可以生成组合层前 `N` 等权固定持有回测结果
- 报表能区分趋势股质量、买点质量、组合层表现
- 测试能覆盖边界条件和未来函数风险
