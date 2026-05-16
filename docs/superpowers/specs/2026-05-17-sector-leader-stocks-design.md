# 板块龙头票识别设计

日期：2026-05-17

## 1. 目标

新增一个板块龙头票分析模块，用本地行业/概念映射和本地日线数据，为每个行业或概念板块找出过去两年内的两类龙头：

1. 长期绝对龙头：过去两年里长期代表该板块的核心票。
2. 板块波段领涨龙头：过去两年板块上涨波段中，经常率先启动并显著跑赢板块的带队票。

该模块第一阶段只生成独立 CSV 报告，不接入 `daily-screening`、`watchlist` 或选股排序。使用方式是先辅助判断“一个主线板块应该重点看哪些股票”，后续验证稳定后再考虑接入盘后和盘中表格。

## 2. 非目标

初版不做以下事情：

- 不用分钟线或实时行情。
- 不预测未来龙头，只衡量历史龙头属性。
- 不自动推荐买入。
- 不把龙头分数直接加入现有 P1/P2/P4/P8 排序。
- 不联网刷新板块成分，直接复用 `data/sector_membership/stock_sector_membership.csv`。
- 不处理历史成分变更，初版使用当前同花顺成分映射回构两年历史，因此报告中必须标注存在成分存活偏差。

## 3. 输入数据

### 3.1 板块成分

使用现有文件：

```text
data/sector_membership/stock_sector_membership.csv
```

字段要求：

- `symbol`
- `name`
- `sector_type`
- `sector_name`
- `sector_label`

读取时继续使用现有忽略规则，排除选股意义较弱的伪题材板块，例如 `2025年报预增`、`2026一季报预增` 和名称包含 `同花顺` 的板块。

### 3.2 股票日线

使用现有目录：

```text
data/daily/{symbol}.parquet
```

最少需要字段：

- `trade_date`
- `open`
- `close`
- `high`
- `low`
- `volume`
- `amount`

计算窗口默认使用最近 `504` 个交易日，近似两年。若某只股票有效交易日少于 `180` 日，不参与该板块龙头计算。

### 3.3 板块指数

初版不使用外部板块指数，而是用当前板块成分的本地日线合成等权板块指数：

```text
sector_return[t] = mean(member_return[t])
sector_close_index[t] = cumprod(1 + sector_return[t])
```

若某日有效成分少于 `min_valid_members`，该日不参与板块指数计算。默认 `min_valid_members = 5`。

## 4. 输出文件

命令输出目录：

```text
reports/sectors/
```

默认输出：

```text
sector_leaders_YYYY-MM-DD.csv
```

每一行是一只股票在一个板块内的一类龙头结果。若同一只股票同时属于多个概念，则可以在多个板块中分别出现。

核心字段：

| 字段 | 含义 |
|---|---|
| `trade_date` | 分析日期 |
| `sector_type` | `industry` 或 `concept` |
| `sector_name` | 板块名称 |
| `sector_label` | 板块标识 |
| `symbol` | 股票代码 |
| `name` | 股票名称 |
| `leader_type` | `long_term` 或 `swing` |
| `leader_rank` | 该榜单内排名 |
| `leader_score` | 当前榜单百分制分数 |
| `long_term_leader_score` | 长期绝对龙头分 |
| `swing_leader_score` | 波段领涨龙头分 |
| `combined_leader_score` | 综合参考分 |
| `is_dual_leader` | 是否同时进入长期 Top5 和波段 Top5 |
| `two_year_return_pct` | 两年累计涨幅 |
| `excess_return_vs_sector_pct` | 两年相对板块超额收益 |
| `outperform_sector_ratio` | 跑赢板块交易日比例 |
| `max_drawdown_pct` | 两年最大回撤 |
| `amount_share_pct` | 两年成交额占该板块成交额比例 |
| `swing_count` | 识别到的板块上涨波段数量 |
| `swing_lead_count` | 该股进入波段领涨 TopN 的次数 |
| `swing_lead_ratio` | `swing_lead_count / swing_count` |
| `avg_swing_excess_return_pct` | 波段内平均相对板块超额收益 |

此外输出一个简表：

```text
sector_leaders_summary_YYYY-MM-DD.csv
```

该文件每个板块一行，列出长期 Top5、波段 Top5、双重龙头，便于人工快速浏览。

## 5. 长期绝对龙头分

长期绝对龙头分用于回答：

```text
过去两年里，谁是这个板块真正绕不开的核心票？
```

它强调长期涨幅、相对板块持续超额、流动性和回撤质量，不强调某一次短线波段。

### 5.1 子指标

对同一板块内全部有效成分做截面百分位归一化，分数范围为 `0-100`，越高越好。

| 子指标 | 权重 | 说明 |
|---|---:|---|
| `return_2y_score` | 25% | 两年累计收益排名 |
| `excess_2y_score` | 25% | 两年累计收益减板块等权收益 |
| `outperform_ratio_score` | 15% | 日收益跑赢板块日收益的比例 |
| `new_high_score` | 10% | 两年内创新高次数和距离两年高点的综合表现 |
| `liquidity_share_score` | 15% | 两年成交额占该板块成交额比例 |
| `drawdown_quality_score` | 10% | 最大回撤越小、从高点修复越强越高 |

长期绝对龙头分：

```text
long_term_leader_score =
  0.25 * return_2y_score
+ 0.25 * excess_2y_score
+ 0.15 * outperform_ratio_score
+ 0.10 * new_high_score
+ 0.15 * liquidity_share_score
+ 0.10 * drawdown_quality_score
```

### 5.2 创新高分

创新高分由两部分组成：

```text
new_high_score = 0.6 * new_high_count_score + 0.4 * distance_to_high_score
```

- `new_high_count_score`：过去两年收盘价创 `120` 日新高的次数，在板块内做百分位。
- `distance_to_high_score`：当前收盘价距离两年最高收盘价越近越高。

### 5.3 回撤质量分

回撤质量分由两部分组成：

```text
drawdown_quality_score = 0.7 * max_drawdown_score + 0.3 * recovery_score
```

- `max_drawdown_score`：两年最大回撤越小越高。
- `recovery_score`：当前价格相对最大回撤后低点的修复程度越强越高。

## 6. 板块波段领涨龙头分

波段领涨龙头分用于回答：

```text
当板块真正启动时，谁经常先涨、涨得更快、涨得更多？
```

它只在板块上涨波段内计算，不用完整两年日线平均掉短期带队能力。

### 6.1 板块上涨波段识别

使用合成板块等权指数识别上涨波段。满足任一条件即可形成候选波段：

1. 从局部低点到后续局部高点涨幅 `>= 8%`。
2. 任意连续 `20` 个交易日板块涨幅 `>= 5%`。

波段约束：

- 最短波段长度：`5` 个交易日。
- 最长波段长度：`60` 个交易日。
- 两个波段重叠时，保留涨幅更高的波段。
- 若两年内有效波段少于 `2` 个，则仍输出结果，但 `swing_count` 会如实记录，人工解读时降低置信度。

### 6.2 单个波段内的领涨分

对每个上涨波段，计算每只成分股：

| 子指标 | 权重 | 说明 |
|---|---:|---|
| `early_return_score` | 25% | 波段前 5 日涨幅排名，衡量是否先启动 |
| `early_excess_score` | 20% | 波段前 5 日相对板块超额 |
| `total_swing_return_score` | 25% | 整个波段涨幅排名 |
| `total_swing_excess_score` | 20% | 整个波段相对板块超额 |
| `volume_expansion_score` | 10% | 波段成交额相对波段前 20 日均值放大程度 |

单波段领涨分：

```text
swing_event_score =
  0.25 * early_return_score
+ 0.20 * early_excess_score
+ 0.25 * total_swing_return_score
+ 0.20 * total_swing_excess_score
+ 0.10 * volume_expansion_score
```

### 6.3 两年波段领涨分

对每只股票聚合全部上涨波段：

```text
swing_leader_score =
  0.35 * avg_top_swing_event_score
+ 0.25 * swing_lead_ratio_score
+ 0.20 * avg_swing_excess_score
+ 0.10 * best_swing_rank_score
+ 0.10 * recent_swing_score
```

说明：

- `avg_top_swing_event_score`：该股在所有波段中最好的若干次分数均值，默认取 Top3。
- `swing_lead_ratio_score`：进入单波段 Top10 的次数占比。
- `avg_swing_excess_score`：波段内平均超额收益。
- `best_swing_rank_score`：历史最好波段排名，排名越靠前越高。
- `recent_swing_score`：最近一个有效上涨波段中的表现，避免只识别很久以前的老龙头。

## 7. 综合参考分和标签

虽然默认输出两张独立榜单，但为了快速排序，额外给综合参考分：

```text
combined_leader_score = 0.55 * long_term_leader_score + 0.45 * swing_leader_score
```

标签规则：

| 标签 | 规则 |
|---|---|
| `长期核心` | 长期绝对龙头榜 Top5 |
| `波段先锋` | 波段领涨龙头榜 Top5 |
| `双重龙头` | 同时进入长期 Top5 和波段 Top5 |
| `长期核心候选` | 长期绝对龙头榜 Top10，但未进 Top5 |
| `波段活跃候选` | 波段领涨龙头榜 Top10，但未进 Top5 |

`双重龙头` 不代表当前可以买，只说明该股在板块历史表现中同时具备长期代表性和波段带队能力。

## 8. 命令设计

新增命令：

```powershell
python -m stocks_analyzer --project-root . analyze-sector-leaders --date 2026-05-15
```

可选参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--lookback-days` | `504` | 历史窗口交易日数 |
| `--min-history-days` | `180` | 股票最少有效日线数 |
| `--min-valid-members` | `5` | 板块指数每日最少有效成分数 |
| `--top-n` | `10` | 每类榜单每个板块输出数量 |
| `--sector-type` | `all` | 可选 `all`、`industry`、`concept` |
| `--sector-name` | 空 | 只分析名称匹配的板块，用于调试 |
| `--progress` | false | 显示进度 |

初版只做分析命令，不自动挂到 `daily-screening`。如果后续人工确认榜单有用，再把最近一次 `sector_leaders_YYYY-MM-DD.csv` 的结果展示到 `选股.md` 或工具中。

## 9. 数据质量与边界处理

1. 成分存活偏差：初版使用当前板块成分回构历史，不能代表历史真实成分。报告中保留 `membership_updated_at` 或读取源文件修改时间，提醒使用者。
2. 复权一致性：沿用本项目当前日线复权口径，不在模块内部二次复权。
3. 停牌和新股：有效交易日不足 `min_history_days` 的股票跳过。
4. 极端值：收益率和成交额放大指标在截面归一化前做 winsorize，默认剪裁到 `1%` 和 `99%` 分位。
5. 板块过小：有效成分少于 `min_valid_members` 的板块跳过，并记录在 `skipped_sector_leaders_YYYY-MM-DD.csv`。
6. 股票多板块归属：同一股票可在多个板块内分别计算，排名只在各自板块内部比较。
7. 名称编码：CSV 使用 `utf-8-sig` 写出，便于 Excel 打开。

## 10. 验证方式

第一阶段验证不做机器学习，只做人工和统计核对：

1. 抽查典型板块，例如商业航天、机器人、钠离子电池、CPO、半导体。
2. 检查长期绝对龙头榜是否符合市场常识。
3. 检查波段领涨榜是否捕捉到历史上涨阶段中的弹性票。
4. 检查 `双重龙头` 是否数量合理，不应每个板块都有很多。
5. 对比 P9/P10 高分板块：如果某板块买入分高，但龙头票全部弱，说明该板块可能只是低位反抽而非强主线。

后续如需量化验证，可以测试：

```text
板块进入 P9/P10 TopN 后，优先买入该板块长期/波段龙头是否优于随机买入该板块成分。
```

该验证不纳入初版实现。

## 11. 与现有流程关系

初版关系如下：

- 依赖 `update-sector-membership` 维护的板块成分。
- 依赖 `data/daily` 的本地股票日线。
- 不依赖 P1/P2/P4/P8/P9/P10 模型。
- 不改变 `watchlist` 入选逻辑。
- 不改变 `intraday-screening`。
- 不改变 `选股.md` 写作指南。

后续可选接入路径：

1. 在 `sector_pullback_metrics`、P9、P10 结果旁边展示每个板块的双重龙头。
2. 在 `watchlist` 中为候选股增加 `sector_leader_tags`。
3. 在 `选股.md` 中，当某只候选股属于当前主线且也是该板块龙头时，作为人工加分理由。

## 12. 初版完成标准

初版实现完成时应满足：

1. `analyze-sector-leaders` 命令可运行。
2. 生成 `sector_leaders_YYYY-MM-DD.csv`。
3. 生成 `sector_leaders_summary_YYYY-MM-DD.csv`。
4. 对每个板块分别输出长期绝对龙头 TopN 和波段领涨龙头 TopN。
5. 输出跳过板块原因文件。
6. 单元测试覆盖：
   - 长期龙头分计算。
   - 板块上涨波段识别。
   - 波段领涨分聚合。
   - 小板块和历史不足股票跳过。
   - CLI 参数解析。

